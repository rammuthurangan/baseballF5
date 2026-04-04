"""
Prediction CLI for F5 runs model.

Usage:
    python -m f5_model.model.predict \
        --pitcher "Corbin Burnes" \
        --lineup "Juan Soto,Aaron Judge,Giancarlo Stanton,Anthony Rizzo,..." \
        --date 2025-09-15

Outputs:
    - Predicted F5 runs (lambda)
    - Full Poisson probability distribution
    - Over/under probabilities for common lines
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson

try:
    from pybaseball import playerid_lookup
except ImportError:
    playerid_lookup = None

from f5_model.utils.statcast_pull import get_processed_data_dir
from f5_model.utils.constants import LINEUP_WEIGHTS, PARK_FACTORS
from f5_model.model.train import get_models_dir

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def lookup_player_id(name: str) -> Optional[int]:
    """
    Look up a player's MLB ID from their name.

    Args:
        name: Player name (e.g., "Corbin Burnes" or "Burnes, Corbin")

    Returns:
        MLB player ID or None if not found
    """
    if playerid_lookup is None:
        logger.warning("pybaseball not available for player lookup")
        return None

    # Parse name
    if ',' in name:
        parts = name.split(',')
        last = parts[0].strip()
        first = parts[1].strip() if len(parts) > 1 else ''
    else:
        parts = name.strip().split()
        if len(parts) >= 2:
            first = parts[0]
            last = ' '.join(parts[1:])
        else:
            first = ''
            last = name

    try:
        result = playerid_lookup(last, first)
        if len(result) > 0:
            # Get the most recent player (highest key_mlbam)
            result = result.sort_values('key_mlbam', ascending=False)
            return int(result.iloc[0]['key_mlbam'])
    except Exception as e:
        logger.warning(f"Error looking up {name}: {e}")

    return None


def load_model_and_features():
    """Load the trained model and feature data."""
    model_path = get_models_dir() / "f5_runs_model.pkl"
    model = joblib.load(model_path)

    feature_path = get_models_dir() / "feature_names.txt"
    with open(feature_path) as f:
        feature_names = [line.strip() for line in f.readlines()]

    return model, feature_names


def get_pitcher_features(pitcher_id: int, date: str) -> Dict:
    """
    Get pitcher features for a given date.

    First tries to load from pre-computed features.
    Falls back to returning NaN values if not found.

    Args:
        pitcher_id: MLB pitcher ID
        date: Date string (YYYY-MM-DD)

    Returns:
        Dictionary of pitcher features
    """
    features_path = get_processed_data_dir() / "pitcher_features.parquet"

    if features_path.exists():
        df = pd.read_parquet(features_path)

        # Find most recent features for this pitcher before/on this date
        pitcher_df = df[
            (df['starter'] == pitcher_id) &
            (df['game_date'] <= date)
        ].sort_values('game_date', ascending=False)

        if len(pitcher_df) > 0:
            row = pitcher_df.iloc[0]
            features = {}
            for col in df.columns:
                if col not in ['game_pk', 'starter', 'game_date']:
                    features[f'p_{col}'] = row[col]
            return features

    # Return empty features if not found
    logger.warning(f"No pitcher features found for ID {pitcher_id}")
    return {}


def get_batter_features(batter_id: int, date: str, vs_hand: str) -> Dict:
    """
    Get batter features for a given date and pitcher handedness.

    Args:
        batter_id: MLB batter ID
        date: Date string
        vs_hand: Pitcher handedness ('L' or 'R')

    Returns:
        Dictionary of batter features
    """
    features_path = get_processed_data_dir() / "batter_features.parquet"

    if features_path.exists():
        df = pd.read_parquet(features_path)

        # Find most recent features for this batter vs this hand
        batter_df = df[
            (df['batter'] == batter_id) &
            (df['vs_hand'] == vs_hand) &
            (df['game_date'] <= date)
        ].sort_values('game_date', ascending=False)

        if len(batter_df) > 0:
            row = batter_df.iloc[0]
            return row.to_dict()

    logger.warning(f"No batter features found for ID {batter_id} vs {vs_hand}HP")
    return {}


def aggregate_lineup(
    lineup_ids: List[int],
    pitcher_hand: str,
    date: str
) -> Dict:
    """
    Aggregate batter features for a lineup.

    Args:
        lineup_ids: List of batter IDs in batting order
        pitcher_hand: Pitcher handedness
        date: Game date

    Returns:
        Dictionary of aggregated lineup features
    """
    feature_cols = ['woba', 'xwoba', 'k_rate', 'bb_rate', 'iso',
                    'barrel_rate', 'avg_exit_velo', 'ops']

    weighted_sums = {f'lineup_{col}': 0.0 for col in feature_cols}
    weighted_counts = {f'lineup_{col}': 0.0 for col in feature_cols}

    weights = LINEUP_WEIGHTS[:len(lineup_ids)]
    batters_found = 0

    for i, batter_id in enumerate(lineup_ids):
        weight = weights[i] if i < len(weights) else weights[-1]

        batter_features = get_batter_features(batter_id, date, pitcher_hand)

        if not batter_features:
            continue

        batters_found += 1

        for col in feature_cols:
            if col in batter_features and pd.notna(batter_features[col]):
                weighted_sums[f'lineup_{col}'] += weight * batter_features[col]
                weighted_counts[f'lineup_{col}'] += weight

    # Compute weighted averages
    features = {}
    for col in feature_cols:
        key = f'lineup_{col}'
        if weighted_counts[key] > 0:
            features[key] = weighted_sums[key] / weighted_counts[key]
        else:
            features[key] = np.nan

    features['lineup_batters_found'] = batters_found

    return features


def build_feature_vector(
    pitcher_features: Dict,
    lineup_features: Dict,
    starter_is_home: bool,
    park: str,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Build the full feature vector for prediction.

    Args:
        pitcher_features: Pitcher feature dict
        lineup_features: Lineup feature dict
        starter_is_home: Whether pitcher is home team
        park: Park code for park factor
        feature_names: Expected feature column names

    Returns:
        DataFrame with single row of features
    """
    features = {
        'starter_is_home': int(starter_is_home),
        'park_factor': PARK_FACTORS.get(park, 1.0),
        **pitcher_features,
        **lineup_features
    }

    # Create DataFrame with correct column order
    df = pd.DataFrame([features])

    # Ensure all expected columns exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan

    # Select only the expected columns in order
    df = df[feature_names]

    return df


def predict_f5_runs(
    model,
    feature_names: List[str],
    pitcher_id: int,
    pitcher_hand: str,
    lineup_ids: List[int],
    date: str,
    starter_is_home: bool = True,
    park: str = "NYY"
) -> Dict:
    """
    Make an F5 runs prediction.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        pitcher_id: Pitcher MLB ID
        pitcher_hand: 'L' or 'R'
        lineup_ids: List of opposing batter IDs
        date: Game date
        starter_is_home: Whether pitcher is home
        park: Park code

    Returns:
        Dictionary with prediction results
    """
    # Get features
    pitcher_features = get_pitcher_features(pitcher_id, date)
    lineup_features = aggregate_lineup(lineup_ids, pitcher_hand, date)

    # Build feature vector
    X = build_feature_vector(
        pitcher_features, lineup_features,
        starter_is_home, park, feature_names
    )

    # Make prediction (lambda for Poisson)
    lambda_pred = model.predict(X)[0]

    # Compute Poisson distribution
    probs = {k: float(poisson.pmf(k, lambda_pred)) for k in range(11)}

    # Over/under probabilities
    over_under = {}
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        p_over = 1 - poisson.cdf(int(line), lambda_pred)
        over_under[f'over_{line}'] = float(p_over)
        over_under[f'under_{line}'] = float(1 - p_over)

    return {
        'predicted_runs': float(lambda_pred),
        'probabilities': probs,
        'over_under': over_under,
        'lineup_batters_found': lineup_features.get('lineup_batters_found', 0)
    }


def format_output(result: Dict, pitcher_name: str, lineup_names: List[str]) -> str:
    """Format prediction results for display."""
    lines = []
    lines.append("=" * 60)
    lines.append("F5 RUNS PREDICTION")
    lines.append("=" * 60)

    lines.append(f"\nPitcher: {pitcher_name}")
    lines.append(f"Lineup ({result['lineup_batters_found']}/9 found): {', '.join(lineup_names[:3])}...")

    lines.append(f"\n{'='*60}")
    lines.append(f"PREDICTED F5 RUNS: {result['predicted_runs']:.2f}")
    lines.append(f"{'='*60}")

    lines.append("\nProbability Distribution:")
    lines.append("-" * 30)
    for runs in range(8):
        prob = result['probabilities'].get(runs, 0)
        bar = "█" * int(prob * 40)
        lines.append(f"  {runs} runs: {prob:5.1%} {bar}")

    lines.append("\nOver/Under Probabilities:")
    lines.append("-" * 30)
    for line in [1.5, 2.5, 3.5]:
        p_over = result['over_under'][f'over_{line}']
        p_under = result['over_under'][f'under_{line}']
        lines.append(f"  {line}: Over {p_over:5.1%} | Under {p_under:5.1%}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Predict F5 runs allowed by a starting pitcher"
    )
    parser.add_argument(
        "--pitcher", "-p",
        required=True,
        help="Pitcher name (e.g., 'Corbin Burnes')"
    )
    parser.add_argument(
        "--pitcher-hand",
        choices=['L', 'R'],
        help="Pitcher handedness (auto-detected if not specified)"
    )
    parser.add_argument(
        "--lineup", "-l",
        required=True,
        help="Comma-separated list of batter names in order"
    )
    parser.add_argument(
        "--date", "-d",
        required=True,
        help="Game date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--home",
        action="store_true",
        help="Pitcher is home team"
    )
    parser.add_argument(
        "--park",
        default="NYY",
        help="Park code for park factor (default: NYY)"
    )
    parser.add_argument(
        "--pitcher-id",
        type=int,
        help="Pitcher MLB ID (skip name lookup)"
    )
    parser.add_argument(
        "--lineup-ids",
        help="Comma-separated batter IDs (skip name lookup)"
    )

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, feature_names = load_model_and_features()

    # Get pitcher ID
    if args.pitcher_id:
        pitcher_id = args.pitcher_id
    else:
        print(f"Looking up pitcher: {args.pitcher}...")
        pitcher_id = lookup_player_id(args.pitcher)
        if pitcher_id is None:
            print(f"ERROR: Could not find pitcher '{args.pitcher}'")
            print("Try using --pitcher-id to specify the MLB ID directly")
            return

    print(f"  Pitcher ID: {pitcher_id}")

    # Get lineup IDs
    lineup_names = [name.strip() for name in args.lineup.split(',')]

    if args.lineup_ids:
        lineup_ids = [int(x) for x in args.lineup_ids.split(',')]
    else:
        print("Looking up lineup...")
        lineup_ids = []
        for name in lineup_names:
            player_id = lookup_player_id(name)
            if player_id:
                lineup_ids.append(player_id)
                print(f"  {name}: {player_id}")
            else:
                print(f"  {name}: NOT FOUND")

    if len(lineup_ids) == 0:
        print("ERROR: No batters found in lineup")
        return

    # Default pitcher hand to R if not specified
    pitcher_hand = args.pitcher_hand or 'R'

    # Make prediction
    print(f"\nMaking prediction...")
    result = predict_f5_runs(
        model=model,
        feature_names=feature_names,
        pitcher_id=pitcher_id,
        pitcher_hand=pitcher_hand,
        lineup_ids=lineup_ids,
        date=args.date,
        starter_is_home=args.home,
        park=args.park
    )

    # Display results
    print(format_output(result, args.pitcher, lineup_names))


if __name__ == "__main__":
    main()
