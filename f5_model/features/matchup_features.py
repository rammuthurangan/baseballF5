"""
Matchup feature engineering for F5 runs prediction model.

Combines pitcher features with aggregated lineup features to create
the final training dataset.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from f5_model.utils.statcast_pull import load_all_raw_data, get_processed_data_dir
from f5_model.utils.f5_processor import filter_f5
from f5_model.utils.constants import LINEUP_WEIGHTS, PARK_FACTORS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_lineup_from_game(game_df: pd.DataFrame, batting_team_is_home: bool) -> List[int]:
    """
    Extract the batting order (lineup) for a team from game data.

    The lineup is determined by the order in which batters first appear
    in the game. We look at the first inning to get the top of the order.

    Args:
        game_df: DataFrame of all pitches in a game
        batting_team_is_home: True if we want the home team's lineup

    Returns:
        List of batter IDs in batting order (up to 9)
    """
    # Filter to the batting team's plate appearances
    if batting_team_is_home:
        # Home team bats in bottom of innings
        team_pitches = game_df[game_df['inning_topbot'] == 'Bot']
    else:
        # Away team bats in top of innings
        team_pitches = game_df[game_df['inning_topbot'] == 'Top']

    if len(team_pitches) == 0:
        return []

    # Sort by at-bat number to get batting order
    team_pitches = team_pitches.sort_values(['at_bat_number', 'pitch_number'])

    # Get unique batters in order of first appearance
    seen = set()
    lineup = []
    for batter in team_pitches['batter']:
        if batter not in seen:
            seen.add(batter)
            lineup.append(batter)
            if len(lineup) >= 9:
                break

    return lineup


def aggregate_lineup_features(
    lineup: List[int],
    pitcher_hand: str,
    batter_features_df: pd.DataFrame,
    game_date: str
) -> Dict:
    """
    Aggregate batter features for a lineup into a single feature vector.

    Args:
        lineup: List of 9 batter IDs in batting order
        pitcher_hand: 'L' or 'R' - the opposing pitcher's handedness
        batter_features_df: DataFrame with batter features
        game_date: Date of the game (to get features as of that date)

    Returns:
        Dictionary of aggregated lineup features
    """
    if len(lineup) == 0:
        return {}

    # Get weights for each lineup position
    weights = LINEUP_WEIGHTS[:len(lineup)]
    total_weight = sum(weights)

    # Features to aggregate
    feature_cols = ['woba', 'xwoba', 'k_rate', 'bb_rate', 'iso',
                    'barrel_rate', 'avg_exit_velo', 'ops']

    # Initialize aggregated values
    weighted_sums = {f'lineup_{col}': 0.0 for col in feature_cols}
    weighted_counts = {f'lineup_{col}': 0.0 for col in feature_cols}

    batters_found = 0

    for i, batter_id in enumerate(lineup):
        weight = weights[i] if i < len(weights) else weights[-1]

        # Look up this batter's features for this date and pitcher hand
        batter_row = batter_features_df[
            (batter_features_df['batter'] == batter_id) &
            (batter_features_df['game_date'] == game_date) &
            (batter_features_df['vs_hand'] == pitcher_hand)
        ]

        if len(batter_row) == 0:
            continue

        batter_row = batter_row.iloc[0]
        batters_found += 1

        for col in feature_cols:
            if col in batter_row and pd.notna(batter_row[col]):
                weighted_sums[f'lineup_{col}'] += weight * batter_row[col]
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


def build_training_data(
    targets_df: pd.DataFrame,
    pitcher_features_df: pd.DataFrame,
    batter_features_df: pd.DataFrame,
    raw_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build the final training dataset by combining all features.

    Each row = one pitcher's F5 outing, with:
    - Target: f5_runs_allowed
    - Pitcher features
    - Aggregated lineup features
    - Context features (home/away, park factor)

    Args:
        targets_df: F5 game targets with game_pk, starter, f5_runs_allowed
        pitcher_features_df: Pitcher features
        batter_features_df: Batter features with platoon splits
        raw_df: Raw Statcast data (for extracting lineups)

    Returns:
        Training DataFrame
    """
    logger.info("Building training data...")

    # Filter raw data to F5 and sort
    df = filter_f5(raw_df)
    df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])

    # Pre-extract lineups for all games
    logger.info("  Extracting lineups for all games...")
    game_lineups = {}

    for game_pk, game_df in df.groupby('game_pk'):
        # Extract both home and away lineups
        home_lineup = extract_lineup_from_game(game_df, batting_team_is_home=True)
        away_lineup = extract_lineup_from_game(game_df, batting_team_is_home=False)
        game_lineups[game_pk] = {
            'home': home_lineup,
            'away': away_lineup
        }

    logger.info(f"  Extracted lineups for {len(game_lineups):,} games")

    # Build training rows
    training_rows = []
    total = len(targets_df)

    logger.info(f"  Building {total:,} training rows...")

    for idx, target_row in enumerate(targets_df.itertuples()):
        if (idx + 1) % 1000 == 0:
            logger.info(f"    Progress: {idx + 1:,} / {total:,}")

        game_pk = target_row.game_pk
        starter = target_row.starter
        game_date = target_row.game_date
        starter_is_home = target_row.starter_is_home
        p_throws = target_row.p_throws
        f5_runs = target_row.f5_runs_allowed

        # Start with target and identifiers
        row = {
            'game_pk': game_pk,
            'game_date': game_date,
            'starter': starter,
            'f5_runs_allowed': f5_runs,
            'starter_is_home': int(starter_is_home),
        }

        # Add park factor
        if starter_is_home:
            park = target_row.home_team
        else:
            park = target_row.away_team
        row['park_factor'] = PARK_FACTORS.get(park, 1.0)

        # Get pitcher features
        pitcher_row = pitcher_features_df[
            (pitcher_features_df['game_pk'] == game_pk) &
            (pitcher_features_df['starter'] == starter)
        ]

        if len(pitcher_row) > 0:
            pitcher_row = pitcher_row.iloc[0]
            # Add all pitcher features except identifiers
            for col in pitcher_features_df.columns:
                if col not in ['game_pk', 'starter', 'game_date']:
                    row[f'p_{col}'] = pitcher_row[col]

        # Get opposing lineup and aggregate features
        if game_pk in game_lineups:
            # If pitcher is home, they face the away lineup (and vice versa)
            if starter_is_home:
                lineup = game_lineups[game_pk]['away']
            else:
                lineup = game_lineups[game_pk]['home']

            # Aggregate lineup features (vs this pitcher's handedness)
            lineup_features = aggregate_lineup_features(
                lineup, p_throws, batter_features_df, game_date
            )
            row.update(lineup_features)

        training_rows.append(row)

    training_df = pd.DataFrame(training_rows)
    logger.info(f"  Built {len(training_df):,} training rows with {len(training_df.columns)} columns")

    return training_df


def process_training_data() -> pd.DataFrame:
    """
    Main function to build and save the training dataset.
    """
    logger.info("=" * 60)
    logger.info("Building training dataset")
    logger.info("=" * 60)

    processed_dir = get_processed_data_dir()

    # Load all required data
    logger.info("\n1. Loading data...")

    logger.info("  Loading raw Statcast data...")
    raw_df = load_all_raw_data()

    logger.info("  Loading F5 targets...")
    targets_df = pd.read_parquet(processed_dir / "f5_game_targets.parquet")
    logger.info(f"    {len(targets_df):,} target games")

    logger.info("  Loading pitcher features...")
    pitcher_df = pd.read_parquet(processed_dir / "pitcher_features.parquet")
    logger.info(f"    {len(pitcher_df):,} pitcher feature rows")

    logger.info("  Loading batter features...")
    batter_df = pd.read_parquet(processed_dir / "batter_features.parquet")
    logger.info(f"    {len(batter_df):,} batter feature rows")

    # Build training data
    logger.info("\n2. Building training data...")
    training_df = build_training_data(targets_df, pitcher_df, batter_df, raw_df)

    # Save
    logger.info("\n3. Saving...")
    output_path = processed_dir / "training_data.parquet"
    training_df.to_parquet(output_path, index=False)
    logger.info(f"   Saved to {output_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total rows: {len(training_df):,}")
    logger.info(f"Total columns: {len(training_df.columns)}")

    # Target distribution
    logger.info(f"\nTarget (f5_runs_allowed) distribution:")
    logger.info(f"  Mean: {training_df['f5_runs_allowed'].mean():.2f}")
    logger.info(f"  Std: {training_df['f5_runs_allowed'].std():.2f}")

    # Missing values in key columns
    logger.info(f"\nMissing values in key columns:")
    key_cols = ['p_szn_f5_era', 'p_days_rest', 'lineup_woba', 'lineup_xwoba']
    for col in key_cols:
        if col in training_df.columns:
            missing = training_df[col].isna().sum()
            pct = missing / len(training_df) * 100
            logger.info(f"  {col}: {missing:,} ({pct:.1f}%)")

    return training_df


def verify_training_data() -> None:
    """Verify the training data file."""
    output_path = get_processed_data_dir() / "training_data.parquet"

    if not output_path.exists():
        print(f"ERROR: {output_path} not found")
        return

    df = pd.read_parquet(output_path)

    print("\n" + "=" * 60)
    print("TRAINING DATA VERIFICATION")
    print("=" * 60)

    print(f"\nShape: {df.shape}")

    print(f"\nColumns ({len(df.columns)}):")
    for col in sorted(df.columns):
        print(f"  - {col}")

    print(f"\nTarget distribution:")
    print(df['f5_runs_allowed'].value_counts().sort_index())

    print(f"\nSample pitcher features:")
    p_cols = [c for c in df.columns if c.startswith('p_')]
    print(df[p_cols[:5]].describe().round(3))

    print(f"\nSample lineup features:")
    l_cols = [c for c in df.columns if c.startswith('lineup_')]
    print(df[l_cols].describe().round(3))


if __name__ == "__main__":
    process_training_data()
