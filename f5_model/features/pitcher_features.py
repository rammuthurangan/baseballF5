"""
Pitcher feature engineering for F5 runs prediction model.

Computes season-to-date and rolling window features for each pitcher
on each game date, using only data available BEFORE that game (no leakage).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from f5_model.utils.statcast_pull import load_all_raw_data, get_processed_data_dir
from f5_model.utils.f5_processor import filter_f5
from f5_model.utils.constants import FIP_CONSTANT, PITCH_TYPES

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_season(date: str) -> int:
    """Extract season year from date string."""
    return int(str(date)[:4])


def compute_era(runs: float, innings: float) -> float:
    """Compute ERA (runs per 9 innings)."""
    if innings == 0:
        return np.nan
    return (runs / innings) * 9


def compute_fip(hr: int, bb: int, hbp: int, k: int, innings: float) -> float:
    """
    Compute FIP (Fielding Independent Pitching).
    FIP = ((13*HR) + (3*(BB+HBP)) - (2*K)) / IP + constant
    """
    if innings == 0:
        return np.nan
    return ((13 * hr) + (3 * (bb + hbp)) - (2 * k)) / innings + FIP_CONSTANT


def identify_event_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns identifying event types from the 'events' column.
    """
    df = df.copy()

    # Strikeouts
    df['is_strikeout'] = df['events'].isin(['strikeout', 'strikeout_double_play'])

    # Walks
    df['is_walk'] = df['events'].isin(['walk'])

    # Hit by pitch
    df['is_hbp'] = df['events'].isin(['hit_by_pitch'])

    # Home runs
    df['is_hr'] = df['events'].isin(['home_run'])

    # Hits (for WHIP, etc.)
    df['is_hit'] = df['events'].isin(['single', 'double', 'triple', 'home_run'])

    # Plate appearances (events that end an at-bat)
    df['is_pa'] = df['events'].notna()

    # At bats (PA minus walks, HBP, sac flies, sac bunts)
    non_ab_events = ['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt', 'sac_fly_double_play',
                     'catcher_interf', 'intent_walk']
    df['is_ab'] = df['events'].notna() & ~df['events'].isin(non_ab_events)

    # Ground balls
    # Approximate using launch_angle < 10 degrees on balls in play
    df['is_bip'] = df['launch_speed'].notna()  # Ball in play
    df['is_gb'] = df['is_bip'] & (df['launch_angle'] < 10)

    # Barrels (high exit velo + optimal launch angle)
    # Statcast barrel: exit velo >= 98 mph and launch angle 26-30 degrees (varies)
    df['is_barrel'] = (df['launch_speed'] >= 98) & (df['launch_angle'] >= 26) & (df['launch_angle'] <= 30)

    return df


def identify_pitch_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns for pitch-level outcomes (swings, whiffs, called strikes).
    """
    df = df.copy()

    # Swinging strike (whiff)
    whiff_desc = ['swinging_strike', 'swinging_strike_blocked', 'foul_tip']
    df['is_whiff'] = df['description'].isin(whiff_desc)

    # Called strike
    df['is_called_strike'] = df['description'] == 'called_strike'

    # Any swing (contact or whiff)
    swing_desc = ['swinging_strike', 'swinging_strike_blocked', 'foul_tip',
                  'foul', 'foul_bunt', 'hit_into_play', 'hit_into_play_no_out',
                  'hit_into_play_score']
    df['is_swing'] = df['description'].isin(swing_desc)

    # CSW (called strike + whiff)
    df['is_csw'] = df['is_whiff'] | df['is_called_strike']

    return df


def compute_pitcher_game_stats(game_pitches: pd.DataFrame) -> Dict:
    """
    Compute stats for a single game's worth of pitches for one pitcher.

    Args:
        game_pitches: DataFrame of pitches for one pitcher in one game (F5 only)

    Returns:
        Dictionary of game-level stats
    """
    if len(game_pitches) == 0:
        return {}

    stats = {}

    # Basic counts
    stats['pitches'] = len(game_pitches)
    stats['innings'] = game_pitches['inning'].nunique()  # Approximate IP

    # Event-based stats
    stats['strikeouts'] = game_pitches['is_strikeout'].sum()
    stats['walks'] = game_pitches['is_walk'].sum()
    stats['hbp'] = game_pitches['is_hbp'].sum()
    stats['hrs'] = game_pitches['is_hr'].sum()
    stats['hits'] = game_pitches['is_hit'].sum()
    stats['pa'] = game_pitches['is_pa'].sum()
    stats['ab'] = game_pitches['is_ab'].sum()

    # Batted ball stats
    stats['bip'] = game_pitches['is_bip'].sum()
    stats['gb'] = game_pitches['is_gb'].sum()
    stats['barrels'] = game_pitches['is_barrel'].sum()

    # Pitch outcome stats
    stats['swings'] = game_pitches['is_swing'].sum()
    stats['whiffs'] = game_pitches['is_whiff'].sum()
    stats['called_strikes'] = game_pitches['is_called_strike'].sum()
    stats['csw'] = game_pitches['is_csw'].sum()

    # Velocity and quality metrics
    stats['avg_velo'] = game_pitches['release_speed'].mean()

    # Exit velocity against (on contact)
    contact = game_pitches[game_pitches['launch_speed'].notna()]
    stats['avg_exit_velo'] = contact['launch_speed'].mean() if len(contact) > 0 else np.nan

    # Expected stats
    xwoba_pitches = game_pitches[game_pitches['estimated_woba_using_speedangle'].notna()]
    stats['xwoba_sum'] = xwoba_pitches['estimated_woba_using_speedangle'].sum()
    stats['xwoba_count'] = len(xwoba_pitches)

    # First inning specific
    inn1 = game_pitches[game_pitches['inning'] == 1]
    if len(inn1) > 0:
        # Runs in first inning (from bat_score changes)
        if len(inn1) > 0:
            start = inn1['bat_score'].iloc[0]
            end = inn1['bat_score'].iloc[-1]
            stats['inn1_runs'] = max(0, end - start)
        else:
            stats['inn1_runs'] = 0
        stats['inn1_pitches'] = len(inn1)
    else:
        stats['inn1_runs'] = 0
        stats['inn1_pitches'] = 0

    # Pitch type counts
    for pt in PITCH_TYPES:
        stats[f'pitch_{pt}'] = (game_pitches['pitch_type'] == pt).sum()

    return stats


def compute_season_stats(games_df: pd.DataFrame) -> Dict:
    """
    Aggregate game-level stats into season-level features.

    Args:
        games_df: DataFrame with one row per game, containing game-level stats

    Returns:
        Dictionary of season-level features
    """
    if len(games_df) == 0:
        return {}

    features = {}

    # Totals
    total_pitches = games_df['pitches'].sum()
    total_innings = games_df['innings'].sum()
    total_pa = games_df['pa'].sum()
    total_ab = games_df['ab'].sum()
    total_bip = games_df['bip'].sum()
    total_swings = games_df['swings'].sum()

    # Rate stats
    features['szn_k_rate'] = games_df['strikeouts'].sum() / total_pa if total_pa > 0 else np.nan
    features['szn_bb_rate'] = games_df['walks'].sum() / total_pa if total_pa > 0 else np.nan
    features['szn_hr_rate'] = games_df['hrs'].sum() / total_pa if total_pa > 0 else np.nan

    # Whiff and CSW rates
    features['szn_whiff_rate'] = games_df['whiffs'].sum() / total_swings if total_swings > 0 else np.nan
    features['szn_csw_rate'] = games_df['csw'].sum() / total_pitches if total_pitches > 0 else np.nan

    # Ground ball rate
    features['szn_gb_rate'] = games_df['gb'].sum() / total_bip if total_bip > 0 else np.nan

    # Barrel rate
    features['szn_barrel_rate'] = games_df['barrels'].sum() / total_bip if total_bip > 0 else np.nan

    # Average velocity (weighted by pitches)
    velo_weighted = (games_df['avg_velo'] * games_df['pitches']).sum()
    features['szn_avg_velo'] = velo_weighted / total_pitches if total_pitches > 0 else np.nan

    # Average exit velocity against
    ev_games = games_df[games_df['avg_exit_velo'].notna()]
    if len(ev_games) > 0:
        ev_weighted = (ev_games['avg_exit_velo'] * ev_games['bip']).sum()
        total_bip_with_ev = ev_games['bip'].sum()
        features['szn_avg_exit_velo'] = ev_weighted / total_bip_with_ev if total_bip_with_ev > 0 else np.nan
    else:
        features['szn_avg_exit_velo'] = np.nan

    # xwOBA against
    total_xwoba = games_df['xwoba_sum'].sum()
    total_xwoba_count = games_df['xwoba_count'].sum()
    features['szn_xwoba_against'] = total_xwoba / total_xwoba_count if total_xwoba_count > 0 else np.nan

    # Pitches per inning
    features['szn_pitches_per_inn'] = total_pitches / total_innings if total_innings > 0 else np.nan

    # First inning ERA
    inn1_runs = games_df['inn1_runs'].sum()
    num_games = len(games_df)
    features['szn_inn1_era'] = (inn1_runs / num_games) * 9 if num_games > 0 else np.nan

    # ERA and FIP
    total_runs = games_df['f5_runs_allowed'].sum() if 'f5_runs_allowed' in games_df.columns else np.nan
    features['szn_f5_era'] = compute_era(total_runs, total_innings) if pd.notna(total_runs) else np.nan

    total_k = games_df['strikeouts'].sum()
    total_bb = games_df['walks'].sum()
    total_hbp = games_df['hbp'].sum()
    total_hr = games_df['hrs'].sum()
    features['szn_f5_fip'] = compute_fip(total_hr, total_bb, total_hbp, total_k, total_innings)

    # Pitch mix (percentage of each type)
    for pt in PITCH_TYPES:
        col = f'pitch_{pt}'
        if col in games_df.columns:
            features[f'szn_pct_{pt}'] = games_df[col].sum() / total_pitches if total_pitches > 0 else 0

    # Workload
    features['szn_pitch_count'] = total_pitches
    features['szn_innings'] = total_innings
    features['szn_games'] = num_games

    return features


def compute_rolling_stats(games_df: pd.DataFrame, n_games: int) -> Dict:
    """
    Compute rolling stats over the last N games.

    Args:
        games_df: DataFrame with one row per game, sorted by date
        n_games: Number of games to include in rolling window

    Returns:
        Dictionary of rolling features
    """
    if len(games_df) < n_games:
        # Not enough games for this rolling window
        return {f'roll_{n_games}g_f5_era': np.nan,
                f'roll_{n_games}g_k_rate': np.nan,
                f'roll_{n_games}g_whiff_rate': np.nan,
                f'roll_{n_games}g_avg_velo': np.nan,
                f'roll_{n_games}g_xwoba_against': np.nan,
                f'roll_{n_games}g_pitches_per_inn': np.nan}

    recent = games_df.tail(n_games)
    prefix = f'roll_{n_games}g'

    features = {}

    # Totals for recent games
    total_pitches = recent['pitches'].sum()
    total_innings = recent['innings'].sum()
    total_pa = recent['pa'].sum()
    total_swings = recent['swings'].sum()

    # ERA
    if 'f5_runs_allowed' in recent.columns:
        total_runs = recent['f5_runs_allowed'].sum()
        features[f'{prefix}_f5_era'] = compute_era(total_runs, total_innings)
    else:
        features[f'{prefix}_f5_era'] = np.nan

    # K rate
    features[f'{prefix}_k_rate'] = recent['strikeouts'].sum() / total_pa if total_pa > 0 else np.nan

    # Whiff rate
    features[f'{prefix}_whiff_rate'] = recent['whiffs'].sum() / total_swings if total_swings > 0 else np.nan

    # Average velocity
    velo_weighted = (recent['avg_velo'] * recent['pitches']).sum()
    features[f'{prefix}_avg_velo'] = velo_weighted / total_pitches if total_pitches > 0 else np.nan

    # xwOBA against
    total_xwoba = recent['xwoba_sum'].sum()
    total_xwoba_count = recent['xwoba_count'].sum()
    features[f'{prefix}_xwoba_against'] = total_xwoba / total_xwoba_count if total_xwoba_count > 0 else np.nan

    # Pitches per inning
    features[f'{prefix}_pitches_per_inn'] = total_pitches / total_innings if total_innings > 0 else np.nan

    return features


def build_pitcher_features(
    raw_df: pd.DataFrame,
    targets_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build pitcher feature table for all games in targets_df.

    For each row in targets_df, compute features using only data
    available BEFORE that game date.

    Args:
        raw_df: Raw Statcast pitch-level data
        targets_df: F5 game targets with game_pk, starter, game_date

    Returns:
        DataFrame with one row per game, containing pitcher features
    """
    logger.info("Building pitcher features...")

    # Filter to F5 innings and add event/outcome columns
    logger.info("  Filtering to F5 and adding event columns...")
    df = filter_f5(raw_df)
    df = identify_event_types(df)
    df = identify_pitch_outcomes(df)

    # Sort by date
    df = df.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number'])
    targets_df = targets_df.sort_values('game_date')

    # Build feature rows
    feature_rows = []
    total = len(targets_df)

    logger.info(f"  Computing features for {total:,} games...")

    for idx, target_row in enumerate(targets_df.itertuples()):
        if (idx + 1) % 500 == 0:
            logger.info(f"    Progress: {idx + 1:,} / {total:,}")

        game_pk = target_row.game_pk
        pitcher = target_row.starter
        game_date = target_row.game_date
        season = get_season(game_date)

        # Get all F5 pitches by this pitcher BEFORE this game in same season
        prior_pitches = df[
            (df['pitcher'] == pitcher) &
            (df['game_date'] < game_date) &
            (df['game_date'].apply(get_season) == season)
        ]

        # Group by game to get game-level stats
        if len(prior_pitches) > 0:
            game_stats_list = []
            for gp, gp_pitches in prior_pitches.groupby('game_pk'):
                game_stats = compute_pitcher_game_stats(gp_pitches)
                game_stats['game_pk'] = gp
                game_stats['game_date'] = gp_pitches['game_date'].iloc[0]
                game_stats_list.append(game_stats)

            if game_stats_list:
                games_df = pd.DataFrame(game_stats_list).sort_values('game_date')

                # Merge in F5 runs allowed from targets if available
                prior_targets = targets_df[
                    (targets_df['starter'] == pitcher) &
                    (targets_df['game_date'] < game_date)
                ][['game_pk', 'f5_runs_allowed']]
                games_df = games_df.merge(prior_targets, on='game_pk', how='left')

                # Compute season stats
                season_features = compute_season_stats(games_df)

                # Compute rolling stats
                roll_3g = compute_rolling_stats(games_df, 3)
                roll_5g = compute_rolling_stats(games_df, 5)

                # Days rest
                last_game_date = games_df['game_date'].iloc[-1]
                days_rest = (pd.to_datetime(game_date) - pd.to_datetime(last_game_date)).days

                features = {
                    'game_pk': game_pk,
                    'starter': pitcher,
                    'game_date': game_date,
                    'days_rest': days_rest,
                    **season_features,
                    **roll_3g,
                    **roll_5g
                }
            else:
                # No prior games found
                features = {
                    'game_pk': game_pk,
                    'starter': pitcher,
                    'game_date': game_date,
                    'days_rest': np.nan
                }
        else:
            # No prior data for this pitcher this season
            features = {
                'game_pk': game_pk,
                'starter': pitcher,
                'game_date': game_date,
                'days_rest': np.nan
            }

        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)
    logger.info(f"  Built features for {len(features_df):,} games")

    return features_df


def process_pitcher_features() -> pd.DataFrame:
    """
    Main function to process pitcher features and save to parquet.
    """
    logger.info("=" * 60)
    logger.info("Processing pitcher features")
    logger.info("=" * 60)

    # Load data
    logger.info("\n1. Loading raw data...")
    raw_df = load_all_raw_data()

    logger.info("\n2. Loading F5 targets...")
    targets_path = get_processed_data_dir() / "f5_game_targets.parquet"
    targets_df = pd.read_parquet(targets_path)
    logger.info(f"   Loaded {len(targets_df):,} target games")

    # Build features
    logger.info("\n3. Building pitcher features...")
    features_df = build_pitcher_features(raw_df, targets_df)

    # Save
    logger.info("\n4. Saving...")
    output_path = get_processed_data_dir() / "pitcher_features.parquet"
    features_df.to_parquet(output_path, index=False)
    logger.info(f"   Saved to {output_path}")

    # Summary stats
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total rows: {len(features_df):,}")
    logger.info(f"Columns: {len(features_df.columns)}")

    # Check for missing values in key columns
    key_cols = ['szn_f5_era', 'szn_k_rate', 'szn_avg_velo', 'days_rest']
    for col in key_cols:
        if col in features_df.columns:
            missing = features_df[col].isna().sum()
            pct = missing / len(features_df) * 100
            logger.info(f"  {col}: {missing:,} missing ({pct:.1f}%)")

    return features_df


def verify_pitcher_features() -> None:
    """Verify the pitcher features file."""
    output_path = get_processed_data_dir() / "pitcher_features.parquet"

    if not output_path.exists():
        print(f"ERROR: {output_path} not found")
        return

    df = pd.read_parquet(output_path)

    print("\n" + "=" * 60)
    print("PITCHER FEATURES VERIFICATION")
    print("=" * 60)

    print(f"\nShape: {df.shape}")
    print(f"\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    print(f"\nSample stats (non-null only):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe().T[['count', 'mean', 'std', 'min', 'max']].round(3).head(15))


if __name__ == "__main__":
    process_pitcher_features()
