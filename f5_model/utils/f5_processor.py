"""
F5 (First 5 Innings) data processing utilities.

This module handles:
1. Filtering Statcast data to innings 1-5
2. Identifying starting pitchers who completed 5 innings
3. Calculating F5 runs allowed (target variable)
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

from .statcast_pull import get_raw_data_dir, get_processed_data_dir, load_all_raw_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def filter_f5(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to only include pitches from innings 1-5.

    Args:
        df: Statcast DataFrame with 'inning' column

    Returns:
        DataFrame filtered to innings 1-5 only
    """
    initial_count = len(df)
    df_f5 = df[df['inning'] <= 5].copy()
    filtered_count = len(df_f5)

    logger.info(f"F5 filter: {initial_count:,} -> {filtered_count:,} pitches "
                f"({filtered_count/initial_count*100:.1f}% retained)")

    return df_f5


def identify_starters(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, identify BOTH starting pitchers (home and away).

    Args:
        df: Statcast DataFrame

    Returns:
        DataFrame with two rows per game (one per starter)
    """
    # Sort by game and pitch order
    df_sorted = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])

    starters_list = []

    for game_pk, game_df in df_sorted.groupby('game_pk'):
        game_date = game_df['game_date'].iloc[0]
        home_team = game_df['home_team'].iloc[0]
        away_team = game_df['away_team'].iloc[0]

        # Home starter = first pitcher in Top of any inning (faces away batters)
        top_pitches = game_df[game_df['inning_topbot'] == 'Top']
        if len(top_pitches) > 0:
            home_starter = top_pitches.iloc[0]['pitcher']
            home_p_throws = top_pitches.iloc[0]['p_throws']
            starters_list.append({
                'game_pk': game_pk,
                'game_date': game_date,
                'starter': home_starter,
                'p_throws': home_p_throws,
                'home_team': home_team,
                'away_team': away_team,
                'starter_is_home': True,
                'starter_team': home_team,
                'opponent_team': away_team,
            })

        # Away starter = first pitcher in Bot of any inning (faces home batters)
        bot_pitches = game_df[game_df['inning_topbot'] == 'Bot']
        if len(bot_pitches) > 0:
            away_starter = bot_pitches.iloc[0]['pitcher']
            away_p_throws = bot_pitches.iloc[0]['p_throws']
            starters_list.append({
                'game_pk': game_pk,
                'game_date': game_date,
                'starter': away_starter,
                'p_throws': away_p_throws,
                'home_team': home_team,
                'away_team': away_team,
                'starter_is_home': False,
                'starter_team': away_team,
                'opponent_team': home_team,
            })

    starters = pd.DataFrame(starters_list)
    logger.info(f"Identified {len(starters):,} starters for {starters['game_pk'].nunique():,} games")

    return starters


def check_starter_completed_f5(df: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    """
    Check which starters pitched in all 5 innings (1-5).

    A starter is considered to have completed F5 if they threw at least one pitch
    in each of innings 1, 2, 3, 4, and 5 in their respective half-innings.

    Args:
        df: Statcast DataFrame (already filtered to F5)
        starters: DataFrame with game_pk, starter, and starter_is_home columns

    Returns:
        starters DataFrame with 'completed_f5' boolean column
    """
    completed_list = []

    for _, row in starters.iterrows():
        game_pk = row['game_pk']
        starter = row['starter']
        starter_is_home = row['starter_is_home']

        # Get game data
        game_df = df[df['game_pk'] == game_pk]

        # Home starters pitch in Top, Away starters pitch in Bot
        starter_half = 'Top' if starter_is_home else 'Bot'

        # Get pitches thrown by this starter in their half-innings
        starter_pitches = game_df[
            (game_df['pitcher'] == starter) &
            (game_df['inning_topbot'] == starter_half)
        ]

        # Check which innings they pitched
        innings_pitched = set(starter_pitches['inning'].unique())
        required_innings = {1, 2, 3, 4, 5}
        completed = required_innings.issubset(innings_pitched)

        completed_list.append({
            'game_pk': game_pk,
            'starter': starter,
            'completed_f5': completed
        })

    completed_df = pd.DataFrame(completed_list)

    # Merge back to starters
    starters = starters.merge(
        completed_df,
        on=['game_pk', 'starter'],
        how='left'
    )
    starters['completed_f5'] = starters['completed_f5'].fillna(False)

    completed = starters['completed_f5'].sum()
    total = len(starters)
    logger.info(f"Starters completing F5: {completed:,} / {total:,} ({completed/total*100:.1f}%)")

    return starters


def calc_f5_runs_allowed(df: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate runs allowed by each starter through the first 5 innings.

    Logic:
    - For each game, we look at the starter's half-innings pitched
    - Track the score changes during those half-innings
    - Sum up all runs scored against the starter in innings 1-5

    Args:
        df: Statcast DataFrame (filtered to F5)
        starters: DataFrame with game_pk, starter, and completed_f5 columns

    Returns:
        starters DataFrame with 'f5_runs_allowed' column
    """
    # Sort df by pitch order for accurate score tracking
    df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number']).copy()

    # Only process games where starter completed F5
    completed_starters = starters[starters['completed_f5']].copy()

    runs_list = []

    for _, row in completed_starters.iterrows():
        game_pk = row['game_pk']
        starter = row['starter']
        starter_is_home = row['starter_is_home']

        # Get all pitches from this game (already sorted)
        game_df = df[df['game_pk'] == game_pk]

        # Determine which half-innings the starter pitched
        # Home starter pitches in Top of innings, Away starter pitches in Bot
        starter_half = 'Top' if starter_is_home else 'Bot'

        # Get all pitches from starter's half-innings (not just starter's pitches)
        # This gives us accurate score tracking even if there was a pitching change
        half_inning_pitches = game_df[game_df['inning_topbot'] == starter_half]

        if len(half_inning_pitches) == 0:
            runs_list.append({'game_pk': game_pk, 'starter': starter, 'f5_runs_allowed': np.nan})
            continue

        # Calculate runs allowed using bat_score changes per inning
        f5_runs = 0
        for inning in range(1, 6):
            inning_pitches = half_inning_pitches[half_inning_pitches['inning'] == inning]
            if len(inning_pitches) == 0:
                continue

            # Get first and last pitch of this half-inning (data is already sorted)
            first_pitch = inning_pitches.iloc[0]
            last_pitch = inning_pitches.iloc[-1]

            # Start score is bat_score at first pitch
            start_score = first_pitch['bat_score']

            # End score: use post_bat_score of last pitch if available
            if pd.notna(last_pitch['post_bat_score']):
                end_score = last_pitch['post_bat_score']
            else:
                end_score = last_pitch['bat_score']

            runs_in_inning = int(end_score - start_score)
            f5_runs += max(0, runs_in_inning)  # Ensure non-negative

        runs_list.append({
            'game_pk': game_pk,
            'starter': starter,
            'f5_runs_allowed': int(f5_runs)
        })

    runs_df = pd.DataFrame(runs_list)

    # Merge back to starters on BOTH game_pk AND starter
    starters = starters.merge(runs_df, on=['game_pk', 'starter'], how='left')

    # Log distribution
    valid_runs = starters[starters['completed_f5']]['f5_runs_allowed'].dropna()
    logger.info(f"F5 runs allowed distribution:")
    logger.info(f"  Mean: {valid_runs.mean():.2f}")
    logger.info(f"  Median: {valid_runs.median():.1f}")
    logger.info(f"  Std: {valid_runs.std():.2f}")
    logger.info(f"  Min: {valid_runs.min():.0f}, Max: {valid_runs.max():.0f}")

    return starters


def process_f5_targets() -> pd.DataFrame:
    """
    Main function to process all raw data and create F5 target variable table.

    Returns:
        DataFrame with columns:
        - game_pk, game_date, starter, p_throws
        - home_team, away_team, starter_is_home
        - completed_f5, f5_runs_allowed
    """
    logger.info("=" * 60)
    logger.info("Processing F5 targets")
    logger.info("=" * 60)

    # Load all raw data
    logger.info("\n1. Loading raw data...")
    df = load_all_raw_data()

    # Filter to F5
    logger.info("\n2. Filtering to innings 1-5...")
    df_f5 = filter_f5(df)

    # Identify starters (both home and away for each game)
    logger.info("\n3. Identifying starting pitchers...")
    starters = identify_starters(df)

    # Check F5 completion
    logger.info("\n4. Checking F5 completion...")
    starters = check_starter_completed_f5(df_f5, starters)

    # Calculate runs allowed
    logger.info("\n5. Calculating F5 runs allowed...")
    starters = calc_f5_runs_allowed(df_f5, starters)

    # Filter to only completed F5 games
    logger.info("\n6. Filtering to completed F5 games only...")
    f5_targets = starters[starters['completed_f5']].copy()
    f5_targets = f5_targets.dropna(subset=['f5_runs_allowed'])

    # Convert runs to int
    f5_targets['f5_runs_allowed'] = f5_targets['f5_runs_allowed'].astype(int)

    # Select final columns
    f5_targets = f5_targets[[
        'game_pk', 'game_date', 'starter', 'p_throws',
        'home_team', 'away_team', 'starter_is_home',
        'starter_team', 'opponent_team', 'f5_runs_allowed'
    ]].copy()

    logger.info(f"\nFinal dataset: {len(f5_targets):,} valid F5 games")

    # Save to processed directory
    output_dir = get_processed_data_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "f5_game_targets.parquet"

    f5_targets.to_parquet(output_file, index=False)
    logger.info(f"Saved to {output_file}")

    return f5_targets


def verify_f5_targets() -> None:
    """
    Verify the F5 targets file and print summary statistics.
    """
    output_file = get_processed_data_dir() / "f5_game_targets.parquet"

    if not output_file.exists():
        print(f"ERROR: {output_file} does not exist. Run process_f5_targets() first.")
        return

    df = pd.read_parquet(output_file)

    print("\n" + "=" * 60)
    print("F5 TARGETS VERIFICATION")
    print("=" * 60)

    print(f"\nTotal games: {len(df):,}")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"Unique pitchers: {df['starter'].nunique():,}")

    print(f"\nF5 Runs Allowed Distribution:")
    print(df['f5_runs_allowed'].describe())

    print(f"\nDistribution by runs:")
    print(df['f5_runs_allowed'].value_counts().sort_index())

    print(f"\nBy pitcher handedness:")
    print(df.groupby('p_throws')['f5_runs_allowed'].agg(['count', 'mean']).round(2))

    print(f"\nBy home/away:")
    print(df.groupby('starter_is_home')['f5_runs_allowed'].agg(['count', 'mean']).round(2))


if __name__ == "__main__":
    process_f5_targets()
