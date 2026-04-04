"""
Statcast data fetching utilities for F5 runs prediction model.

This module handles pulling pitch-level data from Statcast via pybaseball,
with monthly chunking to avoid timeouts and parquet storage for efficiency.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from pybaseball import statcast, cache

from .constants import STATCAST_COLUMNS, MONTHS_2024, MONTHS_2025

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Enable pybaseball caching to avoid re-downloading
cache.enable()


def get_project_root() -> Path:
    """Get the f5_model project root directory."""
    return Path(__file__).parent.parent


def get_raw_data_dir() -> Path:
    """Get the raw data directory path."""
    return get_project_root() / "data" / "raw"


def get_processed_data_dir() -> Path:
    """Get the processed data directory path."""
    return get_project_root() / "data" / "processed"


def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to only include the columns we need.

    Args:
        df: Raw Statcast DataFrame

    Returns:
        DataFrame with only the columns in STATCAST_COLUMNS that exist in df
    """
    available_cols = [col for col in STATCAST_COLUMNS if col in df.columns]
    missing_cols = [col for col in STATCAST_COLUMNS if col not in df.columns]

    if missing_cols:
        logger.warning(f"Missing columns in data: {missing_cols}")

    return df[available_cols].copy()


def pull_month(
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    retry_delay: int = 30
) -> Optional[pd.DataFrame]:
    """
    Pull Statcast data for a date range with retry logic.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_retries: Maximum number of retry attempts
        retry_delay: Seconds to wait between retries

    Returns:
        DataFrame with Statcast data, or None if all retries failed
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Pulling data for {start_date} to {end_date} (attempt {attempt + 1}/{max_retries})")

            df = statcast(start_dt=start_date, end_dt=end_date)

            if df is None or df.empty:
                logger.warning(f"No data returned for {start_date} to {end_date}")
                return None

            # Filter to columns we need
            df = filter_columns(df)

            logger.info(f"Successfully pulled {len(df):,} pitches for {start_date} to {end_date}")
            return df

        except Exception as e:
            logger.error(f"Error pulling data: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to pull data for {start_date} to {end_date} after {max_retries} attempts")
                return None

    return None


def pull_and_save_month(
    start_date: str,
    end_date: str,
    output_dir: Path
) -> bool:
    """
    Pull Statcast data for a month and save to parquet.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_dir: Directory to save parquet files

    Returns:
        True if successful, False otherwise
    """
    # Create output filename from date range
    year = start_date[:4]
    month = start_date[5:7]
    output_file = output_dir / f"statcast_{year}_{month}.parquet"

    # Skip if file already exists
    if output_file.exists():
        logger.info(f"File already exists: {output_file}, skipping...")
        return True

    # Pull the data
    df = pull_month(start_date, end_date)

    if df is None or df.empty:
        logger.error(f"No data to save for {start_date} to {end_date}")
        return False

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    df.to_parquet(output_file, index=False)
    logger.info(f"Saved {len(df):,} rows to {output_file}")

    return True


def pull_season(
    months: List[Tuple[str, str]],
    output_dir: Optional[Path] = None
) -> int:
    """
    Pull all months for a season and save to parquet files.

    Args:
        months: List of (start_date, end_date) tuples for each month
        output_dir: Directory to save files (defaults to data/raw/)

    Returns:
        Number of months successfully pulled
    """
    if output_dir is None:
        output_dir = get_raw_data_dir()

    success_count = 0

    for start_date, end_date in months:
        if pull_and_save_month(start_date, end_date, output_dir):
            success_count += 1

        # Small delay between requests to be nice to the server
        time.sleep(2)

    return success_count


def pull_all_data() -> None:
    """
    Pull all Statcast data for 2024 and 2025 seasons.

    This is the main entry point for data collection.
    """
    output_dir = get_raw_data_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting Statcast data pull for 2024-2025 seasons")
    logger.info("=" * 60)

    # Pull 2024
    logger.info("\n--- Pulling 2024 season ---")
    count_2024 = pull_season(MONTHS_2024, output_dir)
    logger.info(f"2024: Successfully pulled {count_2024}/{len(MONTHS_2024)} months")

    # Pull 2025
    logger.info("\n--- Pulling 2025 season ---")
    count_2025 = pull_season(MONTHS_2025, output_dir)
    logger.info(f"2025: Successfully pulled {count_2025}/{len(MONTHS_2025)} months")

    # Summary
    total_months = len(MONTHS_2024) + len(MONTHS_2025)
    total_success = count_2024 + count_2025

    logger.info("\n" + "=" * 60)
    logger.info(f"COMPLETE: {total_success}/{total_months} months pulled successfully")
    logger.info("=" * 60)


def load_all_raw_data() -> pd.DataFrame:
    """
    Load all raw parquet files and combine into a single DataFrame.

    Returns:
        Combined DataFrame with all Statcast data
    """
    raw_dir = get_raw_data_dir()
    parquet_files = sorted(raw_dir.glob("statcast_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")

    logger.info(f"Loading {len(parquet_files)} parquet files...")

    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        logger.info(f"  {f.name}: {len(df):,} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total: {len(combined):,} pitches loaded")

    return combined


def verify_data() -> None:
    """
    Verify the pulled data exists and has expected structure.
    """
    raw_dir = get_raw_data_dir()
    parquet_files = sorted(raw_dir.glob("statcast_*.parquet"))

    print("\n" + "=" * 60)
    print("DATA VERIFICATION")
    print("=" * 60)

    if not parquet_files:
        print(f"ERROR: No parquet files found in {raw_dir}")
        return

    print(f"\nFound {len(parquet_files)} parquet files:")

    total_rows = 0
    for f in parquet_files:
        df = pd.read_parquet(f)
        total_rows += len(df)
        print(f"  {f.name}: {len(df):,} pitches")

    print(f"\nTotal pitches: {total_rows:,}")

    # Load sample to check columns
    sample = pd.read_parquet(parquet_files[0])
    print(f"\nColumns ({len(sample.columns)}):")
    for col in sorted(sample.columns):
        print(f"  - {col}")

    # Check critical columns
    critical_cols = ["pitcher", "batter", "game_pk", "inning", "events"]
    missing = [c for c in critical_cols if c not in sample.columns]
    if missing:
        print(f"\nWARNING: Missing critical columns: {missing}")
    else:
        print("\nAll critical columns present.")

    # Show sample game_pk values
    print(f"\nSample game_pk values: {sample['game_pk'].unique()[:5].tolist()}")

    # Show date range
    all_dates = sample['game_date'].unique()
    print(f"Date range in first file: {min(all_dates)} to {max(all_dates)}")


if __name__ == "__main__":
    pull_all_data()
