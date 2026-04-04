# F5 Runs Prediction Model

## Instructions for Claude Code

Work through this file step by step, starting at Step 1. Do NOT skip ahead — each step depends on the previous one.

1. Implement and fully complete one step at a time.
2. After finishing each step, verify the output before moving on (e.g., confirm data files exist and look correct, confirm feature tables have the right shape, etc.).
3. Wait for me to confirm before proceeding to the next step.
4. Step 1 (Statcast pull) will take a long time to download. Let it finish completely and save all parquet files before moving on.
5. At each step, write clean, documented Python code with docstrings and type hints.
6. If something fails or data looks wrong, stop and ask — don't try to work around it silently.

## Goal

Predict runs allowed by a starting pitcher through the first 5 innings of an MLB game. The user manually inputs a starting pitcher and the opposing lineup. The model outputs a predicted F5 run total for that side.

## Data Source

Statcast via `pybaseball`. All features are derived from pitch-level Statcast data.

## Project Structure

```
f5_model/
├── data/
│   ├── raw/                  # Raw Statcast pulls
│   └── processed/            # Cleaned feature tables
├── features/
│   ├── pitcher_features.py   # Build pitcher profiles
│   ├── batter_features.py    # Build batter profiles
│   └── matchup_features.py   # Combine pitcher + lineup into game-level rows
├── model/
│   ├── train.py              # Train XGBoost model
│   ├── evaluate.py           # Backtest and evaluation metrics
│   └── predict.py            # CLI prediction interface
├── utils/
│   ├── statcast_pull.py      # Data fetching helpers
│   └── constants.py          # Column mappings, team codes, etc.
├── models/                   # Saved model artifacts (.pkl)
├── requirements.txt
└── README.md
```

## Step 1: Data Collection (`utils/statcast_pull.py`)

Pull Statcast pitch-level data for the 2024 and 2025 seasons.

```python
from pybaseball import statcast
import pandas as pd

# Pull in monthly chunks to avoid timeouts
months_2024 = [
    ("2024-03-28", "2024-04-30"),
    ("2024-05-01", "2024-05-31"),
    ("2024-06-01", "2024-06-30"),
    ("2024-07-01", "2024-07-31"),
    ("2024-08-01", "2024-08-31"),
    ("2024-09-01", "2024-09-29"),
]

# Same structure for 2025
# Save each chunk to data/raw/ as parquet files
```

Key columns to retain from Statcast:
- `pitcher`, `batter`, `game_pk`, `game_date`, `at_bat_number`, `pitch_number`
- `inning`, `inning_topbot` — CRITICAL for filtering to first 5 innings
- `events`, `description`, `des`
- `release_speed`, `effective_speed`, `release_spin_rate`
- `pfx_x`, `pfx_z`, `plate_x`, `plate_z`
- `launch_speed`, `launch_angle`, `hit_distance_sc`
- `estimated_ba_using_speedangle`, `estimated_woba_using_speedangle` (xBA, xwOBA)
- `woba_value`, `woba_denom`, `babip_value`
- `zone`, `stand`, `p_throws`
- `pitch_type`, `pitch_name`
- `balls`, `strikes`, `outs_when_up`
- `home_team`, `away_team`
- `bat_score`, `fld_score`, `post_bat_score`

## Step 2: F5 Filtering

CRITICAL: Filter all data to only innings 1-5 before building features.

```python
def filter_f5(df):
    """Keep only pitches from innings 1-5."""
    return df[df['inning'] <= 5].copy()
```

Calculate F5 runs allowed per pitcher per game:
```python
def calc_f5_runs(df):
    """
    For each game_pk + pitcher, calculate runs allowed through 5 innings.
    Use post_bat_score - bat_score changes, or count events that score runs.
    
    Logic:
    - Group by game_pk, pitcher
    - For each group, find the max bat_score at end of inning 5 minus
      the bat_score at the start of inning 1
    - This is the target variable: f5_runs_allowed
    """
    pass
```

Also track: did the pitcher complete 5 innings? If not, flag it — this affects whether we should include the game or model "early exit" separately.

## Step 3: Pitcher Features (`features/pitcher_features.py`)

For each pitcher on each game date, compute features using ONLY data available before that game (no leakage).

### Season-Level (expanding window up to game date)
- `szn_f5_era`: ERA through 5 innings only
- `szn_f5_fip`: FIP through 5 innings (K, BB, HR, IP through 5)
- `szn_k_rate`: K / BF through 5 innings
- `szn_bb_rate`: BB / BF through 5 innings
- `szn_hr_rate`: HR / BF
- `szn_whiff_rate`: swinging strikes / total swings
- `szn_csw_rate`: called strikes + whiffed strikes / total pitches
- `szn_ground_ball_rate`: GB / balls in play
- `szn_avg_velo`: mean release_speed
- `szn_avg_exit_velo_against`: mean launch_speed on contact
- `szn_barrel_rate`: barrels / batted ball events
- `szn_xwoba_against`: mean estimated_woba_using_speedangle
- `szn_avg_pitches_per_inning`: total pitches / innings through 5
- `szn_first_inning_era`: ERA in inning 1 only (pitchers often perform differently)
- `szn_pitch_mix`: percentage of each pitch type (FF, SL, CH, CU, SI, FC, etc.)

### Rolling Window (last 3 and 5 starts)
- `roll_3g_f5_era`, `roll_5g_f5_era`
- `roll_3g_k_rate`, `roll_5g_k_rate`
- `roll_3g_whiff_rate`, `roll_5g_whiff_rate`
- `roll_3g_avg_velo`, `roll_5g_avg_velo`
- `roll_3g_xwoba_against`, `roll_5g_xwoba_against`
- `roll_3g_avg_pitches_per_inn`

### Rest and Workload
- `days_rest`: days since last start
- `szn_pitch_count_total`: cumulative pitches thrown this season (fatigue proxy)
- `szn_innings_total`: cumulative IP this season

## Step 4: Batter Features (`features/batter_features.py`)

For each batter on each game date, compute season-to-date stats using only pre-game data.

### Per-Batter Season Stats (split by pitcher handedness: vs LHP / vs RHP)
- `woba`: weighted on-base average
- `xwoba`: expected wOBA from Statcast
- `k_rate`: strikeout rate
- `bb_rate`: walk rate
- `iso`: isolated power (SLG - AVG)
- `barrel_rate`: barrels / batted ball events
- `avg_exit_velo`: mean launch_speed
- `ground_ball_rate`: GB%
- `ops`: on-base plus slugging

### Rolling (last 15 games)
- `roll_15g_woba`, `roll_15g_xwoba`, `roll_15g_k_rate`

## Step 5: Lineup Aggregation (`features/matchup_features.py`)

Given a lineup (list of 9 batter IDs) and the opposing pitcher's handedness, aggregate batter features into a single feature vector.

```python
def aggregate_lineup(lineup_batter_ids, pitcher_hand, batter_features_df, game_date):
    """
    For each batter in the lineup:
    1. Look up their platoon-split stats vs pitcher_hand
    2. Weight by lineup position (optional: 1-9 weighting or equal)
    
    Return aggregated features:
    - lineup_avg_woba, lineup_avg_xwoba
    - lineup_avg_k_rate, lineup_avg_bb_rate
    - lineup_avg_iso, lineup_avg_barrel_rate
    - lineup_avg_exit_velo
    - lineup_total_ops (sum or mean)
    """
    pass
```

Weighting options:
- Equal weight across 9 hitters
- PA-weighted by lineup slot (slot 1-4 see ~15% more PA than 7-9)
- Simple linear decay: slot 1 = 1.0, slot 9 = 0.6

## Step 6: Training Data Assembly

Each row = one pitcher's F5 outing in one game.

Columns:
- **Target**: `f5_runs_allowed` (integer, 0-10+)
- **Pitcher features**: all from Step 3, computed as of game_date
- **Lineup features**: aggregated lineup stats vs this pitcher's hand, as of game_date
- **Context features**:
  - `home_away`: is the pitcher home or away (1/0)
  - `park_factor_runs`: ballpark runs factor (can hardcode a lookup table)

Split: train on 2024 full season + 2025 through August 1. Test on 2025 August 1 onward.

## Step 7: Model Training (`model/train.py`)

```python
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

# XGBoost with Poisson objective (count data)
params = {
    'objective': 'count:poisson',
    'eval_metric': 'mae',
    'max_depth': 4,
    'learning_rate': 0.01,
    'subsample': 0.85,
    'colsample_bytree': 0.8,
    'n_estimators': 1500,
    'early_stopping_rounds': 50,
}

# Use TimeSeriesSplit for cross-validation (no future leakage)
tscv = TimeSeriesSplit(n_splits=5)

# Train with early stopping on validation set
model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

# Save model
import joblib
joblib.dump(model, 'models/f5_runs_model.pkl')
```

## Step 8: Evaluation (`model/evaluate.py`)

Metrics to track:
- **MAE**: mean absolute error on F5 runs
- **RMSE**: root mean squared error
- **R²**: variance explained
- **Calibration by bucket**: for predicted ranges (0-1, 1-2, 2-3, 3-4, 4+), what's the actual mean?
- **Over/under accuracy**: given a line (e.g., 2.5 F5 runs), what % does the model get right?
- **Distribution comparison**: histogram of predicted vs actual F5 runs

Also output a Poisson probability distribution per prediction:
```python
import numpy as np
from scipy.stats import poisson

def predict_distribution(model, X):
    """Return predicted lambda and full probability distribution."""
    lam = model.predict(X)[0]
    probs = {k: poisson.pmf(k, lam) for k in range(0, 11)}
    over_under = {
        f"over_{line}": 1 - poisson.cdf(line, lam)
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]
    }
    return lam, probs, over_under
```

## Step 9: Prediction Interface (`model/predict.py`)

CLI tool for daily use:

```
python model/predict.py \
    --pitcher "Corbin Burnes" \
    --pitcher-hand R \
    --opponent-lineup "Juan Soto,Aaron Judge,Giancarlo Stanton,Anthony Rizzo,..." \
    --date 2025-09-15
```

Flow:
1. Look up pitcher ID from name (use `pybaseball.playerid_lookup`)
2. Look up each batter ID from name
3. Load pre-computed pitcher features for that date (or compute on the fly)
4. Load pre-computed batter features, filter to platoon split vs pitcher hand
5. Aggregate lineup features
6. Run model prediction
7. Output:
   - Predicted F5 runs (lambda)
   - Full Poisson distribution (P(0), P(1), ..., P(7+))
   - Over/under probabilities for common lines (1.5, 2.5, 3.5)

## Step 10: Feature Store / Daily Update Pipeline

For ongoing use:
1. After each day's games, pull new Statcast data
2. Update pitcher and batter feature tables (append new rows)
3. Periodically retrain model (weekly or monthly)

Store features as parquet files:
- `data/processed/pitcher_features.parquet` — one row per pitcher per game date
- `data/processed/batter_features.parquet` — one row per batter per game date

## Requirements

```
pybaseball>=2.2.7
xgboost>=2.0
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
scipy>=1.11
joblib
pyarrow  # for parquet
```

## Implementation Order

1. `utils/statcast_pull.py` — fetch and save raw data
2. F5 filtering + target variable calculation
3. `features/pitcher_features.py` — build pitcher feature table
4. `features/batter_features.py` — build batter feature table
5. `features/matchup_features.py` — lineup aggregation
6. `model/train.py` — assemble training data and train
7. `model/evaluate.py` — backtest and metrics
8. `model/predict.py` — CLI prediction tool
9. Iterate: feature engineering, hyperparameter tuning, calibration

## Notes

- Always filter to first 5 innings BEFORE computing any features
- Never use future data — all features must be computed as of the day before the game
- Handle pitchers with < 3 starts carefully (use season averages or priors)
- Park factors can be a simple lookup dict initially; upgrade later if needed
- The Poisson distribution output is what makes this useful for betting — compare P(over X.5) to the book's implied probability