# F5 Runs Prediction Model

## Current Status: COMPLETE

All steps (1-9) have been implemented. The model is trained and ready for 2026 season predictions.

### Model Performance (Test Set)
- **MAE**: 1.504
- **RMSE**: 1.397
- **R²**: 0.034
- **Over/Under 2.5 Accuracy**: 66.4%
- **Top Features**: lineup_xwoba, pitcher whiff rate, velocity

## Quick Start

### Single Pitcher Prediction
```bash
python -m f5_model.model.predict \
    --pitcher "Corbin Burnes" \
    --lineup "Juan Soto,Aaron Judge,Giancarlo Stanton,Anthony Rizzo,Gleyber Torres,Alex Verdugo,Anthony Volpe,Jose Trevino,Oswaldo Cabrera" \
    --date 2026-04-05 \
    --home \
    --park NYY
```

### Full Game Prediction (Both Teams)
```bash
python -m f5_model.model.game_predict \
    --away-pitcher "Hunter Greene" \
    --away-lineup "Friedl,De La Cruz,Steer,Stephenson,Candelario,Fairchild,Espinal,Benson,McLain" \
    --away-team "CIN" \
    --home-pitcher "Nathan Eovaldi" \
    --home-lineup "Semien,Seager,Langford,Jung,Garcia,Smith,Lowe,Heim,Taveras" \
    --home-team "TEX" \
    --date 2026-04-04 \
    --park TEX
```

### Full Game with FanDuel Odds Comparison (Edge Finder)
```bash
python -m f5_model.model.game_predict \
    --away-pitcher "Hunter Greene" \
    --away-lineup "Friedl,De La Cruz,Steer,..." \
    --away-team "CIN" \
    --home-pitcher "Nathan Eovaldi" \
    --home-lineup "Semien,Seager,Langford,..." \
    --home-team "TEX" \
    --date 2026-04-04 \
    --park TEX \
    --fd-away-ml 114 \
    --fd-home-ml -142 \
    --fd-away-3way 150 \
    --fd-home-3way -102 \
    --fd-tie-3way 470 \
    --fd-total 4.5 \
    --fd-over-odds -114 \
    --fd-under-odds -114 \
    --fd-away-rl-odds -128 \
    --fd-home-rl-odds -102
```

## Project Structure

```
f5_model/
├── data/
│   ├── raw/                      # Raw Statcast parquet files (2024-2025)
│   └── processed/
│       ├── f5_game_targets.parquet    # 7,963 valid starter outings
│       ├── pitcher_features.parquet   # 43 pitcher features per game
│       ├── batter_features.parquet    # 174,376 batter rows with platoon splits
│       └── training_data.parquet      # 7,963 rows, 51 features
├── features/
│   ├── pitcher_features.py       # Season + rolling pitcher stats
│   ├── batter_features.py        # Platoon-split batter stats
│   └── matchup_features.py       # Lineup aggregation with linear decay
├── model/
│   ├── train.py                  # XGBoost with Poisson objective
│   ├── evaluate.py               # MAE, calibration, over/under accuracy
│   ├── predict.py                # Single pitcher CLI
│   └── game_predict.py           # Full game CLI with edge finder
├── models/
│   ├── f5_runs_model.pkl         # Trained XGBoost model
│   ├── feature_names.txt         # 51 feature names
│   └── evaluation_results.pkl    # Saved metrics
└── utils/
    ├── statcast_pull.py          # Data fetching + F5 processing
    └── constants.py              # LINEUP_WEIGHTS, PARK_FACTORS, etc.
```

## CLI Reference

### predict.py (Single Pitcher)

Predicts F5 runs allowed by one pitcher against a lineup.

| Flag | Description |
|------|-------------|
| `--pitcher`, `-p` | Pitcher name (required) |
| `--lineup`, `-l` | Comma-separated lineup in batting order (required) |
| `--date`, `-d` | Game date YYYY-MM-DD (required) |
| `--home` | Add if pitcher is home team |
| `--park` | Park code (default: NYY) |
| `--pitcher-hand` | L or R (auto-detected if not specified) |
| `--pitcher-id` | MLB ID to skip name lookup |
| `--lineup-ids` | Comma-separated MLB IDs to skip name lookup |

**Output:**
- Predicted F5 runs (Poisson lambda)
- Probability distribution (0-7 runs)
- Over/under probabilities for 1.5, 2.5, 3.5 lines

### game_predict.py (Full Game)

Predicts full F5 outcome for both teams, outputs all FanDuel betting markets.

**Team Flags:**
| Flag | Description |
|------|-------------|
| `--away-pitcher`, `-ap` | Away pitcher name |
| `--away-lineup`, `-al` | Away lineup (comma-separated) |
| `--away-team` | Away team code for display (e.g., CIN) |
| `--away-hand` | Away pitcher handedness (L/R) |
| `--home-pitcher`, `-hp` | Home pitcher name |
| `--home-lineup`, `-hl` | Home lineup (comma-separated) |
| `--home-team` | Home team code for display (e.g., TEX) |
| `--home-hand` | Home pitcher handedness (L/R) |
| `--date`, `-d` | Game date YYYY-MM-DD |
| `--park` | Park code for park factor |

**FanDuel Odds Flags (for Edge Finder):**
| Flag | Description |
|------|-------------|
| `--fd-away-ml` | 2-way moneyline (e.g., 114 for +114) |
| `--fd-home-ml` | 2-way moneyline (e.g., -142) |
| `--fd-away-3way` | 3-way result odds |
| `--fd-home-3way` | 3-way result odds |
| `--fd-tie-3way` | 3-way tie odds |
| `--fd-total` | Total line (e.g., 4.5) |
| `--fd-over-odds` | Over odds (e.g., -114) |
| `--fd-under-odds` | Under odds (e.g., -114) |
| `--fd-away-rl-odds` | Away run line odds (+0.5) |
| `--fd-home-rl-odds` | Home run line odds (-0.5) |

**Output Markets:**
- Projected F5 Score with run distributions
- First 5 Innings Result (3-way with tie)
- First 5 Innings Money Line (2-way)
- First 5 Innings Run Line (+/- 0.5)
- First 5 Innings Total Runs
- Alternate Run Lines (+/- 3.5 to -3.5)
- Alternate Total Runs (2.5 to 8.5)
- Winning Margin (5-Way and Exact)
- Most Likely F5 Scores
- **Edge Finder** (when FanDuel odds provided): Shows model vs book with edge %

**Edge Labels:**
- `**VALUE**` = 5%+ edge (strong bet)
- `*` = 2-5% edge (potential value)

## Key Implementation Details

### Lineup Weighting
Linear decay: slot 1 = 1.0, slot 2 = 0.95, ..., slot 9 = 0.6
```python
LINEUP_WEIGHTS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
```

### Park Factors
Stored in `utils/constants.py`. Default is 1.0 (neutral).
```python
PARK_FACTORS = {
    'COL': 1.35,  # Coors Field
    'CIN': 1.08,  # Great American
    'TEX': 1.05,  # Globe Life
    # ... etc
}
```

### Model Architecture
- **Algorithm**: XGBoost with `count:poisson` objective
- **Features**: 51 total (pitcher season/rolling stats, lineup aggregated stats, park factor)
- **Train/Test Split**: 80/20 chronological (no future leakage)
- **Cross-Validation**: 5-fold TimeSeriesSplit

### Data
- **Source**: Statcast via pybaseball (2024-2025 seasons, ~1.4M pitches)
- **Valid Games**: 7,963 starter outings where pitcher completed 5 IP
- **Target**: F5 runs allowed (Poisson distributed, mean ~2.0)

## Maintenance / Updates

### To update data for new games:
```python
from f5_model.utils.statcast_pull import pull_month

# Pull recent data
df = pull_month("2026-04-01", "2026-04-30")
# Then rebuild features and optionally retrain
```

### To retrain model:
```bash
python -m f5_model.model.train
python -m f5_model.model.evaluate
```

## Notes

- Player name lookup uses `pybaseball.playerid_lookup` - some names may not be found
- Use `--pitcher-id` and `--lineup-ids` flags to bypass name lookup with MLB IDs
- The model predicts Poisson lambda (expected runs), not exact runs
- Edge finder compares model's fair value odds against sportsbook odds
- All features are computed using only pre-game data (no leakage)
