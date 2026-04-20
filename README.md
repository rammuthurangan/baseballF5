# F5 Runs Prediction Model

Predicts runs allowed by MLB starting pitchers through the first 5 innings using machine learning.

## Overview

An XGBoost model with Poisson objective trained on 2024-2025 Statcast data (~7,963 valid starter outings). The model analyzes pitcher performance, batter matchups, and park factors to generate probability distributions for F5 (first 5 innings) run totals.

**Model Performance**
- MAE: 1.504
- Over/Under 2.5 Accuracy: 66.4%

## Features

- Poisson-based run predictions with full probability distributions
- Automated lineup fetching from MLB Stats API
- Edge detection against sportsbook odds
- Data quality scoring with confidence adjustments
- Bet tracking with performance analytics
- Feature staleness monitoring

## Installation

```bash
git clone https://github.com/yourusername/baseballF5.git
cd baseballF5
pip install -r requirements.txt
```

## Quick Start

### Daily Workflow (Recommended)

Run the full pipeline with one command:

```bash
python -m f5_model.model.daily_workflow --date 2026-04-06 --bankroll 100 --unit 10
```

This orchestrates:
1. Feature staleness check
2. Lineup fetching from MLB API
3. Odds template generation
4. Scanner with edge finding
5. Bet confirmation and logging
6. Pending bet grading

### Single Game Analysis

```bash
python -m f5_model.model.check_game \
    --away STL --home DET \
    --away-pitcher 681517 --away-hand R \
    --home-pitcher 672456 --home-hand R \
    --away-lineup "802139,673962,676475,691023,669357,695336,701675,686780,687363" \
    --home-lineup "682073,805808,650402,681481,682985,693307,678009,679529,595879" \
    --away-ml 106 --home-ml -132 \
    --total 4.5 --over -110 --under -118
```

## Pipeline Components

### Data Collection

```bash
# Fetch lineups and generate games.csv
python -m f5_model.data.lineup_fetcher --date 2026-04-06 --output games.csv

# Build/refresh player ID cache (run weekly)
python -m f5_model.data.player_cache --build

# Fetch F5 scores and grade bets
python -m f5_model.data.results_fetcher --date 2026-04-06 --grade
```

### Scanning & Prediction

```bash
# Run scanner with games file
python -m f5_model.model.daily_scanner --date 2026-04-06 --games-file games.csv

# Validate odds in CSV
python -m f5_model.data.odds_entry --validate games.csv
```

### Bet Tracking

```bash
# Add a bet
python -m f5_model.tracking.bet_tracker add \
    --date 2026-04-05 --game "STL@DET" --type F5_Total --pick U4.5 \
    --odds -118 --size 14 --model-prob 0.630 --data-quality 0.85 --bankroll 148.72

# Record result
python -m f5_model.tracking.bet_tracker result \
    --date 2026-04-05 --game "STL@DET" --type F5_Total --result W

# View performance
python -m f5_model.tracking.bet_tracker summary --days 7
python -m f5_model.tracking.bet_tracker calibration
```

### Data Updates

```bash
python -m f5_model.scripts.daily_update              # Update yesterday's data
python -m f5_model.scripts.daily_update --rebuild-features
```

## Architecture

```
f5_model/
├── data/           # Data fetching and validation
│   ├── lineup_fetcher.py    # MLB Stats API lineup fetcher
│   ├── player_cache.py      # Player ID cache with fuzzy matching
│   ├── odds_entry.py        # Odds validation
│   └── results_fetcher.py   # F5 score fetcher
├── features/       # Feature engineering
│   ├── pitcher_features.py
│   └── batter_features.py
├── model/          # Prediction and scanning
│   ├── predict.py           # Core prediction function
│   ├── check_game.py        # Single game CLI
│   ├── daily_scanner.py     # Batch scanner
│   └── daily_workflow.py    # Full pipeline orchestration
├── models/         # Trained models
│   └── f5_runs_model.pkl
├── tracking/       # Bet tracking
│   └── bet_tracker.py
└── utils/          # Constants and helpers
    └── constants.py         # Park factors, lineup weights, team mappings
```

## Data Quality System

The model uses tiered batter feature lookup with quality scoring:

| Data Type | Quality Score |
|-----------|---------------|
| Split data (vs-RHP/LHP) | 1.0 |
| Overall fallback | 0.75 |
| Missing (rookie) | 0.0 |

Edges are adjusted by confidence: `adjusted_edge = raw_edge × confidence`

## Edge Detection

The scanner identifies betting edges against book odds:
- `**VALUE**` = 5%+ edge
- `*` = 2-5% edge

## CSV Formats

### games.csv (Input)

```csv
away_team,home_team,away_pitcher_id,away_pitcher_hand,home_pitcher_id,home_pitcher_hand,away_lineup_ids,home_lineup_ids,away_ml,home_ml,total,over_odds,under_odds
STL,DET,681517,R,672456,R,"802139,673962,...","682073,805808,...",106,-132,4.5,-110,-118
```

### bets.csv (Tracking)

```csv
date,game,bet_type,pick,odds,size,model_prob,book_prob,raw_edge,adj_edge,data_quality,result,profit,bankroll_after,notes
2026-04-05,STL@DET,F5_Total,U4.5,-118,14,0.630,0.541,0.089,0.076,0.85,W,11.86,160.58,F5: 1-2
```

## Key Details

- All features use only pre-game data (no data leakage)
- Lineup weighting: slot 1 = 1.0, slot 9 = 0.6 (linear decay)
- Probabilities calculated via `scipy.stats.poisson`
- MLB Stats API (`statsapi.mlb.com`) - free/public, uses same player IDs as Baseball Savant

## License

MIT
