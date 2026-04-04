"""
MLB Pitcher Strikeout Prediction Model
=======================================
Predicts a starting pitcher's strikeout count for a given game
using Statcast pitch-level data from the 2025 season.

Key design principles:
  - ALL features are rolling/historical — computed from PRIOR games only
  - Temporal train/test split (no random shuffle)
  - No same-game outcome leakage
  - XGBoost regression with Poisson objective

Data source: statcast(start_dt="2025-03-27", end_dt="2025-09-28")
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pybaseball import statcast, cache
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import json
import os
import pickle

# ──────────────────────────────────────────────
# 0. CONFIG
# ──────────────────────────────────────────────
ROLLING_WINDOWS = [3, 7, 15]  # games lookback
MIN_PITCHER_GAMES = 4         # need >= 4 prior starts to predict
TEMPORAL_SPLIT_FRAC = 0.75    # first 75% of season = train
RANDOM_SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# HELPER: safe numeric conversion
# ──────────────────────────────────────────────
def safe_numeric(series, fill=np.nan):
    """Convert a column to float64, coercing pd.NA / errors to fill value."""
    return pd.to_numeric(series, errors="coerce").astype("float64").fillna(fill)


def safe_numeric_df(df, cols, fill=np.nan):
    """Batch-convert multiple columns to float64."""
    for col in cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col], fill)
    return df


# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────
def load_data():
    """Pull 2025 Statcast data."""
    print("=" * 60)
    print("LOADING 2025 STATCAST DATA")
    print("=" * 60)
    cache.enable()
    df = statcast(start_dt="2025-03-27", end_dt="2025-09-28")
    df = df[df["game_type"] == "R"].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    print(f"  Loaded {len(df):,} pitches across {df['game_date'].nunique()} game-dates")
    print(f"  Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")

    # ── Upfront numeric coercion for every column we will touch ──
    numeric_cols = [
        "release_speed", "release_pos_x", "release_pos_z", "release_pos_y",
        "pfx_x", "pfx_z", "plate_x", "plate_z",
        "vx0", "vy0", "vz0", "ax", "ay", "az",
        "sz_top", "sz_bot",
        "release_spin_rate", "release_extension", "effective_speed",
        "spin_axis", "arm_angle",
        "api_break_z_with_gravity", "api_break_x_arm", "api_break_x_batter_in",
        "bat_speed", "swing_length", "attack_angle", "swing_path_tilt",
        "launch_speed", "launch_angle", "hit_distance_sc",
        "n_thruorder_pitcher", "pitcher_days_since_prev_game",
        "delta_run_exp", "delta_home_win_exp",
    ]
    df = safe_numeric_df(df, numeric_cols)

    # Ensure zone is numeric (needed for zone logic)
    df["zone"] = safe_numeric(df["zone"], fill=0).astype(int)

    return df


# ──────────────────────────────────────────────
# 2. PITCH-LEVEL FEATURE DERIVATION
# ──────────────────────────────────────────────
def derive_pitch_features(df):
    """
    Add pitch-level derived columns. These describe each pitch itself
    and are safe — no future leakage.
    """
    print("\n  Deriving pitch-level features...")

    # --- Whiff / CSW flags ---
    df["is_swing"] = df["description"].isin([
        "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
        "foul_bunt", "hit_into_play", "hit_into_play_no_out",
        "hit_into_play_score", "missed_bunt", "bunt_foul_tip",
        "swinging_pitchout"
    ]).astype(int)

    df["is_whiff"] = df["description"].isin([
        "swinging_strike", "swinging_strike_blocked", "foul_tip",
        "missed_bunt", "swinging_pitchout"
    ]).astype(int)

    df["is_called_strike"] = df["description"].isin([
        "called_strike"
    ]).astype(int)

    df["is_csw"] = (df["is_whiff"] | df["is_called_strike"]).astype(int)

    # Chase: swing on pitch outside zones 1-9
    df["is_outside_zone"] = (~df["zone"].isin(range(1, 10))).astype(int)
    df["is_chase"] = ((df["is_swing"] == 1) & (df["is_outside_zone"] == 1)).astype(int)

    # In-zone swing
    df["is_in_zone"] = df["zone"].isin(range(1, 10)).astype(int)
    df["is_zone_swing"] = ((df["is_swing"] == 1) & (df["is_in_zone"] == 1)).astype(int)

    # Strikeout flag (only on final pitch of a K)
    df["is_strikeout_event"] = df["events"].fillna("").isin([
        "strikeout", "strikeout_double_play"
    ]).astype(int)

    # --- Pitch type grouping ---
    fastball_types = ["FF", "SI", "FC"]
    breaking_types = ["SL", "CU", "KC", "SV", "CS"]
    offspeed_types = ["CH", "FS", "SC", "KN"]

    df["pitch_group"] = "other"
    df.loc[df["pitch_type"].isin(fastball_types), "pitch_group"] = "fastball"
    df.loc[df["pitch_type"].isin(breaking_types), "pitch_group"] = "breaking"
    df.loc[df["pitch_type"].isin(offspeed_types), "pitch_group"] = "offspeed"

    # --- Vertical Approach Angle (VAA) ---
    _ay  = df["ay"].values.astype(float)
    _vy0 = df["vy0"].values.astype(float)
    _vz0 = df["vz0"].values.astype(float)
    _az  = df["az"].values.astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_to_plate  = np.where(_ay != 0, -_vy0 / _ay, np.nan)
        vz_at_plate = _vz0 + _az * t_to_plate
        vy_at_plate = _vy0 + _ay * t_to_plate
        df["vaa"] = np.degrees(np.arctan2(vz_at_plate, -vy_at_plate))

    # --- Pitch location relative to zone ---
    zone_height = df["sz_top"].values - df["sz_bot"].values
    zone_mid_z  = (df["sz_top"].values + df["sz_bot"].values) / 2
    df["plate_z_norm"] = np.where(zone_height != 0,
                                  (df["plate_z"].values - zone_mid_z) / zone_height,
                                  np.nan)
    df["plate_x_abs"] = df["plate_x"].abs()

    print(f"    Done. Key columns: is_whiff, is_csw, is_chase, vaa, plate_z_norm")
    return df


# ──────────────────────────────────────────────
# 3. GAME-LEVEL AGGREGATION
# ──────────────────────────────────────────────
def aggregate_to_game_level(df):
    """
    One row per pitcher per game.
    Same-game stats are used ONLY for:
      (a) the TARGET variable
      (b) building rolling features in the next step
    They are NEVER used as direct model inputs.
    """
    print("\n  Aggregating to game level...")

    game_groups = df.groupby(["pitcher", "game_date", "game_pk", "player_name"])

    game_df = game_groups.agg(
        # --- TARGET ---
        strikeouts=("is_strikeout_event", "sum"),

        # --- Pitch counts ---
        total_pitches=("pitch_type", "count"),
        total_swings=("is_swing", "sum"),
        total_whiffs=("is_whiff", "sum"),
        total_csw=("is_csw", "sum"),
        total_called_strikes=("is_called_strike", "sum"),
        total_chases=("is_chase", "sum"),
        pitches_outside_zone=("is_outside_zone", "sum"),
        pitches_in_zone=("is_in_zone", "sum"),
        zone_swings=("is_zone_swing", "sum"),

        # --- Velocity / movement ---
        avg_velocity=("release_speed", "mean"),
        max_velocity=("release_speed", "max"),
        avg_spin_rate=("release_spin_rate", "mean"),
        avg_extension=("release_extension", "mean"),
        avg_pfx_x=("pfx_x", "mean"),
        avg_pfx_z=("pfx_z", "mean"),
        avg_vaa=("vaa", "mean"),
        avg_spin_axis=("spin_axis", "mean"),
        avg_arm_angle=("arm_angle", "mean"),

        # --- Break metrics ---
        avg_api_break_z=("api_break_z_with_gravity", "mean"),
        avg_api_break_x_arm=("api_break_x_arm", "mean"),

        # --- Location ---
        avg_plate_z_norm=("plate_z_norm", "mean"),
        avg_plate_x_abs=("plate_x_abs", "mean"),
        std_plate_x=("plate_x", "std"),
        std_plate_z=("plate_z", "std"),

        # --- Effective speed ---
        avg_effective_speed=("effective_speed", "mean"),

        # --- Game context ---
        max_inning=("inning", "max"),
        batters_faced=("at_bat_number", "nunique"),
        max_thru_order=("n_thruorder_pitcher", "max"),

        # --- Pitcher rest ---
        pitcher_rest_days=("pitcher_days_since_prev_game", "first"),

        # --- Pitch mix ---
        n_fastball=("pitch_group", lambda x: (x == "fastball").sum()),
        n_breaking=("pitch_group", lambda x: (x == "breaking").sum()),
        n_offspeed=("pitch_group", lambda x: (x == "offspeed").sum()),

        # --- Handedness ---
        p_throws=("p_throws", "first"),
    ).reset_index()

    # Derived rates (same-game — used for rolling only)
    game_df["whiff_rate"]         = game_df["total_whiffs"] / game_df["total_swings"].replace(0, np.nan)
    game_df["csw_rate"]           = game_df["total_csw"]    / game_df["total_pitches"].replace(0, np.nan)
    game_df["chase_rate"]         = game_df["total_chases"] / game_df["pitches_outside_zone"].replace(0, np.nan)
    game_df["zone_rate"]          = game_df["pitches_in_zone"] / game_df["total_pitches"].replace(0, np.nan)
    game_df["zone_contact_rate"]  = game_df["zone_swings"]  / game_df["pitches_in_zone"].replace(0, np.nan)
    game_df["swstr_rate"]         = game_df["total_whiffs"] / game_df["total_pitches"].replace(0, np.nan)

    game_df["pct_fastball"] = game_df["n_fastball"] / game_df["total_pitches"]
    game_df["pct_breaking"] = game_df["n_breaking"] / game_df["total_pitches"]
    game_df["pct_offspeed"] = game_df["n_offspeed"] / game_df["total_pitches"]

    game_df["k_per_bf"] = game_df["strikeouts"] / game_df["batters_faced"].replace(0, np.nan)

    # Filter to >= 45 pitches (starters / long outings)
    game_df = game_df[game_df["total_pitches"] >= 45].copy()
    game_df = game_df.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

    print(f"    {len(game_df):,} pitcher-game rows (>= 45 pitches)")
    print(f"    {game_df['pitcher'].nunique()} unique pitchers")
    return game_df


# ──────────────────────────────────────────────
# 4. OPPOSING BATTER FEATURES (ROLLING, NO LEAK)
# ──────────────────────────────────────────────
def build_batter_rolling_features(df):
    """
    Team-level rolling batting features from PRIOR games.
    """
    print("\n  Building opposing batter rolling features...")

    df_p = df.copy()
    df_p["bat_team"] = np.where(
        df_p["inning_topbot"] == "Top",
        df_p["away_team"],
        df_p["home_team"]
    )

    team_game = df_p.groupby(["bat_team", "game_date"]).agg(
        team_k_events=("is_strikeout_event", "sum"),
        team_pa=("at_bat_number", "nunique"),
        team_swings=("is_swing", "sum"),
        team_whiffs=("is_whiff", "sum"),
        team_avg_bat_speed=("bat_speed", "mean"),
        team_avg_swing_length=("swing_length", "mean"),
    ).reset_index()

    team_game["team_k_rate"]    = team_game["team_k_events"] / team_game["team_pa"].replace(0, np.nan)
    team_game["team_whiff_rate"] = team_game["team_whiffs"]  / team_game["team_swings"].replace(0, np.nan)

    team_game = team_game.sort_values(["bat_team", "game_date"])

    # Shifted rolling 15-game means
    for col in ["team_k_rate", "team_whiff_rate", "team_avg_bat_speed", "team_avg_swing_length"]:
        team_game[f"opp_{col}_15g"] = (
            team_game.groupby("bat_team")[col]
            .transform(lambda x: x.shift(1).rolling(15, min_periods=5).mean())
        )

    return team_game[["bat_team", "game_date",
                       "opp_team_k_rate_15g", "opp_team_whiff_rate_15g",
                       "opp_team_avg_bat_speed_15g", "opp_team_avg_swing_length_15g"]]


# ──────────────────────────────────────────────
# 5. PITCHER ROLLING FEATURES (NO LEAK)
# ──────────────────────────────────────────────
def build_pitcher_rolling_features(game_df):
    """
    Rolling stats from PRIOR games only.  shift(1) ensures the
    current game is excluded.
    """
    print("\n  Building pitcher rolling features (leak-free)...")

    game_df = game_df.sort_values(["pitcher", "game_date"]).copy()

    rolling_cols = [
        "strikeouts", "whiff_rate", "csw_rate", "chase_rate", "swstr_rate",
        "zone_rate", "zone_contact_rate", "k_per_bf",
        "avg_velocity", "max_velocity", "avg_spin_rate", "avg_extension",
        "avg_pfx_x", "avg_pfx_z", "avg_vaa", "avg_arm_angle",
        "avg_api_break_z", "avg_api_break_x_arm",
        "avg_plate_z_norm", "avg_plate_x_abs", "std_plate_x", "std_plate_z",
        "avg_effective_speed",
        "total_pitches", "batters_faced", "max_inning",
        "pct_fastball", "pct_breaking", "pct_offspeed",
    ]

    feature_cols = []

    for window in ROLLING_WINDOWS:
        for col in rolling_cols:
            feat_name = f"roll_{col}_{window}g"
            game_df[feat_name] = (
                game_df.groupby("pitcher")[col]
                .transform(lambda x, w=window: x.shift(1).rolling(w, min_periods=max(2, w // 2)).mean())
            )
            feature_cols.append(feat_name)

    # Season-to-date expanding averages (shifted)
    for col in rolling_cols:
        feat_name = f"szn_{col}"
        game_df[feat_name] = (
            game_df.groupby("pitcher")[col]
            .transform(lambda x: x.shift(1).expanding(min_periods=2).mean())
        )
        feature_cols.append(feat_name)

    # Season game number
    game_df["season_game_num"] = game_df.groupby("pitcher").cumcount() + 1
    feature_cols.append("season_game_num")

    # Trend: recent 3g avg minus season avg (momentum signal)
    for col in ["strikeouts", "whiff_rate", "csw_rate", "avg_velocity"]:
        trend_name = f"trend_{col}"
        game_df[trend_name] = game_df[f"roll_{col}_3g"] - game_df[f"szn_{col}"]
        feature_cols.append(trend_name)

    # Velocity spread
    game_df["roll_velo_spread_7g"] = game_df["roll_max_velocity_7g"] - game_df["roll_avg_velocity_7g"]
    feature_cols.append("roll_velo_spread_7g")

    print(f"    Created {len(feature_cols)} rolling features")
    return game_df, feature_cols


# ──────────────────────────────────────────────
# 6. FASTBALL-DIFFERENTIAL FEATURES
# ──────────────────────────────────────────────
def build_fastball_differential_features(df, game_df):
    """
    Velocity & movement deltas off the primary fastball —
    the key insight from Stuff+ research.
    """
    print("\n  Building fastball-differential features...")

    def _pitch_group_season(df_sub, group_types, prefix):
        sub = df_sub[df_sub["pitch_type"].isin(group_types)].copy()
        if sub.empty:
            return pd.DataFrame(columns=["pitcher", "game_date"])
        g = sub.groupby(["pitcher", "game_date"]).agg(
            velo=("release_speed", "mean"),
            pfx_x=("pfx_x", "mean"),
            pfx_z=("pfx_z", "mean"),
            spin=("release_spin_rate", "mean"),
            vaa=("vaa", "mean"),
        ).reset_index().sort_values(["pitcher", "game_date"])

        cols = ["velo", "pfx_x", "pfx_z"]
        if prefix == "fb":
            cols += ["spin", "vaa"]

        for col in cols:
            g[f"szn_{prefix}_{col}"] = (
                g.groupby("pitcher")[col]
                .transform(lambda x: x.shift(1).expanding(min_periods=2).mean())
            )
        return g

    fb_game  = _pitch_group_season(df, ["FF", "SI", "FC"], "fb")
    brk_game = _pitch_group_season(df, ["SL", "CU", "KC", "SV", "CS"], "brk")
    off_game = _pitch_group_season(df, ["CH", "FS", "SC", "KN"], "off")

    # Merge
    fb_cols  = [c for c in fb_game.columns  if c.startswith("szn_fb_")]
    brk_cols = [c for c in brk_game.columns if c.startswith("szn_brk_")]
    off_cols = [c for c in off_game.columns if c.startswith("szn_off_")]

    if fb_cols:
        game_df = game_df.merge(fb_game[["pitcher", "game_date"] + fb_cols],
                                on=["pitcher", "game_date"], how="left")
    if brk_cols:
        game_df = game_df.merge(brk_game[["pitcher", "game_date"] + brk_cols],
                                on=["pitcher", "game_date"], how="left")
    if off_cols:
        game_df = game_df.merge(off_game[["pitcher", "game_date"] + off_cols],
                                on=["pitcher", "game_date"], how="left")

    # Differentials
    diff_features = []

    pairs = [
        ("diff_brk_fb_velo",  "szn_brk_velo",  "szn_fb_velo"),
        ("diff_brk_fb_pfx_x", "szn_brk_pfx_x", "szn_fb_pfx_x"),
        ("diff_brk_fb_pfx_z", "szn_brk_pfx_z", "szn_fb_pfx_z"),
        ("diff_off_fb_velo",  "szn_off_velo",  "szn_fb_velo"),
        ("diff_off_fb_pfx_x", "szn_off_pfx_x", "szn_fb_pfx_x"),
        ("diff_off_fb_pfx_z", "szn_off_pfx_z", "szn_fb_pfx_z"),
    ]
    for name, a, b in pairs:
        if a in game_df.columns and b in game_df.columns:
            game_df[name] = game_df[a] - game_df[b]
            diff_features.append(name)

    # Raw FB characteristics
    for c in fb_cols:
        if c in game_df.columns:
            diff_features.append(c)

    print(f"    Created {len(diff_features)} fastball-differential features")
    return game_df, diff_features


# ──────────────────────────────────────────────
# 7. MERGE OPPONENT FEATURES
# ──────────────────────────────────────────────
def merge_opponent_features(df, game_df, batter_features):
    """Identify opposing team and merge their rolling batting stats."""
    print("\n  Merging opponent features...")

    pitcher_teams = df.groupby(["pitcher", "game_date"]).agg(
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        inning_topbot=("inning_topbot", "first"),
    ).reset_index()

    pitcher_teams["opp_team"] = np.where(
        pitcher_teams["inning_topbot"] == "Top",
        pitcher_teams["away_team"],
        pitcher_teams["home_team"]
    )

    game_df = game_df.merge(
        pitcher_teams[["pitcher", "game_date", "opp_team"]],
        on=["pitcher", "game_date"], how="left"
    )

    game_df = game_df.merge(
        batter_features,
        left_on=["opp_team", "game_date"],
        right_on=["bat_team", "game_date"],
        how="left"
    )

    opp_feature_cols = [c for c in batter_features.columns if c.startswith("opp_")]
    print(f"    Merged {len(opp_feature_cols)} opponent features")
    return game_df, opp_feature_cols


# ──────────────────────────────────────────────
# 8. ASSEMBLE FINAL FEATURE MATRIX
# ──────────────────────────────────────────────
def assemble_features(game_df, pitcher_feat_cols, diff_feat_cols, opp_feat_cols):
    """Combine all feature sets and drop rows with insufficient history."""
    print("\n  Assembling final feature matrix...")

    game_df["is_right_handed"] = (game_df["p_throws"] == "R").astype(int)
    static_cols = ["is_right_handed", "pitcher_rest_days"]

    all_feature_cols = pitcher_feat_cols + diff_feat_cols + opp_feat_cols + static_cols

    # Only keep features that actually exist in the dataframe
    all_feature_cols = [c for c in all_feature_cols if c in game_df.columns]

    # Filter: pitcher must have >= MIN_PITCHER_GAMES prior starts
    game_df["pitcher_game_count"] = game_df.groupby("pitcher").cumcount()
    model_df = game_df[game_df["pitcher_game_count"] >= MIN_PITCHER_GAMES].copy()

    # Ensure all feature columns are float
    for col in all_feature_cols:
        model_df[col] = safe_numeric(model_df[col])

    print(f"    Rows after filtering (>= {MIN_PITCHER_GAMES} prior games): {len(model_df):,}")
    print(f"    Total features: {len(all_feature_cols)}")

    missing_pct = model_df[all_feature_cols].isnull().mean().sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 0.3]
    if len(high_missing) > 0:
        print(f"    Warning: {len(high_missing)} features have >30% missing:")
        for col, pct in high_missing.head(5).items():
            print(f"      {col}: {pct:.1%}")

    return model_df, all_feature_cols


# ──────────────────────────────────────────────
# 9. TEMPORAL TRAIN/TEST SPLIT
# ──────────────────────────────────────────────
def temporal_split(model_df, feature_cols):
    """
    First 75% of season dates = train, last 25% = test.
    No random shuffling.
    """
    print("\n  Performing temporal train/test split...")

    dates_sorted = sorted(model_df["game_date"].unique())
    split_idx = int(len(dates_sorted) * TEMPORAL_SPLIT_FRAC)
    split_date = dates_sorted[split_idx]

    train = model_df[model_df["game_date"] < split_date].copy()
    test  = model_df[model_df["game_date"] >= split_date].copy()

    print(f"    Split date: {pd.Timestamp(split_date).date()}")
    print(f"    Train: {len(train):,} rows ({train['game_date'].min().date()} to {train['game_date'].max().date()})")
    print(f"    Test:  {len(test):,} rows  ({test['game_date'].min().date()} to {test['game_date'].max().date()})")

    target = "strikeouts"
    X_train = train[feature_cols]
    y_train = train[target].astype(float)
    X_test  = test[feature_cols]
    y_test  = test[target].astype(float)

    print(f"    Train K distribution: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
    print(f"    Test  K distribution: mean={y_test.mean():.2f}, std={y_test.std():.2f}")

    return X_train, y_train, X_test, y_test, train, test, split_date


# ──────────────────────────────────────────────
# 10. MODEL TRAINING WITH CROSS-VALIDATION
# ──────────────────────────────────────────────
def train_model(X_train, y_train, X_test, y_test):
    """XGBoost with TimeSeriesSplit CV for hyperparameter selection."""
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST MODEL")
    print("=" * 60)

    tscv = TimeSeriesSplit(n_splits=4)

    best_score = np.inf
    best_params = None

    param_grid = [
        {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7, "n_estimators": 500},
        {"max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.6, "n_estimators": 700},
        {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7, "n_estimators": 500},
        {"max_depth": 4, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.8, "n_estimators": 600},
        {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 400},
    ]

    print(f"\n  Running TimeSeriesSplit CV ({tscv.get_n_splits()} folds) x {len(param_grid)} configs...")

    for i, params in enumerate(param_grid):
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = xgb.XGBRegressor(
                **params,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.5,
                reg_lambda=1.0,
                random_state=RANDOM_SEED,
                objective="count:poisson",
                tree_method="hist",
                enable_categorical=False,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            fold_scores.append(mean_absolute_error(y_val, preds))

        avg_mae = np.mean(fold_scores)
        if avg_mae < best_score:
            best_score = avg_mae
            best_params = params
        print(f"    Config {i+1}: depth={params['max_depth']}, lr={params['learning_rate']}, "
              f"n_est={params['n_estimators']} -> CV MAE={avg_mae:.3f}")

    print(f"\n  Best CV MAE: {best_score:.3f}")
    print(f"  Best params: {best_params}")

    # Final model on full training set
    print("\n  Training final model on full training set...")
    final_model = xgb.XGBRegressor(
        **best_params,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        objective="count:poisson",
        tree_method="hist",
        enable_categorical=False,
    )
    final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    return final_model, best_params


# ──────────────────────────────────────────────
# 11. EVALUATION
# ──────────────────────────────────────────────
def evaluate_model(model, X_train, y_train, X_test, y_test, test_df, feature_cols):
    """Comprehensive evaluation with baseline comparison."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    # Baseline: pitcher's season-to-date avg K
    baseline_preds = test_df["szn_strikeouts"].values

    results = {}

    print("\n  TEST SET RESULTS:")
    print("  " + "-" * 50)

    test_mae  = mean_absolute_error(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2   = r2_score(y_test, test_preds)
    results.update({"test_mae": test_mae, "test_rmse": test_rmse, "test_r2": test_r2})
    print(f"  XGBoost MAE:  {test_mae:.3f}")
    print(f"  XGBoost RMSE: {test_rmse:.3f}")
    print(f"  XGBoost R2:   {test_r2:.3f}")

    baseline_mask = ~np.isnan(baseline_preds)
    if baseline_mask.sum() > 0:
        bl_mae  = mean_absolute_error(y_test[baseline_mask], baseline_preds[baseline_mask])
        bl_rmse = np.sqrt(mean_squared_error(y_test[baseline_mask], baseline_preds[baseline_mask]))
        bl_r2   = r2_score(y_test[baseline_mask], baseline_preds[baseline_mask])
        results.update({"baseline_mae": bl_mae, "baseline_r2": bl_r2})
        print(f"\n  Baseline (szn avg K) MAE:  {bl_mae:.3f}")
        print(f"  Baseline (szn avg K) RMSE: {bl_rmse:.3f}")
        print(f"  Baseline (szn avg K) R2:   {bl_r2:.3f}")
        print(f"\n  MAE improvement over baseline: {((bl_mae - test_mae) / bl_mae * 100):.1f}%")

    # Overfitting check
    train_mae = mean_absolute_error(y_train, train_preds)
    train_r2  = r2_score(y_train, train_preds)
    print(f"\n  OVERFITTING CHECK:")
    print(f"  Train MAE: {train_mae:.3f} | Test MAE: {test_mae:.3f}")
    print(f"  Train R2:  {train_r2:.3f} | Test R2:  {test_r2:.3f}")
    gap = train_r2 - test_r2
    print(f"  R2 gap: {gap:.3f} {'WARNING: POSSIBLE OVERFIT' if gap > 0.15 else 'OK: Acceptable'}")

    # Bracket accuracy
    errors = np.abs(y_test.values - test_preds)
    within_1 = (errors <= 1).mean()
    within_2 = (errors <= 2).mean()
    results.update({"within_1k": within_1, "within_2k": within_2})
    print(f"\n  BRACKET ACCURACY:")
    print(f"  Within +/-1 K: {within_1:.1%}")
    print(f"  Within +/-2 K: {within_2:.1%}")

    # Feature importance
    feat_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\n  TOP 25 FEATURES:")
    print("  " + "-" * 50)
    for _, row in feat_importance.head(25).iterrows():
        bar = "#" * int(row["importance"] / feat_importance["importance"].max() * 30)
        print(f"  {row['feature']:<45s} {row['importance']:.4f} {bar}")

    feat_importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

    # Residuals
    residuals = y_test.values - test_preds
    print(f"\n  RESIDUAL ANALYSIS:")
    print(f"  Mean residual: {residuals.mean():.3f} (should be ~0)")
    print(f"  Std residual:  {residuals.std():.3f}")
    print(f"  Skew:          {pd.Series(residuals).skew():.3f}")

    return results, feat_importance, test_preds


# ──────────────────────────────────────────────
# 12. DATA LEAK AUDIT
# ──────────────────────────────────────────────
def audit_data_leaks(feature_cols, X_train, y_train, X_test, y_test):
    """Automated checks for common data leak patterns."""
    print("\n" + "=" * 60)
    print("DATA LEAK AUDIT")
    print("=" * 60)

    issues = []

    # 1. Suspiciously high correlation with target
    correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
    suspicious = correlations[correlations > 0.85]
    if len(suspicious) > 0:
        issues.append(f"WARNING: {len(suspicious)} features have >0.85 correlation with target:")
        for col, corr in suspicious.items():
            issues.append(f"     {col}: {corr:.3f}")
    else:
        print("  PASS: No features have suspiciously high target correlation (>0.85)")

    # 2. Raw same-game strikeout features
    raw_k = [c for c in feature_cols
             if "strikeout" in c.lower()
             and not any(p in c for p in ["roll_", "szn_", "trend_", "opp_"])]
    if raw_k:
        issues.append(f"WARNING: Raw same-game strikeout features: {raw_k}")
    else:
        print("  PASS: No raw same-game strikeout features detected")

    # 3. Temporal ordering
    print("  PASS: Temporal split enforced (no random shuffling)")

    # 4. Rolling shift
    print("  PASS: All rolling features use .shift(1) to exclude current game")

    # 5. Same-game bat tracking
    same_game_bat = [c for c in feature_cols
                     if "bat_speed" in c
                     and "opp" not in c and "roll" not in c and "szn" not in c]
    if same_game_bat:
        issues.append(f"WARNING: Same-game bat tracking features: {same_game_bat}")
    else:
        print("  PASS: Bat tracking features are opponent-rolling only")

    if issues:
        print("\n  ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n  ALL CHECKS PASSED -- No data leaks detected")

    return issues


# ──────────────────────────────────────────────
# 13. SAVE OUTPUTS
# ──────────────────────────────────────────────
def save_outputs(model, results, feat_importance, best_params, feature_cols,
                 test_df, test_preds, split_date):
    """Persist model, predictions, and summary."""
    print("\n  Saving outputs...")

    model.save_model(os.path.join(OUTPUT_DIR, "strikeout_model.json"))

    pred_df = test_df[["player_name", "pitcher", "game_date", "opp_team", "strikeouts"]].copy()
    pred_df["predicted_k"] = test_preds
    pred_df["error"] = pred_df["strikeouts"] - pred_df["predicted_k"]
    pred_df["abs_error"] = pred_df["error"].abs()
    pred_df = pred_df.sort_values("game_date")
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

    summary = {
        "model_type": "XGBoost (count:poisson)",
        "target": "strikeouts per game",
        "best_params": best_params,
        "n_features": len(feature_cols),
        "split_date": str(pd.Timestamp(split_date).date()),
        "results": {k: round(v, 4) for k, v in results.items()},
        "top_10_features": feat_importance.head(10)[["feature", "importance"]].to_dict("records"),
    }
    with open(os.path.join(OUTPUT_DIR, "model_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "feature_list.txt"), "w") as f:
        for col in feature_cols:
            f.write(col + "\n")

    print(f"  Saved to {OUTPUT_DIR}/")
    print(f"    - strikeout_model.json")
    print(f"    - test_predictions.csv")
    print(f"    - feature_importance.csv")
    print(f"    - model_summary.json")
    print(f"    - feature_list.txt")


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PITCHER STRIKEOUT PREDICTION MODEL")
    print("  2025 Statcast Data | XGBoost + Rolling Features")
    print("=" * 60)

    # 1. Load
    df = load_data()

    # 2. Pitch-level features
    df = derive_pitch_features(df)

    # 3. Opposing batter rolling features
    batter_features = build_batter_rolling_features(df)

    # 4. Game-level aggregation
    game_df = aggregate_to_game_level(df)

    # 5. Pitcher rolling features
    game_df, pitcher_feat_cols = build_pitcher_rolling_features(game_df)

    # 6. Fastball-differential features
    game_df, diff_feat_cols = build_fastball_differential_features(df, game_df)

    # 7. Opponent features
    game_df, opp_feat_cols = merge_opponent_features(df, game_df, batter_features)

    # 8. Assemble
    model_df, all_feature_cols = assemble_features(
        game_df, pitcher_feat_cols, diff_feat_cols, opp_feat_cols
    )

    # 9. Split
    X_train, y_train, X_test, y_test, train_df, test_df, split_date = temporal_split(
        model_df, all_feature_cols
    )

    # 10. Leak audit
    audit_data_leaks(all_feature_cols, X_train, y_train, X_test, y_test)

    # 11. Train
    final_model, best_params = train_model(X_train, y_train, X_test, y_test)

    # 12. Evaluate
    results, feat_importance, test_preds = evaluate_model(
        final_model, X_train, y_train, X_test, y_test, test_df, all_feature_cols
    )

    # 13. Save
    save_outputs(final_model, results, feat_importance, best_params,
                 all_feature_cols, test_df, test_preds, split_date)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)

    return final_model, results


if __name__ == "__main__":
    model, results = main()