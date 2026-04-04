"""
Constants and configuration for F5 runs prediction model.
"""

from typing import List

# Key columns to retain from Statcast pitch-level data
STATCAST_COLUMNS: List[str] = [
    # Identifiers
    "pitcher",
    "batter",
    "game_pk",
    "game_date",
    "at_bat_number",
    "pitch_number",

    # Inning info - CRITICAL for F5 filtering
    "inning",
    "inning_topbot",

    # Event outcomes
    "events",
    "description",
    "des",

    # Pitch characteristics
    "release_speed",
    "effective_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "pitch_type",
    "pitch_name",

    # Batted ball data
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",

    # Expected stats
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",

    # Outcome values
    "woba_value",
    "woba_denom",
    "babip_value",

    # Context
    "zone",
    "stand",
    "p_throws",
    "balls",
    "strikes",
    "outs_when_up",

    # Teams and scores
    "home_team",
    "away_team",
    "bat_score",
    "fld_score",
    "post_bat_score",
]

# Date ranges for data pulls (monthly chunks to avoid timeouts)
MONTHS_2024 = [
    ("2024-03-28", "2024-04-30"),
    ("2024-05-01", "2024-05-31"),
    ("2024-06-01", "2024-06-30"),
    ("2024-07-01", "2024-07-31"),
    ("2024-08-01", "2024-08-31"),
    ("2024-09-01", "2024-09-29"),
]

MONTHS_2025 = [
    ("2025-03-27", "2025-04-30"),
    ("2025-05-01", "2025-05-31"),
    ("2025-06-01", "2025-06-30"),
    ("2025-07-01", "2025-07-31"),
    ("2025-08-01", "2025-08-31"),
    ("2025-09-01", "2025-09-28"),
]

# Lineup weighting for aggregation (linear decay)
# Slot 1 = 1.0, Slot 9 = 0.6
LINEUP_WEIGHTS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]

# Pitch types for pitch mix features
PITCH_TYPES = ["FF", "SL", "CH", "CU", "SI", "FC", "FS", "KC", "ST", "SV"]

# Park factors (runs, league average = 1.0)
# Source: 2024 park factors, can be updated annually
PARK_FACTORS = {
    "COL": 1.38,  # Coors Field
    "CIN": 1.12,  # Great American Ball Park
    "BOS": 1.10,  # Fenway Park
    "TEX": 1.08,  # Globe Life Field
    "PHI": 1.06,  # Citizens Bank Park
    "ARI": 1.05,  # Chase Field
    "CHC": 1.04,  # Wrigley Field
    "MIL": 1.03,  # American Family Field
    "TOR": 1.02,  # Rogers Centre
    "BAL": 1.01,  # Camden Yards
    "MIN": 1.01,  # Target Field
    "ATL": 1.00,  # Truist Park
    "NYY": 1.00,  # Yankee Stadium
    "LAA": 0.99,  # Angel Stadium
    "CLE": 0.98,  # Progressive Field
    "WSH": 0.98,  # Nationals Park
    "DET": 0.97,  # Comerica Park
    "HOU": 0.97,  # Minute Maid Park
    "KC": 0.96,   # Kauffman Stadium
    "STL": 0.96,  # Busch Stadium
    "SF": 0.95,   # Oracle Park
    "CHW": 0.95,  # Guaranteed Rate Field
    "CWS": 0.95,  # Alternate code for White Sox
    "PIT": 0.94,  # PNC Park
    "SD": 0.93,   # Petco Park
    "NYM": 0.93,  # Citi Field
    "LAD": 0.92,  # Dodger Stadium
    "TB": 0.91,   # Tropicana Field
    "SEA": 0.90,  # T-Mobile Park
    "OAK": 0.89,  # Oakland Coliseum
    "MIA": 0.88,  # LoanDepot Park
}

# FIP constant (approximately 3.2, varies slightly by year)
FIP_CONSTANT = 3.20
