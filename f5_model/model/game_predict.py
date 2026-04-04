"""
Full game F5 prediction CLI.

Takes both teams' pitchers and lineups, outputs all F5 betting markets:
- Projected score and probability distributions
- Moneyline (2-way and 3-way with tie)
- Run lines
- Totals
- Alternate run lines and totals
- Winning margins
- Edge finder (when FanDuel odds provided)

Usage:
    python -m f5_model.model.game_predict \
        --away-pitcher "Hunter Greene" \
        --away-lineup "Batter1,Batter2,..." \
        --away-team "CIN" \
        --home-pitcher "Nathan Eovaldi" \
        --home-lineup "Batter1,Batter2,..." \
        --home-team "TEX" \
        --date 2026-04-04 \
        --park TEX

With FanDuel odds comparison:
    python -m f5_model.model.game_predict \
        ... \
        --fd-away-ml 114 \
        --fd-home-ml -142 \
        --fd-over-odds -114 \
        --fd-under-odds -114 \
        --fd-total 4.5
"""

import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import poisson

from f5_model.model.predict import (
    load_model_and_features,
    lookup_player_id,
    predict_f5_runs
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def prob_to_american_odds(prob: float) -> str:
    """Convert probability to American odds format."""
    if prob <= 0 or prob >= 1:
        return "N/A"

    if prob >= 0.5:
        odds = -100 * prob / (1 - prob)
        return f"{int(round(odds))}"
    else:
        odds = 100 * (1 - prob) / prob
        return f"+{int(round(odds))}"


def american_odds_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def compute_all_probabilities(lambda_away: float, lambda_home: float, max_runs: int = 15) -> Dict:
    """
    Compute all joint probabilities for the game.

    Returns a matrix of P(away=i, home=j) for all i,j.
    """
    away_probs = np.array([poisson.pmf(k, lambda_away) for k in range(max_runs + 1)])
    home_probs = np.array([poisson.pmf(k, lambda_home) for k in range(max_runs + 1)])

    # Joint probability matrix
    joint = np.outer(away_probs, home_probs)

    return {
        'away_probs': away_probs,
        'home_probs': home_probs,
        'joint': joint,
        'max_runs': max_runs
    }


def compute_moneyline_3way(probs: Dict) -> Dict:
    """Compute 3-way moneyline (away win, tie, home win)."""
    joint = probs['joint']
    max_runs = probs['max_runs']

    p_away_win = 0.0
    p_home_win = 0.0
    p_tie = 0.0

    for i in range(max_runs + 1):
        for j in range(max_runs + 1):
            if i > j:
                p_away_win += joint[i, j]
            elif j > i:
                p_home_win += joint[i, j]
            else:
                p_tie += joint[i, j]

    return {
        'away_win': p_away_win,
        'home_win': p_home_win,
        'tie': p_tie
    }


def compute_moneyline_2way(ml_3way: Dict) -> Dict:
    """Compute 2-way moneyline (excluding ties)."""
    total = ml_3way['away_win'] + ml_3way['home_win']
    return {
        'away_win': ml_3way['away_win'] / total,
        'home_win': ml_3way['home_win'] / total
    }


def compute_run_line(probs: Dict, spread: float) -> Dict:
    """
    Compute run line probabilities.

    spread is from the team's perspective:
    - Team +0.5 means team can lose by 0 and still cover
    - Team -0.5 means team must win by 1+
    """
    joint = probs['joint']
    max_runs = probs['max_runs']

    p_away_cover = 0.0
    p_home_cover = 0.0

    for i in range(max_runs + 1):
        for j in range(max_runs + 1):
            margin = i - j  # away margin (positive = away winning)
            if margin > -spread:
                p_away_cover += joint[i, j]
            if -margin > -spread:
                p_home_cover += joint[i, j]

    return {
        'away_cover': p_away_cover,
        'home_cover': p_home_cover
    }


def compute_total(probs: Dict, line: float) -> Dict:
    """Compute over/under probabilities for a total line."""
    joint = probs['joint']
    max_runs = probs['max_runs']

    p_over = 0.0
    for i in range(max_runs + 1):
        for j in range(max_runs + 1):
            if i + j > line:
                p_over += joint[i, j]

    return {
        'over': p_over,
        'under': 1 - p_over
    }


def compute_winning_margin(probs: Dict) -> Dict:
    """Compute winning margin probabilities."""
    joint = probs['joint']
    max_runs = probs['max_runs']

    results = {
        'away_by_1': 0.0,
        'away_by_2': 0.0,
        'away_by_3_plus': 0.0,
        'tie': 0.0,
        'home_by_1': 0.0,
        'home_by_2': 0.0,
        'home_by_3_plus': 0.0,
    }

    for i in range(max_runs + 1):
        for j in range(max_runs + 1):
            margin = i - j
            if margin == 0:
                results['tie'] += joint[i, j]
            elif margin == 1:
                results['away_by_1'] += joint[i, j]
            elif margin == 2:
                results['away_by_2'] += joint[i, j]
            elif margin >= 3:
                results['away_by_3_plus'] += joint[i, j]
            elif margin == -1:
                results['home_by_1'] += joint[i, j]
            elif margin == -2:
                results['home_by_2'] += joint[i, j]
            elif margin <= -3:
                results['home_by_3_plus'] += joint[i, j]

    # Combined margins
    results['away_by_1_2'] = results['away_by_1'] + results['away_by_2']
    results['home_by_1_2'] = results['home_by_1'] + results['home_by_2']

    return results


def compute_exact_scores(probs: Dict, top_n: int = 10) -> List[Tuple]:
    """Get the most likely exact scores."""
    joint = probs['joint']
    max_runs = probs['max_runs']

    scores = []
    for i in range(min(max_runs + 1, 10)):
        for j in range(min(max_runs + 1, 10)):
            scores.append((i, j, joint[i, j]))

    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_n]


def format_odds_line(prob: float) -> str:
    """Format probability with odds."""
    odds = prob_to_american_odds(prob)
    return f"{prob:5.1%} ({odds:>5})"


def calculate_edge(model_prob: float, book_odds: int) -> float:
    """Calculate edge: model probability - implied probability."""
    implied = american_odds_to_prob(book_odds)
    return model_prob - implied


def format_edge(edge: float) -> str:
    """Format edge with color indicator."""
    if edge > 0.05:
        return f"+{edge:.1%} **VALUE**"
    elif edge > 0.02:
        return f"+{edge:.1%} *"
    elif edge > 0:
        return f"+{edge:.1%}"
    else:
        return f"{edge:.1%}"


def format_game_output(
    away_team: str,
    home_team: str,
    away_pitcher: str,
    home_pitcher: str,
    away_lambda: float,
    home_lambda: float,
    probs: Dict,
    fd_odds: Optional[Dict] = None
) -> str:
    """Format the full game prediction output matching FanDuel markets."""
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append(f"F5 PREDICTION: {away_team} @ {home_team}")
    lines.append("=" * 70)
    lines.append(f"\n  {away_team} Pitcher: {away_pitcher}")
    lines.append(f"  {home_team} Pitcher: {home_pitcher}")

    # =========================================================================
    # PROJECTED SCORE & DISTRIBUTION
    # =========================================================================
    lines.append(f"\n{'=' * 70}")
    lines.append("PROJECTED F5 SCORE")
    lines.append("=" * 70)

    lines.append(f"\n  {away_team:>6}:  {away_lambda:.2f} runs")
    lines.append(f"  {home_team:>6}:  {home_lambda:.2f} runs")
    lines.append(f"  {'Total':>6}:  {away_lambda + home_lambda:.2f} runs")

    # Score differential
    diff = home_lambda - away_lambda
    if abs(diff) < 0.1:
        spread_str = "PICK'EM"
    elif diff > 0:
        spread_str = f"{home_team} favored by {diff:.2f}"
    else:
        spread_str = f"{away_team} favored by {-diff:.2f}"
    lines.append(f"  {'Spread':>6}:  {spread_str}")

    # Probability Distribution
    lines.append(f"\n  {away_team} Run Distribution:")
    lines.append("  " + "-" * 40)
    for runs in range(7):
        prob = probs['away_probs'][runs]
        bar = "#" * int(prob * 50)
        lines.append(f"    {runs} runs: {prob:5.1%} {bar}")
    lines.append(f"    7+:    {1 - sum(probs['away_probs'][:7]):5.1%}")

    lines.append(f"\n  {home_team} Run Distribution:")
    lines.append("  " + "-" * 40)
    for runs in range(7):
        prob = probs['home_probs'][runs]
        bar = "#" * int(prob * 50)
        lines.append(f"    {runs} runs: {prob:5.1%} {bar}")
    lines.append(f"    7+:    {1 - sum(probs['home_probs'][:7]):5.1%}")

    # =========================================================================
    # EDGE FINDER (if FanDuel odds provided)
    # =========================================================================
    if fd_odds:
        ml_3way = compute_moneyline_3way(probs)
        ml_2way = compute_moneyline_2way(ml_3way)

        lines.append(f"\n{'=' * 70}")
        lines.append("EDGE FINDER - Model vs FanDuel")
        lines.append("=" * 70)
        lines.append(f"\n  {'Market':<25} {'Model':>8} {'FanDuel':>10} {'Edge':>15}")
        lines.append("  " + "-" * 60)

        # Moneyline edges
        if fd_odds.get('away_ml') is not None:
            edge = calculate_edge(ml_2way['away_win'], fd_odds['away_ml'])
            lines.append(f"  {away_team + ' ML':<25} {prob_to_american_odds(ml_2way['away_win']):>8} {fd_odds['away_ml']:>+10} {format_edge(edge):>15}")

        if fd_odds.get('home_ml') is not None:
            edge = calculate_edge(ml_2way['home_win'], fd_odds['home_ml'])
            lines.append(f"  {home_team + ' ML':<25} {prob_to_american_odds(ml_2way['home_win']):>8} {fd_odds['home_ml']:>+10} {format_edge(edge):>15}")

        # 3-way edges
        if fd_odds.get('away_3way') is not None:
            edge = calculate_edge(ml_3way['away_win'], fd_odds['away_3way'])
            lines.append(f"  {away_team + ' (3-way)':<25} {prob_to_american_odds(ml_3way['away_win']):>8} {fd_odds['away_3way']:>+10} {format_edge(edge):>15}")

        if fd_odds.get('tie_3way') is not None:
            edge = calculate_edge(ml_3way['tie'], fd_odds['tie_3way'])
            lines.append(f"  {'Tie (3-way)':<25} {prob_to_american_odds(ml_3way['tie']):>8} {fd_odds['tie_3way']:>+10} {format_edge(edge):>15}")

        if fd_odds.get('home_3way') is not None:
            edge = calculate_edge(ml_3way['home_win'], fd_odds['home_3way'])
            lines.append(f"  {home_team + ' (3-way)':<25} {prob_to_american_odds(ml_3way['home_win']):>8} {fd_odds['home_3way']:>+10} {format_edge(edge):>15}")

        # Total edges
        if fd_odds.get('total') is not None:
            total_probs = compute_total(probs, fd_odds['total'])

            if fd_odds.get('over_odds') is not None:
                edge = calculate_edge(total_probs['over'], fd_odds['over_odds'])
                lines.append(f"  {'Over ' + str(fd_odds['total']):<25} {prob_to_american_odds(total_probs['over']):>8} {fd_odds['over_odds']:>+10} {format_edge(edge):>15}")

            if fd_odds.get('under_odds') is not None:
                edge = calculate_edge(total_probs['under'], fd_odds['under_odds'])
                lines.append(f"  {'Under ' + str(fd_odds['total']):<25} {prob_to_american_odds(total_probs['under']):>8} {fd_odds['under_odds']:>+10} {format_edge(edge):>15}")

        # Run line edges
        if fd_odds.get('away_rl_odds') is not None:
            rl_spread = fd_odds.get('away_rl_spread', 0.5)
            rl = compute_run_line(probs, rl_spread)
            edge = calculate_edge(rl['away_cover'], fd_odds['away_rl_odds'])
            lines.append(f"  {away_team + ' +' + str(rl_spread):<25} {prob_to_american_odds(rl['away_cover']):>8} {fd_odds['away_rl_odds']:>+10} {format_edge(edge):>15}")

        if fd_odds.get('home_rl_odds') is not None:
            rl_spread = fd_odds.get('home_rl_spread', -0.5)
            rl = compute_run_line(probs, -rl_spread)
            edge = calculate_edge(rl['home_cover'], fd_odds['home_rl_odds'])
            lines.append(f"  {home_team + ' ' + str(rl_spread):<25} {prob_to_american_odds(rl['home_cover']):>8} {fd_odds['home_rl_odds']:>+10} {format_edge(edge):>15}")

        lines.append("\n  Legend: ** VALUE ** = 5%+ edge, * = 2-5% edge")

    # =========================================================================
    # BETTING MARKETS - MODEL FAIR VALUE
    # =========================================================================

    # 3-Way Result
    ml_3way = compute_moneyline_3way(probs)
    lines.append(f"\n{'=' * 70}")
    lines.append("FIRST 5 INNINGS RESULT (3-Way)")
    lines.append("=" * 70)
    lines.append(f"\n  {away_team:<15} {format_odds_line(ml_3way['away_win'])}")
    lines.append(f"  {'Tie':<15} {format_odds_line(ml_3way['tie'])}")
    lines.append(f"  {home_team:<15} {format_odds_line(ml_3way['home_win'])}")

    # 2-Way Moneyline
    ml_2way = compute_moneyline_2way(ml_3way)
    lines.append(f"\n{'=' * 70}")
    lines.append("FIRST 5 INNINGS MONEY LINE (2-Way)")
    lines.append("=" * 70)
    lines.append(f"\n  {away_team:<15} {format_odds_line(ml_2way['away_win'])}")
    lines.append(f"  {home_team:<15} {format_odds_line(ml_2way['home_win'])}")

    # Standard Run Line (+/- 0.5)
    rl = compute_run_line(probs, 0.5)
    rl_rev = compute_run_line(probs, -0.5)
    lines.append(f"\n{'=' * 70}")
    lines.append("FIRST 5 INNINGS RUN LINE")
    lines.append("=" * 70)
    lines.append(f"\n  {away_team} +0.5      {format_odds_line(rl['away_cover'])}")
    lines.append(f"  {home_team} -0.5      {format_odds_line(rl['home_cover'])}")
    lines.append("")
    lines.append(f"  {away_team} -0.5      {format_odds_line(rl_rev['away_cover'])}")
    lines.append(f"  {home_team} +0.5      {format_odds_line(rl_rev['home_cover'])}")

    # Standard Total
    total_45 = compute_total(probs, 4.5)
    lines.append(f"\n{'=' * 70}")
    lines.append("FIRST 5 INNINGS TOTAL RUNS")
    lines.append("=" * 70)
    lines.append(f"\n  Over 4.5        {format_odds_line(total_45['over'])}")
    lines.append(f"  Under 4.5       {format_odds_line(total_45['under'])}")

    # Alternate Run Lines
    lines.append(f"\n{'=' * 70}")
    lines.append("FIRST 5 INNINGS ALTERNATE RUN LINES")
    lines.append("=" * 70)
    lines.append(f"\n  {'Line':<18} {'Prob':>7} {'Odds':>8}")
    lines.append("  " + "-" * 35)

    for spread in [3.5, 2.5, 1.5, 0.5, -0.5, -1.5, -2.5, -3.5]:
        rl = compute_run_line(probs, spread)
        sign = "+" if spread >= 0 else ""
        lines.append(f"  {away_team} {sign}{spread:<12} {rl['away_cover']:>6.1%} {prob_to_american_odds(rl['away_cover']):>8}")

    lines.append("")
    for spread in [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]:
        rl = compute_run_line(probs, spread)
        sign = "+" if spread >= 0 else ""
        lines.append(f"  {home_team} {sign}{spread:<12} {rl['home_cover']:>6.1%} {prob_to_american_odds(rl['home_cover']):>8}")

    # Alternate Totals
    lines.append(f"\n{'=' * 70}")
    lines.append("FIRST 5 INNINGS ALTERNATE TOTAL RUNS")
    lines.append("=" * 70)
    lines.append(f"\n  {'Line':<12} {'Over':>8} {'Odds':>8}  |  {'Under':>8} {'Odds':>8}")
    lines.append("  " + "-" * 55)

    for total_line in [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]:
        t = compute_total(probs, total_line)
        lines.append(f"  {total_line:<12} {t['over']:>7.1%} {prob_to_american_odds(t['over']):>8}  |  {t['under']:>7.1%} {prob_to_american_odds(t['under']):>8}")

    # Winning Margin (5-Way)
    wm = compute_winning_margin(probs)
    lines.append(f"\n{'=' * 70}")
    lines.append("FIRST 5 INNINGS WINNING MARGIN (5-Way)")
    lines.append("=" * 70)
    lines.append(f"\n  {away_team} Win By 1-2 Runs     {format_odds_line(wm['away_by_1_2'])}")
    lines.append(f"  {away_team} Win By 3+ Runs      {format_odds_line(wm['away_by_3_plus'])}")
    lines.append(f"  Tie                       {format_odds_line(wm['tie'])}")
    lines.append(f"  {home_team} Win By 1-2 Runs     {format_odds_line(wm['home_by_1_2'])}")
    lines.append(f"  {home_team} Win By 3+ Runs      {format_odds_line(wm['home_by_3_plus'])}")

    # Winning Margin (Exact)
    lines.append(f"\n{'=' * 70}")
    lines.append("FIRST 5 INNINGS WINNING MARGIN (Exact)")
    lines.append("=" * 70)
    lines.append(f"\n  {away_team} Win By 1 Run        {format_odds_line(wm['away_by_1'])}")
    lines.append(f"  {away_team} Win By 2 Runs       {format_odds_line(wm['away_by_2'])}")
    lines.append(f"  {away_team} Win By 3+ Runs      {format_odds_line(wm['away_by_3_plus'])}")
    lines.append(f"  Tie                       {format_odds_line(wm['tie'])}")
    lines.append(f"  {home_team} Win By 1 Run        {format_odds_line(wm['home_by_1'])}")
    lines.append(f"  {home_team} Win By 2 Runs       {format_odds_line(wm['home_by_2'])}")
    lines.append(f"  {home_team} Win By 3+ Runs      {format_odds_line(wm['home_by_3_plus'])}")

    # Most Likely Scores
    exact_scores = compute_exact_scores(probs)
    lines.append(f"\n{'=' * 70}")
    lines.append("MOST LIKELY F5 SCORES")
    lines.append("=" * 70)
    lines.append(f"\n  {'Score':<15} {'Prob':>7} {'Odds':>8}")
    lines.append("  " + "-" * 32)
    for away_score, home_score, prob in exact_scores:
        score_str = f"{away_team} {away_score} - {home_team} {home_score}"
        lines.append(f"  {score_str:<15} {prob:>6.1%} {prob_to_american_odds(prob):>8}")

    # Summary
    lines.append(f"\n{'=' * 70}")
    lines.append("SUMMARY")
    lines.append("=" * 70)

    if ml_2way['home_win'] > ml_2way['away_win']:
        fav_team, fav_prob = home_team, ml_2way['home_win']
        dog_team, dog_prob = away_team, ml_2way['away_win']
    else:
        fav_team, fav_prob = away_team, ml_2way['away_win']
        dog_team, dog_prob = home_team, ml_2way['home_win']

    lines.append(f"\n  Projected Score: {away_team} {away_lambda:.1f} - {home_team} {home_lambda:.1f}")
    lines.append(f"  Projected Total: {away_lambda + home_lambda:.1f}")
    lines.append(f"  Favorite: {fav_team} ({prob_to_american_odds(fav_prob)})")
    lines.append(f"  Underdog: {dog_team} ({prob_to_american_odds(dog_prob)})")
    lines.append(f"  Tie Probability: {ml_3way['tie']:.1%}")

    return "\n".join(lines)


def parse_lineup(lineup_str: str) -> List[str]:
    """Parse comma-separated lineup string."""
    return [name.strip() for name in lineup_str.split(',')]


def lookup_lineup_ids(names: List[str]) -> Tuple[List[int], List[str]]:
    """Look up MLB IDs for a list of player names."""
    ids = []
    found_names = []

    for name in names:
        player_id = lookup_player_id(name)
        if player_id:
            ids.append(player_id)
            found_names.append(name)
            print(f"    {name}: {player_id}")
        else:
            print(f"    {name}: NOT FOUND")

    return ids, found_names


def main():
    parser = argparse.ArgumentParser(
        description="Predict F5 outcome for a full game matchup"
    )

    # Away team
    parser.add_argument("--away-pitcher", "-ap", required=True, help="Away team starting pitcher name")
    parser.add_argument("--away-lineup", "-al", required=True, help="Away team lineup (comma-separated names)")
    parser.add_argument("--away-pitcher-id", type=int, help="Away pitcher MLB ID (skip name lookup)")
    parser.add_argument("--away-lineup-ids", help="Away lineup MLB IDs (comma-separated)")
    parser.add_argument("--away-hand", choices=['L', 'R'], default='R', help="Away pitcher handedness")
    parser.add_argument("--away-team", default="AWAY", help="Away team name for display")

    # Home team
    parser.add_argument("--home-pitcher", "-hp", required=True, help="Home team starting pitcher name")
    parser.add_argument("--home-lineup", "-hl", required=True, help="Home team lineup (comma-separated names)")
    parser.add_argument("--home-pitcher-id", type=int, help="Home pitcher MLB ID (skip name lookup)")
    parser.add_argument("--home-lineup-ids", help="Home lineup MLB IDs (comma-separated)")
    parser.add_argument("--home-hand", choices=['L', 'R'], default='R', help="Home pitcher handedness")
    parser.add_argument("--home-team", default="HOME", help="Home team name for display")

    # Game context
    parser.add_argument("--date", "-d", required=True, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--park", default="NYY", help="Park code for park factor")

    # FanDuel odds for comparison (optional)
    parser.add_argument("--fd-away-ml", type=int, help="FanDuel away team moneyline odds (e.g., +114)")
    parser.add_argument("--fd-home-ml", type=int, help="FanDuel home team moneyline odds (e.g., -142)")
    parser.add_argument("--fd-away-3way", type=int, help="FanDuel away team 3-way odds")
    parser.add_argument("--fd-home-3way", type=int, help="FanDuel home team 3-way odds")
    parser.add_argument("--fd-tie-3way", type=int, help="FanDuel tie 3-way odds")
    parser.add_argument("--fd-total", type=float, help="FanDuel total line (e.g., 4.5)")
    parser.add_argument("--fd-over-odds", type=int, help="FanDuel over odds (e.g., -114)")
    parser.add_argument("--fd-under-odds", type=int, help="FanDuel under odds (e.g., -114)")
    parser.add_argument("--fd-away-rl-spread", type=float, default=0.5, help="FanDuel away run line spread")
    parser.add_argument("--fd-away-rl-odds", type=int, help="FanDuel away run line odds")
    parser.add_argument("--fd-home-rl-spread", type=float, default=-0.5, help="FanDuel home run line spread")
    parser.add_argument("--fd-home-rl-odds", type=int, help="FanDuel home run line odds")

    args = parser.parse_args()

    # Build FanDuel odds dict if any provided
    fd_odds = None
    if any([args.fd_away_ml, args.fd_home_ml, args.fd_total]):
        fd_odds = {
            'away_ml': args.fd_away_ml,
            'home_ml': args.fd_home_ml,
            'away_3way': args.fd_away_3way,
            'home_3way': args.fd_home_3way,
            'tie_3way': args.fd_tie_3way,
            'total': args.fd_total,
            'over_odds': args.fd_over_odds,
            'under_odds': args.fd_under_odds,
            'away_rl_spread': args.fd_away_rl_spread,
            'away_rl_odds': args.fd_away_rl_odds,
            'home_rl_spread': args.fd_home_rl_spread,
            'home_rl_odds': args.fd_home_rl_odds,
        }

    # Load model
    print("Loading model...")
    model, feature_names = load_model_and_features()

    # Parse lineups
    away_lineup_names = parse_lineup(args.away_lineup)
    home_lineup_names = parse_lineup(args.home_lineup)

    # Look up away pitcher
    if args.away_pitcher_id:
        away_pitcher_id = args.away_pitcher_id
    else:
        print(f"\nLooking up away pitcher: {args.away_pitcher}...")
        away_pitcher_id = lookup_player_id(args.away_pitcher)
        if away_pitcher_id is None:
            print(f"ERROR: Could not find pitcher '{args.away_pitcher}'")
            return
        print(f"  {args.away_pitcher}: {away_pitcher_id}")

    # Look up home pitcher
    if args.home_pitcher_id:
        home_pitcher_id = args.home_pitcher_id
    else:
        print(f"\nLooking up home pitcher: {args.home_pitcher}...")
        home_pitcher_id = lookup_player_id(args.home_pitcher)
        if home_pitcher_id is None:
            print(f"ERROR: Could not find pitcher '{args.home_pitcher}'")
            return
        print(f"  {args.home_pitcher}: {home_pitcher_id}")

    # Look up away lineup
    if args.away_lineup_ids:
        away_lineup_ids = [int(x) for x in args.away_lineup_ids.split(',')]
    else:
        print(f"\nLooking up away lineup...")
        away_lineup_ids, _ = lookup_lineup_ids(away_lineup_names)

    if len(away_lineup_ids) == 0:
        print("ERROR: No batters found in away lineup")
        return

    # Look up home lineup
    if args.home_lineup_ids:
        home_lineup_ids = [int(x) for x in args.home_lineup_ids.split(',')]
    else:
        print(f"\nLooking up home lineup...")
        home_lineup_ids, _ = lookup_lineup_ids(home_lineup_names)

    if len(home_lineup_ids) == 0:
        print("ERROR: No batters found in home lineup")
        return

    # Predict F5 runs for each side
    print(f"\nPredicting F5 runs...")

    # Away team scoring = Home pitcher vs Away lineup
    away_result = predict_f5_runs(
        model=model,
        feature_names=feature_names,
        pitcher_id=home_pitcher_id,
        pitcher_hand=args.home_hand,
        lineup_ids=away_lineup_ids,
        date=args.date,
        starter_is_home=True,
        park=args.park
    )
    away_lambda = away_result['predicted_runs']

    # Home team scoring = Away pitcher vs Home lineup
    home_result = predict_f5_runs(
        model=model,
        feature_names=feature_names,
        pitcher_id=away_pitcher_id,
        pitcher_hand=args.away_hand,
        lineup_ids=home_lineup_ids,
        date=args.date,
        starter_is_home=False,
        park=args.park
    )
    home_lambda = home_result['predicted_runs']

    # Compute all probabilities
    probs = compute_all_probabilities(away_lambda, home_lambda)

    # Format and print output
    output = format_game_output(
        away_team=args.away_team,
        home_team=args.home_team,
        away_pitcher=args.away_pitcher,
        home_pitcher=args.home_pitcher,
        away_lambda=away_lambda,
        home_lambda=home_lambda,
        probs=probs,
        fd_odds=fd_odds
    )

    print(output)


if __name__ == "__main__":
    main()
