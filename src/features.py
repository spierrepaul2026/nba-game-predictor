"""Build pregame features for every game.

The golden rule: a feature for a game may only use information from BEFORE
tip-off. Every rolling average below is shifted by one game, so a team's
stats from tonight never leak into tonight's prediction.
"""

import numpy as np
import pandas as pd

from data import STAT_COLS

ROLL_WINDOW = 10  # "recent form" = average of the last 10 games


# ---------------------------------------------------------------- Elo ratings
def add_elo(team_games, k=20, home_adv=100, start=1500, carry=0.75):
    """Pregame Elo rating for each team in each game.

    Elo is a running strength rating: beat a strong team and you gain a lot,
    lose to a weak team and you drop a lot. Between seasons, ratings move
    25% of the way back toward average (carry=0.75).
    """
    games = (team_games[team_games["team_home_away"] == "home"]
             .sort_values(["game_date", "game_id"]))
    rating, last_season = {}, {}
    pre_home, pre_away = {}, {}

    for g in games.itertuples():
        for team in (g.team_id, g.opponent_team_id):
            r = rating.get(team, start)
            if team in last_season and last_season[team] != g.season:
                r = start + carry * (r - start)  # new season: regress to the mean
            rating[team], last_season[team] = r, g.season

        home, away = g.team_id, g.opponent_team_id
        pre_home[g.game_id], pre_away[g.game_id] = rating[home], rating[away]

        expected_home = 1 / (1 + 10 ** (-(rating[home] + home_adv - rating[away]) / 400))
        # Bigger wins move ratings more (FiveThirtyEight-style margin multiplier).
        margin = abs(g.point_diff)
        mult = np.log(margin + 1) * 2.2 / ((abs(rating[home] - rating[away]) * 0.001) + 2.2)
        change = k * mult * ((1.0 if g.won else 0.0) - expected_home)
        rating[home] += change
        rating[away] -= change

    is_home = team_games["team_home_away"] == "home"
    team_games["elo_pre"] = np.where(
        is_home,
        team_games["game_id"].map(pre_home),
        team_games["game_id"].map(pre_away),
    )
    return team_games


# ------------------------------------------------------------ rolling form
def add_rolling_form(team_games):
    """Each team's recent and season-to-date averages, using past games only."""
    tg = team_games.sort_values(["team_id", "game_date", "game_id"]).copy()
    by_team_season = tg.groupby(["team_id", "season"], group_keys=False)

    cols = STAT_COLS + ["point_diff", "won"]
    for col in cols:
        past = by_team_season[col].shift(1)  # shift(1) = exclude tonight's game
        tg[f"{col}_last{ROLL_WINDOW}"] = (
            past.groupby([tg["team_id"], tg["season"]])
            .transform(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())
        )
    tg["win_pct_season"] = by_team_season["won"].transform(
        lambda s: s.shift(1).expanding().mean())
    tg["games_played"] = by_team_season.cumcount()

    # Rest: days since the team's previous game (capped at 7), and back-to-backs.
    prev = tg.groupby("team_id")["game_date"].shift(1)
    tg["rest_days"] = (tg["game_date"] - prev).dt.days.clip(upper=7)
    tg["back_to_back"] = (tg["rest_days"] == 1).astype(float)
    return tg


# --------------------------------------------------------- one row per game
def build_game_table(team_games):
    """Put home and away features side by side and add home-minus-away differences."""
    tg = add_rolling_form(add_elo(team_games.copy()))
    feature_cols = [c for c in tg.columns if c.endswith(f"_last{ROLL_WINDOW}")] + [
        "elo_pre", "win_pct_season", "rest_days", "back_to_back", "games_played"]

    home = tg[tg["team_home_away"] == "home"]
    away = tg[tg["team_home_away"] == "away"]
    base = home[["game_id", "season", "season_type", "game_date", "team_abbreviation", "won"]].rename(
        columns={"team_abbreviation": "home_team", "won": "home_win"})
    away_name = away[["game_id", "team_abbreviation"]].rename(columns={"team_abbreviation": "away_team"})

    h = home[["game_id"] + feature_cols].set_index("game_id").add_prefix("home_")
    a = away[["game_id"] + feature_cols].set_index("game_id").add_prefix("away_")
    games = base.merge(away_name, on="game_id").set_index("game_id").join(h).join(a).reset_index()

    for c in feature_cols:
        games[f"diff_{c}"] = games[f"home_{c}"] - games[f"away_{c}"]
    games["home_win"] = games["home_win"].astype(int)
    return games.sort_values("game_date").reset_index(drop=True)
