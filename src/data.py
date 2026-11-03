"""Download NBA team box scores and turn them into one row per game.

Data source: sportsdataverse/hoopR-nba-data (ESPN box scores), hosted on GitHub.
Each raw file has TWO rows per game (one from each team's point of view).
We keep regular-season and playoff games and reshape to one row per game,
with the home team's and away team's stats side by side.
"""

from pathlib import Path
import urllib.request

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
URL = ("https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/"
       "nba/team_box/parquet/team_box_{season}.parquet")

# ESPN season_type codes: 2 = regular season, 3 = playoffs (5 = play-in, skipped)
KEEP_SEASON_TYPES = {2, 3}

# Box-score columns we use to describe how a team plays.
STAT_COLS = [
    "team_score", "opponent_team_score", "field_goal_pct", "three_point_field_goal_pct",
    "free_throw_pct", "total_rebounds", "offensive_rebounds", "assists",
    "steals", "blocks", "turnovers", "points_in_paint", "fast_break_points",
]


def download(seasons):
    """Download any season files that aren't already in data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        path = RAW_DIR / f"team_box_{season}.parquet"
        if not path.exists():
            print(f"Downloading {season} ...")
            urllib.request.urlretrieve(URL.format(season=season), path)


def load_team_games(seasons):
    """One row per team per game, sorted by date."""
    frames = [pd.read_parquet(RAW_DIR / f"team_box_{s}.parquet") for s in seasons]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["season_type"].isin(KEEP_SEASON_TYPES)].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    for col in STAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["won"] = df["team_score"] > df["opponent_team_score"]
    df["point_diff"] = df["team_score"] - df["opponent_team_score"]
    keep = ["game_id", "season", "season_type", "game_date", "team_id",
            "team_abbreviation", "opponent_team_id", "team_home_away", "won", "point_diff"] + STAT_COLS
    return df[keep].sort_values(["game_date", "game_id"]).reset_index(drop=True)


if __name__ == "__main__":
    seasons = range(2015, 2027)
    download(seasons)
    games = load_team_games(seasons)
    print(games.shape)
    print(games.head())
