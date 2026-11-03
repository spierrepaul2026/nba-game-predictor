"""Train and compare models on seasons they have never seen.

Split by time, never at random: train on 2015-2023, tune on 2024,
and report final results on 2025 and 2026. A random split would let the
model learn from games played after the ones it is predicting.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import download, load_team_games
from features import build_game_table

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SEASONS = range(2015, 2027)
TRAIN, VALID, TEST = range(2015, 2024), [2024], [2025, 2026]


def score(name, y, p):
    return {"model": name, "accuracy": accuracy_score(y, p >= 0.5),
            "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), "brier": brier_score_loss(y, p)}


def main():
    download(SEASONS)
    games = build_game_table(load_team_games(SEASONS))
    # Skip each team's first 5 games of a season: there's no form to measure yet.
    games = games[(games["home_games_played"] >= 5) & (games["away_games_played"] >= 5)]

    features = [c for c in games.columns if c.startswith("diff_")]
    train = games[games["season"].isin(TRAIN)]
    valid = games[games["season"].isin(VALID)]
    test = games[games["season"].isin(TEST)]
    print(f"games  train={len(train)}  valid={len(valid)}  test={len(test)}")

    X_tr, y_tr = train[features], train["home_win"]
    models = {
        "Logistic regression": make_pipeline(
            SimpleImputer(), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.03, max_depth=3, l2_regularization=1.0, random_state=0),
    }
    for m in models.values():
        m.fit(X_tr, y_tr)

    rows = []
    for split_name, df in [("valid (2024)", valid), ("test (2025-26)", test)]:
        y = df["home_win"]
        home_rate = y_tr.mean()  # how often the home team won in training
        elo_p = 1 / (1 + 10 ** (-(df["diff_elo_pre"] + 100) / 400))
        preds = {"Always pick home team": np.full(len(df), home_rate),
                 "Elo rating only": elo_p.to_numpy()}
        preds.update({n: m.predict_proba(df[features])[:, 1] for n, m in models.items()})
        for name, p in preds.items():
            rows.append({"split": split_name, **score(name, y, p)})
    results = pd.DataFrame(rows)

    REPORTS.mkdir(exist_ok=True)
    table = results.round(3).to_markdown(index=False)
    (REPORTS / "results.md").write_text("# Results\n\n" + table + "\n")
    print(table)

    # Calibration: when the model says 70%, does the home team win ~70% of the time?
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    for name, m in models.items():
        frac, mean_p = calibration_curve(test["home_win"], m.predict_proba(test[features])[:, 1], n_bins=10)
        ax.plot(mean_p, frac, marker="o", label=name)
    ax.set_xlabel("Predicted home-win probability")
    ax.set_ylabel("Actual home-win rate")
    ax.set_title("Calibration on 2025-26 test games")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "calibration.png", dpi=150)

    # Which features matter most to the logistic regression?
    lr = models["Logistic regression"][-1]
    coefs = pd.Series(lr.coef_[0], index=features).sort_values(key=abs, ascending=False)
    print("\nTop logistic-regression features:\n", coefs.head(8).round(3).to_string())


if __name__ == "__main__":
    main()
