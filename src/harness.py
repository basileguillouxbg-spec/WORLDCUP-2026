"""Reusable walk-forward evaluation harness for the 3-way outcome target.

Target label: 0 = away win, 1 = draw, 2 = home win (regulation / 90-min result).

The model is the stacked Poisson regressor (one row per attacking side). We
predict each side's expected goals, build the score grid, optionally apply the
Dixon-Coles low-score correction (rho != 0), then collapse the grid into
W/D/L probabilities.

Everything is vectorised so 5-fold walk-forward over ~49k matches runs in
seconds. Import `walk_forward` from this module; run it directly for Step 1
(rho=0, the honest version of the current model).
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

FEATS = ["elo_diff", "att", "opp_def", "is_home"]


def load():
    return pd.read_csv("data/features.csv", parse_dates=["date"])


def stack(d):
    """One match -> two attacking rows (home view, away view)."""
    home = pd.DataFrame({"goals": d.home_score, "elo_diff": d.home_elo - d.away_elo,
                         "att": d.home_gf, "opp_def": d.away_ga, "is_home": d.home_advantage})
    away = pd.DataFrame({"goals": d.away_score, "elo_diff": d.away_elo - d.home_elo,
                         "att": d.away_gf, "opp_def": d.home_ga, "is_home": 0})
    return pd.concat([home, away], ignore_index=True)


def fit(train_df, alpha=0.1):
    m = make_pipeline(StandardScaler(), PoissonRegressor(alpha=alpha, max_iter=300))
    tr = stack(train_df)
    m.fit(tr[FEATS], tr["goals"])
    return m


def expected_goals(model, d):
    """Return (lambda_home, lambda_away) arrays for a match dataframe."""
    home = pd.DataFrame({"elo_diff": d.home_elo - d.away_elo, "att": d.home_gf,
                         "opp_def": d.away_ga, "is_home": d.home_advantage})[FEATS]
    away = pd.DataFrame({"elo_diff": d.away_elo - d.home_elo, "att": d.away_gf,
                         "opp_def": d.home_ga, "is_home": 0})[FEATS]
    return model.predict(home), model.predict(away)


def grid_probs(lh, la, rho=0.0, maxg=10):
    """Vectorised score-grid -> [P(away), P(draw), P(home)] per match.

    rho is the Dixon-Coles low-score correction (0 = independent Poisson).
    Negative rho lifts 0-0 and 1-1 (draws) and trims 1-0 / 0-1.
    """
    lh = np.asarray(lh, float); la = np.asarray(la, float)
    ks = np.arange(maxg)
    ph = poisson.pmf(ks[None, :], lh[:, None])           # N x maxg (home goals)
    pa = poisson.pmf(ks[None, :], la[:, None])            # N x maxg (away goals)
    joint = ph[:, :, None] * pa[:, None, :]              # N x i(home) x j(away)

    if rho != 0.0:
        tau = np.ones_like(joint)
        tau[:, 0, 0] = 1.0 - lh * la * rho
        tau[:, 0, 1] = 1.0 + lh * rho
        tau[:, 1, 0] = 1.0 + la * rho
        tau[:, 1, 1] = 1.0 - rho
        joint = np.clip(joint * tau, 0.0, None)          # floor tiny negatives

    i_idx = ks[:, None]; j_idx = ks[None, :]
    home = (joint * (i_idx > j_idx)).sum(axis=(1, 2))
    draw = (joint * (i_idx == j_idx)).sum(axis=(1, 2))
    away = (joint * (i_idx < j_idx)).sum(axis=(1, 2))
    P = np.column_stack([away, draw, home])
    return P / P.sum(axis=1, keepdims=True)


def outcomes(d):
    """3-way label: 0=away, 1=draw, 2=home."""
    return ((d.home_score > d.away_score).astype(int) * 2
            + (d.home_score == d.away_score).astype(int)).values


def brier_multi(y, p, n_classes=3):
    onehot = np.eye(n_classes)[y]
    return np.mean(np.sum((p - onehot) ** 2, axis=1))


def walk_forward(rho=0.0, alpha=0.1, n_splits=5, label="", verbose=True):
    """TimeSeriesSplit walk-forward. Returns dict of averaged metrics."""
    df = load()
    tss = TimeSeriesSplit(n_splits=n_splits)
    ll, br, ac = [], [], []
    cm = np.zeros((3, 3), dtype=int)
    naive_ll = []
    pred_draw_rate, true_draw_rate, n_test = [], [], 0

    for tr_idx, te_idx in tss.split(df):
        train, test = df.iloc[tr_idx], df.iloc[te_idx]
        model = fit(train, alpha=alpha)
        lh, la = expected_goals(model, test)
        P = grid_probs(lh, la, rho=rho)
        y = outcomes(test)
        ll.append(log_loss(y, P, labels=[0, 1, 2]))
        br.append(brier_multi(y, P))
        ac.append(accuracy_score(y, P.argmax(1)))
        cm += confusion_matrix(y, P.argmax(1), labels=[0, 1, 2])
        # naive base-rate baseline: predict TRAIN outcome frequencies for every test row
        base = np.bincount(outcomes(train), minlength=3) / len(train)
        naive_ll.append(log_loss(y, np.tile(base, (len(y), 1)), labels=[0, 1, 2]))
        pred_draw_rate.append((P.argmax(1) == 1).mean())
        true_draw_rate.append((y == 1).mean()); n_test += len(y)

    res = {"log_loss": np.mean(ll), "brier": np.mean(br), "accuracy": np.mean(ac),
           "naive_log_loss": np.mean(naive_ll), "cm": cm,
           "pred_draw_rate": np.mean(pred_draw_rate), "true_draw_rate": np.mean(true_draw_rate)}
    if verbose:
        tag = f" [{label}]" if label else ""
        print(f"Walk-forward ({n_splits} folds){tag}  rho={rho}  alpha={alpha}")
        print(f"  log-loss : {res['log_loss']:.4f}   (primary; naive base-rate = {res['naive_log_loss']:.4f})")
        print(f"  brier    : {res['brier']:.4f}")
        print(f"  accuracy : {res['accuracy']:.4f}")
        print(f"  draw rate: pred {res['pred_draw_rate']:.3f}  vs true {res['true_draw_rate']:.3f}")
        print("  confusion (rows=true, cols=pred) [away,draw,home]:")
        for row in res["cm"]:
            print("   ", row)
    return res


if __name__ == "__main__":
    print("=" * 64)
    print("STEP 1 — honest walk-forward, independent Poisson (rho=0)")
    print("=" * 64)
    walk_forward(rho=0.0, label="independent Poisson")
