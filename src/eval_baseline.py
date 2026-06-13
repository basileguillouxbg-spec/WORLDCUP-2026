"""Step 0: complete the BASELINE for the current Poisson model.

05_model.py already prints accuracy + log-loss on its single 2022+ split, but
not Brier. This reproduces that exact model + split and reports all three
3-class metrics (log loss = primary, accuracy, multiclass Brier) so we have a
clear number to beat. Label order: 0=away win, 1=draw, 2=home win.
"""
import numpy as np
import pandas as pd
from math import exp, factorial
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

df = pd.read_csv("data/features.csv", parse_dates=["date"])

def stack(d):
    home = pd.DataFrame({"goals": d.home_score, "elo_diff": d.home_elo - d.away_elo,
                         "att": d.home_gf, "opp_def": d.away_ga, "is_home": d.home_advantage})
    away = pd.DataFrame({"goals": d.away_score, "elo_diff": d.away_elo - d.home_elo,
                         "att": d.away_gf, "opp_def": d.home_ga, "is_home": 0})
    return pd.concat([home, away], ignore_index=True)

FEATS = ["elo_diff", "att", "opp_def", "is_home"]
train, test = df[df.date < "2022-01-01"], df[df.date >= "2022-01-01"]
model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.1, max_iter=300))
model.fit(stack(train)[FEATS], stack(train)["goals"])

def pmf(k, lam): return exp(-lam) * lam**k / factorial(k)
def outcome_probs(lh, la, maxg=10):
    ph = [pmf(i, lh) for i in range(maxg)]
    pa = [pmf(j, la) for j in range(maxg)]
    h = d = a = 0.0
    for i in range(maxg):
        for j in range(maxg):
            p = ph[i] * pa[j]
            h += p if i > j else 0; d += p if i == j else 0; a += p if i < j else 0
    tot = a + d + h
    return [a / tot, d / tot, h / tot]   # [away, draw, home]

lh = model.predict(pd.DataFrame({"elo_diff": test.home_elo - test.away_elo,
        "att": test.home_gf, "opp_def": test.away_ga, "is_home": test.home_advantage})[FEATS])
la = model.predict(pd.DataFrame({"elo_diff": test.away_elo - test.home_elo,
        "att": test.away_gf, "opp_def": test.home_ga, "is_home": 0})[FEATS])
probs = np.array([outcome_probs(a, b) for a, b in zip(lh, la)])
ytrue = ((test.home_score > test.away_score).astype(int) * 2
         + (test.home_score == test.away_score).astype(int)).values

def brier_multi(y, p, n_classes=3):
    onehot = np.eye(n_classes)[y]
    return np.mean(np.sum((p - onehot) ** 2, axis=1))

print("=" * 60)
print("BASELINE — current Poisson model, single 2022+ split")
print(f"  test matches : {len(ytrue)}")
print(f"  log-loss     : {log_loss(ytrue, probs, labels=[0,1,2]):.4f}   (primary, lower better)")
print(f"  accuracy     : {accuracy_score(ytrue, probs.argmax(1)):.4f}")
print(f"  Brier (3cls) : {brier_multi(ytrue, probs):.4f}   (0=perfect, 2=worst)")
print("=" * 60)

cm = confusion_matrix(ytrue, probs.argmax(1), labels=[0, 1, 2])
print("confusion matrix (rows=true, cols=pred) order [away, draw, home]:")
print(cm)
print(f"\n  true draw rate      : {(ytrue == 1).mean():.3f}")
print(f"  predicted draw rate : {(probs.argmax(1) == 1).mean():.3f}   <- note: independent-Poisson under-predicts draws")
