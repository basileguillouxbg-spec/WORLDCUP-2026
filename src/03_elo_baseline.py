"""Detour: how good is Elo alone? Logistic regression, walk-forward CV."""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss

# data is already sorted by date from step 2
df = pd.read_csv("data/processed_matches.csv", parse_dates=["date"])

# one feature: Elo gap (home minus away)
X = (df["home_elo"] - df["away_elo"]).to_frame("elo_diff")
# target: 0 = away win, 1 = draw, 2 = home win
y = (df["home_score"] > df["away_score"]).astype(int) * 2 \
    + (df["home_score"] == df["away_score"]).astype(int)

# walk-forward: train on past, test on next chunk, roll forward (5 times)
accs, losses = [], []
for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(X):
    model = LogisticRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])
    proba = model.predict_proba(X.iloc[test_idx])
    accs.append(accuracy_score(y.iloc[test_idx], pred))
    losses.append(log_loss(y.iloc[test_idx], proba, labels=[0, 1, 2]))

# baseline: always guess home win
home_win_rate = (y == 2).mean()

print(f"Elo-only logistic regression (5 walk-forward folds):")
print(f"  accuracy : {sum(accs)/len(accs):.3f}")
print(f"  log-loss : {sum(losses)/len(losses):.3f}  (lower is better)")
print(f"  baseline : {home_win_rate:.3f}  (always predict home win)")
