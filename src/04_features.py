"""Phase 2 - Step 3: add recent-form features + venue flag -> final table.

Recent form = average goals scored / conceded by a team over its last N
matches, computed in date order so we only ever use the past (no leakage).
"""
import pandas as pd
from collections import deque, defaultdict

N = 5  # how many recent matches define "form"
HOSTS = {"United States", "Mexico", "Canada"}  # 2026 host nations

df = pd.read_csv("data/processed_matches.csv", parse_dates=["date"])  # already date-sorted

# per-team rolling memory of (goals_for, goals_against) for last N matches
hist = defaultdict(lambda: deque(maxlen=N))

def form(team):
    """Average goals for / against over recent matches; 1.0/1.0 if no history yet."""
    if not hist[team]:
        return 1.0, 1.0
    gf = sum(g[0] for g in hist[team]) / len(hist[team])
    ga = sum(g[1] for g in hist[team]) / len(hist[team])
    return gf, ga

rows = []
for r in df.itertuples():
    h_gf, h_ga = form(r.home_team)   # form BEFORE this match
    a_gf, a_ga = form(r.away_team)
    rows.append((h_gf, h_ga, a_gf, a_ga))
    # then update memory with what actually happened
    hist[r.home_team].append((r.home_score, r.away_score))
    hist[r.away_team].append((r.away_score, r.home_score))

f = pd.DataFrame(rows, columns=["home_gf", "home_ga", "away_gf", "away_ga"])
df = pd.concat([df, f], axis=1)

# venue: real historical home advantage (used for TRAINING)
df["home_advantage"] = (~df["neutral"]).astype(int)
df["elo_diff"] = df["home_elo"] - df["away_elo"]

df.to_csv("data/features.csv", index=False)
print(f"saved data/features.csv ({len(df)} matches)")
print("\nfeature columns:", ["elo_diff", "home_gf", "home_ga", "away_gf", "away_ga", "home_advantage"])
print("\nsample (last 3 played matches):")
cols = ["date", "home_team", "away_team", "elo_diff", "home_gf", "away_gf", "home_advantage"]
print(df[cols].tail(3).to_string(index=False))
