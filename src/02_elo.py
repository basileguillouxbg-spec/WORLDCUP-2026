"""Phase 2 - Step 2: compute pre-match Elo ratings for every match.

Elo idea: every team starts at 1500. After each match the winner takes
points from the loser. How many points move depends on how surprising the
result was, the goal margin, and how important the match was.
"""
import pandas as pd
from collections import defaultdict

# --- reload + clean (same as step 1), keep played matches in date order ---
df = pd.read_csv("data/raw/results.csv", parse_dates=["date"])
names = pd.read_csv("data/raw/former_names.csv")
rename = dict(zip(names["former"], names["current"]))
df["home_team"] = df["home_team"].replace(rename)
df["away_team"] = df["away_team"].replace(rename)
df = df[df["home_score"].notna()].sort_values("date").reset_index(drop=True)

# --- Elo settings ---
START, HOME_ADV = 1500.0, 60.0      # base rating; home edge in rating points
rating = defaultdict(lambda: START)

def expected(ra, rb):                # win probability of A vs B (logistic curve)
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

home_elo, away_elo = [], []          # pre-match ratings, stored per match

for r in df.itertuples():
    rh, ra = rating[r.home_team], rating[r.away_team]
    home_elo.append(rh)              # record rating BEFORE the match (no leakage)
    away_elo.append(ra)

    # K = how much a match moves ratings; bigger for important games.
    k = 60 if "FIFA World Cup" in r.tournament else (40 if r.tournament != "Friendly" else 20)
    gd = abs(r.home_score - r.away_score)        # blowouts move ratings more
    k *= 1.0 if gd <= 1 else (1.5 if gd == 2 else (1 + gd / 5))

    res = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
    exp = expected(rh + (0 if r.neutral else HOME_ADV), ra)
    change = k * (res - exp)
    rating[r.home_team] = rh + change
    rating[r.away_team] = ra - change

df["home_elo"], df["away_elo"] = home_elo, away_elo
df.to_csv("data/processed_matches.csv", index=False)

print(f"saved data/processed_matches.csv ({len(df)} matches, with pre-match Elo)")
print("\nCurrent Elo top 10:")
for i, (team, rt) in enumerate(sorted(rating.items(), key=lambda x: -x[1])[:10], 1):
    print(f"{i:2}. {team:<15} {rt:6.0f}")
