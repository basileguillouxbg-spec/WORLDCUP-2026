"""Phase 2 - Step 1: load the raw results and clean them."""
import pandas as pd

RAW = "data/raw"

# 1. Load matches, parsing the date column into real datetimes.
df = pd.read_csv(f"{RAW}/results.csv", parse_dates=["date"])

# 2. Reconcile former country names (e.g. Dahomey -> Benin) so a team's
#    whole history sits under one current name.
names = pd.read_csv(f"{RAW}/former_names.csv")
rename = dict(zip(names["former"], names["current"]))
df["home_team"] = df["home_team"].replace(rename)
df["away_team"] = df["away_team"].replace(rename)

# 3. Split played matches (have scores) from future fixtures (scores are NA).
played = df[df["home_score"].notna()].copy()
future = df[df["home_score"].isna()].copy()

# 4. Quick report.
print(f"total rows   : {len(df)}")
print(f"played       : {len(played)}  ({played['date'].min().date()} -> {played['date'].max().date()})")
print(f"future (WC26): {len(future)}")
print(f"teams seen   : {pd.concat([played.home_team, played.away_team]).nunique()}")
