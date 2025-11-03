# 8  update_features.py
# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import pandas as pd
from datetime import timedelta
from feast import FeatureStore
from preprocess import preprocess_data 

PROCESSED_PATH = "data/khi_air_quality_clean.parquet"   
LIVE_RAW = "data/live_khi_raw.csv"                     
FEAST_REPO_PATH = "."  


if not os.path.exists(LIVE_RAW):
    raise FileNotFoundError(f"Live raw file not found: {LIVE_RAW}. Run fetch_live_khi.py first.")

live_raw = pd.read_csv(LIVE_RAW)
print(f"✅ Loaded live raw rows: {len(live_raw)}")


live_processed = preprocess_data(live_raw)
print(f"✅ Live rows after preprocessing: {len(live_processed)}")
print(live_processed.head().to_string(index=False))


if os.path.exists(PROCESSED_PATH):
    hist_df = pd.read_parquet(PROCESSED_PATH)
    print(f"✅ Loaded existing processed parquet: {len(hist_df)} rows")
else:
    hist_df = pd.DataFrame(columns=live_processed.columns)
    print("⚠️ No existing processed parquet found. Will create new one.")

# aligning cols
hist_df = hist_df.copy()
live_processed = live_processed.copy()

combined = pd.concat([hist_df, live_processed], ignore_index=True, sort=False)

combined["time"] = pd.to_datetime(combined["time"], utc=True)
combined = combined.sort_values("time").drop_duplicates(subset=["time", "city_id"], keep="last").reset_index(drop=True)

combined.to_parquet(PROCESSED_PATH, index=False)
print(f"✅ Saved combined processed parquet: {PROCESSED_PATH} (rows: {len(combined)})")

fs = FeatureStore(repo_path=FEAST_REPO_PATH)
print("✅ Loaded Feast FeatureStore")

start_ts = live_processed["time"].min() - timedelta(minutes=1)
end_ts = live_processed["time"].max() + timedelta(minutes=1)
print(f"Materializing from {start_ts} to {end_ts} ...")

fs.materialize(start_ts, end_ts)
print("✅ Feast materialize completed for the new window.")
