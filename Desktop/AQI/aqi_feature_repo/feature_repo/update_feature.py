# 8 update_feature.py
# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import pandas as pd
from datetime import timedelta
from feast import FeatureStore
from preprocess import preprocess_data 
from s3_utils import download_from_s3, upload_to_s3
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==============================
# Paths
# ==============================
PROCESSED_PATH = "data/khi_air_quality_clean.parquet"   
LIVE_RAW = "data/live_khi_raw.csv"                     
FEAST_REPO_PATH = "."  

# Ensure directories exist safely
for path in [LIVE_RAW, PROCESSED_PATH]:
    dir_name = os.path.dirname(path)
    if dir_name:  # only make dir if there is a parent
        os.makedirs(dir_name, exist_ok=True)

# ==============================
# Download live raw CSV from S3
# ==============================
download_from_s3("data/live_khi_raw.csv", LIVE_RAW)

live_raw = pd.read_csv(LIVE_RAW)
print(f"✅ Loaded live raw rows: {len(live_raw)}")

live_processed = preprocess_data(live_raw)
print(f"✅ Live rows after preprocessing: {len(live_processed)}")
print(live_processed.head().to_string(index=False))

# ==============================
# Download existing processed parquet
# ==============================
try:
    download_from_s3("data/khi_air_quality_clean.parquet", PROCESSED_PATH)
    print(f"✅ Downloaded existing processed parquet from S3: {PROCESSED_PATH}")
except FileNotFoundError:
    print("⚠️ No existing processed parquet in S3. Will create a new one.")

if os.path.exists(PROCESSED_PATH):
    hist_df = pd.read_parquet(PROCESSED_PATH)
    print(f"✅ Loaded existing processed parquet: {len(hist_df)} rows")
else:
    hist_df = pd.DataFrame(columns=live_processed.columns)
    print("⚠️ No existing processed parquet found. Will create new one.")

# ==============================
# Combine historical and live
# ==============================
hist_df = hist_df.copy()
live_processed = live_processed.copy()

combined = pd.concat([hist_df, live_processed], ignore_index=True, sort=False)
combined["time"] = pd.to_datetime(combined["time"], utc=True)
combined = combined.sort_values("time").drop_duplicates(subset=["time", "city_id"], keep="last").reset_index(drop=True)

combined.to_parquet(PROCESSED_PATH, index=False)
print(f"✅ Saved combined processed parquet: {PROCESSED_PATH} (rows: {len(combined)})")

upload_to_s3(PROCESSED_PATH, "data/khi_air_quality_clean.parquet")
print("✅ Uploaded updated processed parquet to S3")

# ==============================
# Materialize Feast features
# ==============================
fs = FeatureStore(repo_path=FEAST_REPO_PATH)
print("✅ Loaded Feast FeatureStore")

start_ts = live_processed["time"].min() - timedelta(minutes=1)
end_ts = live_processed["time"].max() + timedelta(minutes=1)
print(f"Materializing from {start_ts} to {end_ts} ...")

fs.materialize(start_ts, end_ts)
print("✅ Feast materialize completed for the new window.")
