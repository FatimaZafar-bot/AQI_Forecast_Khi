# 2 preprocess.py

import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import os
from s3_utils import download_from_s3, upload_to_s3
import sys

# Add the parent folder (aqi_feature_repo/) to Python path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = "karachi_air_quality.csv"
PROCESSED_DIR = "data"
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "khi_air_quality_clean.parquet")

os.makedirs(PROCESSED_DIR, exist_ok=True)

download_from_s3("data/karachi_air_quality.csv", RAW_PATH)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, preprocess, and engineer features for air quality data.
    Works for both historical and live batches."""
    df = df.copy()

    # Normalize column names
    cols = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df.rename(columns=cols, inplace=True)

    map_variants = {
        "pm2_5": ["pm2_5", "pm2.5", "pm25", "pm_2_5", "pm_25"],
        "pm10": ["pm10", "pm_10"],
        "no2": ["no2", "nitrogen_dioxide", "n02"],
        "so2": ["so2", "sulphur_dioxide", "sulfur_dioxide"],
        "co": ["co", "carbon_monoxide", "co2"],
        "o3": ["o3", "ozone"],
        "time": ["time", "datetime", "date", "timestamp"]
    }

    found = {}
    for canon, variants in map_variants.items():
        for v in variants:
            if v in df.columns:
                found[canon] = v
                break

    if "time" not in found:
        raise ValueError(f"❌ No timestamp column found. Got: {list(df.columns)}")

    
    for canon, actual in found.items():
        if canon != actual:
            df.rename(columns={actual: canon}, inplace=True)

   
    numeric_cols = ["pm2_5", "pm10", "no2", "so2", "co", "o3"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan

    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan
        if df[col].notna().sum() > 0:
            upper = df[col].quantile(0.995)
            df.loc[df[col] > upper, col] = np.nan

    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            df[col] = df[col].ffill().bfill()

    
    # ENGINEERED FEATURES
   
    # Rolling averages
    df["pm2_5_3h_avg"] = df["pm2_5"].rolling(window=3, min_periods=1).mean()
    df["pm10_3h_avg"] = df["pm10"].rolling(window=3, min_periods=1).mean()
    df["pm2_5_6h_avg"] = df["pm2_5"].rolling(window=6, min_periods=1).mean()
    df["pm10_6h_avg"] = df["pm10"].rolling(window=6, min_periods=1).mean()

    # Normalization 
    for col in numeric_cols:
        mean = df[col].mean(skipna=True)
        std = df[col].std(skipna=True)
        if pd.isna(std) or std == 0:
            df[f"{col}_norm"] = np.nan
        else:
            df[f"{col}_norm"] = (df[col] - mean) / (std + 1e-9)

  
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["dayofweek"] = df["time"].dt.dayofweek

    # Derived AQI estimate 
    df["aqi_estimate"] = (
        df["pm2_5"] * 0.4 +
        df["pm10"] * 0.3 +
        df["no2"] * 0.1 +
        df["so2"] * 0.1 +
        df["o3"] * 0.1
    )

 
    df["aqi_change_rate"] = df["aqi_estimate"].diff().fillna(0)
    df["city_id"] = 1
    df["city_name"] = "Karachi"

    return df



if __name__ == "__main__":
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"❌ Raw CSV not found at: {os.path.abspath(RAW_PATH)}")

    raw_df = pd.read_csv(RAW_PATH)
    print(f"✅ Loaded raw data: {len(raw_df)} rows, {len(raw_df.columns)} columns")

    processed_df = preprocess_data(raw_df)
    print("✅ Processed sample:")
    print(processed_df.head().to_string(index=False))

    processed_df.to_parquet(PROCESSED_PATH, index=False)
    print(f"✅ Saved processed data to: {os.path.abspath(PROCESSED_PATH)}")
    upload_to_s3(PROCESSED_PATH, "data/khi_air_quality_clean.parquet")

