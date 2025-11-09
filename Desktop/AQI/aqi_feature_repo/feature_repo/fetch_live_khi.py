# fetch_live_khi.py
# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8")
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
from s3_utils import upload_to_s3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


LAT = 24.8607
LON = 67.0011
LOOKBACK_HOURS = 24
API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OUT_DIR = "data"
OUT_RAW = os.path.join(OUT_DIR, "live_khi_raw.csv")
os.makedirs(OUT_DIR, exist_ok=True)

end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(hours=LOOKBACK_HOURS)

params = {
    "latitude": LAT,
    "longitude": LON,
    "hourly": [
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ],
    # avoiding timestamps mismatching
    "timezone": "UTC",
    "start": start_time.isoformat(timespec="hours"),
    "end": end_time.isoformat(timespec="hours"),
}

print(f"🌍 Fetching Open-Meteo live AQI model data for Karachi ({start_time} → {end_time}) ...")

resp = requests.get(API_URL, params=params, timeout=30)
resp.raise_for_status()
data = resp.json()

if "hourly" not in data or not any(data["hourly"].values()):
    print("[INFO] No hourly data returned.")
    exit(0)

df = pd.DataFrame(data["hourly"])
df["time"] = pd.to_datetime(df["time"], utc=True)

df = df[df["time"] <= end_time].reset_index(drop=True)

# append in csv
if os.path.exists(OUT_RAW):
    old_df = pd.read_csv(OUT_RAW)
    old_df["time"] = pd.to_datetime(old_df["time"], utc=True)
    df = pd.concat([old_df, df], ignore_index=True)

# Removing same data (only unique)
df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

df.to_csv(OUT_RAW, index=False)
print(f"✅ Saved AQI model data to: {OUT_RAW} ({len(df)} rows)")

upload_to_s3("data/live_khi_raw.csv", "data/live_khi_raw.csv")
print("✅ Uploaded latest live raw CSV to S3")


print("Hourly variables and counts:")
for k, v in data["hourly"].items():
    print(f" - {k}: {len(v)} values")
