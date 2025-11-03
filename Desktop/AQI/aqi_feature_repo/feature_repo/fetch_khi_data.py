# 1 to run fetch_khi_data.py
import requests
import pandas as pd
from datetime import datetime, timedelta

#khi coordinates
LAT = 24.8607
LON = 67.0011

end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=365)

# api endpoint
url = "https://air-quality-api.open-meteo.com/v1/air-quality"

params = {
    "latitude": LAT,
    "longitude": LON,
    "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"],
    "start_date": start_date.isoformat(),
    "end_date": end_date.isoformat(),
    "timezone": "auto"
}

print(f"Fetching air quality data for Karachi from {start_date} to {end_date}...")

# api request
response = requests.get(url, params=params)
if response.status_code == 200:
    data = response.json()
    if "hourly" in data:
        hourly_data = data["hourly"]
        df = pd.DataFrame(hourly_data)
        df["time"] = pd.to_datetime(df["time"])
        df.to_csv("karachi_air_quality.csv", index=False)
        print(f"✅ Data saved to karachi_air_quality.csv ({len(df)} rows)")
    else:
        print("⚠️ No hourly data found in response.")
else:
    print(f"❌ Error {response.status_code}: {response.text}")
