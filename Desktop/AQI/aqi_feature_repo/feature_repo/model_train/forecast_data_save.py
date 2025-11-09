import pandas as pd
import os
import sys
from feature_repo.s3_utils import download_from_s3, upload_to_s3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(_file_))))

NEW_FORECAST_FILE = "forecast_next3days_all_models.csv"
HISTORICAL_FILE = "historical_forecast.csv"

S3_FORECAST_NEW = "forecasts/forecast_next3days_all_models.csv"
S3_FORECAST_HISTORY = "forecasts/historical_forecast.csv"


os.makedirs("forecasts", exist_ok=True)

print("\n--- Downloading forecast files from S3 ---")
download_from_s3(S3_FORECAST_NEW, NEW_FORECAST_FILE)

try:
    download_from_s3(S3_FORECAST_HISTORY, HISTORICAL_FILE)
except Exception as e:
    print("No historical forecast file found in S3 (will create a new one).")


def save_new_forecast():
    """
    Loads the latest 72-hour forecast, merges it with the historical data,
    and saves the de-duplicated result back to the historical file.
    """
    try:
        # Load new 72-hour forecast
        df_new_forecast = pd.read_csv(NEW_FORECAST_FILE)
        df_new_forecast['timestamp'] = pd.to_datetime(df_new_forecast['timestamp'])
        print(f"Loaded new forecast starting at: {df_new_forecast['timestamp'].min()}")

    except FileNotFoundError:
        print(f"Error: Could not find {NEW_FORECAST_FILE}. Ensure forecast.py ran correctly.")
        return
    except Exception as e:
        print(f"Error reading new forecast file: {e}")
        return

    df_history = pd.DataFrame()
    
    if os.path.exists(HISTORICAL_FILE):
        df_history = pd.read_csv(HISTORICAL_FILE)
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
        print(f"Loaded historical data (Total rows: {len(df_history)})")
        
        new_timestamps = df_new_forecast[~df_new_forecast['timestamp'].isin(df_history['timestamp'])]
        df_combined = pd.concat([df_history, new_timestamps], ignore_index=True)
        
    else:
        df_combined = df_new_forecast
        print("Historical file not found. Creating new history from the latest forecast.")

    df_combined = df_combined.sort_values(by='timestamp').reset_index(drop=True)
    df_combined.to_csv(HISTORICAL_FILE, index=False)

    print("---")
    print(f"Successfully updated {HISTORICAL_FILE}")
    print(f"Total unique forecast points now in history: {len(df_combined)}")
    added_points = len(df_combined) - len(df_history) if not df_history.empty else len(df_combined)
    print(f"New forecast points added: {added_points}")


if __name__ == "__main__":
    save_new_forecast()

    print("\n--- Uploading updated historical forecast to S3 ---")
    upload_to_s3(HISTORICAL_FILE, S3_FORECAST_HISTORY)

