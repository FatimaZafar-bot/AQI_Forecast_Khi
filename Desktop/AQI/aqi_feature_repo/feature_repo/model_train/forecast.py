import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import os
import subprocess
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from s3_utils import download_from_s3, upload_to_s3
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

RAW_PARQUET_S3 = "data/khi_air_quality_clean.parquet"
RAW_PARQUET_LOCAL = "data/khi_air_quality_clean.parquet"

MODEL_S3_PREFIX = "models/"
MODEL_LOCAL = "model_registry"

FORECAST_S3_PATH = "forecasts/forecast_next3days_all_models.csv"
NEW_FORECAST_FILE = "forecast_next3days_all_models.csv"

os.makedirs("data", exist_ok=True)
os.makedirs(MODEL_LOCAL, exist_ok=True)

print("\n--- Downloading latest model artifacts and data from S3 ---")
download_from_s3(RAW_PARQUET_S3, RAW_PARQUET_LOCAL)
download_from_s3(f"{MODEL_S3_PREFIX}scaler.pkl", os.path.join(MODEL_LOCAL, "scaler.pkl"))
download_from_s3(f"{MODEL_S3_PREFIX}train_test.pkl", os.path.join(MODEL_LOCAL, "train_test.pkl"))

for model_name in ["RandomForest", "GradientBoosting", "LightGBM"]:
    download_from_s3(f"{MODEL_S3_PREFIX}{model_name}_model.pkl",
                     os.path.join(MODEL_LOCAL, f"{model_name}_model.pkl"))


SCALER_PATH = os.path.join(MODEL_LOCAL, "scaler.pkl")
TRAIN_TEST_PATH = os.path.join(MODEL_LOCAL, "train_test.pkl")
DATA_PATH = RAW_PARQUET_LOCAL
FORECAST_HORIZON_HOURS = 72

try:
    scaler = joblib.load(SCALER_PATH)
    X_train_full, _, _, _ = joblib.load(TRAIN_TEST_PATH)
    feature_names = X_train_full.columns.tolist()
    df_full = pd.read_parquet(DATA_PATH)
    df_full = df_full.sort_values("time").reset_index(drop=True)
except FileNotFoundError as e:
    print(f"Error: Required file not found. Please run training.py first. Missing: {e.filename}")
    exit()

MODELS = ["RandomForest", "GradientBoosting", "LightGBM"]


def get_cyclical_features(dt_time):
    """Calculates Sine/Cosine features for hour and day of week for a given timestamp."""
    hour = dt_time.hour
    dayofweek = dt_time.dayofweek
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "dayofweek_sin": np.sin(2 * np.pi * dayofweek / 7.0),
        "dayofweek_cos": np.cos(2 * np.pi * dayofweek / 7.0),
    }


last_time_known = df_full.iloc[-1]["time"]
timestamps = [last_time_known + timedelta(hours=i) for i in range(1, FORECAST_HORIZON_HOURS + 1)]
combined_forecast = pd.DataFrame({"timestamp": timestamps})

aqi_history_known = df_full["aqi_estimate"].iloc[-25:].tolist()
last_pollutant_row = df_full.iloc[-1].copy().to_dict()

for name in MODELS:
    model_path = os.path.join(MODEL_LOCAL, f"{name}_model.pkl")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Model file not found for {name}. Skipping.")
        continue
    
    print(f"\nForecasting {FORECAST_HORIZON_HOURS} hours with {name} (Recursive Approach)...")
    preds = []
    aqi_history = aqi_history_known[:]
    current_time = last_time_known + timedelta(hours=1)
    
    for step in range(FORECAST_HORIZON_HOURS):
        cyclical_features = get_cyclical_features(current_time)
        features = {}

        # Non-AQI features
        for col in feature_names:
            if col not in ["lag_1", "lag_2", "lag_24", "rolling_mean_6h", "rolling_mean_24h",
                           "aqi_change_24h", "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos"]:
                features[col] = last_pollutant_row.get(col, 0)

        current_aqi = aqi_history[-1]
        aqi_25h_ago = aqi_history[0]

        features["lag_1"] = current_aqi
        features["lag_2"] = aqi_history[-2]
        features["lag_24"] = aqi_history[-24]
        features["rolling_mean_6h"] = np.mean(aqi_history[-6:])
        features["rolling_mean_24h"] = np.mean(aqi_history[-24:])
        features["aqi_change_24h"] = current_aqi - aqi_25h_ago
        features.update(cyclical_features)

        X_input = pd.DataFrame([features], columns=feature_names)

        if name != "LightGBM":
            X_input_scaled = scaler.transform(X_input)
            y_pred = model.predict(X_input_scaled)[0]
        else:
            y_pred = model.predict(X_input)[0]

        preds.append(y_pred)
        aqi_history.pop(0)
        aqi_history.append(y_pred)
        current_time += timedelta(hours=1)

    combined_forecast[name] = np.round(preds, 2)

if all(model_name in combined_forecast.columns for model_name in MODELS):
    combined_forecast['Ensemble_Prediction'] = combined_forecast[MODELS].mean(axis=1).round(2)
    print(f"Added 'Ensemble_Prediction' column (average of {', '.join(MODELS)})")
else:
    print("Warning: Not all model columns found to calculate 'Ensemble_Prediction'.")

combined_forecast.to_csv(NEW_FORECAST_FILE, index=False)
print(f"\nCombined {FORECAST_HORIZON_HOURS}-hour forecast saved to {NEW_FORECAST_FILE}")
print(f"Forecast starting at {timestamps[0]} saved.")

print("\n--- Uploading new forecast to S3 ---")
upload_to_s3(NEW_FORECAST_FILE, FORECAST_S3_PATH)


print("\n--- Starting Automatic History Update (Running forecast_data_save.py) ---")
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    saving_script_path = os.path.join(script_dir, "forecast_data_save.py")
    result = subprocess.run(
        [sys.executable, saving_script_path],
        capture_output=True,
        text=True,
        check=True
    )
    print("forecast_data_save.py Output:")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error executing forecast_data_save.py:")
    print(e.stderr)
except FileNotFoundError:
    print(f"Error: The 'python' command or 'forecast_data_save.py' script was not found.")
