#forecast.py
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import os
import subprocess
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys 

MODEL_REGISTRY = "model_registry"
SCALER_PATH = os.path.join(MODEL_REGISTRY, "scaler.pkl")
TRAIN_TEST_PATH = "model_registry/train_test.pkl"

DATA_PATH = r"C:\Users\Hp\Desktop\AQI\aqi_feature_repo\feature_repo\data\khi_air_quality_clean.parquet"
FORECAST_HORIZON_HOURS = 72 

NEW_FORECAST_FILE = "forecast_next3days_all_models.csv"

try:
    # Load assets saved by training.py
    scaler = joblib.load(SCALER_PATH)
    _, _, _, _ = joblib.load(TRAIN_TEST_PATH) 
    
    # Reload X_train to get feature names 
    X_train_full, _, _, _ = joblib.load(TRAIN_TEST_PATH)
    feature_names = X_train_full.columns.tolist()

    # Load full, original dataframe to get latest row ->feature initialisation
    df_full = pd.read_parquet(DATA_PATH)
    df_full = df_full.sort_values("time").reset_index(drop=True)
    
except FileNotFoundError as e:
    print(f"Error: Required file not found. Please run training.py Canvas first. Missing: {e.filename}")
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



# Finding last know time n row
last_time_known = df_full.iloc[-1]["time"]

# Generating 72 future timestamps (t+1 to t+72)
timestamps = [last_time_known + timedelta(hours=i) for i in range(1, FORECAST_HORIZON_HOURS + 1)]
combined_forecast = pd.DataFrame({"timestamp": timestamps})

# Get the last 25 hours of known AQI data for history tracking (needed for lag_24 and aqi_change_24h)
aqi_history_known = df_full["aqi_estimate"].iloc[-25:].tolist()

# Get the last row of pollutant data to use as a frozen base for forecast
last_pollutant_row = df_full.iloc[-1].copy().to_dict()

for name in MODELS:
    model_path = os.path.join(MODEL_REGISTRY, f"{name}_model.pkl")
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
        
        # Update time-based features for the current step
        cyclical_features = get_cyclical_features(current_time)
        
        # Construct the feature vector (X_input)
        features = {}
        
        # Non-AQI features 
        for col in feature_names:
            if col not in ["lag_1", "lag_2", "lag_24", "rolling_mean_6h", "rolling_mean_24h", "aqi_change_24h",
                           "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos"]:
                features[col] = last_pollutant_row.get(col, 0)

        # Update Lag Rolling  features
        current_aqi = aqi_history[-1]
        aqi_25h_ago = aqi_history[0]
        
        features["lag_1"] = current_aqi
        features["lag_2"] = aqi_history[-2]
        features["lag_24"] = aqi_history[-24]
        
        features["rolling_mean_6h"] = np.mean(aqi_history[-6:])
        features["rolling_mean_24h"] = np.mean(aqi_history[-24:])
        
        features["aqi_change_24h"] = current_aqi - aqi_25h_ago
        
        # Update Cyclical features
        features.update(cyclical_features)

        # Convert feature dictionary to DataFrame
        X_input = pd.DataFrame([features], columns=feature_names)
        
        # Scale and Predict
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
print(f"\n Combined {FORECAST_HORIZON_HOURS}-hour recursive forecast saved to {NEW_FORECAST_FILE}")
print(f"Forecast starting at {timestamps[0]} saved.")

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