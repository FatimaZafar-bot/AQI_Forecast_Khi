# train_model.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import timedelta
from s3_utils import download_from_s3, upload_to_s3
import sys
import os
# Add the parent folder (aqi_feature_repo/) to Python path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(_file_))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def manual_mean_squared_error(y_true, y_pred):
    """
    Manually calculates the Mean Squared Error (MSE).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)

# ==============================
# 0️⃣ S3 Paths and Local Directories
# ==============================
RAW_PARQUET_LOCAL = "data/khi_air_quality_clean.parquet"
RAW_PARQUET_S3 = "data/khi_air_quality_clean.parquet"  # S3 key
MODEL_LOCAL = "model_registry"
MODEL_S3_PREFIX = "models/"

os.makedirs("data", exist_ok=True)
os.makedirs(MODEL_LOCAL, exist_ok=True)

# Download latest parquet from S3
download_from_s3(RAW_PARQUET_S3, RAW_PARQUET_LOCAL)

# ==============================
# 1️⃣ Load Data
# ==============================
path = RAW_PARQUET_LOCAL
df = pd.read_parquet(path)
df = df.sort_values("time").reset_index(drop=True)

# ==============================
# 2️⃣ Create lag and rolling features & SHIFT TARGET
# ==============================
df["lag_1"] = df["aqi_estimate"].shift(1)
df["lag_2"] = df["aqi_estimate"].shift(2)
df["lag_24"] = df["aqi_estimate"].shift(24)
df["rolling_mean_6h"] = df["aqi_estimate"].shift(1).rolling(6, min_periods=1).mean()

FORECAST_HORIZON = 1
df[f"target_{FORECAST_HORIZON}h"] = df["aqi_estimate"].shift(-FORECAST_HORIZON)

df["rolling_mean_24h"] = df["aqi_estimate"].shift(1).rolling(24, min_periods=1).mean()
df["aqi_change_24h"] = df["aqi_estimate"].shift(1) - df["aqi_estimate"].shift(25)

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

df = df.dropna().reset_index(drop=True)

# ==============================
# 3️⃣ Feature selection
# ==============================
feature_cols = [
    "pm10", "pm2_5", "co", "no2", "so2", "o3",
    "pm2_5_3h_avg", "pm10_3h_avg", "pm2_5_6h_avg", "pm10_6h_avg",
    "lag_1", "lag_2", "lag_24", "rolling_mean_6h",
    "rolling_mean_24h", "aqi_change_24h",
    "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos"
]
target_col = f"target_{FORECAST_HORIZON}h"

X = df[feature_cols]
y = df[target_col]

# ==============================
# 4️⃣ Train/test split
# ==============================
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Save train/test splits
train_test_local = f"{MODEL_LOCAL}/train_test.pkl"
joblib.dump([X_train, X_test, y_train, y_test], train_test_local)

# ==============================
# 5️⃣ Scale features
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
scaler_local = f"{MODEL_LOCAL}/scaler.pkl"
joblib.dump(scaler, scaler_local)

# ==============================
# 6️⃣ Define models
# ==============================
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1)
}

# ==============================
# 7️⃣ Train models, evaluate, save
# ==============================
results = {}

for name, model in models.items():
    print(f"Training {name} ...")
    if name == "LightGBM":
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

    try:
        mse = mean_squared_error(y_test, preds)
    except NameError:
        mse = manual_mean_squared_error(y_test, preds)

    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results[name] = {"Models": name, "RMSE": rmse, "MAE": mae, "R2": r2}
    model_file = f"{MODEL_LOCAL}/{name}_model.pkl"
    joblib.dump(model, model_file)

# Save results
results_file_local = f"{MODEL_LOCAL}/results.pkl"
joblib.dump(results, results_file_local)
pd.DataFrame(results.values()).to_csv(f"{MODEL_LOCAL}/model_results.csv", index=False)

print("\nTraining complete. Model performance:")
for k, v in results.items():
    print(f"{k}: RMSE={v['RMSE']:.2f}, MAE={v['MAE']:.2f}, R2={v['R2']:.3f}")

# ==============================
# 8️⃣ Upload all artifacts to S3
# ==============================
# Models
for name in models.keys():
    upload_to_s3(f"{MODEL_LOCAL}/{name}_model.pkl", f"{MODEL_S3_PREFIX}{name}_model.pkl")

# Scaler
upload_to_s3(scaler_local, f"{MODEL_S3_PREFIX}scaler.pkl")

# Train/Test split
upload_to_s3(train_test_local, f"{MODEL_S3_PREFIX}train_test.pkl")

# Results CSV
upload_to_s3(f"{MODEL_LOCAL}/model_results.csv", f"{MODEL_S3_PREFIX}model_results.csv")
