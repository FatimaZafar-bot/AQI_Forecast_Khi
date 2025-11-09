# train_model.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from s3_utils import download_from_s3, upload_to_s3
import tempfile

def manual_mean_squared_error(y_true, y_pred):
    """Manually calculates the Mean Squared Error (MSE)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)



RAW_PARQUET_S3 = "data/khi_air_quality_clean.parquet"
MODEL_S3_PREFIX = "models/"

tmp_dir = tempfile.mkdtemp()

raw_parquet_local = os.path.join(tmp_dir, "khi_air_quality_clean.parquet")
download_from_s3(RAW_PARQUET_S3, raw_parquet_local)


# ==============================
# 1️⃣ Load Data
# ==============================
df = pd.read_parquet(raw_parquet_local)
df = df.sort_values("time").reset_index(drop=True)

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


# Feature selection
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


split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1)
}


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

    # Save model temporarily and upload to S3
    model_tmp = os.path.join(tmp_dir, f"{name}_model.pkl")
    joblib.dump(model, model_tmp)
    upload_to_s3(model_tmp, f"{MODEL_S3_PREFIX}{name}_model.pkl")



scaler_tmp = os.path.join(tmp_dir, "scaler.pkl")
joblib.dump(scaler, scaler_tmp)
upload_to_s3(scaler_tmp, f"{MODEL_S3_PREFIX}scaler.pkl")

# Train/Test split
train_test_tmp = os.path.join(tmp_dir, "train_test.pkl")
joblib.dump([X_train, X_test, y_train, y_test], train_test_tmp)
upload_to_s3(train_test_tmp, f"{MODEL_S3_PREFIX}train_test.pkl")

results_tmp = os.path.join(tmp_dir, "results.pkl")
joblib.dump(results, results_tmp)
upload_to_s3(results_tmp, f"{MODEL_S3_PREFIX}results.pkl")

results_csv_tmp = os.path.join(tmp_dir, "model_results.csv")
pd.DataFrame(results.values()).to_csv(results_csv_tmp, index=False)
upload_to_s3(results_csv_tmp, f"{MODEL_S3_PREFIX}model_results.csv")

print("\n✅ Training complete. Model performance:")
for k, v in results.items():
    print(f"{k}: RMSE={v['RMSE']:.2f}, MAE={v['MAE']:.2f}, R2={v['R2']:.3f}")
