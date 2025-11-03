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

def manual_mean_squared_error(y_true, y_pred):
    """
    Manually calculates the Mean Squared Error (MSE).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


path = r"C:\Users\Hp\Desktop\AQI\aqi_feature_repo\feature_repo\data\khi_air_quality_clean.parquet"
df = pd.read_parquet(path)
df = df.sort_values("time").reset_index(drop=True)

# ==============================
# 2️⃣ Create lag and rolling features & SHIFT TARGET
# ==============================
# Features are calculated based on current/past AQI
df["lag_1"] = df["aqi_estimate"].shift(1)
df["lag_2"] = df["aqi_estimate"].shift(2)
df["lag_24"] = df["aqi_estimate"].shift(24)
df["rolling_mean_6h"] = df["aqi_estimate"].shift(1).rolling(6, min_periods=1).mean()

# *** ADJUSTMENT 1: CHANGED TO 1-hour forecast horizon for the Recursive (Reversing) approach ***
# The reversing approach trains a 1-step model and applies it 72 times in a loop.
FORECAST_HORIZON = 1
df[f"target_{FORECAST_HORIZON}h"] = df["aqi_estimate"].shift(-FORECAST_HORIZON)

# *** ADJUSTMENT 2: Adding powerful long-term features for 72h prediction ***
df["rolling_mean_24h"] = df["aqi_estimate"].shift(1).rolling(24, min_periods=1).mean()
df["aqi_change_24h"] = df["aqi_estimate"].shift(1) - df["aqi_estimate"].shift(25) # Change from 25h ago to 1h ago

# *** ADJUSTMENT 3: Adding powerful cyclical time features (Sine/Cosine) ***
# This replaces the linear 'hour' and 'dayofweek' features in the feature_cols list below.
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)


# Drop rows with NaNs from lagging AND the target shift
df = df.dropna().reset_index(drop=True)

# ==============================
# 3️⃣ Feature selection
# ==============================
feature_cols = [
    "pm10", "pm2_5", "co", "no2", "so2", "o3",
    "pm2_5_3h_avg", "pm10_3h_avg", "pm2_5_6h_avg", "pm10_6h_avg",
    "lag_1", "lag_2", "lag_24", "rolling_mean_6h",
    "rolling_mean_24h", "aqi_change_24h", # Long-term features
    "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos" # Cyclical features
    ]
target_col = f"target_{FORECAST_HORIZON}h" # Use the new shifted target column for 1-hour prediction

X = df[feature_cols]
y = df[target_col]

# ==============================
# 4️⃣ Train/test split (chronological)
# ==============================
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
# Save train/test splits for forecast.py
joblib.dump([X_train, X_test, y_train, y_test], "model_registry/train_test.pkl")


# ==============================
# 5️⃣ Scale features
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
os.makedirs("model_registry", exist_ok=True)
joblib.dump(scaler, "model_registry/scaler.pkl")

# ==============================
# 6️⃣ Define models
# ==============================
# *** ADJUSTMENT 4: Reverting model parameters to friend's defaults (n_estimators=200, simpler learning rates) ***
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
        # Note: LightGBM is trained on unscaled features
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    else:
        # RF and GB are trained on scaled features
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

    # Re-using the manual_mean_squared_error if needed
    try:
        mse = mean_squared_error(y_test, preds)
    except NameError:
        # Fallback to manual implementation if sklearn.metrics is still failing
        mse = manual_mean_squared_error(y_test, preds)

    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results[name] = {"Models": name, "RMSE": rmse, "MAE": mae, "R2": r2}
    joblib.dump(model, f"model_registry/{name}_model.pkl")

# Save results
joblib.dump(results, "model_registry/results.pkl")
pd.DataFrame(results.values()).to_csv("model_registry/model_results.csv", index=False)

print("\nTraining complete. Model performance:")
for k, v in results.items():
    print(f"{k}: RMSE={v['RMSE']:.2f}, MAE={v['MAE']:.2f}, R2={v['R2']:.3f}")
