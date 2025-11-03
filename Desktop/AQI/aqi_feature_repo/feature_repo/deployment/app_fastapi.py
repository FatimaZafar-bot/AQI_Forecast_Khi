import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import List, Optional
from datetime import datetime # ✅ FIX: Import datetime for Pydantic models and time functions
import uvicorn
import pytz # Added for reliable time zone handling

# --- CONFIGURATION ---
# ➡️ FIX: Correct relative path from 'deployment/' to 'model_train/'
FORECAST_FILE = "../model_train/forecast_next3days_all_models.csv" 
KARACHI_TZ = pytz.timezone("Asia/Karachi")

# --- Pydantic Data Models ---
class AQIForecast(BaseModel):
    timestamp: datetime # ✅ FIX: Now uses imported datetime
    RandomForest: Optional[float]
    GradientBoosting: Optional[float]
    LightGBM: Optional[float]
    Ensemble_Prediction: float 

class ForecastResponse(BaseModel):
    last_update_time: datetime # ✅ FIX: Now uses imported datetime
    forecasts: List[AQIForecast]

# --- FastAPI App Initialization ---
app = FastAPI(title="AQI Forecast API", version="1.0.0")

# --- Helper Function to Load Forecast Data ---
def load_latest_forecast() -> pd.DataFrame:
    """Loads and preprocesses the latest 72-hour forecast CSV."""
    
    # Use os.path functions to resolve the correct file location relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    absolute_path = os.path.abspath(os.path.join(script_dir, FORECAST_FILE))
    
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f"Forecast file not found at {absolute_path}")
        
    df = pd.read_csv(absolute_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df

# --- API Endpoint ---
@app.get("/api/v1/forecast/karachi", response_model=ForecastResponse)
def get_karachi_forecast():
    """Returns the latest 72-hour AQI forecast, including the ensemble."""
    try:
        df = load_latest_forecast()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Forecast data not available. Run forecast.py.")

    # Get the file path again to check modification time
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    file_path = os.path.abspath(os.path.join(script_dir, FORECAST_FILE))
    
    # Get the file modification time and localize it to Karachi
    mod_time_utc = datetime.fromtimestamp(os.path.getmtime(file_path), tz=pytz.utc)
    last_update_ts = mod_time_utc.astimezone(KARACHI_TZ)


    # Convert DataFrame records to the Pydantic model structure
    forecast_list = df.to_dict('records')

    return {
        "last_update_time": last_update_ts,
        "forecasts": forecast_list
    }

# --- Root Endpoint (optional check) ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "AQI Forecast API is running."}

# To run this file: uvicorn app_fastapi:app --reload --port 8000