#train_pipeline.py
import os
import time
import subprocess
import sys
from datetime import datetime
import pandas as pd
import numpy as np

DATA_PATH = r"C:\Users\Hp\Desktop\AQI\aqi_feature_repo\feature_repo\data\khi_air_quality_clean.parquet" 

# File to store the timestamp last successful pipeline run
LAST_RUN_FILE = ".pipeline_last_run" 


TRAIN_MODEL_SCRIPT = "train_model.py"
FORECAST_SCRIPT = "forecast.py" 

MIN_INTERVAL_SECONDS = 24 * 60 * 60 
CHECK_INTERVAL_SECONDS = 60 * 60 

def get_last_run_time():
    """Reads the timestamp of the last successful pipeline execution."""
    if os.path.exists(LAST_RUN_FILE):
        with open(LAST_RUN_FILE, 'r') as f:
            try:
                return float(f.read().strip())
            except ValueError:
                return 0.0
    return 0.0

def update_last_run_time(timestamp):
    """Updates the last run timestamp after a successful pipeline execution."""
    with open(LAST_RUN_FILE, 'w') as f:
        f.write(str(timestamp))

def execute_script(script_name):
    """
    Executes a given script using subprocess, ensuring the same Python 
    environment is used and handling errors.
    """
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    print(f"\n>>>> Executing {script_name}...")
    

    if not os.path.exists(script_path):
        print(f"CRITICAL ERROR: Script not found at {script_path}. Check file names and location.")
        return False
        
    try:
        result = subprocess.run(
            [sys.executable, script_path], 
            capture_output=True, 
            text=True, 
            check=True 
        )
        print(f"SUCCESS: {script_name} finished.")
        # Print output of the executed script for monitoring
        print("Stdout:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nFAILURE: {script_name} FAILED!")
        print("-" * 30)
        print("Error Message (Stderr):")
        print(e.stderr)
        print("-" * 30)
        return False

# 3. MAIN SCHEDULER LOGIC
def run_full_pipeline():
    """
    Checks if the data has been updated and runs the full training and 
    forecasting sequence if new data is available AND if the minimum time 
    interval (24 hours) has passed since the last run.
    """
    
    if not os.path.exists(DATA_PATH):
        print(f"CRITICAL ERROR: Data file not found at {DATA_PATH}. Cannot proceed.")
        return

    current_time = time.time()
    data_mod_time = os.path.getmtime(DATA_PATH)
    last_run_time = get_last_run_time()
    
    data_mod_datetime = datetime.fromtimestamp(data_mod_time).strftime('%Y-%m-%d %H:%M:%S')
    last_run_datetime = datetime.fromtimestamp(last_run_time).strftime('%Y-%m-%d %H:%M:%S')

    print(f"\n[Pipeline Check @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print(f"  Data last updated: {data_mod_datetime}")
    print(f"  Pipeline last run: {last_run_datetime}")
    
    if (current_time - last_run_time) < MIN_INTERVAL_SECONDS:
        print("\n** SKIPPED: Less than 24 hours have passed since the last successful run. **")
        return

    # Data Modification Check (Original Logic)
    if data_mod_time > last_run_time:
        print("\n** TRIGGERED: New data detected AND 24-hour interval passed. Starting full pipeline... **")
        print("-" * 50)
        
        # Execute Training (train_model.py)
        if not execute_script(TRAIN_MODEL_SCRIPT):
            print("\nPipeline Halted: Model training failed.")
            return

        #  Execute Forecasting (forecast.py)
        if not execute_script(FORECAST_SCRIPT):
            print("\nPipeline Halted: Forecasting failed.")
            return
        
        update_last_run_time(time.time())
        print("-" * 50)
        print("\n** SUCCESS: Pipeline completed and last run time updated. **")
        print("Forecasts are saved and historical data is updated.")
        
    else:
        print("\n** SKIPPED: 24-hour interval passed, but data has not been updated since last run. **")


if __name__ == "__main__":
    # NOTE: We use a short sleep (CHECK_INTERVAL_SECONDS = 1 hour) so the script 
    # is responsive to data updates soon after the 24-hour training gate opens.
    print("-" * 50)
    print(f"Starting continuous pipeline checker. Checking every {CHECK_INTERVAL_SECONDS/3600} hour(s)...")
    print(f"24-hour training gate is enforced by MIN_INTERVAL_SECONDS={MIN_INTERVAL_SECONDS}s.")
    print("-" * 50)

    while True:
        try:
            run_full_pipeline()
        except Exception as e:
            print(f"\nCRITICAL UNHANDLED ERROR in main loop: {e}")
            print("Attempting to restart check cycle after delay.")
            

        time.sleep(CHECK_INTERVAL_SECONDS)
