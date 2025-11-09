#train_pipelineimport os
import time
import subprocess
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from s3_utils import download_from_s3, upload_to_s3

# ===============================
# S3 and Local Paths
# ===============================
S3_PARQUET_PATH = "data/khi_air_quality_clean.parquet"
LOCAL_PARQUET_PATH = "data/khi_air_quality_clean.parquet"

LAST_RUN_FILE = ".pipeline_last_run"
TRAIN_MODEL_SCRIPT = "train_model.py"
FORECAST_SCRIPT = "forecast.py"

# Minimum intervals
MIN_INTERVAL_SECONDS = 24 * 60 * 60      # 24 hours
CHECK_INTERVAL_SECONDS = 60 * 60         # 1 hour

# ===============================
# Helper functions
# ===============================
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
    """Executes a given script using subprocess, preserving environment."""
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

# ===============================
# Main Scheduler Logic
# ===============================
def run_full_pipeline():
    """
    Checks if the data has been updated and runs the full training and forecasting
    sequence if new data is available AND if 24 hours have passed since the last run.
    """

    os.makedirs("data", exist_ok=True)

    print("\n--- Downloading latest data from S3 ---")
    try:
        download_from_s3(S3_PARQUET_PATH, LOCAL_PARQUET_PATH)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to download data from S3 — {e}")
        return

    if not os.path.exists(LOCAL_PARQUET_PATH):
        print(f"CRITICAL ERROR: Data file not found locally at {LOCAL_PARQUET_PATH}. Cannot proceed.")
        return

    current_time = time.time()
    data_mod_time = os.path.getmtime(LOCAL_PARQUET_PATH)
    last_run_time = get_last_run_time()

    data_mod_datetime = datetime.fromtimestamp(data_mod_time).strftime('%Y-%m-%d %H:%M:%S')
    last_run_datetime = datetime.fromtimestamp(last_run_time).strftime('%Y-%m-%d %H:%M:%S')

    print(f"\n[Pipeline Check @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print(f"  Data last updated (S3 downloaded): {data_mod_datetime}")
    print(f"  Pipeline last run: {last_run_datetime}")

    if (current_time - last_run_time) < MIN_INTERVAL_SECONDS:
        print("\n** SKIPPED: Less than 24 hours since last successful run. **")
        return

    if data_mod_time > last_run_time:
        print("\n** TRIGGERED: New data detected AND 24-hour interval passed. Starting full pipeline... **")
        print("-" * 50)

        # Step 1: Train model
        if not execute_script(TRAIN_MODEL_SCRIPT):
            print("\nPipeline Halted: Model training failed.")
            return

        # Step 2: Forecast generation
        if not execute_script(FORECAST_SCRIPT):
            print("\nPipeline Halted: Forecasting failed.")
            return

        # Step 3: Upload the latest processed parquet back to S3 (if updated by feature scripts)
        print("\n--- Uploading updated dataset to S3 ---")
        try:
            upload_to_s3(LOCAL_PARQUET_PATH, S3_PARQUET_PATH)
        except Exception as e:
            print(f"Warning: Upload to S3 failed — {e}")

        # Step 4: Update timestamp
        update_last_run_time(time.time())
        print("-" * 50)
        print("\n** SUCCESS: Pipeline completed and timestamp updated. **")
        print("Forecasts saved and historical data updated.")
    else:
        print("\n** SKIPPED: 24-hour interval passed, but no new data detected since last run. **")

# ===============================
# Continuous Monitoring
# ===============================
if __name__ == "__main__":
    print("-" * 50)
    print(f"Starting continuous pipeline checker. Checking every {CHECK_INTERVAL_SECONDS/3600} hour(s)...")
    print(f"24-hour training gate enforced by MIN_INTERVAL_SECONDS={MIN_INTERVAL_SECONDS}s.")
    print("-" * 50)

    while True:
        try:
            run_full_pipeline()
        except Exception as e:
            print(f"\nCRITICAL UNHANDLED ERROR in main loop: {e}")
            print("Attempting to restart check cycle after delay.")

        time.sleep(CHECK_INTERVAL_SECONDS)

