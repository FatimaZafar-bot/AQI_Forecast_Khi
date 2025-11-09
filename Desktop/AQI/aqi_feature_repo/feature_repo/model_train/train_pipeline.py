import os
import subprocess
import sys
from datetime import datetime
import time
from feature_repo.s3_utils import download_from_s3, upload_to_s3
import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(_file_))))

S3_PARQUET_PATH = "data/khi_air_quality_clean.parquet"
LOCAL_PARQUET_PATH = "data/khi_air_quality_clean.parquet"

LAST_RUN_FILE = ".pipeline_last_run"
TRAIN_MODEL_SCRIPT = "train_model.py"
FORECAST_SCRIPT = "forecast.py"


def get_last_run_time():
    if os.path.exists(LAST_RUN_FILE):
        with open(LAST_RUN_FILE, 'r') as f:
            try:
                return float(f.read().strip())
            except ValueError:
                return 0.0
    return 0.0

def update_last_run_time(timestamp):
    with open(LAST_RUN_FILE, 'w') as f:
        f.write(str(timestamp))

def execute_script(script_name):
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
        print(f"SUCCESS: {script_name} finished.\nStdout:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nFAILURE: {script_name} FAILED!\nStderr:\n{e.stderr}")
        return False


def run_full_pipeline():
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

    # Step 1: Train model
    if not execute_script(TRAIN_MODEL_SCRIPT):
        print("\nPipeline Halted: Model training failed.")
        return

    # Step 2: Forecast generation
    if not execute_script(FORECAST_SCRIPT):
        print("\nPipeline Halted: Forecasting failed.")
        return

    # Step 3: Upload updated dataset back to S3
    print("\n--- Uploading updated dataset to S3 ---")
    try:
        upload_to_s3(LOCAL_PARQUET_PATH, S3_PARQUET_PATH)
    except Exception as e:
        print(f"Warning: Upload to S3 failed — {e}")

    update_last_run_time(time.time())
    print("\n** SUCCESS: Pipeline completed and timestamp updated. **")


if __name__ == "__main__":
    print(f"Starting single-run AQI pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    run_full_pipeline()
