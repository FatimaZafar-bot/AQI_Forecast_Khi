# automate_pipeline_khi.py
import subprocess
import sys
import traceback
from datetime import datetime
import pytz
from s3_utils import download_from_s3, upload_to_s3
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import os 
RAW_DATA_S3 = "data/live_khi_raw.csv"
FEATURES_S3 = "data/khi_air_quality_clean.parquet"

# Scripts
FETCH_SCRIPT = "fetch_live_khi.py"
UPDATE_SCRIPT = "feature_repo/update_features.py"

def log(msg: str):
    """Log messages with timestamp (prints only, no local file)."""
    karachi_tz = pytz.timezone("Asia/Karachi")
    ts = datetime.now(karachi_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{ts}] {msg}")

def run_step(script):
    """Run a script as subprocess and log output."""
    log(f" Running {script} ...")
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            check=True
        )
        log(f"{script} completed successfully")
        log(result.stdout)
    except subprocess.CalledProcessError as e:
        log(f"{script} failed")
        log(e.stdout)
        log(e.stderr)
        log(traceback.format_exc())
        raise e

def main():
    log("🌐 Starting live AQI auto-update pipeline...")

    try:
        # 1️⃣ Fetch latest live data
        run_step(FETCH_SCRIPT)
        upload_to_s3("data/live_khi_raw.csv", RAW_DATA_S3)
        log(f"✅ Uploaded latest live CSV to S3: {RAW_DATA_S3}")

        # 2️⃣ Update features from S3 raw data
        run_step(UPDATE_SCRIPT)
        upload_to_s3("data/khi_air_quality_clean.parquet", FEATURES_S3)
        log(f"✅ Uploaded updated features to S3: {FEATURES_S3}")

        log("✅ Live AQI pipeline completed successfully.")

    except KeyboardInterrupt:
        log("Pipeline stopped manually by user. Goodbye!")
    except Exception as e:
        log(f"❌ Pipeline failed: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
