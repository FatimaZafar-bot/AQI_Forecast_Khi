import subprocess
import traceback
import sys
from datetime import datetime
import pytz
from s3_utils import download_from_s3, upload_to_s3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Paths
RAW_PARQUET_S3 = "data/khi_air_quality_clean.parquet"

FETCH_SCRIPT = "fetch_live_khi.py"
UPDATE_SCRIPT = "update_feature.py"
LOG_FILE = "data/live_pipeline.log"

def log(msg: str):
    karachi_tz = pytz.timezone("Asia/Karachi")           
    ts = datetime.now(karachi_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(script):
    """Run a script as subprocess and log results."""
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
        log(f" {script} failed")
        log(e.stdout)
        log(e.stderr)
        log(traceback.format_exc())
# train_pipeline.py
import subprocess
import sys
import traceback
from datetime import datetime
import pytz
from s3_utils import download_from_s3, upload_to_s3

# Paths
RAW_PARQUET_S3 = "data/khi_air_quality_clean.parquet"
RAW_PARQUET_LOCAL = "data/khi_air_quality_clean.parquet"
MODEL_S3 = "models/model_rf.pkl"
MODEL_LOCAL = "model_rf.pkl"

FETCH_SCRIPT = "fetch_live_khi.py"
UPDATE_SCRIPT = "update_features.py"
TRAIN_SCRIPT = "train_model.py"
LOG_FILE = "data/train_pipeline.log"

def log(msg: str):
    karachi_tz = pytz.timezone("Asia/Karachi")           
    ts = datetime.now(karachi_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_step(script):
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

def main():
    log("🌐 Starting full AQI pipeline with training...")

    try:
        # 1️⃣ Fetch & Update features
        log("============== NEW CYCLE START ==============")
        run_step(FETCH_SCRIPT)
        run_step(UPDATE_SCRIPT)

        # 2️⃣ Ensure latest data from S3
        log("📥 Downloading latest processed parquet and model (if exists) from S3...")
        download_from_s3(RAW_PARQUET_S3, RAW_PARQUET_LOCAL)
        try:
            download_from_s3(MODEL_S3, MODEL_LOCAL)
            log(f"✅ Downloaded previous model: {MODEL_LOCAL}")
        except FileNotFoundError:
            log("⚠️ No previous model found. Will train new model from scratch.")

        # 3️⃣ Train model
        run_step(TRAIN_SCRIPT)

        # 4️⃣ Upload trained model back to S3
        upload_to_s3(MODEL_LOCAL, MODEL_S3)
        log(f"✅ Uploaded trained model to S3: {MODEL_S3}")

        log("✅ Full AQI pipeline cycle completed successfully.")

    except KeyboardInterrupt:
        log("\nPipeline stopped manually by user. Goodbye!")

if __name__ == "__main__":
    main()

def main():
    log("🌐 Starting live AQI auto-update pipeline...")
    try:
        log("============== NEW CYCLE START ==============")
        run_step(FETCH_SCRIPT)
        run_step(UPDATE_SCRIPT)
        log(" Cycle completed successfully.")
    except KeyboardInterrupt:
        log("\nPipeline stopped manually by user. Goodbye!")

if __name__ == "__main__":
    main()
