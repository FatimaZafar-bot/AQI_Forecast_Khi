import subprocess
import traceback
import sys
from datetime import datetime
import pytz

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
