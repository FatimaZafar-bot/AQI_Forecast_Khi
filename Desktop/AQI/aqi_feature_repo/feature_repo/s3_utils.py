import boto3
import os

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
)

BUCKET_NAME = "aqidata11"

def download_from_s3(s3_key, local_path):
    """Download a file from S3 to local path."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(BUCKET_NAME, s3_key, local_path)
    print(f"✅ Downloaded {s3_key} → {local_path}")

def upload_to_s3(local_path, s3_key):
    """Upload a local file to S3."""
    s3.upload_file(local_path, BUCKET_NAME, s3_key)
    print(f"✅ Uploaded {local_path} → {s3_key}")
