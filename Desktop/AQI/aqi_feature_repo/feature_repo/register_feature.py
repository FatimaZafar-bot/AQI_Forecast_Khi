# 3 register_features.py
from feast import FeatureStore, Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from datetime import timedelta
import pandas as pd
from s3_utils import download_from_s3

PARQUET_PATH = "data/khi_air_quality_clean.parquet"

# 🔹 Always download the latest processed data parquet from S3
print("⬇️ Downloading latest processed parquet from S3...")
download_from_s3("data/khi_air_quality_clean.parquet", PARQUET_PATH)
print("✅ Downloaded parquet from S3")


city = Entity(name="city", join_keys=["city_id"])

khi_source = FileSource(
    path=PARQUET_PATH,
    timestamp_field="time",
)

khi_air_features = FeatureView(
    name="khi_air_features",
    entities=[city],
    ttl=timedelta(days=365),
    schema=[
        Field(name="pm2_5", dtype=Float32),
        Field(name="pm10", dtype=Float32),
        Field(name="no2", dtype=Float32),
        Field(name="so2", dtype=Float32),
        Field(name="co", dtype=Float32),
        Field(name="o3", dtype=Float32),
        Field(name="pm2_5_3h_avg", dtype=Float32),
        Field(name="pm10_3h_avg", dtype=Float32),
        Field(name="pm2_5_6h_avg", dtype=Float32),
        Field(name="pm10_6h_avg", dtype=Float32),
        Field(name="pm2_5_norm", dtype=Float32),
        Field(name="pm10_norm", dtype=Float32),
        Field(name="no2_norm", dtype=Float32),
        Field(name="so2_norm", dtype=Float32),
        Field(name="co_norm", dtype=Float32),
        Field(name="o3_norm", dtype=Float32),
        Field(name="hour", dtype=Int64),
        Field(name="day", dtype=Int64),
        Field(name="month", dtype=Int64),
        Field(name="dayofweek", dtype=Int64),
        Field(name="aqi_estimate", dtype=Float32),
        Field(name="aqi_change_rate", dtype=Float32),
        Field(name="city_name", dtype=String),
    ],
    source=khi_source,
)


store = FeatureStore(repo_path=".")
store.apply([city, khi_air_features])
print("✅ Features registered successfully!")

# Materializing
df = pd.read_parquet(PARQUET_PATH)
store.materialize(df["time"].min(), df["time"].max())
print("✅ Feast materialization completed successfully!")
