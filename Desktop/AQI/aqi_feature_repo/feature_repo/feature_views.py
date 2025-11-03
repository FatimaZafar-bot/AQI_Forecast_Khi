from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

# --- Entity ---
city = Entity(
    name="city_id",
    join_keys=["city_id"],
    description="City identifier for AQI data"
)

# --- Data Source ---
khi_source = FileSource(
    path="data/khi_air_quality_clean.parquet",  # ✅ update if your parquet path differs
    timestamp_field="event_timestamp"
)

# --- Feature View ---
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
    online=True,
    source=khi_source,
)
