# 4 validate_feast_features.py 
from feast import FeatureStore

store = FeatureStore(repo_path=".")
print("✅ Feature store loaded successfully!")

fv = store.get_feature_view("khi_air_features")

print("\n📋 Registered features in 'khi_air_features':")
for f in fv.features:
    print("-", f.name)
