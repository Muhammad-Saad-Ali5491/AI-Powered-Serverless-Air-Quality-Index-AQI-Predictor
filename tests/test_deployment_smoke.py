import json

import pandas as pd

from src.inference.predict import load_champion
from src.utils.paths import FEATURE_STORE_DIR, MODELS_DIR


def test_checked_in_deployment_assets_are_loadable():
    registry_path = MODELS_DIR / "registry.json"
    feature_path = FEATURE_STORE_DIR / "aqi_features.parquet"

    assert registry_path.exists(), "models/registry.json is required by Streamlit Cloud"
    assert feature_path.exists(), "data/features/aqi_features.parquet is required by Streamlit Cloud"

    registry = json.loads(registry_path.read_text())
    artifact = registry["champion"]["artifact"]
    assert (MODELS_DIR / artifact).exists(), f"Champion artifact is missing: {artifact}"

    bundle, champion = load_champion()
    assert bundle["model"] is not None
    assert champion["feature_columns"]

    features = pd.read_parquet(feature_path)
    assert not features.empty
    assert {"city", "timestamp", "aqi"}.issubset(features.columns)