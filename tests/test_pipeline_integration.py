"""
Full end-to-end smoke test of the pipeline, using synthetic data so it
needs no live API keys or GCP project. This is the test that proves the
whole system (feature engineering -> feature store -> training -> model
registry -> inference -> SHAP) actually works together, on the OS running
the test (Windows, Linux, or macOS).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src import config
from src.features.feature_engineering import build_feature_table
from src.features.feature_store import get_feature_store
from src.training.train_model import (
    run_training,
    build_training_matrix,
    _prune_stale_artifacts,
    MODELS_DIR,
)
from src.inference.predict import forecast_city, ModelNotTrainedError
from src.features.feature_store import LocalFeatureStore
from scripts.generate_synthetic_data import generate_raw_history


@pytest.fixture(scope="module")
def trained_pipeline():
    """Generate a small synthetic dataset, train models on it, and leave
    the feature store + model registry populated for downstream tests.

    The store is reset first so this test is deterministic and isolated
    from any data left behind by manual runs or other test sessions
    (e.g. running scripts/generate_synthetic_data.py locally beforehand).
    """
    cities = [config.get_city("Lahore"), config.get_city("Karachi")]
    raw_df = generate_raw_history(hours=24 * 20, cities=cities)  # 20 days, small & fast
    feature_df = build_feature_table(raw_df)

    store = get_feature_store()
    if isinstance(store, LocalFeatureStore) and store.path.exists():
        store.path.unlink()
    store.write_features(feature_df)

    run_record = run_training(df=store.read_features(), epochs=3)
    return run_record


def test_feature_table_has_expected_columns(trained_pipeline):
    store = get_feature_store()
    df = store.read_features()
    assert not df.empty
    for col in config.ALL_FEATURES:
        assert col in df.columns, f"missing feature column {col}"


def test_training_matrix_builds_without_error(trained_pipeline):
    store = get_feature_store()
    df = store.read_features()
    X, y, feature_cols = build_training_matrix(df)
    assert len(X) > 0
    assert y.shape[1] == 3  # 24h/48h/72h horizons
    assert len(feature_cols) > 0


def test_training_produces_champion_with_metrics(trained_pipeline):
    run_record = trained_pipeline
    assert run_record["model_type"] in ("ridge", "random_forest", "xgboost", "tensorflow")
    assert "overall" in run_record["metrics"][run_record["model_type"]]
    rmse = run_record["metrics"][run_record["model_type"]]["overall"]["rmse"]
    assert rmse >= 0
    assert rmse < 1000  # sanity bound, not a tight accuracy requirement


def test_inference_produces_3day_forecast(trained_pipeline):
    result = forecast_city("Lahore")
    assert result["city"] == "Lahore"
    assert len(result["forecast"]) == 3
    horizons = sorted(f["horizon_hours"] for f in result["forecast"])
    assert horizons == [24, 48, 72]
    for f in result["forecast"]:
        assert 0 <= f["predicted_aqi"] <= 500
        assert f["category"] in (
            "Good", "Moderate", "Unhealthy for Sensitive Groups",
            "Unhealthy", "Very Unhealthy", "Hazardous",
        )


def test_inference_raises_for_city_with_no_data(trained_pipeline):
    with pytest.raises(ValueError):
        forecast_city("Quetta")  # not in the synthetic dataset generated above


def test_inference_raises_meaningful_error_for_unknown_city(trained_pipeline):
    with pytest.raises(ValueError):
        forecast_city("Not A Real City")


def test_stale_model_artifacts_are_pruned_after_training(trained_pipeline):
    """
    Guards against unbounded repo growth: after training, models/ should
    contain ONLY the current champion's artifact (plus registry.json and
    .gitkeep) — not every historical run's model file. This matters because
    the daily GitHub Actions training workflow commits models/ back to the
    repo; without pruning, the repo would grow by ~tens of MB every day
    forever.
    """
    run_record = trained_pipeline
    champion_artifact = run_record["artifact"]

    model_files = [
        p.name for p in MODELS_DIR.glob("*")
        if p.suffix in (".joblib", ".keras") and p.name != "registry.json"
    ]
    # every remaining model file must belong to the champion (main artifact,
    # or its companion scaler file if it's a tensorflow model)
    for fname in model_files:
        is_champion_file = (
            fname == champion_artifact
            or fname == champion_artifact.replace(".keras", "_scaler.joblib")
        )
        assert is_champion_file, f"Stale model artifact was not pruned: {fname}"


def test_prune_stale_artifacts_keeps_only_named_champion(tmp_path, monkeypatch):
    """Unit-level check of the pruning function in isolation."""
    monkeypatch.setattr("src.training.train_model.MODELS_DIR", tmp_path)

    (tmp_path / "champion_model.joblib").write_bytes(b"keep-me")
    (tmp_path / "old_run_1.joblib").write_bytes(b"delete-me")
    (tmp_path / "old_run_2.keras").write_bytes(b"delete-me")
    (tmp_path / "old_run_2_scaler.joblib").write_bytes(b"delete-me")
    (tmp_path / "registry.json").write_text("{}")
    (tmp_path / ".gitkeep").write_bytes(b"")

    _prune_stale_artifacts("champion_model.joblib")

    remaining = {p.name for p in tmp_path.glob("*")}
    assert remaining == {"champion_model.joblib", "registry.json", ".gitkeep"}
