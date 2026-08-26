import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.features.feature_store import LocalFeatureStore


@pytest.fixture
def tmp_store(tmp_path):
    return LocalFeatureStore(path=tmp_path / "features.parquet")


def _sample_df(city="Lahore", n=5):
    return pd.DataFrame(
        {
            "city": [city] * n,
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
            "aqi": list(range(50, 50 + n)),
        }
    )


def test_write_and_read_roundtrip(tmp_store):
    df = _sample_df()
    tmp_store.write_features(df)
    result = tmp_store.read_features()
    assert len(result) == len(df)


def test_read_filters_by_city(tmp_store):
    df = pd.concat([_sample_df("Lahore", 3), _sample_df("Karachi", 4)], ignore_index=True)
    tmp_store.write_features(df)
    lahore_only = tmp_store.read_features(city="Lahore")
    assert len(lahore_only) == 3
    assert (lahore_only["city"] == "Lahore").all()


def test_write_deduplicates_on_city_and_timestamp(tmp_store):
    df1 = _sample_df(n=3)
    tmp_store.write_features(df1)
    # overlapping write with same timestamps should not double rows
    df2 = _sample_df(n=3)
    tmp_store.write_features(df2)
    result = tmp_store.read_features()
    assert len(result) == 3


def test_get_latest_returns_most_recent_row(tmp_store):
    df = _sample_df(n=5)
    tmp_store.write_features(df)
    latest = tmp_store.get_latest("Lahore")
    assert latest is not None
    assert latest["aqi"] == 54  # last row in the generated sequence


def test_get_latest_returns_none_when_empty(tmp_store):
    assert tmp_store.get_latest("Lahore") is None


def test_read_features_empty_store_returns_empty_df(tmp_store):
    result = tmp_store.read_features()
    assert result.empty
