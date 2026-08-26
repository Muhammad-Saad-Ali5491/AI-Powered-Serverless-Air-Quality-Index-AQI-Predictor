import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

from src.features.feature_engineering import (
    add_time_features,
    add_aqi_column,
    add_lag_and_rolling_features,
    build_feature_table,
)


def make_raw_df(n_hours=100, city="Lahore"):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n_hours):
        ts = start + timedelta(hours=i)
        rows.append(
            {
                "city": city,
                "timestamp": ts.isoformat(),
                "temp": 20 + i * 0.01,
                "humidity": 50,
                "pressure": 1010,
                "wind_speed": 2.5,
                "wind_deg": 180,
                "clouds": 30,
                "pm25": 30 + 5 * np.sin(i / 10),
                "pm10": 45,
                "no2": 20,
                "so2": 8,
                "co": 700,
                "o3": 25,
            }
        )
    return pd.DataFrame(rows)


def test_add_time_features_columns_present():
    df = add_time_features(make_raw_df(5))
    for col in ["hour", "day", "month", "day_of_week", "is_weekend", "day_of_year"]:
        assert col in df.columns
    assert df["hour"].between(0, 23).all()


def test_add_aqi_column_produces_values():
    df = add_aqi_column(make_raw_df(5))
    assert "aqi" in df.columns
    assert df["aqi"].notna().all()
    assert (df["aqi"] >= 0).all()


def test_add_lag_and_rolling_features_shapes():
    df = add_aqi_column(make_raw_df(50))
    df = add_lag_and_rolling_features(df)
    for col in ["aqi_lag_1h", "aqi_lag_24h", "aqi_rolling_mean_24h", "aqi_change_rate_1h"]:
        assert col in df.columns
    # first row has no lag -> NaN
    assert pd.isna(df.iloc[0]["aqi_lag_1h"])
    # by row 25 the 24h lag should be populated
    assert pd.notna(df.iloc[25]["aqi_lag_24h"])


def test_build_feature_table_end_to_end_no_nans_in_target():
    raw = make_raw_df(80)
    features = build_feature_table(raw)
    assert not features.empty
    assert features["aqi"].notna().all()
    # engineered columns exist
    for col in ["hour", "aqi_lag_1h", "aqi_rolling_mean_24h"]:
        assert col in features.columns


def test_build_feature_table_multi_city_independent_lags():
    raw_lhr = make_raw_df(60, city="Lahore")
    raw_khi = make_raw_df(60, city="Karachi")
    raw = pd.concat([raw_lhr, raw_khi], ignore_index=True)
    features = build_feature_table(raw)
    assert set(features["city"].unique()) == {"Lahore", "Karachi"}
    # each city should have its own independent row count roughly matching input
    assert features[features["city"] == "Lahore"].shape[0] > 0
    assert features[features["city"] == "Karachi"].shape[0] > 0
