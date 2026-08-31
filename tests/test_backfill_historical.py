"""
Tests for src/data/backfill_historical.py's reshape_openaq_rows, focused on
the real production bug where cities with partial sensor coverage (e.g. a
station reporting only pm25, with no10/no2/so2/co/o3 sensors at all) caused
missing pollutant columns to be filled with pandas' pd.NA sentinel instead
of a real float NaN — which crashed downstream AQI computation with
"TypeError: boolean value of NA is ambiguous".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import pandas as pd

from src.data.backfill_historical import reshape_openaq_rows
from src.features.feature_engineering import build_feature_table
from src import config


def _rows_for_city_with_only_pm25(city="Lahore", n_hours=80):
    """Simulates exactly what happens when a station only has a pm25
    sensor: every other pollutant column is entirely absent from the raw
    OpenAQ rows for this city."""
    from datetime import datetime, timedelta, timezone

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n_hours):
        ts = start + timedelta(hours=i)
        rows.append(
            {
                "city": city,
                "location": "Test Station",
                "parameter": "pm25",
                "value": 30.0 + (i % 10),
                "unit": "µg/m³",
                "date_utc": ts.isoformat(),
            }
        )
    return rows


def test_reshape_fills_missing_pollutants_with_real_nan_not_pd_na():
    rows = _rows_for_city_with_only_pm25()
    wide = reshape_openaq_rows(rows)

    assert "pm25" in wide.columns
    for pollutant in config.POLLUTANTS:
        if pollutant == "pm25":
            continue
        assert pollutant in wide.columns
        missing_value = wide[pollutant].iloc[0]
        # Must be a real float NaN (math.isnan works), NOT pandas' pd.NA
        # sentinel (which raises in boolean numeric comparisons downstream).
        assert isinstance(missing_value, float), (
            f"{pollutant} missing-value fill is {type(missing_value)}, expected float('nan')"
        )
        assert math.isnan(missing_value)
        assert pd.isna(missing_value)  # sanity: pd.isna still correctly detects it as missing


def test_full_pipeline_with_partial_sensor_coverage_does_not_crash():
    """
    End-to-end regression test: a city where the only available data is a
    single pollutant (as happens with real OpenAQ stations that only run
    a pm25 sensor) must flow all the way through reshape -> feature
    engineering -> AQI computation without raising.
    """
    rows = _rows_for_city_with_only_pm25(n_hours=80)
    wide = reshape_openaq_rows(rows)

    # This is the exact call chain that crashed in production.
    features = build_feature_table(wide)

    assert not features.empty
    assert features["aqi"].notna().all()
    # AQI should be computable from pm25 alone even though every other
    # pollutant column is entirely missing for this city.
    assert (features["aqi"] > 0).all()


def test_reshape_handles_multiple_cities_with_different_sensor_coverage():
    """Different cities can have completely different available pollutants
    — this must not cause cross-contamination or crashes."""
    from datetime import datetime, timedelta, timezone

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = list(_rows_for_city_with_only_pm25("Lahore", n_hours=80))
    # Karachi has pm25 AND no2 sensors
    for i in range(80):
        ts = start + timedelta(hours=i)
        rows.append(
            {"city": "Karachi", "location": "Test Station 2", "parameter": "pm25",
             "value": 25.0, "unit": "µg/m³", "date_utc": ts.isoformat()}
        )
        rows.append(
            {"city": "Karachi", "location": "Test Station 2", "parameter": "no2",
             "value": 15.0, "unit": "ppm", "date_utc": ts.isoformat()}
        )

    wide = reshape_openaq_rows(rows)
    features = build_feature_table(wide)

    assert set(features["city"].unique()) == {"Lahore", "Karachi"}
    assert features["aqi"].notna().all()
