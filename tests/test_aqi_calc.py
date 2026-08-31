import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.aqi_calc import compute_aqi, aqi_category, convert_units


def test_compute_aqi_good_air():
    # Low PM2.5 concentration -> "Good" range (0-50)
    aqi = compute_aqi({"pm25": 5.0})
    assert aqi is not None
    assert 0 <= aqi <= 50


def test_compute_aqi_moderate():
    aqi = compute_aqi({"pm25": 20.0})
    assert 51 <= aqi <= 100


def test_compute_aqi_unhealthy():
    aqi = compute_aqi({"pm25": 100.0})
    assert 151 <= aqi <= 200


def test_compute_aqi_takes_max_across_pollutants():
    # pm25 alone would be "Good" but pm10 pushes it much higher
    low_pm25 = compute_aqi({"pm25": 5.0})
    combined = compute_aqi({"pm25": 5.0, "pm10": 400})
    assert combined > low_pm25


def test_compute_aqi_none_when_no_data():
    assert compute_aqi({}) is None
    assert compute_aqi({"pm25": None}) is None


def test_compute_aqi_negative_ignored():
    assert compute_aqi({"pm25": -5}) is None


def test_aqi_category_boundaries():
    assert aqi_category(0) == "Good"
    assert aqi_category(50) == "Good"
    assert aqi_category(51) == "Moderate"
    assert aqi_category(101) == "Unhealthy for Sensitive Groups"
    assert aqi_category(151) == "Unhealthy"
    assert aqi_category(201) == "Very Unhealthy"
    assert aqi_category(301) == "Hazardous"
    assert aqi_category(None) == "Unknown"


def test_convert_units_pm_unchanged():
    assert convert_units("pm25", 35.0) == 35.0


def test_convert_units_gas_converted():
    val = convert_units("no2", 100.0)
    assert val != 100.0
    assert val > 0


def test_compute_aqi_extreme_value_caps_at_500():
    aqi = compute_aqi({"pm25": 900.0})
    assert aqi == 500


def test_compute_aqi_handles_pandas_na_without_crashing():
    """
    Regression test for a real production crash: when a pollutant column
    is entirely missing (e.g. a station only has a pm25 sensor, so no10/
    no2/so2/co/o3 are absent), the backfill pipeline used to fill those
    columns with pandas' pd.NA sentinel. Passing pd.NA through a numeric
    comparison (`if concentration < 0`) raises
    "TypeError: boolean value of NA is ambiguous" instead of behaving like
    a normal missing value. compute_aqi must treat pd.NA exactly like None.
    """
    import pandas as pd

    # Should not raise, and should compute AQI from the one real value.
    aqi = compute_aqi({"pm25": 20.0, "pm10": pd.NA, "no2": pd.NA, "so2": pd.NA, "co": pd.NA, "o3": pd.NA})
    assert aqi is not None
    assert 51 <= aqi <= 100

    # All-NA input should return None, not raise.
    assert compute_aqi({"pm25": pd.NA, "pm10": pd.NA}) is None


def test_sub_index_handles_pandas_na_directly():
    import pandas as pd
    from src.utils.aqi_calc import _sub_index

    assert _sub_index("pm25", pd.NA) is None
    assert _sub_index("no2", pd.NA) is None
