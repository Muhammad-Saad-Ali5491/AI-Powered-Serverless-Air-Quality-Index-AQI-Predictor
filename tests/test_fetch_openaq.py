"""
Tests for src/data/fetch_openaq.py against the REAL OpenAQ v3 API shape.

These use mocked HTTP responses (no live network calls, no API key
needed) built from OpenAQ's actual documented response schemas:
https://docs.openaq.org/api/operations/sensors_get_v3_locations__locations_id__sensors_get
https://docs.openaq.org/api/operations/sensor_measurements_get_v3_sensors__sensors_id__measurements_get

This guards specifically against the bug where the code called the
non-existent v3 endpoint `/locations/{id}/measurements` (a v2-only
endpoint) instead of the real `/sensors/{id}/measurements` endpoint —
these tests would fail loudly if that regression were reintroduced.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.data import fetch_openaq


def _mock_sensors_response():
    """Shape of GET /v3/locations/{id}/sensors, per OpenAQ docs."""
    return {
        "meta": {"name": "openaq-api", "page": 1, "limit": 100, "found": 2},
        "results": [
            {"id": 1001, "name": "pm25 sensor", "parameter": {"id": 2, "name": "pm25", "units": "µg/m³", "displayName": "PM2.5"}},
            {"id": 1002, "name": "no2 sensor", "parameter": {"id": 5, "name": "no2", "units": "ppm", "displayName": "NO2"}},
            {"id": 1003, "name": "unrelated sensor", "parameter": {"id": 99, "name": "relativehumidity", "units": "%", "displayName": "RH"}},
        ],
    }


def _mock_measurements_response(value=42.5, param_name="pm25"):
    """Shape of GET /v3/sensors/{id}/measurements, per OpenAQ docs."""
    return {
        "meta": {"name": "openaq-api", "page": 1, "limit": 1000, "found": 1},
        "results": [
            {
                "value": value,
                "flagInfo": {"hasFlags": False},
                "parameter": {"id": 2, "name": param_name, "units": "µg/m³", "displayName": None},
                "period": {
                    "label": "1hour",
                    "interval": "01:00:00",
                    "datetimeFrom": {"utc": "2026-01-01T00:00:00Z", "local": "2026-01-01T05:00:00+05:00"},
                    "datetimeTo": {"utc": "2026-01-01T01:00:00Z", "local": "2026-01-01T06:00:00+05:00"},
                },
                "coordinates": {"latitude": 31.55, "longitude": 74.34},
                "summary": None,
                "coverage": None,
            }
        ],
    }


def test_fetch_location_sensors_uses_correct_v3_endpoint():
    """Guards against calling a nonexistent endpoint for listing sensors."""
    with patch.object(fetch_openaq, "_get", return_value=_mock_sensors_response()) as mock_get:
        sensors = fetch_openaq.fetch_location_sensors(123)

    mock_get.assert_called_once_with("/locations/123/sensors", {})
    assert len(sensors) == 3
    assert sensors[0]["parameter"]["name"] == "pm25"


def test_fetch_sensor_measurements_uses_sensor_endpoint_and_correct_params():
    """
    This is THE regression test for the original bug: the code must call
    /sensors/{id}/measurements with datetime_from/datetime_to — not the
    nonexistent /locations/{id}/measurements with date_from/date_to.
    """
    with patch.object(fetch_openaq, "_get", return_value=_mock_measurements_response()) as mock_get:
        results = fetch_openaq.fetch_sensor_measurements(
            1001,
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 2, 1, tzinfo=timezone.utc),
        )

    call_path, call_params = mock_get.call_args[0]
    assert call_path == "/sensors/1001/measurements"
    assert "datetime_from" in call_params
    assert "datetime_to" in call_params
    assert "date_from" not in call_params  # the old, wrong param name
    assert "date_to" not in call_params    # the old, wrong param name
    assert len(results) == 1
    assert results[0]["value"] == 42.5


def test_fetch_sensor_measurements_paginates_until_found_is_reached():
    page1 = _mock_measurements_response(value=1.0)
    page1["meta"]["found"] = 1500  # more than one page's worth
    page2 = _mock_measurements_response(value=2.0)
    page2["meta"]["found"] = 1500

    with patch.object(fetch_openaq, "_get", side_effect=[page1, page2, {"meta": {"found": 1500}, "results": []}]) as mock_get:
        with patch("time.sleep"):  # skip real pacing delay in tests
            results = fetch_openaq.fetch_sensor_measurements(
                1001,
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 2, 1, tzinfo=timezone.utc),
                limit=1000,
            )

    assert mock_get.call_count == 2  # stops once page*limit >= found, doesn't need the empty 3rd page
    assert len(results) == 2


def test_fetch_city_historical_filters_sensors_to_tracked_pollutants(monkeypatch):
    """Sensors for untracked parameters (e.g. relative humidity) must be skipped."""
    from src import config

    city = config.get_city("Lahore")

    with patch.object(fetch_openaq, "find_locations", return_value=[{"id": 123, "name": "Test Station"}]):
        with patch.object(fetch_openaq, "fetch_location_sensors", return_value=_mock_sensors_response()["results"]):
            with patch.object(
                fetch_openaq, "fetch_sensor_measurements",
                return_value=_mock_measurements_response()["results"],
            ) as mock_measurements:
                with patch("time.sleep"):
                    rows = fetch_openaq.fetch_city_historical(city, years=1, max_locations=1)

    # only pm25 and no2 sensors should have been queried (2 sensors x N windows),
    # never the "relativehumidity" sensor
    queried_sensor_ids = {call.args[0] for call in mock_measurements.call_args_list}
    assert queried_sensor_ids == {1001, 1002}
    assert len(rows) > 0
    assert all(r["parameter"] in config.POLLUTANTS for r in rows)


def test_fetch_city_historical_returns_empty_list_when_no_locations_found():
    from src import config
    city = config.get_city("Quetta")

    with patch.object(fetch_openaq, "find_locations", return_value=[]):
        rows = fetch_openaq.fetch_city_historical(city, years=1)

    assert rows == []


def test_row_date_utc_extracted_from_nested_period_structure():
    """
    The v3 measurement schema nests the timestamp under
    period.datetimeFrom.utc — not a top-level 'date' field like v2 had.
    This test locks in correct extraction of that nested value.
    """
    from src import config
    city = config.get_city("Lahore")

    with patch.object(fetch_openaq, "find_locations", return_value=[{"id": 123, "name": "Test Station"}]):
        with patch.object(fetch_openaq, "fetch_location_sensors", return_value=[_mock_sensors_response()["results"][0]]):
            with patch.object(
                fetch_openaq, "fetch_sensor_measurements",
                return_value=_mock_measurements_response()["results"],
            ):
                with patch("time.sleep"):
                    rows = fetch_openaq.fetch_city_historical(city, years=1, max_locations=1)

    assert len(rows) > 0
    assert rows[0]["date_utc"] == "2026-01-01T00:00:00Z"
