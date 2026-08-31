"""
Fetch historical pollutant measurements from OpenAQ (v3 API) for the
4-year backfill used to build the initial training dataset.

IMPORTANT — OpenAQ v3 API shape:
OpenAQ v3 does NOT have a `/locations/{id}/measurements` endpoint (that
was v2). In v3, a "location" (monitoring station) has one or more
"sensors", each tracking a single parameter (pm25, no2, etc.), and
measurements are fetched per-sensor via `/v3/sensors/{sensors_id}/measurements`
with `datetime_from`/`datetime_to` query params (not `date_from`/`date_to`).

So the real flow is: find nearby locations -> list each location's sensors
-> filter sensors to the pollutants we care about -> page through each
sensor's measurements.

Docs: https://docs.openaq.org/resources/locations
      https://docs.openaq.org/resources/sensors
      https://docs.openaq.org/api/operations/sensor_measurements_get_v3_sensors__sensors_id__measurements_get
"""
from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional

import requests

from src import config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SESSION = requests.Session()
_HEADERS = {"X-API-Key": config.OPENAQ_API_KEY} if config.OPENAQ_API_KEY else {}

# OpenAQ's free-tier API key allows ~60 requests/minute. The per-sensor,
# per-window pagination in this module can issue a lot of calls, so we
# pace ourselves rather than relying solely on 429 retries.
_REQUEST_PACING_SECONDS = 1.1


class OpenAQError(RuntimeError):
    pass


def _get(path: str, params: dict, retries: int = 3, backoff: float = 2.0) -> dict:
    url = f"{config.OPENAQ_BASE_URL}{path}"
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = _SESSION.get(url, params=params, headers=_HEADERS, timeout=20)
            if resp.status_code == 401:
                raise OpenAQError(
                    "OpenAQ API returned 401 Unauthorized. "
                    "Set OPENAQ_API_KEY (get one free at explore.openaq.org)."
                )
            if resp.status_code == 404:
                # A genuinely missing resource (e.g. a sensor/location that
                # was removed) — don't burn retries on a 404, it won't change.
                raise OpenAQError(f"404 Not Found for {resp.url}")
            if resp.status_code == 429:
                logger.warning("OpenAQ rate limit hit, sleeping before retry...")
                time.sleep(backoff * attempt * 2)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, OpenAQError) as exc:
            last_exc = exc
            if isinstance(exc, OpenAQError) and "404" in str(exc):
                raise  # don't retry 404s
            logger.warning("OpenAQ request failed (attempt %d/%d): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise OpenAQError(f"OpenAQ request failed after {retries} attempts: {last_exc}")


def find_locations(city: "config.City", limit: int = 10) -> list[dict]:
    """Find OpenAQ monitoring stations near a city's coordinates."""
    data = _get(
        "/locations",
        {
            "coordinates": f"{city.lat},{city.lon}",
            "radius": city.openaq_radius_m,
            "limit": limit,
        },
    )
    return data.get("results", [])


def fetch_location_sensors(location_id: int) -> list[dict]:
    """
    List the sensors at a location (one sensor per parameter, e.g. one for
    pm25, one for no2, etc.). Each sensor dict includes an 'id' and a
    'parameter' object with the parameter's name (e.g. "pm25").
    """
    data = _get(f"/locations/{location_id}/sensors", {})
    return data.get("results", [])


def _iter_date_windows(start: datetime, end: datetime, window_days: int = 90) -> Iterator[tuple[datetime, datetime]]:
    cursor = start
    while cursor < end:
        window_end = min(cursor + timedelta(days=window_days), end)
        yield cursor, window_end
        cursor = window_end


def _fmt_datetime(dt: datetime) -> str:
    # OpenAQ v3 accepts ISO 8601; using an explicit UTC offset avoids any
    # ambiguity about which timezone a naive-looking string is in.
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_sensor_measurements(
    sensor_id: int,
    date_from: datetime,
    date_to: datetime,
    limit: int = 1000,
    max_pages: int = 20,
) -> list[dict]:
    """
    Fetch raw measurements for one sensor within a date range (paginated).
    `max_pages` is a safety cap so a single dense window can't spin forever.
    """
    all_results: list[dict] = []
    page = 1
    while page <= max_pages:
        data = _get(
            f"/sensors/{sensor_id}/measurements",
            {
                "datetime_from": _fmt_datetime(date_from),
                "datetime_to": _fmt_datetime(date_to),
                "limit": limit,
                "page": page,
            },
        )
        results = data.get("results", [])
        if not results:
            break
        all_results.extend(results)
        meta = data.get("meta", {})
        found = meta.get("found", 0)
        if not isinstance(found, int) or page * limit >= found:
            break
        page += 1
        time.sleep(_REQUEST_PACING_SECONDS)
    return all_results


def fetch_city_historical(
    city: "config.City",
    years: int = config.BACKFILL_YEARS,
    max_locations: int = 3,
) -> list[dict]:
    """
    Fetch up to `years` of historical pollutant measurements for a city by
    combining data from its nearest OpenAQ monitoring stations.

    Returns a flat list of dict rows: {city, location, parameter, value,
    unit, date_utc} — same shape regardless of the v3 sensor-based fetch
    happening underneath, so downstream code (reshape_openaq_rows) is
    unaffected by this module's internals.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * years)

    locations = find_locations(city, limit=max_locations)
    if not locations:
        logger.warning("No OpenAQ stations found near %s — skipping historical fetch.", city.name)
        return []

    rows: list[dict] = []
    for loc in locations:
        loc_id = loc.get("id")
        loc_name = loc.get("name", str(loc_id))

        try:
            sensors = fetch_location_sensors(loc_id)
        except OpenAQError as exc:
            logger.error("Could not list sensors for %s station '%s' (id=%s): %s", city.name, loc_name, loc_id, exc)
            continue

        pollutant_sensors = [
            s for s in sensors
            if (s.get("parameter") or {}).get("name") in config.POLLUTANTS
        ]
        if not pollutant_sensors:
            logger.warning(
                "Station '%s' (id=%s) near %s has no sensors for our tracked pollutants %s — skipping.",
                loc_name, loc_id, city.name, config.POLLUTANTS,
            )
            continue

        logger.info(
            "Fetching OpenAQ history for %s station '%s' (id=%s), sensors: %s",
            city.name, loc_name, loc_id,
            [s["parameter"]["name"] for s in pollutant_sensors],
        )

        for sensor in pollutant_sensors:
            sensor_id = sensor["id"]
            param_name = sensor["parameter"]["name"]
            param_units = sensor["parameter"].get("units")

            for window_start, window_end in _iter_date_windows(start, end, window_days=90):
                try:
                    measurements = fetch_sensor_measurements(sensor_id, window_start, window_end)
                except OpenAQError as exc:
                    logger.error(
                        "Failed fetching %s window %s-%s for sensor %s (%s) at %s: %s",
                        param_name, window_start.date(), window_end.date(), sensor_id, loc_name, city.name, exc,
                    )
                    continue

                for m in measurements:
                    value = m.get("value")
                    period = m.get("period") or {}
                    dt_from = period.get("datetimeFrom") or {}
                    date_utc = dt_from.get("utc")
                    rows.append(
                        {
                            "city": city.name,
                            "location": loc_name,
                            "parameter": param_name,
                            "value": value,
                            "unit": param_units,
                            "date_utc": date_utc,
                        }
                    )
                time.sleep(_REQUEST_PACING_SECONDS)

    logger.info("Fetched %d historical rows for %s", len(rows), city.name)
    return rows
