"""
Fetch historical pollutant measurements from OpenAQ (v3 API) for the
4-year backfill used to build the initial training dataset.

OpenAQ organizes data by "locations" (monitoring stations). For each
Pakistani city we find nearby stations, then page through their
historical measurements.

Docs: https://docs.openaq.org/
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
            if resp.status_code == 429:
                logger.warning("OpenAQ rate limit hit, sleeping before retry...")
                time.sleep(backoff * attempt * 2)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, OpenAQError) as exc:
            last_exc = exc
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


def _iter_date_windows(start: datetime, end: datetime, window_days: int = 30) -> Iterator[tuple[datetime, datetime]]:
    cursor = start
    while cursor < end:
        window_end = min(cursor + timedelta(days=window_days), end)
        yield cursor, window_end
        cursor = window_end


def fetch_location_measurements(
    location_id: int,
    date_from: datetime,
    date_to: datetime,
    limit: int = 1000,
) -> list[dict]:
    """Fetch raw measurements for one station within a date range (paginated)."""
    all_results = []
    page = 1
    while True:
        data = _get(
            f"/locations/{location_id}/measurements",
            {
                "date_from": date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "date_to": date_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        if page * limit >= (found if isinstance(found, int) else len(all_results)):
            break
        page += 1
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
    unit, date_utc}.
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
        logger.info("Fetching OpenAQ history for %s station '%s' (id=%s)", city.name, loc_name, loc_id)
        for window_start, window_end in _iter_date_windows(start, end, window_days=30):
            try:
                measurements = fetch_location_measurements(loc_id, window_start, window_end)
            except OpenAQError as exc:
                logger.error("Failed fetching window %s-%s for %s: %s", window_start, window_end, loc_name, exc)
                continue
            for m in measurements:
                param = (m.get("parameter") or {}).get("name") if isinstance(m.get("parameter"), dict) else m.get("parameter")
                value = m.get("value")
                date_info = m.get("date", {})
                date_utc = date_info.get("utc") if isinstance(date_info, dict) else m.get("date_utc")
                rows.append(
                    {
                        "city": city.name,
                        "location": loc_name,
                        "parameter": param,
                        "value": value,
                        "unit": m.get("unit"),
                        "date_utc": date_utc,
                    }
                )
    logger.info("Fetched %d historical rows for %s", len(rows), city.name)
    return rows
