"""
Fetch real-time weather + air pollution data from OpenWeather.

Used by the hourly/daily feature pipeline to get the latest observation
for each Pakistani city. Also used by the inference pipeline to build the
"current" feature row a forecast is anchored to.
"""
from __future__ import annotations
import time
from datetime import datetime, timezone
from typing import Optional

import requests

from src import config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SESSION = requests.Session()


class OpenWeatherError(RuntimeError):
    pass


def _get(url: str, params: dict, retries: int = 3, backoff: float = 2.0) -> dict:
    params = {**params, "appid": config.OPENWEATHER_API_KEY}
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = _SESSION.get(url, params=params, timeout=15)
            if resp.status_code == 401:
                raise OpenWeatherError(
                    "OpenWeather API returned 401 Unauthorized. "
                    "Check that OPENWEATHER_API_KEY is set correctly."
                )
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, OpenWeatherError) as exc:
            last_exc = exc
            logger.warning("OpenWeather request failed (attempt %d/%d): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise OpenWeatherError(f"OpenWeather request failed after {retries} attempts: {last_exc}")


def fetch_current_weather(city: "config.City") -> dict:
    """Fetch current weather (temp, humidity, wind, pressure, clouds)."""
    data = _get(
        config.OPENWEATHER_CURRENT_URL,
        {"lat": city.lat, "lon": city.lon, "units": "metric"},
    )
    main = data.get("main", {})
    wind = data.get("wind", {})
    clouds = data.get("clouds", {})
    return {
        "city": city.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "temp": main.get("temp"),
        "humidity": main.get("humidity"),
        "pressure": main.get("pressure"),
        "wind_speed": wind.get("speed"),
        "wind_deg": wind.get("deg"),
        "clouds": clouds.get("all"),
        "weather_main": (data.get("weather") or [{}])[0].get("main"),
    }


def fetch_air_pollution(city: "config.City") -> dict:
    """Fetch current air-pollution component concentrations (ug/m3)."""
    data = _get(
        config.OPENWEATHER_AIR_POLLUTION_URL,
        {"lat": city.lat, "lon": city.lon},
    )
    entries = data.get("list", [])
    if not entries:
        raise OpenWeatherError(f"No air pollution data returned for {city.name}")
    components = entries[0].get("components", {})
    return {
        "city": city.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pm25": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "no2": components.get("no2"),
        "so2": components.get("so2"),
        "co": components.get("co"),
        "o3": components.get("o3"),
        "openweather_aqi_index": entries[0].get("main", {}).get("aqi"),  # OWM's own 1-5 scale (kept for reference)
    }


def fetch_forecast(city: "config.City") -> list[dict]:
    """5-day / 3-hour forecast — used to enrich features with 'known future weather'."""
    data = _get(
        config.OPENWEATHER_FORECAST_URL,
        {"lat": city.lat, "lon": city.lon, "units": "metric"},
    )
    records = []
    for item in data.get("list", []):
        main = item.get("main", {})
        wind = item.get("wind", {})
        clouds = item.get("clouds", {})
        records.append(
            {
                "city": city.name,
                "forecast_time": datetime.fromtimestamp(item["dt"], tz=timezone.utc).isoformat(),
                "temp": main.get("temp"),
                "humidity": main.get("humidity"),
                "pressure": main.get("pressure"),
                "wind_speed": wind.get("speed"),
                "wind_deg": wind.get("deg"),
                "clouds": clouds.get("all"),
            }
        )
    return records


def fetch_city_snapshot(city: "config.City") -> dict:
    """Combine weather + pollution into one row for the feature pipeline."""
    weather = fetch_current_weather(city)
    pollution = fetch_air_pollution(city)
    snapshot = {**weather}
    snapshot.update({k: v for k, v in pollution.items() if k not in ("city", "timestamp")})
    return snapshot
