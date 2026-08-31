"""
Central configuration for the Pearls AQI Predictor.

Loads secrets from environment variables (populated locally via a .env
file, and in CI via GitHub Actions "secrets"). Also supports Streamlit
Community Cloud secrets (st.secrets) when running under Streamlit.

Nothing sensitive is hard-coded here.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # no-op in CI where real env vars are already set


def _get_secret(name: str, default: str = "") -> str:
    """Read a secret from env, falling back to Streamlit secrets if available.
    This makes the same code work locally (.env), in GitHub Actions (env),
    and on Streamlit Community Cloud (st.secrets).
    """
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
OPENWEATHER_API_KEY = _get_secret("OPENWEATHER_API_KEY", "")
OPENAQ_API_KEY = _get_secret("OPENAQ_API_KEY", "")  # OpenAQ v3 requires an API key

# ---------------------------------------------------------------------------
# Feature store: Hopsworks (default, free serverless tier) settings.
# A local Parquet cache is ALWAYS kept in sync as a fast offline fallback —
# see src/features/feature_store.py — so the pipeline still works even
# without a Hopsworks account, just with local storage instead.
# ---------------------------------------------------------------------------
# Prefer explicit USE_HOPSWORKS; if unset and no API key, default to local
# store so Streamlit Cloud / local runs without Hopsworks never fail.
_raw_use_hw = os.getenv("USE_HOPSWORKS")
HOPSWORKS_API_KEY = _get_secret("HOPSWORKS_API_KEY", "")
if _raw_use_hw is None:
    USE_HOPSWORKS = bool(HOPSWORKS_API_KEY)  # auto-enable only when key present
else:
    USE_HOPSWORKS = _raw_use_hw.lower() == "true"
HOPSWORKS_PROJECT_NAME = _get_secret("HOPSWORKS_PROJECT_NAME", "")
HOPSWORKS_HOST = _get_secret("HOPSWORKS_HOST", "")  # blank = c.app.hopsworks.ai (managed serverless)
HOPSWORKS_FEATURE_GROUP_NAME = os.getenv("HOPSWORKS_FEATURE_GROUP_NAME", "aqi_features")
HOPSWORKS_FEATURE_GROUP_VERSION = int(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "1"))

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
OPENWEATHER_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_AIR_POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
OPENWEATHER_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
OPENAQ_BASE_URL = "https://api.openaq.org/v3"

# ---------------------------------------------------------------------------
# Cities: major Pakistani cities. lat/lon required by OpenWeather; OpenAQ is
# queried by coordinate radius since station coverage varies by city.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class City:
    name: str
    country: str = "PK"
    lat: float = 0.0
    lon: float = 0.0
    openaq_radius_m: int = 25000  # search radius for OpenAQ stations


CITIES = [
    City("Lahore", lat=31.5497, lon=74.3436),
    City("Karachi", lat=24.8608, lon=67.0104),
    City("Islamabad", lat=33.7215, lon=73.0433),
    City("Rawalpindi", lat=33.5973, lon=73.0479),
    City("Faisalabad", lat=31.4167, lon=73.0911),
    City("Multan", lat=30.1978, lon=71.4697),
    City("Peshawar", lat=34.0151, lon=71.5675),
    City("Quetta", lat=30.1841, lon=67.0014),
]

CITY_NAMES = [c.name for c in CITIES]


def get_city(name: str) -> City:
    for c in CITIES:
        if c.name.lower() == name.lower():
            return c
    raise ValueError(f"Unknown city '{name}'. Available: {CITY_NAMES}")


# ---------------------------------------------------------------------------
# Pollutants tracked (matches both OpenWeather Air Pollution API and OpenAQ)
# ---------------------------------------------------------------------------
POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3"]

# ---------------------------------------------------------------------------
# Backfill window for historical data (OpenAQ)
# ---------------------------------------------------------------------------
BACKFILL_YEARS = 4

# ---------------------------------------------------------------------------
# Forecast horizon
# ---------------------------------------------------------------------------
FORECAST_HORIZON_DAYS = 3

# ---------------------------------------------------------------------------
# Hazardous AQI threshold (US EPA scale) used for alerting
# ---------------------------------------------------------------------------
HAZARDOUS_AQI_THRESHOLD = 200  # "Very Unhealthy" and above

# ---------------------------------------------------------------------------
# Feature columns used by the models (kept in one place so training +
# inference always agree on schema)
# ---------------------------------------------------------------------------
TIME_FEATURES = ["hour", "day", "month", "day_of_week", "is_weekend", "day_of_year"]
WEATHER_FEATURES = ["temp", "humidity", "pressure", "wind_speed", "wind_deg", "clouds"]
POLLUTANT_FEATURES = POLLUTANTS
DERIVED_FEATURES = [
    "aqi_lag_1h", "aqi_lag_24h", "aqi_lag_72h",
    "aqi_rolling_mean_24h", "aqi_rolling_std_24h",
    "aqi_change_rate_1h", "aqi_change_rate_24h",
]
TARGET_COLUMN = "aqi"

ALL_FEATURES = TIME_FEATURES + WEATHER_FEATURES + POLLUTANT_FEATURES + DERIVED_FEATURES
