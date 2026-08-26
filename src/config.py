"""
Central configuration for the Pearls AQI Predictor.

Loads secrets from environment variables (populated locally via a .env
file, and in CI via GitHub Actions "secrets"). Nothing sensitive is
hard-coded here.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # no-op in CI where real env vars are already set

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", "")  # OpenAQ v3 requires an API key

# ---------------------------------------------------------------------------
# Feature store: Hopsworks (default, free serverless tier) settings.
# A local Parquet cache is ALWAYS kept in sync as a fast offline fallback —
# see src/features/feature_store.py — so the pipeline still works even
# without a Hopsworks account, just with local storage instead.
# ---------------------------------------------------------------------------
USE_HOPSWORKS = os.getenv("USE_HOPSWORKS", "true").lower() == "true"
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "")  # blank = app.hopsworks.ai (managed serverless tier)
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
    City("Karachi", lat=24.8607, lon=67.0011),
    City("Islamabad", lat=33.6844, lon=73.0479),
    City("Rawalpindi", lat=33.5651, lon=73.0169),
    City("Faisalabad", lat=31.4504, lon=73.1350),
    City("Multan", lat=30.1575, lon=71.5249),
    City("Peshawar", lat=34.0151, lon=71.5249),
    City("Quetta", lat=30.1798, lon=66.9750),
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
