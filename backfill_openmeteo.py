"""
Historical backfill using Open-Meteo.

Open-Meteo requires no API key and provides:
  - Historical weather archive: https://archive-api.open-meteo.com/v1/archive
  - Historical air quality (CAMS reanalysis): https://air-quality-api.open-meteo.com/v1/air-quality

This script pulls BACKFILL_YEARS of hourly data for every city in
config.CITIES, merges weather + pollution on timestamp, reshapes into the
same schema as the live feature_pipeline.py, and inserts into Hopsworks.

IMPORTANT SCALE CAVEAT:
Open-Meteo's `us_aqi` field is on the 0-500 US EPA scale.
OpenWeather's `aqi` field (used by the live pipeline) is on a 1-5 categorical
scale. These are NOT the same scale. Mixing them directly as one 'aqi'
target column will corrupt your training data.

This script therefore does NOT populate 'aqi' from Open-Meteo's us_aqi.
Instead, it leaves 'aqi' null for backfilled historical rows, and your
training pipeline should primarily use pm2_5 (which IS on a consistent,
comparable concentration scale across both sources) as the modeling target,
or you must explicitly convert one scale to the other before combining them.
See training_pipeline.py for how pm2_5 is used as the primary target.

Run manually (not scheduled) - this is a one-time (or occasional) backfill,
not a recurring job.
"""

import os
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import hopsworks

from config import CITIES, BACKFILL_YEARS, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

FLOAT_COLS = ["pm2_5", "pm10", "no2", "o3", "co", "temp", "wind_speed"]
INT_COLS = ["hour", "day", "month", "day_of_week", "humidity", "pressure"]


def fetch_weather_history(lat, lon, start_date, end_date):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "timezone": "UTC",
    }
    r = requests.get(WEATHER_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_air_quality_history(lat, lon, start_date, end_date):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide",
        "timezone": "UTC",
    }
    r = requests.get(AIR_QUALITY_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def build_city_dataframe(city, lat, lon, start_date, end_date):
    weather_json = fetch_weather_history(lat, lon, start_date, end_date)
    air_json = fetch_air_quality_history(lat, lon, start_date, end_date)

    weather_df = pd.DataFrame({
        "timestamp": pd.to_datetime(weather_json["hourly"]["time"]),
        "temp": weather_json["hourly"]["temperature_2m"],
        "humidity": weather_json["hourly"]["relative_humidity_2m"],
        "pressure": weather_json["hourly"]["surface_pressure"],
        "wind_speed": weather_json["hourly"]["wind_speed_10m"],
    })

    air_df = pd.DataFrame({
        "timestamp": pd.to_datetime(air_json["hourly"]["time"]),
        "pm2_5": air_json["hourly"]["pm2_5"],
        "pm10": air_json["hourly"]["pm10"],
        "no2": air_json["hourly"]["nitrogen_dioxide"],
        "o3": air_json["hourly"]["ozone"],
        "co": air_json["hourly"]["carbon_monoxide"],
    })

    df = pd.merge(weather_df, air_df, on="timestamp", how="inner")
    df["city"] = city

    # Drop rows where core pollutant data is missing (Open-Meteo occasionally
    # has gaps at the very start/end of its coverage window)
    df = df.dropna(subset=["pm2_5", "temp"]).reset_index(drop=True)

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["aqi"] = pd.NA  # see module docstring - scale mismatch, left null on purpose

    return df


def enforce_dtypes(df):
    for col in FLOAT_COLS:
        df[col] = df[col].astype("float64")
    for col in INT_COLS:
        df[col] = df[col].astype("int64")
    df["city"] = df["city"].astype("str")
    # aqi stays nullable (pandas "Int64" nullable int, since it's all-NA here)
    df["aqi"] = df["aqi"].astype("Int64")
    return df


def get_feature_group():
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        description="Hourly AQI + weather features for major Pakistani cities",
    )
    return fg


def insert_in_chunks(fg, df, chunk_size=5000):
    total = len(df)
    for start in range(0, total, chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        for attempt in range(1, 4):
            try:
                fg.insert(chunk)
                print(f"  inserted rows {start}-{start+len(chunk)} / {total}")
                break
            except Exception as e:
                print(f"  chunk insert attempt {attempt} failed: {e}")
                if attempt < 3:
                    time.sleep(5 * attempt)
                else:
                    raise


def main():
    end_date = date.today() - timedelta(days=1)  # yesterday, since today's data may be incomplete
    start_date = end_date - timedelta(days=365 * BACKFILL_YEARS)

    fg = get_feature_group()

    for city, coords in CITIES.items():
        print(f"\nBackfilling {city} from {start_date} to {end_date}...")
        try:
            df = build_city_dataframe(city, coords["lat"], coords["lon"], str(start_date), str(end_date))
            df = enforce_dtypes(df)
            print(f"  {len(df)} rows fetched")
            insert_in_chunks(fg, df)
        except Exception as e:
            print(f"  FAILED for {city}: {e}")
            continue

    print("\nBackfill complete.")


if __name__ == "__main__":
    main()
