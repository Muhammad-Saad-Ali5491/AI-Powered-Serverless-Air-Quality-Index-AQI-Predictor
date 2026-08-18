"""
Live feature pipeline - multi-city version.

Runs hourly (via GitHub Actions). For every city in config.CITIES:
  1. Fetches current pollution + weather from OpenWeather
  2. Builds a feature row (with explicit, schema-matching dtypes)
  3. Inserts it into the shared Hopsworks feature group, tagged by city

Schema note: 'city' is part of the primary key alongside 'timestamp', so
the same feature group holds all cities' data, distinguished by that column.
Numeric dtypes are forced explicitly because OpenWeather sometimes returns
whole numbers as JSON ints (e.g. temp: 25 instead of 25.4), which otherwise
causes intermittent Hopsworks schema-mismatch errors.
"""

import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import hopsworks

from config import CITIES, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

API_KEY = os.environ["OPENWEATHER_API_KEY"]

# Columns that are genuinely continuous -> always cast to float64
FLOAT_COLS = ["pm2_5", "pm10", "no2", "o3", "co", "temp", "wind_speed"]
# Columns that OpenWeather always returns as whole numbers -> always cast to int64
INT_COLS = ["aqi", "hour", "day", "month", "day_of_week", "humidity", "pressure"]


def fetch_raw(lat, lon):
    pollution = requests.get(
        f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}",
        timeout=15,
    ).json()
    weather = requests.get(
        f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric",
        timeout=15,
    ).json()
    return pollution, weather


def build_features(city, pollution, weather):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return {
        "city": city,
        "timestamp": now,
        "hour": int(now.hour),
        "day": int(now.day),
        "month": int(now.month),
        "day_of_week": int(now.weekday()),
        "aqi": int(pollution["list"][0]["main"]["aqi"]),
        "pm2_5": float(pollution["list"][0]["components"]["pm2_5"]),
        "pm10": float(pollution["list"][0]["components"]["pm10"]),
        "no2": float(pollution["list"][0]["components"]["no2"]),
        "o3": float(pollution["list"][0]["components"]["o3"]),
        "co": float(pollution["list"][0]["components"]["co"]),
        "temp": float(weather["main"]["temp"]),
        "humidity": int(weather["main"]["humidity"]),
        "pressure": int(weather["main"]["pressure"]),
        "wind_speed": float(weather["wind"]["speed"]),
    }


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


def enforce_dtypes(df):
    for col in FLOAT_COLS:
        df[col] = df[col].astype("float64")
    for col in INT_COLS:
        df[col] = df[col].astype("int64")
    df["city"] = df["city"].astype("str")
    return df


def insert_with_retry(fg, df, max_retries=3):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            fg.insert(df)
            return True
        except Exception as e:
            last_error = e
            print(f"  insert attempt {attempt} failed: {e}")
            if attempt < max_retries:
                wait = 5 * attempt
                print(f"  retrying in {wait}s...")
                time.sleep(wait)
    print(f"  giving up after {max_retries} attempts: {last_error}")
    return False


def main():
    fg = get_feature_group()

    rows = []
    for city, coords in CITIES.items():
        try:
            pollution, weather = fetch_raw(coords["lat"], coords["lon"])
            row = build_features(city, pollution, weather)
            rows.append(row)
            print(f"Fetched {city}: AQI={row['aqi']} PM2.5={row['pm2_5']}")
        except Exception as e:
            # One city's API hiccup shouldn't take down the other 6
            print(f"Failed to fetch {city}: {e}")

    if not rows:
        raise RuntimeError("No cities were successfully fetched this run - aborting insert.")

    df = pd.DataFrame(rows)
    df = enforce_dtypes(df)

    success = insert_with_retry(fg, df)
    if not success:
        raise RuntimeError("Failed to insert batch into Hopsworks after retries.")

    print(f"Inserted {len(df)} rows ({len(rows)} cities) for {df['timestamp'].iloc[0]}")


if __name__ == "__main__":
    main()
