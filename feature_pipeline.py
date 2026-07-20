import requests, os, time
import pandas as pd
from datetime import datetime
import hopsworks

API_KEY = os.environ["OPENWEATHER_API_KEY"]
LAT = os.environ.get("LATITUDE").strip()
LON = os.environ.get("LONGITUDE").strip()


def fetch_raw():
    pollution = requests.get(
        f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
    ).json()
    weather = requests.get(
        f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
    ).json()
    return pollution, weather


def build_features(pollution, weather):
    now = datetime.utcnow()
    return {
        "timestamp": now,
        "hour": int(now.hour),
        "day": int(now.day),
        "month": int(now.month),
        "day_of_week": int(now.weekday()),
        # Explicit float() casts below - OpenWeather sometimes returns whole
        # numbers as JSON ints (e.g. temp: 25 instead of 25.4), which pandas
        # then infers as int64 instead of float64, breaking Hopsworks' fixed
        # 'double' schema on those rows. Forcing float() makes every row
        # consistent regardless of what the API happened to return.
        "aqi": int(pollution["list"][0]["main"]["aqi"]),
        "pm2_5": float(pollution["list"][0]["components"]["pm2_5"]),
        "pm10": float(pollution["list"][0]["components"]["pm10"]),
        "no2": float(pollution["list"][0]["components"]["no2"]),
        "o3": float(pollution["list"][0]["components"]["o3"]),
        "co": float(pollution["list"][0]["components"]["co"]),
        "temp": float(weather["main"]["temp"]),
        "humidity": float(weather["main"]["humidity"]),
        "pressure": float(weather["main"]["pressure"]),
        "wind_speed": float(weather["wind"]["speed"]),
    }


def push_to_feature_store(row: dict, max_retries: int = 3):
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="Hourly AQI features"
    )

    df = pd.DataFrame([row])

    # Belt-and-braces: force the dtypes explicitly too, in case pandas
    # still infers something unexpected from a single-row dataframe.
    float_cols = ["pm2_5", "pm10", "no2", "o3", "co", "temp", "humidity", "pressure", "wind_speed"]
    for col in float_cols:
        df[col] = df[col].astype("float64")

    int_cols = ["aqi", "hour", "day", "month", "day_of_week"]
    for col in int_cols:
        df[col] = df[col].astype("int64")

    # Retry logic for transient connection drops (RemoteDisconnected, etc.)
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            fg.insert(df)
            print(f"Inserted row for {row['timestamp']} (attempt {attempt})")
            return
        except Exception as e:
            last_error = e
            print(f"Insert attempt {attempt} failed: {e}")
            if attempt < max_retries:
                wait = 5 * attempt  # simple backoff: 5s, 10s, 15s...
                print(f"Retrying in {wait}s...")
                time.sleep(wait)

    # If we exhausted all retries, raise the last error so the Action
    # still shows as failed (better than silently losing an hour of data)
    raise last_error


if __name__ == "__main__":
    pollution, weather = fetch_raw()
    row = build_features(pollution, weather)
    push_to_feature_store(row)