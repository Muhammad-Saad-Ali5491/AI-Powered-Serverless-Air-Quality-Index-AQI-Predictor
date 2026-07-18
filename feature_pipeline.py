import requests, os
import pandas as pd
from datetime import datetime
import hopsworks

API_KEY = os.environ["OPENWEATHER_API_KEY"]
LAT = os.environ.get("LATITUDE", "31.5497").strip()
LON = os.environ.get("LONGITUDE", "74.3436").strip()

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
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),
        "aqi": pollution["list"][0]["main"]["aqi"],
        "pm2_5": pollution["list"][0]["components"]["pm2_5"],
        "pm10": pollution["list"][0]["components"]["pm10"],
        "no2": pollution["list"][0]["components"]["no2"],
        "o3": pollution["list"][0]["components"]["o3"],
        "co": pollution["list"][0]["components"]["co"],
        "temp": weather["main"]["temp"],
        "humidity": weather["main"]["humidity"],
        "pressure": weather["main"]["pressure"],
        "wind_speed": weather["wind"]["speed"],
    }

def push_to_feature_store(row: dict):
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
    fg.insert(df)
    print(f"Inserted row for {row['timestamp']}")

if __name__ == "__main__":
    pollution, weather = fetch_raw()
    row = build_features(pollution, weather)
    push_to_feature_store(row)