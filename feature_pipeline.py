import requests, os, json
from datetime import datetime, timezone

API_KEY = os.environ["OPENWEATHER_API_KEY"]
LAT = os.environ.get("LATITUDE", "31.5497")
LON = os.environ.get("LONGITUDE", "74.3436")

def fetch():
    pollution = requests.get(
        f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
    ).json()
    print("POLLUTION RESPONSE:", pollution)  # ADD THIS LINE TEMPORARILY

    weather = requests.get(
        f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
    ).json()

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),   # <- confirm this line exists
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
    print(json.dumps(row))
    # TODO (tomorrow): push `row` to Hopsworks instead of printing

if __name__ == "__main__":
    fetch()
    
 