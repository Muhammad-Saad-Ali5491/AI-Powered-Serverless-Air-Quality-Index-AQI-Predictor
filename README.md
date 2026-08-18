# Pakistan AQI Predictor

Multi-city (Lahore, Karachi, Islamabad, Faisalabad, Rawalpindi, Multan, Peshawar)
PM2.5 / AQI forecasting pipeline. Serverless: GitHub Actions + Hopsworks.

## Files

- `config.py` - city list + coordinates, feature group settings
- `feature_pipeline.py` - hourly live fetch (OpenWeather) for all cities, writes to Hopsworks
- `backfill_openmeteo.py` - one-time historical backfill (Open-Meteo, ~4 years, no API key needed)
- `feature_engineering.py` - city-aware, gap-aware lags/rolling stats/change rate/cyclical encoding
- `training_pipeline.py` - trains Ridge/RandomForest/XGBoost per 24h/48h/72h horizon, evaluates, promotes best to Model Registry
- `dashboard.py` - Streamlit app: city selector, forecast, SHAP explanation, hazard alerts
- `.github/workflows/` - hourly fetch, daily training, manual backfill

## Setup

1. `pip install -r requirements.txt` (inside a venv - see note below)
2. Set env vars: `OPENWEATHER_API_KEY`, `HOPSWORKS_API_KEY`
3. Add the same as GitHub repo secrets (Settings -> Secrets and variables -> Actions)
4. Run the backfill workflow manually first (Actions tab -> "Backfill historical data (Open-Meteo)" -> Run workflow)
5. Let the hourly + daily workflows take over

## Important design notes

- **Target variable is PM2.5 (ug/m3), not OpenWeather's 1-5 `aqi` field.**
  PM2.5 is on a consistent scale across both the live feed and the Open-Meteo
  historical backfill; `aqi` is not comparable between the two sources.
  See docstrings in `backfill_openmeteo.py` and `training_pipeline.py`.
- **All time-series features are computed per city** - never across the
  whole dataframe at once - to avoid one city's history leaking into another's
  lag/rolling features.
- **Model promotion is conditional**: a newly trained model only replaces the
  production one in the registry if its RMSE is actually better.

## Local Windows setup
Hopsworks' client has some dependencies that don't build cleanly on native
Windows. Recommended: create a dedicated venv and test there first; if it
fails, WSL2 is the reliable fallback (see project chat history for full
step-by-step). Either way, GitHub Actions runs everything on Linux
regardless of your local setup, so the pipeline works even if local testing
is inconvenient.
