# Setup Instructions for AI-Powered AQI Predictor

## ⚠️ Prerequisites (MUST COMPLETE FIRST)

### 1. Get API Keys

#### OpenWeather API Key
1. Go to https://openweathermap.org/api
2. Sign up for a free account
3. Navigate to "API keys" section
4. Copy your default API key (starts with alphanumeric characters)

#### Hopsworks API Key
1. Go to https://www.hopsworks.ai/ 
2. Create a FREE account (no credit card needed)
3. Create a new project or use existing one
4. Go to Settings → API keys
5. Create a new API key and copy it

### 2. Add Secrets to GitHub

1. Go to your repository: `Settings` → `Secrets and variables` → `Actions`
2. Click "New repository secret" for each:

| Secret Name | Value |
|---|---|
| `OPENWEATHER_API_KEY` | Your OpenWeather API key |
| `HOPSWORKS_API_KEY` | Your Hopsworks API key |

⚠️ **Don't include quotes or extra spaces!**

## 🚀 Running the Project

### Step 1: Run Backfill Workflow (ONE TIME)
This populates your feature store with historical data:

1. Go to **Actions** tab
2. Find **"Backfill historical data (Open-Meteo)"** workflow
3. Click **"Run workflow"** → Select branch → **"Run workflow"**
4. ⏳ Wait 5-10 minutes for completion
5. Check logs for success (should download 4 years of data)

### Step 2: Run Feature Pipeline (ONE TIME)
Fetches today's data and inserts into feature store:

1. Go to **Actions** tab
2. Find **"Fetch AQI features (multi-city)"** workflow
3. Click **"Run workflow"** → **"Run workflow"**
4. ⏳ Wait 1-2 minutes
5. Check logs - should show: "Fetched Lahore: AQI=... PM2.5=..."

### Step 3: Run Training Pipeline (ONE TIME)
Trains initial models:

1. Go to **Actions** tab
2. Find **"Train AQI models"** workflow
3. Click **"Run workflow"** → **"Run workflow"**
4. ⏳ Wait 3-5 minutes
5. Check logs - should show models trained for 24h/48h/72h horizons

### Step 4: Verify Workflows Are Scheduled
After successful runs, workflows will run automatically:
- **Feature fetch**: Every hour (see `.github/workflows/feature_pipeline.yml`)
- **Model training**: Daily at midnight (see `.github/workflows/training_pipeline.yml`)

## 🔧 Troubleshooting

### "Could not find any project"
- ✅ Verify `HOPSWORKS_API_KEY` is correct (no quotes)
- ✅ Make sure Hopsworks project exists
- ✅ Try regenerating the API key

### "No files found that match the pattern"
- ✅ Run **Backfill** workflow first
- ✅ Then run **Feature Pipeline**
- ✅ Then run **Training Pipeline**
- ⚠️ Order matters!

### "Invalid API key for OpenWeather"
- ✅ Verify `OPENWEATHER_API_KEY` is correct
- ✅ Key should be ~32 characters
- ✅ Check OpenWeather dashboard

### Jobs still failing?
1. Click on the failed workflow run
2. Expand "Fetch" or "Train" job
3. Look for red error messages
4. Common causes:
   - Secrets not set correctly
   - Hopsworks project doesn't exist
   - APIs rate-limited (wait 1 hour)

## 📊 Local Testing (Optional)

Test locally before relying on scheduled runs:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENWEATHER_API_KEY="your_key_here"
export HOPSWORKS_API_KEY="your_key_here"

# Test feature pipeline
python feature_pipeline.py

# Test backfill (optional)
python backfill_openmeteo.py

# Test training
python training_pipeline.py
```

## 📚 Project Structure

| File | Purpose |
|---|---|
| `config.py` | City coordinates + Hopsworks settings |
| `feature_pipeline.py` | Hourly: Fetch live AQI/weather data |
| `backfill_openmeteo.py` | One-time: Load 4 years historical data |
| `feature_engineering.py` | Build time-series features (lags, rolling stats) |
| `training_pipeline.py` | Daily: Train Ridge/RandomForest/XGBoost models |
| `dashboard.py` | Streamlit app: View forecasts & SHAP explanations |
| `.github/workflows/` | GitHub Actions CI/CD |

## ✅ Success Indicators

Once running successfully, you should see:
- ✅ Feature store populated with hourly data (city, timestamp, PM2.5, weather)
- ✅ Models trained for 24h, 48h, 72h prediction horizons
- ✅ Best model promoted to Hopsworks Model Registry
- ✅ Dashboard accessible with predictions

