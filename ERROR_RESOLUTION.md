# AI-Powered AQI Predictor - Complete Error Resolution Guide

## ❌ Identified Issues & Root Causes

Based on workflow run analysis, here are the **specific errors** and their fixes:

---

## Issue 1: "Could not find any project"

**Error Log:**
```
hopsworks_common.client.exceptions.ProjectException: Could not find any project
```

**Root Cause:**
- `HOPSWORKS_API_KEY` secret is either:
  - Not set in GitHub repository secrets
  - Set with incorrect format (extra spaces, quotes, or wrong key)
  - Regenerated but old key still in use

**Fix:**
1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Delete the old `HOPSWORKS_API_KEY` if it exists
3. Create a NEW secret:
   - **Name:** `HOPSWORKS_API_KEY`
   - **Value:** Go to https://www.hopsworks.ai → your project → Settings → API keys → copy the full key (no quotes, no spaces)
4. Save and verify the secret appears in the list

**Verification:**
```bash
# Test locally first
export HOPSWORKS_API_KEY="your-actual-key-here"
python feature_pipeline.py
```

---

## Issue 2: "IO Error: No files found that match the pattern"

**Error Log:**
```
pyarrow._flight.FlightServerError: IO Error: No files found that match the pattern 
"hdfs:///apps/hive/warehouse/aqi_index_featurestore.db/aqi_features_1/..."
```

**Root Cause:**
- Feature group `aqi_features` (version 1) exists but is **empty** or **has no data yet**
- Training pipeline runs before backfill and feature pipeline complete
- Expected order: Backfill → Feature Fetch → Training

**Fix - CRITICAL ORDER:**
1. **FIRST:** Run `Backfill historical data (Open-Meteo)` workflow manually
   - Go to **Actions** → **Backfill historical data** → **Run workflow**
   - Wait for completion (5-15 minutes)
   - Check logs: should see "Backfilling Lahore... X rows fetched"

2. **SECOND:** Run `Fetch AQI features (multi-city)` workflow manually
   - Go to **Actions** → **Fetch AQI features** → **Run workflow**
   - Should see: "Fetched Lahore: AQI=X PM2.5=Y"

3. **THIRD:** Run `Train AQI models` workflow manually
   - Go to **Actions** → **Train AQI models** → **Run workflow**
   - Should see model training metrics

4. **THEN:** Let hourly/daily scheduled workflows run

---

## Issue 3: "Could not read data using Hopsworks Query Service"

**Error Log:**
```
hopsworks_common.client.exceptions.FeatureStoreException: Could not read data 
using Hopsworks Query Service.
```

**Root Cause:**
- Network timeout or Hopsworks server briefly unavailable
- Feature group schema mismatch between insert and read

**Fix:**
1. Verify feature pipeline has data:
   ```bash
   # In Hopsworks UI: 
   # Project → Feature Store → aqi_features → Data
   # Should show recent rows
   ```

2. If empty, run feature pipeline again:
   - **Actions** → **Fetch AQI features** → **Run workflow** → Wait

3. If issue persists, retry training pipeline:
   - **Actions** → **Train AQI models** → **Run workflow**

---

## Issue 4: Schema Mismatches in Feature Group

**Error Log:**
```
Schema mismatch: expected type X got type Y
```

**Root Cause:**
- OpenWeather API sometimes returns integers as floats or vice versa
- Multiple inserts with inconsistent dtypes

**Fix Applied:**
✅ Already in code:
```python
# feature_pipeline.py line 81-87
def enforce_dtypes(df):
    for col in FLOAT_COLS:
        df[col] = df[col].astype("float64")
    for col in INT_COLS:
        df[col] = df[col].astype("int64")
    df["city"] = df["city"].astype("str")
    return df
```

**If error persists:**
1. Delete the old feature group from Hopsworks UI
2. Rerun backfill (creates fresh feature group)

---

## Issue 5: "Could not find any project" in Training

**When:** Training pipeline runs before any data is collected
**Why:** Training tries to read an empty feature store

**Fix:**
✅ Code handles this in `training_pipeline.py` line 272-275:
```python
if not results:
    print("No horizon had enough data to train on yet. "
          "This is expected early on - accumulate more history and re-run.")
    return
```

This is expected. Just run again after 24+ hours of data collection.

---

## Complete Setup & Execution Checklist

### ✅ Step 1: Create Hopsworks Project
- [ ] Visit https://www.hopsworks.ai
- [ ] Sign up (free account, no credit card)
- [ ] Create a new project
- [ ] Go to Settings → API keys → Create key
- [ ] Copy and save the key

### ✅ Step 2: Create OpenWeather API Key
- [ ] Visit https://openweathermap.org/api
- [ ] Sign up for free account
- [ ] Create API key
- [ ] Copy and save the key

### ✅ Step 3: Add GitHub Secrets
- [ ] Go to your repo → Settings → Secrets and variables → Actions
- [ ] Create `HOPSWORKS_API_KEY` with your Hopsworks key
- [ ] Create `OPENWEATHER_API_KEY` with your OpenWeather key
- [ ] Verify both appear in the secrets list

### ✅ Step 4: Run Workflows IN ORDER

**4a. Backfill Historical Data (One-time)**
```
Actions → Backfill historical data (Open-Meteo) → Run workflow
⏳ Wait 10-15 minutes
✅ Check: Logs show "Backfilling Lahore... X rows" for each city
```

**4b. Fetch Live AQI Data (One-time test)**
```
Actions → Fetch AQI features (multi-city) → Run workflow
⏳ Wait 1-2 minutes
✅ Check: Logs show "Fetched Lahore: AQI=X PM2.5=Y"
```

**4c. Train Initial Models (One-time)**
```
Actions → Train AQI models → Run workflow
⏳ Wait 3-5 minutes
✅ Check: Logs show "[24h] random_forest: RMSE=..."
```

**4d. Verify Automatic Scheduling**
```
After success, workflows should run automatically:
- Feature fetch: Every hour (0 * * * *)
- Model training: Daily at 2 AM UTC (0 2 * * *)
```

---

## Troubleshooting Matrix

| Error | First Check | Next Step | Last Resort |
|---|---|---|---|
| "Could not find any project" | Is `HOPSWORKS_API_KEY` set? | Regenerate key | Delete & recreate project |
| "No files found" | Has backfill run? | Run backfill first | Delete feature group |
| "Could not read data" | Is there data in feature store? | Run feature pipeline | Regenerate keys |
| Models won't train | Is test data sufficient? | Wait 24h for data | Check feature columns |
| Scheduled jobs skip | Check cron syntax | Check workflow triggers | Re-enable workflows |

---

## Local Testing (Before Relying on GitHub Actions)

### Test Feature Pipeline Locally
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

pip install -r requirements.txt

export OPENWEATHER_API_KEY="your-key"
export HOPSWORKS_API_KEY="your-key"

python feature_pipeline.py
```

Expected output:
```
Fetched Lahore: AQI=2 PM2.5=23.5
Fetched Karachi: AQI=3 PM2.5=18.2
...
Inserted 7 rows (7 cities)
```

### Test Backfill Locally
```bash
export HOPSWORKS_API_KEY="your-key"
python backfill_openmeteo.py
```

Expected output:
```
Backfilling Lahore from 2022-08-18 to 2026-08-18...
  35040 rows fetched
  inserted rows 0-5000 / 35040
  ...
Backfill complete.
```

### Test Training Locally (after backfill)
```bash
export HOPSWORKS_API_KEY="your-key"
python training_pipeline.py
```

Expected output:
```
Fetching data from Hopsworks...
  35280 raw rows across 7 cities
Engineering features...
Building forecast targets...
  560 feature columns
[24h] random_forest: RMSE=12.34 MAE=8.91 R2=0.756
[24h] promoted random_forest (RMSE=12.34, first model registered)
Training pipeline complete.
```

---

## GitHub Actions Workflow Files

### 🔄 `.github/workflows/feature_pipeline.yml`
**Runs:** Every hour at minute 0
**Purpose:** Fetch latest OpenWeather data for all cities
**On Failure:** Logs show API key or network issue

### 🔄 `.github/workflows/training_pipeline.yml`
**Runs:** Daily at 2 AM UTC
**Purpose:** Retrain models, compare against production, promote if better
**On Failure:** Likely not enough data yet (expected early on)

### 🔄 `.github/workflows/backfill_openmeteo.yml`
**Runs:** Manual only (workflow_dispatch)
**Purpose:** One-time historical data load
**Expected Duration:** 10-15 minutes

---

## Success Indicators ✅

You'll know everything is working when:

1. **Backfill Complete**
   - Feature group shows ~168,000 rows (7 cities × 4 years × ~8760 hours)
   - Each city has full hourly coverage

2. **Hourly Fetches Working**
   - 7 new rows appear every hour (one per city)
   - GitHub Actions log shows "Inserted 7 rows"

3. **Models Trained**
   - Model Registry shows 3 models: `aqi_pm25_24h`, `aqi_pm25_48h`, `aqi_pm25_72h`
   - Metrics log CSV has entries

4. **Dashboard Ready** (optional)
   ```bash
   export HOPSWORKS_API_KEY="your-key"
   streamlit run dashboard.py
   # Open http://localhost:8501
   ```

---

## Next Steps

1. ✅ Follow the **Setup Checklist** above
2. ✅ Run workflows **IN ORDER** (backfill → fetch → train)
3. ✅ Verify **GitHub Actions** logs after each step
4. ✅ Check **Hopsworks** UI to see data accumulating
5. ✅ Wait 24+ hours for automatic workflows
6. ✅ (Optional) Launch dashboard with Streamlit

For additional help, see **SETUP_INSTRUCTIONS.md** for step-by-step guidance.
