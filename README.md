# Pearls AQI Predictor 🌫️🇵🇰

Predict the Air Quality Index (AQI) for the next **3 days** across major Pakistani
cities, using a **100% serverless** ML stack with **Hopsworks** as the feature store.

- **Feature pipeline** — hourly, via GitHub Actions → OpenWeather live data → engineered features → written to **Hopsworks Feature Store** (default) with an automatic local Parquet fallback
- **Historical backfill** — up to 4 years of pollutant history from **OpenAQ**
- **Training pipeline** — daily, via GitHub Actions → Ridge / Random Forest / Extra Trees / Histogram Gradient Boosting / XGBoost / TensorFlow, best model auto-promoted
- **Dashboard** — Streamlit (interactive charts, model comparison, SHAP explainability, alerts) + a Flask REST API
- **Explainability** — SHAP feature importance for every forecast
- **Cities covered**: Lahore, Karachi, Islamabad, Rawalpindi, Faisalabad, Multan, Peshawar, Quetta

### Live Dashboard

Open the deployed Streamlit dashboard: [Pearls AQI Predictor](https://ai-powered-serverless-air-quality-index-aqi-predictor-5kdujskm.streamlit.app/)

Built to run identically on **Windows, macOS, and Linux** — all paths use `pathlib`,
no symlinks, no shell-only tooling required.

---

## 0. Run this entirely on GitHub + Streamlit Cloud (no local Python needed)

Everything below is done through the GitHub and Streamlit web UIs — you never
run a pipeline script on your own machine.

**Step 1 — Push this repo to GitHub** (or use GitHub's "Upload files" web UI
if you don't want to use git locally at all).

**Step 2 — Create a free Hopsworks account** at https://app.hopsworks.ai,
create a project, and generate an API key (**Account Settings → API Keys →
New API Key**). This is the default feature store.

**Step 3 — Add repo secrets** (Settings → Secrets and variables → Actions →
"New repository secret"):

| Secret | Get it from |
|---|---|
| `OPENWEATHER_API_KEY` | https://openweathermap.org/api (free) |
| `OPENAQ_API_KEY` | https://explore.openaq.org/register (free) |
| `HOPSWORKS_API_KEY` | https://app.hopsworks.ai → Account Settings → API Keys (free) |

**Step 4 — Bootstrap the historical data + first model.** Go to the
**Actions** tab → **"One-Time Bootstrap (Backfill + Initial Training)"** →
**Run workflow**. This fetches ~4 years of OpenAQ history for all 8 cities,
writes it to your Hopsworks feature store, trains the first model, and
commits the local cache + `models/` back to the repo. It can take a while
(large OpenAQ pull) — watch it finish with a green check.

**Step 5 — From here on, it's fully automatic:**
- `feature_pipeline.yml` runs **every hour**, pulling fresh OpenWeather data into Hopsworks
- `training_pipeline.yml` runs **daily**, retraining and auto-promoting the best model
- Both commit their local-cache results back to the repo too — nothing to run, ever

**Step 6 — Deploy the dashboard on Streamlit Community Cloud** (free):
1. Go to https://share.streamlit.io → sign in with GitHub
2. **New app** → pick this repo/branch → set **Main file path** to `app/streamlit_app.py`
3. Click **Deploy**

The dashboard doesn't strictly need any secrets to run — it reads from the
local Parquet cache that's always kept in sync and committed to the repo
(see "How the feature store works" below). If you'd like it to read
directly from Hopsworks instead, add `HOPSWORKS_API_KEY` (and optionally
`HOPSWORKS_PROJECT_NAME`) under your Streamlit app's **Settings → Secrets**.
Streamlit Community Cloud automatically redeploys the running app whenever
new commits land on the watched branch, so the live dashboard picks up each
hourly/daily update on its own.

That's the whole loop: GitHub Actions keeps Hopsworks (and the local cache)
current, Streamlit Cloud keeps the dashboard current, and neither requires
your computer to be on or Python to be installed anywhere locally.

> **If you want to run it locally too** (for development/debugging), the
> rest of this README covers that — but it's optional.

---

## 1. How the feature store works (Hopsworks-first, local-fallback)

`src/features/feature_store.py` exposes one function, `get_feature_store()`,
used everywhere else in the codebase:

- **Default (`USE_HOPSWORKS=true`):** every write goes to your Hopsworks
  Feature Store (a real managed feature group with online + offline
  storage, versioning, and a query API) **and** to a local Parquet cache
  at the same time. Bulk reads (training, backfill) always come from the
  fast local cache; call `store.read_from_hopsworks(city=...)` if you
  specifically need to read straight from Hopsworks instead.
- **Automatic fallback:** if `HOPSWORKS_API_KEY` isn't set, or Hopsworks is
  briefly unreachable, everything transparently falls back to the local
  Parquet file — the pipeline never hard-fails for lack of a Hopsworks
  account. You'll see an info/warning log line explaining which path was
  used.
- **Local-only mode:** set `USE_HOPSWORKS=false` to skip Hopsworks
  entirely and always use the local, git-versioned Parquet store.

---

## 2. Project layout

```
pearls-aqi-predictor/
├── .github/workflows/
│   ├── bootstrap_pipeline.yml # one-time, manual: backfill + first training run
│   ├── feature_pipeline.yml   # hourly: live OpenWeather data → Hopsworks + local cache
│   ├── training_pipeline.yml  # daily: retrain + auto-promote best model
│   └── ci_tests.yml           # every push/PR: pytest
├── .streamlit/
│   └── config.toml            # Streamlit Cloud / local theme + server config
├── app/
│   ├── streamlit_app.py       # interactive dashboard
│   └── flask_api.py           # REST API
├── src/
│   ├── config.py              # cities, API endpoints, feature schema
│   ├── data/                  # OpenWeather + OpenAQ fetchers, backfill
│   ├── features/              # feature engineering + feature store (Hopsworks / local Parquet)
│   ├── training/               # model training + evaluation
│   ├── inference/              # real-time prediction
│   ├── explainability/         # SHAP
│   └── utils/                  # AQI math, logging, cross-platform paths
├── scripts/                    # pipeline entry points + synthetic data generator
├── tests/                      # pytest unit + integration tests
├── data/features/               # local feature store cache (Parquet, versioned by CI)
├── models/                      # trained model artifact (champion only) + registry.json
└── requirements.txt              # Streamlit/ML/runtime dependencies
```

---

## 3. Quickstart (optional local development)

### Prerequisites
- Python 3.10 or 3.11
- Free API keys:
  - **OpenWeather**: https://openweathermap.org/api (Current Weather + Air Pollution API)
  - **OpenAQ**: https://explore.openaq.org/register (v3 API key)
  - **Hopsworks**: https://app.hopsworks.ai (optional — leave `HOPSWORKS_API_KEY`
    blank in `.env` to automatically use the local Parquet store instead)

### Setup

**Windows (PowerShell):**
```powershell
git clone <your-repo-url> pearls-aqi-predictor
cd pearls-aqi-predictor
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# edit .env and paste your API keys
```

**macOS / Linux:**
```bash
git clone <your-repo-url> pearls-aqi-predictor
cd pearls-aqi-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your API keys
```

> The Streamlit/CI requirements use current pandas and NumPy releases with
> prebuilt wheels for modern Python versions, including Python 3.14. Hopsworks
> is not installed in the dashboard environment; workflows use the local
> Parquet cache by default. Enable Hopsworks deliberately with
> `USE_HOPSWORKS=true` and install `requirements-hopsworks.txt` separately.

### Run the pipeline end-to-end

```bash
# 1. Backfill ~4 years of historical data from OpenAQ (one-time, can take a while)
python -m src.data.backfill_historical

# 2. (Optional but recommended first) Fetch one live snapshot to prime the feature store
python scripts/run_feature_pipeline.py

# 3. Train the models (Ridge, Random Forest, Extra Trees, Histogram Gradient Boosting, XGBoost, TensorFlow) and pick the champion
python scripts/run_training_pipeline.py

# 4. Launch the dashboard
streamlit run app/streamlit_app.py

# ...or the REST API
python app/flask_api.py
```

### Try it instantly without any API keys (synthetic demo data)

```bash
python scripts/generate_synthetic_data.py
python -m src.training.train_model
streamlit run app/streamlit_app.py
```

This populates the local feature store cache and trains a model on
realistic synthetic data so you can see the whole app working before wiring
up real API keys. (`USE_HOPSWORKS` defaults to `true`, but with no
`HOPSWORKS_API_KEY` set it automatically uses the local store — see
section 1.)

---

## 4. Running tests (debugging / CI)

```bash
pip install -r requirements.txt
pytest tests/ -v
```

The test suite includes:
- `test_aqi_calc.py` — EPA AQI breakpoint math
- `test_feature_engineering.py` — time features, lag/rolling features, AQI derivation
- `test_feature_store.py` — local Parquet feature store read/write/dedupe
- `test_pipeline_integration.py` — full smoke test: synthetic data → feature store → training → inference → registry → model-artifact pruning, proving the entire pipeline works together

All tests use synthetic in-memory/temp-file data and the local feature
store fallback — no live API calls and no real Hopsworks account needed —
so they run the same way locally and in GitHub Actions CI.

---

## 5. Automated pipelines (GitHub Actions)

| Workflow | Trigger | What it does |
|---|---|---|
| `.github/workflows/bootstrap_pipeline.yml` | manual (Actions tab → Run workflow), once | Backfills ~4 years of OpenAQ history into Hopsworks (+ local cache), fetches one live OpenWeather snapshot, trains the first model, commits everything back |
| `.github/workflows/feature_pipeline.yml` | every hour | Fetches live OpenWeather data for all 8 cities, engineers features, writes to Hopsworks + the local Parquet cache, commits the cache back to the repo |
| `.github/workflows/training_pipeline.yml` | daily @ 02:00 UTC | Retrains Ridge / Random Forest / Extra Trees / Histogram Gradient Boosting / XGBoost / TensorFlow on accumulated feature history, evaluates with RMSE/MAE/R², promotes the best model (pruning stale model files so the repo doesn't grow unbounded), commits `models/` back to the repo |
| `.github/workflows/ci_tests.yml` | every push/PR | Runs the full pytest suite on Python 3.10 & 3.11 |

### Required GitHub secrets

Set these under **Repo → Settings → Secrets and variables → Actions**:

| Name | Required | Purpose |
|---|---|---|
| `OPENWEATHER_API_KEY` | ✅ | live weather + pollution data (`feature_pipeline.yml`, `bootstrap_pipeline.yml`) |
| `OPENAQ_API_KEY` | ✅ | historical pollutant data (`bootstrap_pipeline.yml`) |
| `HOPSWORKS_API_KEY` | recommended | enables the default Hopsworks feature store; without it, workflows automatically fall back to the local Parquet store |

Optionally set repo **variables** (not secrets): `USE_HOPSWORKS` (defaults to
`false`), `HOPSWORKS_PROJECT_NAME`, `HOPSWORKS_HOST`. Set `USE_HOPSWORKS=true`
only after adding the Hopsworks API key and validating the feature-store
connection; otherwise the workflows use the committed local Parquet cache.

The workflows use `permissions: contents: write` and push commits with
`[skip ci]` in the message so the hourly/daily jobs don't trigger the test
workflow in a loop.

---

## 6. AQI methodology

Raw pollutant concentrations (PM2.5, PM10, NO₂, SO₂, CO, O₃) from OpenAQ /
OpenWeather are converted to the **US EPA Air Quality Index** using the
official breakpoint tables (`src/utils/aqi_calc.py`). The overall AQI for a
given hour/city is the **maximum** sub-index across all available
pollutants, per EPA methodology.

---

## 7. Alerts

The Flask API exposes `GET /alerts`, and the Streamlit dashboard surfaces a
red banner, whenever any city's 3-day forecast crosses
`HAZARDOUS_AQI_THRESHOLD` (default: AQI ≥ 200, "Very Unhealthy"+),
configurable in `src/config.py`.

---

## 8. Tech stack

Python • scikit-learn • TensorFlow • GitHub Actions • **Hopsworks Feature
Store** • Streamlit • Flask • OpenWeather API • OpenAQ API • SHAP • Git

## 9. Submission report

The generated project report is available at
`reports/Pearls_AQI_Project_Report.pdf`. Rebuild it after changing the model
registry or project description with:

```bash
python scripts/generate_report.py
```
