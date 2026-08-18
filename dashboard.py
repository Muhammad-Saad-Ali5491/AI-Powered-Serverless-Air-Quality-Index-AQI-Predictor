"""
AQI Predictor dashboard - multi-city version.

Loads the latest features + registered models from Hopsworks, lets the
user pick a city, and shows:
  - Current PM2.5 / AQI reading
  - 24h / 48h / 72h PM2.5 forecast
  - A hazard alert banner if forecasted levels are unhealthy
  - SHAP feature importance for the selected forecast

Run locally with: streamlit run dashboard.py
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import hopsworks

from config import CITIES, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
from feature_engineering import engineer_features

HORIZONS = {"24h": 24, "48h": 48, "72h": 72}

# PM2.5 (ug/m3) thresholds for the hazard banner - US EPA breakpoints
PM25_THRESHOLDS = [
    (0, 12, "Good", "#00e400"),
    (12, 35.4, "Moderate", "#ffff00"),
    (35.4, 55.4, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (55.4, 150.4, "Unhealthy", "#ff0000"),
    (150.4, 250.4, "Very Unhealthy", "#8f3f97"),
    (250.4, 99999, "Hazardous", "#7e0023"),
]


def classify_pm25(value):
    for low, high, label, color in PM25_THRESHOLDS:
        if low <= value < high:
            return label, color
    return "Unknown", "#888888"


@st.cache_resource
def get_project():
    return hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])


@st.cache_data(ttl=1800)  # refresh every 30 minutes
def load_latest_data():
    project = get_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df = engineer_features(df)
    return df


@st.cache_resource
def load_model(horizon):
    project = get_project()
    mr = project.get_model_registry()
    try:
        hw_model = mr.get_model(f"aqi_pm25_{horizon}", version=None)
        model_dir = hw_model.download()
        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        with open(os.path.join(model_dir, "feature_columns.json")) as f:
            feature_cols = json.load(f)
        return model, feature_cols
    except Exception as e:
        return None, None


def main():
    st.set_page_config(page_title="Pakistan AQI Predictor", layout="wide")
    st.title("Pakistan AQI Predictor")
    st.caption("3-day PM2.5 / AQI forecast for major Pakistani cities")

    city = st.selectbox("Select city", list(CITIES.keys()))

    with st.spinner("Loading latest data..."):
        df = load_latest_data()

    city_df = df[df["city"] == city].sort_values("timestamp")
    if city_df.empty:
        st.warning(f"No data yet for {city}.")
        return

    latest = city_df.iloc[-1]

    # --- current readings ---
    col1, col2, col3 = st.columns(3)
    label, color = classify_pm25(latest["pm2_5"])
    col1.metric("Current PM2.5 (ug/m3)", f"{latest['pm2_5']:.1f}")
    col2.metric("Category", label)
    col3.metric("Last updated", latest["timestamp"].strftime("%Y-%m-%d %H:%M UTC"))

    st.markdown(
        f"<div style='background-color:{color};padding:10px;border-radius:6px;color:black;'>"
        f"<b>Current air quality: {label}</b></div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # --- forecasts ---
    st.subheader("Forecast")
    forecast_cols = st.columns(len(HORIZONS))
    forecasts = {}
    worst_label = "Good"
    worst_rank = 0
    rank_order = [t[2] for t in PM25_THRESHOLDS]

    for i, (label_h, hours) in enumerate(HORIZONS.items()):
        model, feature_cols = load_model(label_h)
        if model is None:
            forecast_cols[i].info(f"{label_h}: model not trained yet")
            continue

        X_latest = city_df[feature_cols].iloc[[-1]].fillna(0)
        pred = model.predict(X_latest)[0]
        cat_label, cat_color = classify_pm25(pred)
        forecasts[label_h] = (pred, cat_label, model, feature_cols, X_latest)

        rank = rank_order.index(cat_label) if cat_label in rank_order else 0
        if rank > worst_rank:
            worst_rank = rank
            worst_label = cat_label

        forecast_cols[i].metric(f"+{label_h} PM2.5", f"{pred:.1f}", cat_label)

    # --- hazard alert banner ---
    if worst_rank >= 3:  # Unhealthy or worse
        st.error(f"⚠️ Hazard alert: forecasted air quality reaches **{worst_label}** "
                 f"within the next 72 hours in {city}. Consider limiting outdoor exposure.")
    elif worst_rank >= 2:
        st.warning(f"Forecasted air quality reaches **{worst_label}** within 72 hours in {city}.")

    st.divider()

    # --- trend chart ---
    st.subheader("Recent PM2.5 trend")
    st.line_chart(city_df.set_index("timestamp")["pm2_5"].tail(24 * 14))

    st.divider()

    # --- SHAP feature importance ---
    st.subheader("Why this forecast? (feature importance)")
    horizon_choice = st.selectbox("Explain forecast for horizon:", list(HORIZONS.keys()))
    if horizon_choice in forecasts:
        pred, cat_label, model, feature_cols, X_latest = forecasts[horizon_choice]
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_latest)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_latest, plot_type="bar", show=False)
            st.pyplot(fig)
        except Exception:
            st.info("SHAP explanation only available for tree-based models "
                     "(Random Forest / XGBoost), not Ridge.")
    else:
        st.info("No model available yet for this horizon.")


if __name__ == "__main__":
    main()
