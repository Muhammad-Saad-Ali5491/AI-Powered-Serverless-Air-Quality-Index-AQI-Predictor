"""
Pearls AQI Predictor — Streamlit Dashboard

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.features.feature_store import get_feature_store
from src.inference.predict import forecast_city, ModelNotTrainedError
from src.explainability.shap_explain import explain_model
from src.utils.aqi_calc import aqi_category

st.set_page_config(page_title="Pearls AQI Predictor — Pakistan", page_icon="🌫️", layout="wide")

AQI_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
    "Unknown": "#9e9e9e",
}


@st.cache_data(ttl=300)
def _load_history(city: str) -> pd.DataFrame:
    store = get_feature_store()
    df = store.read_features(city=city)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")
    return df


def render_header():
    st.title("🌫️ Pearls AQI Predictor")
    st.caption("3-day Air Quality Index forecasts for major Pakistani cities — 100% serverless ML pipeline")


def render_sidebar() -> str:
    st.sidebar.header("Settings")
    city = st.sidebar.selectbox("City", config.CITY_NAMES, index=0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Stack:** OpenWeather + OpenAQ • scikit-learn / TensorFlow • "
        "GitHub Actions • Hopsworks Feature Store • SHAP"
    )
    st.sidebar.markdown(f"Hazardous alert threshold: **AQI ≥ {config.HAZARDOUS_AQI_THRESHOLD}**")
    return city


def render_current_and_forecast(city: str):
    try:
        result = forecast_city(city)
    except ModelNotTrainedError:
        st.warning(
            "⚠️ No trained model yet. Run the backfill + training pipeline first "
            "(see README → Quickstart), or wait for the daily GitHub Actions training run."
        )
        return
    except ValueError as exc:
        st.warning(f"⚠️ {exc}")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        cat = result["current_category"]
        st.metric("Current AQI", result["current_aqi"] if result["current_aqi"] is not None else "N/A")
        st.markdown(
            f"<span style='background-color:{AQI_COLORS.get(cat, '#999')}; "
            f"padding:4px 10px; border-radius:6px; color:black; font-weight:600;'>{cat}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"As of {result['as_of']}")
        st.caption(f"Model: {result['model_type']} ({result['model_run_id']})")

    with col2:
        forecast_df = pd.DataFrame(result["forecast"])
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[f"+{h}h" for h in forecast_df["horizon_hours"]],
                y=forecast_df["predicted_aqi"],
                marker_color=[AQI_COLORS.get(c, "#999") for c in forecast_df["category"]],
                text=forecast_df["predicted_aqi"],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=f"3-Day AQI Forecast — {city}",
            yaxis_title="Predicted AQI",
            height=350,
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    hazardous = [f for f in result["forecast"] if f["is_hazardous"]]
    if hazardous:
        for f in hazardous:
            st.error(
                f"🚨 **Hazardous AQI Alert** — {city} is forecast to reach AQI "
                f"{f['predicted_aqi']} ({f['category']}) at +{f['horizon_hours']}h "
                f"({f['target_time']})."
            )


def render_history(city: str):
    df = _load_history(city)
    if df.empty:
        st.info("No historical feature data yet for this city.")
        return
    st.subheader("Historical AQI trend")
    fig = px.line(df, x="timestamp", y="aqi", title=f"AQI history — {city}")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw pollutant concentrations"):
        pollutant_cols = [c for c in config.POLLUTANTS if c in df.columns]
        st.plotly_chart(
            px.line(df, x="timestamp", y=pollutant_cols, title="Pollutant concentrations (µg/m³)"),
            use_container_width=True,
        )


def render_shap(city: str):
    st.subheader("🔍 Why this forecast? (SHAP feature importance)")
    try:
        explanation = explain_model(city=city)
    except Exception as exc:
        st.info(f"SHAP explanation unavailable yet: {exc}")
        return

    imp_df = pd.DataFrame(explanation["feature_importance"]).head(12)
    fig = px.bar(
        imp_df.sort_values("mean_abs_shap"),
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title=f"Top features driving the {explanation['horizon']} forecast ({explanation['model_type']})",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_all_cities_overview():
    st.subheader("🇵🇰 All cities overview")
    rows = []
    for c in config.CITY_NAMES:
        try:
            r = forecast_city(c)
            row = {"City": c, "Current AQI": r["current_aqi"], "Category": r["current_category"]}
            for f in r["forecast"]:
                row[f"+{f['horizon_hours']}h"] = f["predicted_aqi"]
            rows.append(row)
        except Exception:
            rows.append({"City": c, "Current AQI": "N/A", "Category": "N/A"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main():
    render_header()
    city = render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📊 History & EDA", "🔍 Explainability", "🇵🇰 All Cities"])
    with tab1:
        render_current_and_forecast(city)
    with tab2:
        render_history(city)
    with tab3:
        render_shap(city)
    with tab4:
        render_all_cities_overview()


if __name__ == "__main__":
    main()
