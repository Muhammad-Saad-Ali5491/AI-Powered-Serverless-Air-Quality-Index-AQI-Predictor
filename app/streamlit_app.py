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

AQI_RANGES = [
    (0, 50, "Good", "Low risk"),
    (51, 100, "Moderate", "Manageable"),
    (101, 150, "Sensitive groups", "Take care"),
    (151, 200, "Unhealthy", "Limit exposure"),
    (201, 300, "Very unhealthy", "Avoid outdoors"),
    (301, 500, "Hazardous", "Stay indoors"),
]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #132b35; --muted: #66808a; --teal: #0d7773; --orange: #f27a38; --paper: #f5f8f6; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(245,248,246,.86); }
    [data-testid="stSidebar"] { background: #102f38; }
    [data-testid="stSidebar"] * { color: #eef8f5 !important; }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div { background: #194752; border-color: #34717a; }
    h1, h2, h3, [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    p, label, [data-testid="stCaptionContainer"] { font-family: 'DM Sans', sans-serif; }
    h1 { color: var(--ink); font-size: 2.6rem !important; margin-bottom: .1rem; }
    h2 { color: var(--ink); margin-top: 1.2rem; }
    .eyebrow { color: var(--orange); font: 700 .75rem 'DM Sans', sans-serif; letter-spacing: .12em; text-transform: uppercase; }
    .hero { background: linear-gradient(120deg, #123b45 0%, #0d7773 72%, #47a28b 100%); border-radius: 20px; padding: 1.55rem 1.8rem; color: white; margin: .4rem 0 1.1rem; box-shadow: 0 14px 35px rgba(24,66,70,.16); }
    .hero h2 { color: white; margin: .2rem 0 .35rem; font-size: 1.65rem; }
    .hero p { color: #d9f1eb; margin: 0; }
    .mini-card { background: white; border: 1px solid #dce9e5; border-radius: 14px; padding: 1rem 1.1rem; min-height: 84px; box-shadow: 0 5px 18px rgba(19,43,53,.04); }
    .mini-label { color: var(--muted); font: 600 .72rem 'DM Sans', sans-serif; text-transform: uppercase; letter-spacing: .08em; }
    .mini-value { color: var(--ink); font: 700 1.55rem 'Space Grotesk', sans-serif; margin-top: .2rem; }
    .mini-note { color: var(--muted); font: .76rem 'DM Sans', sans-serif; }
    .section-kicker { color: var(--teal); font: 700 .74rem 'DM Sans', sans-serif; text-transform: uppercase; letter-spacing: .1em; margin: .8rem 0 .25rem; }
    .status-pill { display: inline-block; border-radius: 999px; padding: .36rem .72rem; color: #132b35; font: 700 .78rem 'DM Sans', sans-serif; }
    .alert-strip { background: #fff2e9; border-left: 5px solid #e85f38; border-radius: 10px; padding: .75rem 1rem; color: #713022; margin: .6rem 0 1rem; }
    div[data-testid="stTabs"] button { font-family: 'DM Sans', sans-serif; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300)
def _load_history(city: str) -> pd.DataFrame:
    store = get_feature_store()
    df = store.read_features(city=city)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")
    return df


@st.cache_data(ttl=300, show_spinner=False)
def _load_forecast(city: str) -> dict:
    return forecast_city(city)


def _chart_history(df: pd.DataFrame, max_points: int = 1500) -> pd.DataFrame:
    """Keep Plotly responsive when the repository contains years of hourly data."""
    if len(df) <= max_points:
        return df
    numeric_cols = list(df.select_dtypes(include="number").columns)
    compact = (
        df.set_index("timestamp")[numeric_cols]
        .resample("6h")
        .mean()
        .dropna(subset=["aqi"])
        .reset_index()
    )
    return compact


@st.cache_data(ttl=600, show_spinner=False)
def _load_explanation(city: str) -> dict:
    from src.explainability.shap_explain import explain_model

    return explain_model(city=city)


def render_header():
    st.markdown('<div class="eyebrow">Pearls intelligence / Pakistan</div>', unsafe_allow_html=True)
    st.title("Air quality, with a little foresight.")
    st.caption("A three-day AQI outlook built from live pollutant signals, weather context, and a trained forecasting model.")


def render_sidebar() -> str:
    st.sidebar.markdown("## PEARLS AQI")
    st.sidebar.caption("Forecast console · v1.0")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Location")
    city = st.sidebar.selectbox("City", config.CITY_NAMES, index=0)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Signal guide")
    st.sidebar.markdown("Live observations\n\nForecast horizon\n\nModel explanations")
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Alert threshold: AQI ≥ {config.HAZARDOUS_AQI_THRESHOLD}")
    return city


def render_current_and_forecast(city: str):
    try:
        result = _load_forecast(city)
    except ModelNotTrainedError:
        st.warning(
            "⚠️ No trained model yet. Run the backfill + training pipeline first "
            "(see README → Quickstart), or wait for the daily GitHub Actions training run."
        )
        return
    except ValueError as exc:
        st.warning(f"⚠️ {exc}")
        return

    current_aqi = result["current_aqi"]
    current_category = result["current_category"]
    forecast_df = pd.DataFrame(result["forecast"])
    forecast_df["label"] = forecast_df["horizon_hours"].map({24: "Tomorrow", 48: "In 2 days", 72: "In 3 days"})
    peak = int(forecast_df["predicted_aqi"].max())
    peak_category = forecast_df.loc[forecast_df["predicted_aqi"].idxmax(), "category"]
    color = AQI_COLORS.get(current_category, AQI_COLORS["Unknown"])
    st.markdown(
        f'<div class="hero"><div class="eyebrow" style="color:#ffcf9d">Selected station</div>'
        f'<h2>{city} / atmospheric outlook</h2><p>Latest signal: {result["as_of"]} · Champion: '
        f'{result["model_type"].replace("_", " ").title()}</p></div>',
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_data = [
        ("Current AQI", "N/A" if current_aqi is None else str(current_aqi), current_category),
        ("72-hour peak", str(peak), peak_category),
        ("Forecast points", str(len(forecast_df)), "24h · 48h · 72h"),
        ("Model run", result["model_run_id"].split("_")[-1].replace("Z", " UTC"), result["model_type"].replace("_", " ").title()),
    ]
    for column, (label, value, note) in zip(metric_cols, metric_data):
        with column:
            st.markdown(
                f'<div class="mini-card"><div class="mini-label">{label}</div>'
                f'<div class="mini-value">{value}</div><div class="mini-note">{note}</div></div>',
                unsafe_allow_html=True,
            )

    chart_col, gauge_col = st.columns([2.4, 1])
    with chart_col:
        st.markdown('<div class="section-kicker">The next 72 hours</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df["label"], y=forecast_df["predicted_aqi"], mode="lines+markers+text",
            text=forecast_df["predicted_aqi"], textposition="top center",
            line=dict(color="#0d7773", width=4), marker=dict(size=13, color=[AQI_COLORS.get(c, "#9e9e9e") for c in forecast_df["category"]], line=dict(color="white", width=2)),
            hovertemplate="%{x}<br>AQI <b>%{y}</b><extra></extra>",
        ))
        for low, high, _, _ in AQI_RANGES:
            fig.add_hrect(y0=low, y1=high, fillcolor=AQI_COLORS.get(aqi_category(low), "#ddd"), opacity=.08, line_width=0)
        fig.update_layout(height=330, margin=dict(l=8, r=8, t=18, b=8), yaxis=dict(title="AQI", range=[0, max(180, peak + 40)], gridcolor="#e5efeb"), xaxis=dict(title=None), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with gauge_col:
        st.markdown('<div class="section-kicker">Current signal</div>', unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=current_aqi or 0,
            number={"font": {"size": 42, "color": "#132b35"}},
            gauge={"axis": {"range": [0, 500], "tickwidth": 0, "tickcolor": "#dce9e5"}, "bar": {"color": color}, "bgcolor": "#edf4f1", "borderwidth": 0, "steps": [{"range": [0, 50], "color": "#e9f6df"}, {"range": [50, 100], "color": "#fff9d9"}, {"range": [100, 200], "color": "#fff0df"}, {"range": [200, 500], "color": "#f8e2e3"}]},
        ))
        gauge.update_layout(height=250, margin=dict(l=12, r=12, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<span class="status-pill" style="background:{color}">{current_category}</span>', unsafe_allow_html=True)

    hazardous = [f for f in result["forecast"] if f["is_hazardous"]]
    if hazardous:
        for f in hazardous:
            st.markdown(f'<div class="alert-strip"><b>Air quality alert</b><br>{city} is forecast to reach AQI {f["predicted_aqi"]} ({f["category"]}) at +{f["horizon_hours"]}h.</div>', unsafe_allow_html=True)


def render_history(city: str):
    df = _load_history(city)
    if df.empty:
        st.info("No historical feature data yet for this city.")
        return
    st.markdown('<div class="section-kicker">Observed history</div>', unsafe_allow_html=True)
    history_cols = st.columns(3)
    for column, label, value in zip(
        history_cols,
        ["Latest observed", "30-day average", "Observed peak"],
        [int(df["aqi"].iloc[-1]), int(df["aqi"].tail(24 * 30).mean()), int(df["aqi"].max())],
    ):
        with column:
            st.markdown(f'<div class="mini-card"><div class="mini-label">{label}</div><div class="mini-value">{value}</div><div class="mini-note">AQI index</div></div>', unsafe_allow_html=True)

    chart_df = _chart_history(df)
    fig = px.area(chart_df, x="timestamp", y="aqi", title=None)
    fig.update_traces(line_color="#0d7773", fillcolor="rgba(13,119,115,.14)")
    fig.update_layout(height=330, margin=dict(l=8, r=8, t=18, b=8), yaxis_title="AQI", xaxis_title=None, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Explore pollutant signals"):
        pollutant_cols = [c for c in config.POLLUTANTS if c in df.columns]
        if pollutant_cols:
            pollutant_df = chart_df.melt(id_vars="timestamp", value_vars=[c for c in pollutant_cols if c in chart_df], var_name="Pollutant", value_name="Concentration")
            pollutant_fig = px.line(pollutant_df, x="timestamp", y="Concentration", color="Pollutant")
            pollutant_fig.update_layout(height=330, margin=dict(l=8, r=8, t=18, b=8), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(pollutant_fig, use_container_width=True, config={"displayModeBar": False})


def render_shap(city: str):
    st.markdown('<div class="section-kicker">Model transparency</div>', unsafe_allow_html=True)
    st.header("Why the model expects this outlook")
    st.caption("SHAP measures how each input moves the 24-hour forecast. Longer bars mean stronger influence; green pushes the estimate up and orange pushes it down.")
    try:
        with st.spinner("Calculating feature contributions..."):
            explanation = _load_explanation(city)
    except Exception as exc:
        st.warning("The explanation could not be calculated for this model yet.")
        st.caption(f"Diagnostic: {exc}")
        return

    imp_df = pd.DataFrame(explanation["feature_importance"]).head(12).sort_values("mean_abs_shap")
    contribution_df = pd.DataFrame(explanation["contributions"]).head(12).sort_values("shap_value")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Global importance")
        fig = px.bar(imp_df, x="mean_abs_shap", y="feature", orientation="h", color="mean_abs_shap", color_continuous_scale=["#d6eee6", "#0d7773"])
        fig.update_layout(height=440, margin=dict(l=8, r=8, t=10, b=8), xaxis_title="Mean |SHAP value|", yaxis_title=None, coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown("#### This city's latest row")
        fig = px.bar(contribution_df, x="shap_value", y="feature", orientation="h", color="shap_value", color_continuous_scale=["#ef9b62", "#f7f3e8", "#0d7773"], color_continuous_midpoint=0)
        fig.update_layout(height=440, margin=dict(l=8, r=8, t=10, b=8), xaxis_title="Contribution to 24h forecast", yaxis_title=None, coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", zeroline=True, zerolinecolor="#879c9d")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with st.expander("View explanation data"):
        st.dataframe(pd.DataFrame(explanation["feature_importance"]), use_container_width=True, hide_index=True)


def render_all_cities_overview():
    st.markdown('<div class="section-kicker">Pakistan at a glance</div>', unsafe_allow_html=True)
    st.header("City comparison")
    st.caption("Compare current AQI with the next 24-hour forecast across every supported city.")
    rows = []
    for c in config.CITY_NAMES:
        try:
            r = _load_forecast(c)
            row = {"City": c, "Current AQI": r["current_aqi"], "Category": r["current_category"]}
            for f in r["forecast"]:
                row[f"+{f['horizon_hours']}h"] = f["predicted_aqi"]
            rows.append(row)
        except Exception:
            rows.append({"City": c, "Current AQI": "N/A", "Category": "N/A"})
    overview_df = pd.DataFrame(rows)
    chart_df = overview_df[pd.to_numeric(overview_df["+24h"], errors="coerce").notna()].copy()
    if not chart_df.empty:
        chart_df["+24h"] = pd.to_numeric(chart_df["+24h"])
        chart_df = chart_df.sort_values("+24h")
        fig = px.bar(chart_df, x="+24h", y="City", orientation="h", color="Category", text="+24h", color_discrete_map=AQI_COLORS)
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(height=390, margin=dict(l=8, r=35, t=12, b=8), xaxis_title="Predicted AQI in 24 hours", yaxis_title=None, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", legend_title=None)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.dataframe(overview_df, use_container_width=True, hide_index=True)


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
