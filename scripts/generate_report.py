"""Generate the project submission report as a PDF.

Run from the repository root:
    python scripts/generate_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import MODELS_DIR


REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORT_PATH = REPORT_DIR / "Pearls_AQI_Project_Report.pdf"


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("&", "&amp;"), style)


def _metrics_table(registry: dict, styles: dict) -> Table:
    champion = registry.get("champion", {})
    metrics = champion.get("metrics", {}).get(champion.get("model_type"), {}).get("overall", {})
    rows = [
        ["Champion model", champion.get("model_type", "Unavailable")],
        ["RMSE", f"{metrics.get('rmse', 0):.3f}"],
        ["MAE", f"{metrics.get('mae', 0):.3f}"],
        ["R2", f"{metrics.get('r2', 0):.3f}"],
        ["Forecast horizons", ", ".join(f"{h}h" for h in champion.get("horizons_hours", [24, 48, 72]))],
        ["Training rows", str(champion.get("n_train_rows", "Unavailable"))],
        ["Test rows", str(champion.get("n_test_rows", "Unavailable"))],
    ]
    table = Table(rows, colWidths=[1.8 * inch, 4.7 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f3f1")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#172a3a")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b5c8c5")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def _table(rows: list[list[str]], widths: list[float], header: bool = True) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163f49")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c6d8d4")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#172a3a")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]
    for row in range(1, len(rows)):
        if row % 2 == 0:
            style.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#f1f7f5")))
    if not header:
        style = style[2:]
    table.setStyle(TableStyle(style))
    return table


def _model_comparison_table(registry: dict) -> Table:
    champion = registry.get("champion", {})
    metrics_by_model = champion.get("metrics", {})
    rows = [["Model", "RMSE", "MAE", "R2"]]
    for model_name, metrics in metrics_by_model.items():
        overall = metrics.get("overall", {})
        rows.append([
            model_name.replace("_", " ").title(),
            f"{overall.get('rmse', 0):.3f}",
            f"{overall.get('mae', 0):.3f}",
            f"{overall.get('r2', 0):.3f}",
        ])
    return _table(rows, [2.7 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch])


def _horizon_table(registry: dict) -> Table:
    champion = registry.get("champion", {})
    metrics = champion.get("metrics", {}).get(champion.get("model_type"), {}).get("per_horizon", {})
    rows = [["Horizon", "RMSE", "MAE", "R2"]]
    for horizon in ("24h", "48h", "72h"):
        values = metrics.get(horizon, {})
        rows.append([horizon, f"{values.get('rmse', 0):.3f}", f"{values.get('mae', 0):.3f}", f"{values.get('r2', 0):.3f}"])
    return _table(rows, [2.7 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch])


def _footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#c6d8d4"))
    canvas.line(document.leftMargin, 0.48 * inch, A4[0] - document.rightMargin, 0.48 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#66808a"))
    canvas.drawString(document.leftMargin, 0.3 * inch, "Pearls AQI Predictor | Project report")
    canvas.drawRightString(A4[0] - document.rightMargin, 0.3 * inch, f"Page {document.page}")
    canvas.restoreState()


def build_story(styles: dict) -> list:
    registry_path = MODELS_DIR / "registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    body = styles["BodyText"]
    heading = styles["Heading2"]
    story = [
        Spacer(1, 1.0 * inch),
        Paragraph("Pearls AQI Predictor", styles["Title"]),
        Paragraph("End-to-end serverless Air Quality Index prediction system", styles["Subtitle"]),
        Spacer(1, 0.35 * inch),
        Paragraph("Project Report", styles["CoverLabel"]),
        Spacer(1, 3.2 * inch),
        Paragraph("Prepared for the AQI prediction project submission", styles["CoverLabel"]),
        PageBreak(),
        Paragraph("1. Executive Summary", heading),
        _paragraph("Pearls AQI Predictor forecasts AQI 24, 48, and 72 hours ahead for eight major Pakistani cities. The system combines weather and pollutant observations, feature engineering, supervised model training, local Parquet storage with optional Hopsworks integration, GitHub Actions automation, and a Streamlit dashboard.", body),
        _paragraph("The deployment design is deliberately resilient: the dashboard can run from the checked-in feature cache and champion model without API keys, while scheduled workflows can enrich the cache with live data when secrets are configured.", body),
        Spacer(1, 0.15 * inch),
        Paragraph("2. System Architecture", heading),
        _paragraph("Raw weather and air-pollution data are fetched from OpenWeather for current observations. Historical pollutant observations are obtained from OpenAQ. The feature pipeline computes EPA-style AQI, calendar features, lag features, rolling statistics, and AQI change rates. Features are written to the local Parquet cache and, when explicitly enabled, to Hopsworks.", body),
        _paragraph("The training pipeline creates multi-output targets for 24h, 48h, and 72h horizons, compares Ridge Regression, Random Forest, XGBoost, and optionally TensorFlow, and records metrics in the model registry. Streamlit loads the champion model and latest city row to render forecasts, history, alerts, model comparisons, and optional SHAP explanations.", body),
        Spacer(1, 0.15 * inch),
        Paragraph("3. Feature Pipeline", heading),
        _paragraph("The engineered schema includes hour, day, month, day of week, weekend flag, day of year, temperature, humidity, pressure, wind speed and direction, cloud cover, PM2.5, PM10, NO2, SO2, CO, O3, AQI lags at 1h/24h/72h, 24-hour rolling mean and standard deviation, and AQI change rates at 1h/24h.", body),
        _paragraph("Feature imputation is performed within each city so one city's history cannot fill another city's lag or rolling values. This is important for a multi-city time-series dataset.", body),
        Spacer(1, 0.15 * inch),
        Paragraph("4. Historical Backfill", heading),
        _paragraph("The backfill workflow runs the feature-generation path over a configurable historical window and writes the resulting training rows to the feature store. The default project configuration supports up to four years, subject to OpenAQ station coverage and API limits.", body),
        PageBreak(),
        Paragraph("5. Training Pipeline and Evaluation", heading),
        _paragraph("The model-selection process uses a chronological train/test split to reduce leakage from future observations. Candidate models are evaluated using RMSE, MAE, and R2. The best candidate is stored as the champion and its feature schema is recorded with the artifact.", body),
        Spacer(1, 0.15 * inch),
        _metrics_table(registry, styles),
        Spacer(1, 0.18 * inch),
        Paragraph("Candidate comparison", styles["Heading3"]),
        _model_comparison_table(registry),
        Spacer(1, 0.18 * inch),
        Paragraph("Champion performance by horizon", styles["Heading3"]),
        _horizon_table(registry),
        Spacer(1, 0.2 * inch),
        Paragraph("6. Automation and Deployment", heading),
        _paragraph("GitHub Actions provides four operational workflows: CI tests on pushes and pull requests, a manual bootstrap for historical backfill and initial training, an hourly feature pipeline, and daily model training. The workflows default to the local cache unless the USE_HOPSWORKS repository variable is explicitly set to true, making the default path reliable without an external feature-store account.", body),
        _paragraph("Streamlit Community Cloud should be configured with app/streamlit_app.py as the main file and a supported modern Python runtime. The checked-in data/features/aqi_features.parquet, models/registry.json, and champion artifact allow the dashboard to open before the first live pipeline run. API keys belong in Streamlit Secrets or GitHub Actions Secrets and are never embedded in source code.", body),
        Paragraph("Operational workflow", styles["Heading3"]),
        _table([
            ["Workflow", "Cadence", "Responsibility"],
            ["CI test suite", "Push / pull request", "Compile code and run unit/integration tests"],
            ["Feature pipeline", "Every hour", "Fetch observations, engineer features, update cache"],
            ["Training pipeline", "Daily", "Compare candidates, promote champion, prune artifacts"],
            ["Bootstrap pipeline", "Manual", "Backfill history and create the first model"],
        ], [1.45 * inch, 1.35 * inch, 3.5 * inch]),
        PageBreak(),
        Paragraph("7. Feature Inventory", heading),
        _paragraph("The feature design combines short-term persistence, seasonal context, atmospheric conditions, and pollutant measurements. This gives the model both the current state of the city and signals that help it distinguish recurring daily patterns from sudden changes.", body),
        _table([
            ["Group", "Features", "Purpose"],
            ["Time", "hour, day, month, day_of_week, is_weekend, day_of_year", "Capture daily and seasonal cycles"],
            ["Weather", "temp, humidity, pressure, wind_speed, wind_deg, clouds", "Represent dispersion and atmospheric conditions"],
            ["Pollutants", "PM2.5, PM10, NO2, SO2, CO, O3", "Measure the current pollution mixture"],
            ["Dynamics", "AQI lags, rolling mean/std, change rates", "Capture persistence, volatility, and direction"],
        ], [1.0 * inch, 3.0 * inch, 2.3 * inch]),
        Spacer(1, 0.2 * inch),
        Paragraph("8. Dashboard and Explainability", heading),
        _paragraph("The dashboard is organized around five user questions: What is the current AQI? What will happen over the next three days? What patterns are visible in the historical data? Why did the model make this forecast? Which candidate model won? The forecast view uses a gauge, KPI cards, colored AQI bands, and a compact timeline. The EDA view adds an AQI distribution and an hour-by-weekday heatmap. The explainability view provides global feature importance and signed current-row contributions using SHAP. The Model Lab compares candidate metrics and shows the champion's horizon-level errors.", body),
        _paragraph("SHAP is loaded only when the explainability tab is opened, its calculations are cached, and the chart input is bounded. This keeps the main dashboard responsive while preserving model transparency for users who need it.", body),
        Spacer(1, 0.15 * inch),
        Paragraph("9. Data Quality and Reliability", heading),
        _paragraph("The local Parquet cache is the deployment safety net. Feature writes are de-duplicated by city and timestamp, and Hopsworks is optional. Derived imputation is performed within each city to prevent cross-city leakage. Training targets are joined by timestamp offsets so missing observations do not silently turn a 24-row shift into a false 24-hour horizon.", body),
        Spacer(1, 0.15 * inch),
        Paragraph("10. Limitations and Future Work", heading),
        _paragraph("Forecast quality depends on station coverage, API freshness, and the availability of complete hourly observations. Timestamp-based target joins correctly omit unavailable future observations, but longer-term production work should add explicit hourly resampling and missingness indicators. Future work should add calibrated uncertainty intervals, richer station-level spatial features, drift monitoring, and a dedicated alert delivery channel.", body),
        Paragraph("11. Reproduction Checklist", heading),
        _paragraph("1. Configure OPENWEATHER_API_KEY and optionally OPENAQ_API_KEY and HOPSWORKS_API_KEY. 2. Run the bootstrap workflow once. 3. Confirm the CI workflow passes. 4. Deploy app/streamlit_app.py on Streamlit Community Cloud using Python 3.11. 5. Run the hourly and daily workflows manually once to verify repository write permissions.", body),
    ]
    return story


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontSize=15, leading=20, alignment=TA_CENTER, textColor=colors.HexColor("#367c78")))
    styles.add(ParagraphStyle(name="CoverLabel", parent=styles["Normal"], fontSize=11, leading=15, alignment=TA_CENTER, textColor=colors.HexColor("#52616b")))
    styles["Title"].fontSize = 30
    styles["Title"].leading = 36
    styles["Title"].alignment = TA_CENTER
    styles["Heading2"].textColor = colors.HexColor("#1f5f5b")
    styles["Heading2"].spaceBefore = 8
    styles["Heading2"].spaceAfter = 10
    styles["Heading3"].textColor = colors.HexColor("#367c78")
    styles["Heading3"].spaceBefore = 6
    styles["Heading3"].spaceAfter = 7
    styles["BodyText"].fontSize = 10.5
    styles["BodyText"].leading = 16
    styles["BodyText"].spaceAfter = 9

    document = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=A4,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Pearls AQI Predictor Project Report",
        author="Pearls AQI Predictor",
    )
    document.build(build_story(styles), onFirstPage=_footer, onLaterPages=_footer)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()