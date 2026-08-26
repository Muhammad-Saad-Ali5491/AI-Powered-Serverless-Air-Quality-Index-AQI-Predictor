"""
Generate a synthetic (but realistic-shaped) raw dataset so the entire
pipeline — feature engineering, feature store, training, inference, SHAP —
can be exercised and tested end-to-end WITHOUT live OpenWeather/OpenAQ API
keys. This is what the test suite and CI "smoke test" job run against.

Not used in production; production data comes from real API calls in
scripts/run_feature_pipeline.py and src/data/backfill_historical.py.

Run:  python scripts/generate_synthetic_data.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from src import config
from src.features.feature_engineering import build_feature_table
from src.features.feature_store import get_feature_store, LocalFeatureStore
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

RNG = np.random.default_rng(42)


def generate_raw_history(hours: int = 24 * 120, cities=None) -> pd.DataFrame:
    """~120 days of hourly synthetic weather/pollution readings per city,
    with a diurnal + seasonal-ish pattern so lag/rolling features are
    meaningful and models have real signal to learn from."""
    cities = cities or config.CITIES
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    timestamps = [end - timedelta(hours=h) for h in range(hours)][::-1]

    rows = []
    for city in cities:
        base_pm25 = RNG.uniform(40, 120)  # Pakistani cities skew toward higher baseline AQI
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            diurnal = 20 * np.sin((hour - 6) / 24 * 2 * np.pi) + 20  # worse in morning/evening traffic
            noise = RNG.normal(0, 8)
            pm25 = max(5, base_pm25 + diurnal + noise + 0.01 * i * RNG.normal(0, 1))
            rows.append(
                {
                    "city": city.name,
                    "timestamp": ts.isoformat(),
                    "temp": 25 + 10 * np.sin((ts.timetuple().tm_yday / 365) * 2 * np.pi) + RNG.normal(0, 2),
                    "humidity": float(np.clip(50 + RNG.normal(0, 15), 5, 100)),
                    "pressure": float(1010 + RNG.normal(0, 5)),
                    "wind_speed": float(max(0, RNG.normal(3, 1.5))),
                    "wind_deg": float(RNG.uniform(0, 360)),
                    "clouds": float(np.clip(RNG.normal(40, 25), 0, 100)),
                    "pm25": pm25,
                    "pm10": pm25 * RNG.uniform(1.3, 1.8),
                    "no2": max(1, RNG.normal(25, 8)),
                    "so2": max(1, RNG.normal(10, 4)),
                    "co": max(100, RNG.normal(800, 200)),
                    "o3": max(1, RNG.normal(30, 10)),
                }
            )
    return pd.DataFrame(rows)


def main(hours: int = 24 * 120) -> pd.DataFrame:
    raw_df = generate_raw_history(hours=hours)
    feature_df = build_feature_table(raw_df)

    store = get_feature_store()
    if isinstance(store, LocalFeatureStore) and store.path.exists():
        store.path.unlink()  # clean slate for reproducible demo/testing runs

    store.write_features(feature_df)
    logger.info("Synthetic dataset generated: %d rows across %d cities.", len(feature_df), raw_df["city"].nunique())
    return feature_df


if __name__ == "__main__":
    main()
