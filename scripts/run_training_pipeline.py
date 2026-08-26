"""
Daily training pipeline entry point (invoked by GitHub Actions on a daily
cron schedule, or manually / locally).

Run:  python scripts/run_training_pipeline.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train_model import run_training
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    try:
        run_record = run_training(epochs=30)
        logger.info("Champion candidate this run: %s", run_record["model_type"])
    except ValueError as exc:
        logger.error("Training skipped: %s", exc)
        sys.exit(0)  # don't fail CI hard if there simply isn't enough data yet
