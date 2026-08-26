"""
Cross-platform path utilities.

All paths in this project are built with pathlib.Path so the exact same
code works on Windows, macOS and Linux (no hard-coded '/' separators,
no reliance on symlinks, no POSIX-only permissions).
"""
from pathlib import Path

# Project root = two levels up from this file (src/utils/paths.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_STORE_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

for _dir in (DATA_DIR, RAW_DIR, PROCESSED_DIR, FEATURE_STORE_DIR, MODELS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    """Strip characters that are illegal in Windows filenames."""
    illegal = '<>:"/\\|?*'
    for ch in illegal:
        name = name.replace(ch, "_")
    return name.strip().replace(" ", "_")
