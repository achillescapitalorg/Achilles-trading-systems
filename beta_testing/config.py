"""
Beta Testing Configuration
==========================
Paths and constants for the gold data pipeline.
"""
import os
from pathlib import Path

# Base data directory (project-root/data/beta_testing/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data" / "beta_testing"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Kaggle
KAGGLE_DATASET = "novandraanugrah/xauusd-gold-price-historical-data-2004-2024"
KAGGLE_RAW_DIR = DATA_DIR / "kaggle_raw"
KAGGLE_RAW_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE_1M_CSV = KAGGLE_RAW_DIR / "XAU_1m_data.csv"

# Dukascopy
DUKASCOPY_DIR = DATA_DIR / "dukascopy"
DUKASCOPY_DIR.mkdir(parents=True, exist_ok=True)
DUKASCOPY_TICK_DIR = DUKASCOPY_DIR / "ticks"
DUKASCOPY_TICK_DIR.mkdir(parents=True, exist_ok=True)
DUKASCOPY_1M_CSV = DUKASCOPY_DIR / "XAUUSD_1m_aggregated.csv"

# Processed / unified
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
UNIFIED_1M_CSV = PROCESSED_DIR / "gold_1m_unified.csv"

# Validation reports
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Time constants
EXPECTED_BAR_MINUTES = 1
MAX_GAP_MINUTES = 2
WEEKEND_GAP_HOURS = 72

# Gold price sanity bounds
GOLD_PRICE_MIN = 200.0
GOLD_PRICE_MAX = 10000.0
