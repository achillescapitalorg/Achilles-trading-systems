"""
Beta Testing Module — Gold 1-Minute Data Pipeline
==================================================
Downloads, validates, and prepares Kaggle + Dukascopy datasets
for upcoming model training.  No models here yet — data only.
"""

from .config import DATA_DIR, KAGGLE_DATASET, DUKASCOPY_DIR
from .data_loader import BetaDataLoader
from .data_validator import validate_gold_data
from .kaggle_downloader import KaggleGoldDownloader
from .dukascopy_downloader import DukascopyGoldDownloader

__all__ = [
    "DATA_DIR",
    "KAGGLE_DATASET",
    "DUKASCOPY_DIR",
    "BetaDataLoader",
    "validate_gold_data",
    "KaggleGoldDownloader",
    "DukascopyGoldDownloader",
]
