"""
Beta Testing Unified Data Loader
=================================
High-level interface that loads Kaggle, Dukascopy, or merges both.
Also provides train/validation/test splits and feature engineering hooks.
"""
from typing import Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

from .config import UNIFIED_1M_CSV, KAGGLE_1M_CSV, DUKASCOPY_1M_CSV
from .kaggle_downloader import KaggleGoldDownloader
from .dukascopy_downloader import DukascopyGoldDownloader
from .data_validator import validate_gold_data


class BetaDataLoader:
    """
    Loads and prepares gold 1-minute data for upcoming model training.
    """

    def __init__(self):
        self.kaggle = KaggleGoldDownloader()
        self.dukascopy = DukascopyGoldDownloader()
        self._kaggle_df: Optional[pd.DataFrame] = None
        self._dukascopy_df: Optional[pd.DataFrame] = None
        self._unified_df: Optional[pd.DataFrame] = None

    # ── Loaders ──────────────────────────────────────────────────────────────

    def load_kaggle(self, force_reload: bool = False) -> Optional[pd.DataFrame]:
        if self._kaggle_df is not None and not force_reload:
            return self._kaggle_df
        self._kaggle_df = self.kaggle.load()
        return self._kaggle_df

    def load_dukascopy(self, force_reload: bool = False) -> Optional[pd.DataFrame]:
        if self._dukascopy_df is not None and not force_reload:
            return self._dukascopy_df
        self._dukascopy_df = self.dukascopy.load()
        return self._dukascopy_df

    def load_unified(
        self,
        prefer: str = "kaggle",
        force_rebuild: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Merge both sources into a single cleaned 1-minute DataFrame.
        prefer='kaggle' uses Kaggle as base and fills gaps with Dukascopy.
        """
        if self._unified_df is not None and not force_rebuild:
            return self._unified_df

        kdf = self.load_kaggle()
        ddf = self.load_dukascopy()

        if kdf is None and ddf is None:
            print("[BetaDataLoader] No data available from either source.")
            return None

        if kdf is not None and ddf is None:
            unified = kdf.copy()
        elif ddf is not None and kdf is None:
            unified = ddf.copy()
        else:
            # Both available — merge with preference
            if prefer == "kaggle":
                base = kdf.copy()
                filler = ddf.copy()
            else:
                base = ddf.copy()
                filler = kdf.copy()

            # Align columns
            cols = ["open", "high", "low", "close", "volume"]
            base = base[[c for c in cols if c in base.columns]]
            filler = filler[[c for c in cols if c in filler.columns]]

            # Reindex filler to base's full range, forward-fill small gaps
            filler = filler.reindex(base.index, method="ffill", limit=5)
            unified = base.combine_first(filler)

        # Standardise
        unified = unified[[c for c in ["open", "high", "low", "close", "volume"] if c in unified.columns]]
        unified = unified.dropna()
        unified = unified.sort_index()

        self._unified_df = unified
        unified.to_csv(UNIFIED_1M_CSV)
        print(f"[BetaDataLoader] Unified dataset saved: {UNIFIED_1M_CSV} ({len(unified):,} rows)")
        return unified

    # ── Validation ───────────────────────────────────────────────────────────

    def validate(self, df: Optional[pd.DataFrame] = None) -> dict:
        if df is None:
            df = self.load_unified()
        if df is None:
            return {"error": "No data loaded."}
        return validate_gold_data(df)

    # ── Splits ───────────────────────────────────────────────────────────────

    @staticmethod
    def time_series_split(
        df: pd.DataFrame,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Chronological split (no shuffle).
        Returns train, validation, test DataFrames.
        """
        n = len(df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:].copy()

        print(
            f"[BetaDataLoader] Split: train={len(train):,} | val={len(val):,} | test={len(test):,}"
        )
        return train, val, test

    # ── Quick stats ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        out = {
            "kaggle": self.kaggle.get_status(),
            "dukascopy": self.dukascopy.get_status(),
        }
        if self._unified_df is not None:
            out["unified_rows"] = len(self._unified_df)
            out["unified_range"] = f"{self._unified_df.index.min()} to {self._unified_df.index.max()}"
        return out
