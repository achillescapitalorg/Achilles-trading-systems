"""
Kaggle Gold Data Downloader
============================
Downloads the XAU/USD 1-minute dataset from Kaggle.
Falls back to instructions if kaggle API credentials are missing.
"""
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import KAGGLE_DATASET, KAGGLE_RAW_DIR, KAGGLE_1M_CSV


class KaggleGoldDownloader:
    """Handles download and extraction of the Kaggle gold dataset."""

    def __init__(self, dataset: str = KAGGLE_DATASET, output_dir: Path = KAGGLE_RAW_DIR):
        self.dataset = dataset
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.zip_path = self.output_dir / "kaggle_gold.zip"

    def is_available(self) -> bool:
        """Check if the 1-minute CSV already exists locally."""
        return KAGGLE_1M_CSV.exists() and KAGGLE_1M_CSV.stat().st_size > 0

    def download(self, method: str = "auto") -> Path:
        """
        Download dataset.

        method:
            'auto'   — try kaggle API first, then print manual instructions
            'api'    — force kaggle API (requires ~/.kaggle/kaggle.json)
            'manual' — only print manual download link
        """
        if self.is_available():
            print(f"[Kaggle] Already downloaded: {KAGGLE_1M_CSV}")
            return KAGGLE_1M_CSV

        if method in ("auto", "api"):
            try:
                self._download_via_api()
                self._extract()
                return KAGGLE_1M_CSV
            except Exception as exc:
                print(f"[Kaggle] API download failed: {exc}")
                if method == "api":
                    raise

        print("[Kaggle] Manual download required.")
        print(f"  1. Create free account at https://www.kaggle.com")
        print(f"  2. Visit: https://www.kaggle.com/datasets/{self.dataset}")
        print(f"  3. Click Download and place ZIP in: {self.output_dir}")
        print(f"  4. Extract so that {KAGGLE_1M_CSV} exists.")
        return self.output_dir

    def _download_via_api(self) -> None:
        """Use kaggle CLI to download dataset."""
        creds = Path.home() / ".kaggle" / "kaggle.json"
        if not creds.exists():
            raise FileNotFoundError(f"Kaggle API credentials not found at {creds}")

        cmd = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            self.dataset,
            "-p",
            str(self.output_dir),
            "--force",
        ]
        print(f"[Kaggle] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        # Rename the generic zip to our known name
        generic_zip = self.output_dir / f"{self.dataset.replace('/', '_')}.zip"
        if generic_zip.exists():
            generic_zip.rename(self.zip_path)

    def _extract(self) -> None:
        """Extract ZIP archive."""
        if not self.zip_path.exists():
            # Try to find any zip in the directory
            zips = list(self.output_dir.glob("*.zip"))
            if not zips:
                raise FileNotFoundError(f"No ZIP found in {self.output_dir}")
            self.zip_path = zips[0]

        print(f"[Kaggle] Extracting {self.zip_path} ...")
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            zf.extractall(self.output_dir)
        print("[Kaggle] Extraction complete.")

    def load(self) -> Optional[pd.DataFrame]:
        """Load the 1-minute CSV into a DataFrame."""
        if not self.is_available():
            print(f"[Kaggle] Data not found at {KAGGLE_1M_CSV}")
            return None

        print(f"[Kaggle] Loading {KAGGLE_1M_CSV} ...")
        # Kaggle dataset sometimes uses semicolon delimiter
        df = pd.read_csv(KAGGLE_1M_CSV, sep=None, engine="python")
        # Standardise columns
        df.columns = [c.lower().strip() for c in df.columns]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
        print(f"[Kaggle] Loaded {len(df):,} rows x {len(df.columns)} cols")
        print(f"[Kaggle] Range: {df.index.min()} to {df.index.max()}")
        return df

    def get_status(self) -> dict:
        """Return quick status dict for UI display."""
        return {
            "source": "Kaggle",
            "dataset": self.dataset,
            "available": self.is_available(),
            "path": str(KAGGLE_1M_CSV),
            "size_mb": round(KAGGLE_1M_CSV.stat().st_size / (1024 * 1024), 2)
            if self.is_available()
            else 0,
        }
