"""
Dukascopy Gold Tick Data Downloader
====================================
Downloads XAU/USD tick data from Dukascopy HTTP feed and
resamples to 1-minute OHLCV bars.
"""
import lzma
import struct
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

from .config import DUKASCOPY_DIR, DUKASCOPY_TICK_DIR, DUKASCOPY_1M_CSV


class DukascopyGoldDownloader:
    """
    Downloads tick-by-tick gold data from Dukascopy and
    aggregates to 1-minute OHLCV.
    """

    BASE_URL = "http://datafeed.dukascopy.com/datafeed/XAUUSD/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"

    def __init__(
        self,
        output_dir: Path = DUKASCOPY_DIR,
        tick_dir: Path = DUKASCOPY_TICK_DIR,
        n_workers: int = 4,
    ):
        self.output_dir = output_dir
        self.tick_dir = tick_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tick_dir.mkdir(parents=True, exist_ok=True)
        self.n_workers = n_workers

    def is_available(self) -> bool:
        """Check if aggregated 1-minute CSV already exists."""
        return DUKASCOPY_1M_CSV.exists() and DUKASCOPY_1M_CSV.stat().st_size > 0

    def _download_hour(
        self, year: int, month: int, day: int, hour: int
    ) -> Optional[pd.DataFrame]:
        """Download and parse one hour of tick data."""
        url = self.BASE_URL.format(year=year, month=month - 1, day=day, hour=hour)
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200 or len(resp.content) == 0:
                return None
            decompressed = lzma.decompress(resp.content)
        except Exception:
            return None

        ticks: List[dict] = []
        for i in range(0, len(decompressed), 20):
            chunk = decompressed[i : i + 20]
            if len(chunk) < 20:
                break
            ms, ask_points, bid_points, ask_vol, bid_vol = struct.unpack(
                ">IIIff", chunk
            )
            timestamp = datetime(year, month, day, hour) + timedelta(
                milliseconds=int(ms)
            )
            ask = ask_points / 1000.0
            bid = bid_points / 1000.0
            ticks.append(
                {
                    "timestamp": timestamp,
                    "bid": bid,
                    "ask": ask,
                    "spread": ask - bid,
                    "mid": (bid + ask) / 2.0,
                    "bid_volume": bid_vol,
                    "ask_volume": ask_vol,
                }
            )

        if not ticks:
            return None
        return pd.DataFrame(ticks)

    def download_range(
        self,
        start: datetime,
        end: datetime,
        save_ticks: bool = True,
    ) -> pd.DataFrame:
        """
        Download all tick data in [start, end) and return as DataFrame.
        Also saves hourly parquet files to tick_dir.
        """
        hours: List[tuple] = []
        current = start.replace(minute=0, second=0, microsecond=0)
        while current < end:
            hours.append((current.year, current.month, current.day, current.hour))
            current += timedelta(hours=1)

        all_ticks: List[pd.DataFrame] = []

        print(f"[Dukascopy] Downloading {len(hours)} hours with {self.n_workers} workers ...")
        with ThreadPoolExecutor(max_workers=self.n_workers) as exe:
            futures = {
                exe.submit(self._download_hour, y, m, d, h): (y, m, d, h)
                for y, m, d, h in hours
            }
            for future in as_completed(futures):
                y, m, d, h = futures[future]
                try:
                    df_hour = future.result()
                    if df_hour is not None and len(df_hour) > 0:
                        all_ticks.append(df_hour)
                        if save_ticks:
                            fname = self.tick_dir / f"XAUUSD_{y}{m:02d}{d:02d}_{h:02d}.parquet"
                            df_hour.to_parquet(fname, index=False)
                        if h % 6 == 0:
                            print(f"  OK {y}-{m:02d}-{d:02d} {h:02d}h ({len(df_hour)} ticks)")
                    else:
                        if h % 6 == 0:
                            print(f"  EMPTY {y}-{m:02d}-{d:02d} {h:02d}h")
                except Exception as exc:
                    print(f"  ERR {y}-{m:02d}-{d:02d} {h:02d}h: {exc}")

        if not all_ticks:
            return pd.DataFrame()
        full = pd.concat(all_ticks, ignore_index=True)
        full = full.sort_values("timestamp").reset_index(drop=True)
        print(f"[Dukascopy] Total ticks downloaded: {len(full):,}")
        return full

    def resample_to_1m(self, tick_df: pd.DataFrame) -> pd.DataFrame:
        """Convert tick DataFrame to 1-minute OHLCV bars using mid price."""
        if tick_df.empty:
            return pd.DataFrame()

        tick_df = tick_df.copy()
        tick_df["timestamp"] = pd.to_datetime(tick_df["timestamp"])
        tick_df = tick_df.set_index("timestamp").sort_index()

        # Resample to 1-minute bars from mid price
        ohlc = tick_df["mid"].resample("1min").ohlc()
        volume = tick_df["bid_volume"].resample("1min").sum() + tick_df["ask_volume"].resample("1min").sum()

        df_1m = ohlc.copy()
        df_1m["volume"] = volume
        df_1m = df_1m.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"})
        df_1m = df_1m.dropna()
        return df_1m

    def build_1m_csv(self, tick_df: Optional[pd.DataFrame] = None) -> Path:
        """
        Build or refresh the 1-minute aggregated CSV.
        If tick_df is None, loads all saved parquet files from tick_dir.
        """
        if tick_df is None:
            files = sorted(self.tick_dir.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No tick parquet files found in {self.tick_dir}")
            print(f"[Dukascopy] Loading {len(files)} parquet files ...")
            tick_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

        df_1m = self.resample_to_1m(tick_df)
        df_1m.to_csv(DUKASCOPY_1M_CSV)
        print(f"[Dukascopy] Saved 1-minute bars to {DUKASCOPY_1M_CSV} ({len(df_1m):,} rows)")
        return DUKASCOPY_1M_CSV

    def load(self) -> Optional[pd.DataFrame]:
        """Load the aggregated 1-minute CSV."""
        if not self.is_available():
            print(f"[Dukascopy] Data not found at {DUKASCOPY_1M_CSV}")
            return None
        df = pd.read_csv(DUKASCOPY_1M_CSV, index_col=0, parse_dates=True)
        print(f"[Dukascopy] Loaded {len(df):,} rows")
        return df

    def get_status(self) -> dict:
        return {
            "source": "Dukascopy",
            "available": self.is_available(),
            "path": str(DUKASCOPY_1M_CSV),
            "size_mb": round(DUKASCOPY_1M_CSV.stat().st_size / (1024 * 1024), 2)
            if self.is_available()
            else 0,
            "tick_files": len(list(self.tick_dir.glob("*.parquet"))),
        }
