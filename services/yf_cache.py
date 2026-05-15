"""
Robust Yahoo Finance fetcher with caching, deduplication, and retry.
=================================================================
Solves the concurrent-request rate-limit problem when Dash fires
multiple callbacks simultaneously.

Features:
  - Persistent CSV cache with TTL (5m for 1m, 30m for 15m, 1h for 1h, 1d for daily)
  - Request deduplication: parallel callers for the same key share one download
  - Exponential backoff + jitter on transient failures
  - Rate limiting: max 2 yfinance calls per second
  - Automatic fallback to synthetic data if Yahoo fails repeatedly
  - Chunked 1m download for periods > 7 days (respects Yahoo 8-day cap)
"""
from __future__ import annotations

import os
import sys
import time
import hashlib
import threading
from contextlib import redirect_stderr
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

# yfinance prints misleading "delisted" errors to stderr before returning
# empty data. Redirect to null to keep terminal clean.
_YF_NULL = open(os.devnull, "w")


def _suppress_yf_stderr():
    """Context manager that silences yfinance's stderr spam."""
    return redirect_stderr(_YF_NULL)

# ── Config ───────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "yf_cache"
)
os.makedirs(CACHE_DIR, exist_ok=True)

YF_SYMBOLS = {
    "XAUUSD": "GC=F",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "SPX500": "^GSPC",
    "NAS100": "^NDX",
}

BASE_PRICES = {
    "XAUUSD": 2650, "BTCUSD": 95000, "ETHUSD": 3400,
    "EURUSD": 1.05, "GBPUSD": 1.26, "USDJPY": 153.0,
    "SPX500": 5900, "NAS100": 20500,
}

# TTL in seconds per interval
TTL_MAP = {
    "1m": 300, "2m": 300, "5m": 600, "15m": 1800,
    "30m": 1800, "60m": 3600, "90m": 3600, "1h": 3600,
    "1d": 86400, "5d": 86400, "1wk": 86400 * 7, "1mo": 86400 * 7,
}

# ── Thread-safe state ────────────────────────────────────────────────────────
_lock = threading.Lock()
_pending: Dict[str, threading.Event] = {}
_cache_memory: Dict[str, Tuple[pd.DataFrame, float]] = {}
_last_request_time = 0.0
_min_request_interval = 0.5  # seconds between yfinance calls


def _cache_key(symbol: str, period: str, interval: str) -> str:
    return hashlib.md5(f"{symbol}:{period}:{interval}".encode()).hexdigest()


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.csv")


def _ttl(interval: str) -> int:
    return TTL_MAP.get(interval, 1800)


def _load_from_disk(key: str, ttl: int) -> Optional[pd.DataFrame]:
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    age = time.time() - os.path.getmtime(path)
    if age > ttl:
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=["timestamp"])
        if len(df) > 10:
            return df
    except Exception:
        pass
    return None


def _save_to_disk(key: str, df: pd.DataFrame):
    try:
        path = _cache_path(key)
        df.copy().to_csv(path)
    except Exception:
        pass


def _rate_limit():
    global _last_request_time
    with _lock:
        elapsed = time.time() - _last_request_time
        if elapsed < _min_request_interval:
            time.sleep(_min_request_interval - elapsed)
        _last_request_time = time.time()


def _yf_download_single(yf_symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """Single yfinance download with retry."""
    for attempt in range(3):
        _rate_limit()
        try:
            with _suppress_yf_stderr():
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(period=period, interval=interval)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            err = str(e).lower()
            if "delisted" in err or "no data" in err:
                pass
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt) + np.random.uniform(0, 0.3))
    return None


def _yf_download_chunked_1m(yf_symbol: str, days: int) -> Optional[pd.DataFrame]:
    """Download 1m data in 7-day chunks."""
    chunks: List[pd.DataFrame] = []
    end = datetime.now()
    start = end - timedelta(days=days)
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=7, hours=23), end)
        _rate_limit()
        try:
            with _suppress_yf_stderr():
                df = yf.download(
                    yf_symbol, start=cursor.strftime("%Y-%m-%d"),
                    end=chunk_end.strftime("%Y-%m-%d"), interval="1m",
                    progress=False, auto_adjust=True, threads=False,
                )
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df.columns = [c.lower() for c in df.columns]
                df = df[["open", "high", "low", "close", "volume"]].dropna()
                df.index = pd.to_datetime(df.index, utc=True)
                chunks.append(df)
        except Exception:
            pass
        cursor = chunk_end
        time.sleep(0.3)
    if not chunks:
        return None
    out = pd.concat(chunks)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def _generate_fallback(symbol: str, periods: int = 500) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV for fallback."""
    base = BASE_PRICES.get(symbol, 100)
    np.random.seed(int(datetime.now().timestamp()) % 2**32)
    # Random walk with slight mean reversion
    returns = np.random.normal(0.00005, 0.0015, periods)
    prices = base * np.cumprod(1 + returns)
    timestamps = pd.date_range(
        end=datetime.now(), periods=periods, freq="1min"
    )
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices * (1 + np.random.normal(0, 0.0003, periods)),
        "high": prices * (1 + abs(np.random.normal(0, 0.0005, periods))),
        "low": prices * (1 - abs(np.random.normal(0, 0.0005, periods))),
        "close": prices,
        "volume": np.random.lognormal(12, 1, periods),
    })
    df["low"] = np.minimum(df["low"], df[["open", "close"]].min(axis=1))
    df["high"] = np.maximum(df["high"], df[["open", "close"]].max(axis=1))
    return df


def fetch_cached(symbol: str, period: str = "5d", interval: str = "15m",
                 use_fallback: bool = True) -> pd.DataFrame:
    """
    Fetch historical data with caching, deduplication, retry, and rate limiting.
    This is a drop-in replacement for app.fetch_yahoo_finance_data.
    """
    key = _cache_key(symbol, period, interval)
    ttl = _ttl(interval)

    # 1. Check memory cache
    with _lock:
        if key in _cache_memory:
            df, cached_at = _cache_memory[key]
            if time.time() - cached_at < ttl:
                return df.copy()

    # 2. Check disk cache
    df = _load_from_disk(key, ttl)
    if df is not None:
        with _lock:
            _cache_memory[key] = (df.copy(), time.time())
        return df.copy()

    # 3. Request deduplication: if same key in flight, wait for it
    with _lock:
        if key in _pending:
            event = _pending[key]
            is_leader = False
        else:
            event = threading.Event()
            _pending[key] = event
            is_leader = True

    if not is_leader:
        event.wait(timeout=30.0)
        # After waiting, try cache again
        with _lock:
            if key in _cache_memory:
                df, _ = _cache_memory[key]
                return df.copy()
        # If still nothing, proceed as leader
        with _lock:
            if key not in _pending:
                event = threading.Event()
                _pending[key] = event
                is_leader = True

    try:
        yf_symbol = YF_SYMBOLS.get(symbol, symbol)
        days = 5
        if period.endswith("d") and period[:-1].isdigit():
            days = int(period[:-1])
        elif period == "1mo":
            days = 30
        elif period == "3mo":
            days = 90
        elif period == "6mo":
            days = 180
        elif period == "1y":
            days = 365

        # 4. Fetch from Yahoo
        if interval in ("1m", "1min") and days > 7:
            raw = _yf_download_chunked_1m(yf_symbol, days)
        else:
            raw = _yf_download_single(yf_symbol, period, interval)

        if raw is not None and not raw.empty:
            df = raw.reset_index()
            # Normalize columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    str(c[0]).lower() if isinstance(c, tuple) else str(c).lower()
                    for c in df.columns
                ]
            else:
                df.columns = [str(c).lower() for c in df.columns]

            if "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
            elif "datetime" in df.columns:
                df["timestamp"] = pd.to_datetime(df["datetime"])
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            else:
                df["timestamp"] = pd.to_datetime(df.iloc[:, 0])

            required = ["timestamp", "open", "high", "low", "close", "volume"]
            for col in required:
                if col not in df.columns:
                    raise ValueError(f"Missing column: {col}")

            df = df[required].copy()
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

            # Save to caches
            _save_to_disk(key, df)
            with _lock:
                _cache_memory[key] = (df.copy(), time.time())
            return df.copy()

    except Exception as e:
        print(f"[YFCache] Fetch failed for {symbol} {period} {interval}: {e}")

    finally:
        # Signal waiting threads
        if is_leader:
            with _lock:
                event = _pending.pop(key, None)
            if event is not None:
                event.set()

    # 5. Fallback
    if use_fallback:
        print(f"[YFCache] Using fallback data for {symbol} {period} {interval}")
        periods_map = {"1d": 390, "5d": 1950, "1mo": 8500, "3mo": 25000,
                       "6mo": 50000, "1y": 100000, "2y": 200000}
        periods = periods_map.get(period, 1000)
        if interval == "1m":
            pass
        elif interval in ("5m", "15m", "1h", "1d"):
            periods = max(100, periods // 5)
        df = _generate_fallback(symbol, periods)
        return df

    return pd.DataFrame()


def warm_cache(symbol: str = "XAUUSD", period: str = "5d", interval: str = "1m"):
    """Pre-fetch data in background to warm the cache."""
    try:
        fetch_cached(symbol, period, interval)
    except Exception:
        pass
