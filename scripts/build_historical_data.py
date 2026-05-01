"""
Build canonical historical OHLCV CSVs for training.

Sources (all free, no API key needed):
  - Yahoo Finance via yfinance     — primary (daily 20y, 1h 2y, 1m 8d)
  - ECB euro reference rates       — FX close-price cross-check (since 1999)
  - Binance public klines          — BTC daily cross-check (since 2017)
  - Stooq                          — attempted, often blocked from non-EU IPs

For each pair, writes up to three CSVs to data/historical/<PAIR>_<INTERVAL>.csv:
    *_daily.csv  — 1d bars, ~20-26 years
    *_1h.csv     — 1h bars, ~2 years
    *_1m.csv     — 1m bars, ~8 days (yfinance per-request cap)

Cross-verification: where two sources overlap on daily bars, we report the
median absolute % difference. >0.5% is flagged as suspicious.

Output schema:
    timestamp, open, high, low, close, volume, source

Run:
    python scripts/build_historical_data.py            # all pairs
    python scripts/build_historical_data.py XAUUSD     # one pair
    python scripts/build_historical_data.py --intervals daily   # only daily
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Cache ECB once per run (it's a 600KB zip with all FX rates back to 1999)
_ECB_CACHE: Optional[pd.DataFrame] = None

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "historical",
)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Pair → ticker mappings ───────────────────────────────────────────────────
# yfinance uses different conventions per asset class:
#   FX:       "EURUSD=X" (spot)
#   Gold:     "GC=F" (Comex futures continuous) or "XAUUSD=X" (spot, less history)
#   Crypto:   "BTC-USD"
# Stooq: lowercase pair name (eurusd, gbpusd, xauusd, btcusd, ^spx, etc.)
PAIRS: Dict[str, Dict[str, Optional[str]]] = {
    "XAUUSD": {"yf": "GC=F",     "stooq": "xauusd",  "binance": None,      "ecb": None},
    "BTCUSD": {"yf": "BTC-USD",  "stooq": "btcusd",  "binance": "BTCUSDT", "ecb": None},
    "EURUSD": {"yf": "EURUSD=X", "stooq": "eurusd",  "binance": None,      "ecb": "EURUSD"},
    "GBPUSD": {"yf": "GBPUSD=X", "stooq": "gbpusd",  "binance": None,      "ecb": "GBPUSD"},
    "USDJPY": {"yf": "USDJPY=X", "stooq": "usdjpy",  "binance": None,      "ecb": "USDJPY"},
    "AUDUSD": {"yf": "AUDUSD=X", "stooq": "audusd",  "binance": None,      "ecb": "AUDUSD"},
}


# ── Source 1: yfinance ───────────────────────────────────────────────────────

def fetch_yf(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval,
                         progress=False, auto_adjust=True, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "timestamp"
        return df
    except Exception as e:
        print(f"  [yf] {symbol} {interval} failed: {e}")
        return pd.DataFrame()


def fetch_yf_1m_chunked(symbol: str, days: int = 60) -> pd.DataFrame:
    """yfinance caps 1m at 8 days/request — chunk over the requested window."""
    chunks = []
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=7, hours=23), end)
        try:
            df = yf.download(symbol, start=cursor, end=chunk_end, interval="1m",
                             progress=False, auto_adjust=True, threads=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df.columns = [c.lower() for c in df.columns]
                df = df[["open", "high", "low", "close", "volume"]].dropna()
                df.index = pd.to_datetime(df.index, utc=True)
                chunks.append(df)
        except Exception as e:
            print(f"  [yf 1m] {symbol} chunk {cursor.date()} failed: {e}")
        cursor = chunk_end
        time.sleep(0.4)
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks)
    out.index.name = "timestamp"
    return out[~out.index.duplicated(keep="first")].sort_index()


# ── Source 2: Stooq (free CSV download, daily only) ──────────────────────────

def fetch_stooq(symbol: str) -> pd.DataFrame:
    """Stooq returns daily OHLCV CSV. Often blocked from non-EU IPs — best-effort."""
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or not r.text or "Date" not in r.text[:200]:
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        df["timestamp"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("timestamp").drop(columns=["date"])
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df[["open", "high", "low", "close", "volume"]].dropna(
            subset=["open", "high", "low", "close"])
    except Exception:
        return pd.DataFrame()


# ── Source 4: ECB euro reference rates (FX close cross-check, since 1999) ────

def _load_ecb() -> pd.DataFrame:
    """Download and cache ECB historical FX reference rates (close prices only).
    Returns DataFrame indexed by date, columns are 3-letter currency codes
    (USD, GBP, JPY, AUD...) — each value is X units of that currency per 1 EUR.
    """
    global _ECB_CACHE
    if _ECB_CACHE is not None:
        return _ECB_CACHE
    url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            _ECB_CACHE = pd.DataFrame()
            return _ECB_CACHE
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            df = pd.read_csv(z.open(name))
        df = df.dropna(axis=1, how="all")
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.set_index("Date").sort_index()
        df = df.replace("N/A", np.nan)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        _ECB_CACHE = df
    except Exception as e:
        print(f"  [ecb] hist load failed: {e}")
        _ECB_CACHE = pd.DataFrame()
    return _ECB_CACHE


def fetch_ecb_pair(pair: str) -> pd.DataFrame:
    """Derive a USD-quoted FX series from ECB EUR-reference rates.
    Returns OHLC = close × 4 (no intraday range data) so cross-check still works.
    """
    df = _load_ecb()
    if df.empty:
        return pd.DataFrame()
    # ECB columns are currency codes in upper case (USD, GBP, JPY, AUD).
    # Each row: 1 EUR = N units of that currency. So:
    #   EURUSD = USD column directly
    #   GBPUSD = USD ÷ GBP
    #   USDJPY = JPY ÷ USD
    #   AUDUSD = USD ÷ AUD
    if "USD" not in df.columns:
        return pd.DataFrame()
    if pair == "EURUSD":
        s = df["USD"].dropna()
    elif pair == "GBPUSD" and "GBP" in df.columns:
        s = (df["USD"] / df["GBP"]).dropna()
    elif pair == "USDJPY" and "JPY" in df.columns:
        s = (df["JPY"] / df["USD"]).dropna()
    elif pair == "AUDUSD" and "AUD" in df.columns:
        s = (df["USD"] / df["AUD"]).dropna()
    else:
        return pd.DataFrame()
    out = pd.DataFrame({
        "open": s, "high": s, "low": s, "close": s,
        "volume": np.zeros(len(s)),
    }, index=s.index)
    out.index.name = "timestamp"
    return out


# ── Source 3: Binance public klines (BTC) ────────────────────────────────────

def fetch_binance(symbol: str, interval: str = "1d") -> pd.DataFrame:
    """Binance daily klines from launch (~Aug 2017) to now. No key required."""
    base = "https://api.binance.com/api/v3/klines"
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = int(datetime(2017, 8, 17, tzinfo=timezone.utc).timestamp() * 1000)
    out = []
    cursor = start
    while cursor < end:
        try:
            r = requests.get(base, timeout=30, params={
                "symbol": symbol, "interval": interval,
                "startTime": cursor, "limit": 1000,
            })
            data = r.json()
            if not isinstance(data, list) or not data:
                break
            out.extend(data)
            cursor = int(data[-1][6]) + 1   # close_time + 1ms
            if len(data) < 1000:
                break
            time.sleep(0.05)
        except Exception as e:
            print(f"  [binance] {symbol} chunk failed: {e}")
            break
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "tb_base", "tb_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
    return df.astype(float)


# ── Cross-verification ───────────────────────────────────────────────────────

def cross_check(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str) -> Dict:
    """Compare two daily series by reindexing to common dates."""
    if a.empty or b.empty:
        return {"overlap_bars": 0, "median_pct_diff": None}
    # Normalize to date (drop time component) for cross-source comparison
    a_d = a.copy(); a_d.index = a_d.index.normalize()
    b_d = b.copy(); b_d.index = b_d.index.normalize()
    common = a_d.index.intersection(b_d.index)
    if len(common) == 0:
        return {"overlap_bars": 0, "median_pct_diff": None}
    pct = ((a_d.loc[common, "close"] - b_d.loc[common, "close"])
           / b_d.loc[common, "close"]).abs()
    return {
        "overlap_bars": int(len(common)),
        "median_pct_diff": float(np.nanmedian(pct.values)),
        "p95_pct_diff": float(np.nanquantile(pct.values, 0.95)),
    }


# ── Merge logic ──────────────────────────────────────────────────────────────

def merge_daily(yf_df: pd.DataFrame, stooq_df: pd.DataFrame,
                binance_df: pd.DataFrame, ecb_df: pd.DataFrame) -> pd.DataFrame:
    """Use yfinance as canonical; fall back to Stooq/Binance/ECB for missing dates.
    Tag the source per row. ECB has only close prices (open=high=low=close)."""
    frames = []
    if not yf_df.empty:
        d = yf_df.copy(); d["source"] = "yfinance"
        frames.append(d)
    if not stooq_df.empty:
        d = stooq_df.copy(); d["source"] = "stooq"
        frames.append(d)
    if not binance_df.empty:
        d = binance_df.copy(); d["source"] = "binance"
        frames.append(d)
    if not ecb_df.empty:
        d = ecb_df.copy(); d["source"] = "ecb"
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames)
    merged.index = merged.index.normalize()
    merged = merged[~merged.index.duplicated(keep="first")]
    return merged.sort_index()


# ── Pipeline per pair ────────────────────────────────────────────────────────

def build_pair(pair: str, intervals: List[str]) -> Dict:
    cfg = PAIRS[pair]
    summary = {"pair": pair}
    print(f"\n=== {pair} (yf={cfg['yf']}, ecb={cfg['ecb']}, binance={cfg['binance']}) ===")

    # Daily — pull all sources, cross-check, merge
    if "daily" in intervals:
        yf_d  = fetch_yf(cfg["yf"], period="max", interval="1d")
        time.sleep(0.3)
        st_d  = fetch_stooq(cfg["stooq"])  # silent best-effort
        time.sleep(0.2)
        bn_d  = fetch_binance(cfg["binance"]) if cfg["binance"] else pd.DataFrame()
        ecb_d = fetch_ecb_pair(cfg["ecb"]) if cfg["ecb"] else pd.DataFrame()

        def _range(df):
            if df.empty: return "—"
            return f"{df.index.min().date()} → {df.index.max().date()}"

        print(f"  yfinance: {len(yf_d):>6} bars  ({_range(yf_d)})")
        if not st_d.empty:
            print(f"  stooq:    {len(st_d):>6} bars  ({_range(st_d)})")
        if not bn_d.empty:
            print(f"  binance:  {len(bn_d):>6} bars  ({_range(bn_d)})")
        if not ecb_d.empty:
            print(f"  ecb:      {len(ecb_d):>6} bars  ({_range(ecb_d)})")

        # Cross-checks against yfinance (the canonical source)
        diffs = {}
        for label, src in (("stooq", st_d), ("binance", bn_d), ("ecb", ecb_d)):
            if not src.empty:
                cc = cross_check(yf_d, src, "yf", label)
                diffs[label] = cc.get("median_pct_diff")
                if cc["median_pct_diff"] is not None:
                    flag = " ⚠️" if cc["median_pct_diff"] > 0.005 else ""
                    print(f"  yf↔{label}: overlap={cc['overlap_bars']} bars, "
                          f"median |Δ|={cc['median_pct_diff']:.4%}, "
                          f"p95={cc.get('p95_pct_diff', 0):.4%}{flag}")

        merged = merge_daily(yf_d, st_d, bn_d, ecb_d)
        if not merged.empty:
            out_path = os.path.join(OUT_DIR, f"{pair}_daily.csv")
            merged.to_csv(out_path)
            print(f"  → {out_path}: {len(merged)} bars  "
                  f"({merged.index.min().date()} → {merged.index.max().date()})")
            summary["daily"] = {
                "bars": len(merged),
                "first": str(merged.index.min().date()),
                "last":  str(merged.index.max().date()),
                "diffs": diffs,
            }
        else:
            print(f"  ⚠️  daily merge empty for {pair}")
            summary["daily"] = None

    # Hourly — yfinance only (~2y limit)
    if "1h" in intervals:
        yf_h = fetch_yf(cfg["yf"], period="730d", interval="1h")
        if not yf_h.empty:
            yf_h["source"] = "yfinance"
            out_path = os.path.join(OUT_DIR, f"{pair}_1h.csv")
            yf_h.to_csv(out_path)
            print(f"  → {out_path}: {len(yf_h)} bars  "
                  f"({yf_h.index.min().date()} → {yf_h.index.max().date()})")
            summary["1h"] = {"bars": len(yf_h),
                             "first": str(yf_h.index.min().date()),
                             "last":  str(yf_h.index.max().date())}
        else:
            print(f"  ⚠️  1h empty for {pair}")
            summary["1h"] = None
        time.sleep(0.3)

    # 1-minute — chunked across 60 days (yfinance per-request cap is 8 days)
    if "1m" in intervals:
        yf_m = fetch_yf_1m_chunked(cfg["yf"], days=60)
        if not yf_m.empty:
            yf_m["source"] = "yfinance"
            out_path = os.path.join(OUT_DIR, f"{pair}_1m.csv")
            yf_m.to_csv(out_path)
            print(f"  → {out_path}: {len(yf_m)} bars  "
                  f"({yf_m.index.min().date()} → {yf_m.index.max().date()})")
            summary["1m"] = {"bars": len(yf_m),
                             "first": str(yf_m.index.min().date()),
                             "last":  str(yf_m.index.max().date())}
        else:
            print(f"  ⚠️  1m empty for {pair}")
            summary["1m"] = None

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*", help="Pairs to fetch (default: all)")
    ap.add_argument("--intervals", default="daily,1h,1m",
                    help="Comma-separated: daily,1h,1m")
    args = ap.parse_args()

    pairs = args.pairs if args.pairs else list(PAIRS.keys())
    intervals = [x.strip() for x in args.intervals.split(",") if x.strip()]
    invalid = [p for p in pairs if p not in PAIRS]
    if invalid:
        print(f"Unknown pairs: {invalid}. Available: {list(PAIRS.keys())}")
        sys.exit(1)

    print(f"Output dir: {OUT_DIR}")
    print(f"Pairs:      {pairs}")
    print(f"Intervals:  {intervals}")

    summaries = []
    for p in pairs:
        try:
            summaries.append(build_pair(p, intervals))
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ❌ {p} failed: {e}")

    # Final summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for s in summaries:
        d = s.get("daily") or {}
        h = s.get("1h") or {}
        m = s.get("1m") or {}
        diffs = (d.get("diffs") or {}) if isinstance(d, dict) else {}
        diff_strs = []
        for src in ("ecb", "binance", "stooq"):
            v = diffs.get(src)
            if v is not None:
                diff_strs.append(f"yf↔{src}={v:.3%}")
        diff_str = "  ".join(diff_strs) if diff_strs else "no cross-check"
        print(f"{s['pair']:<8}  "
              f"daily={d.get('bars', 0):>6} ({d.get('first', '—')}→{d.get('last', '—')})  "
              f"1h={h.get('bars', 0):>5}  "
              f"1m={m.get('bars', 0):>6}  "
              f"{diff_str}")


if __name__ == "__main__":
    main()
