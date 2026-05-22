"""
Preprocess & Combine Data Sources
=================================
1. Load old CSV (gold_2025_2026.csv)
2. Load new Dukascopy M1 data
3. Validate both (price sanity, gap detection, duplicates)
4. Combine into unified dataset
5. Save as gold_unified_16m.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

OLD_CSV = Path("data/beta_testing/processed/gold_2025_2026.csv")
NEW_CSV = Path("data/beta_testing/dukascopy/XAUUSD_1m_feb_may_2026.csv")
OUTPUT_CSV = Path("data/beta_testing/processed/gold_unified_16m.csv")

def load_old():
    print("[Preprocess] Loading old CSV...")
    df = pd.read_csv(OLD_CSV, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df = df.rename(columns=str.lower)
    # Keep only OHLCV
    df = df[["open", "high", "low", "close", "volume"]]
    print(f"  Rows: {len(df):,} | Range: {df.index.min()} to {df.index.max()}")
    return df

def load_new():
    print("[Preprocess] Loading Dukascopy M1 data...")
    if not NEW_CSV.exists():
        print(f"  ERROR: {NEW_CSV} not found!")
        return pd.DataFrame()
    df = pd.read_csv(NEW_CSV, index_col=0, parse_dates=True)
    df = df.rename(columns=str.lower)
    # Ensure columns match
    expected = ["open", "high", "low", "close", "volume"]
    for c in expected:
        if c not in df.columns:
            print(f"  ERROR: Missing column '{c}'")
            return pd.DataFrame()
    df = df[expected]
    print(f"  Rows: {len(df):,} | Range: {df.index.min()} to {df.index.max()}")
    return df

def validate(df, label):
    print(f"\n[Validate] {label}")
    issues = []

    # Price sanity
    for col in ["open", "high", "low", "close"]:
        invalid = (df[col] < 1000) | (df[col] > 10000)
        if invalid.any():
            issues.append(f"  {invalid.sum()} prices outside 1000-10000 in {col}")

    # OHLC logic
    invalid_ohlc = (df["high"] < df["low"]) | (df["close"] > df["high"]) | (df["close"] < df["low"])
    if invalid_ohlc.any():
        issues.append(f"  {invalid_ohlc.sum()} bars with invalid OHLC logic")

    # Negative volume
    neg_vol = df["volume"] < 0
    if neg_vol.any():
        issues.append(f"  {neg_vol.sum()} bars with negative volume")

    # Duplicates
    dups = df.index.duplicated().sum()
    if dups > 0:
        issues.append(f"  {dups} duplicate timestamps")

    # Gaps > 5 minutes (during trading hours)
    gaps = df.index.to_series().diff().dt.total_seconds() / 60
    big_gaps = gaps[gaps > 5]
    if len(big_gaps) > 0:
        issues.append(f"  {len(big_gaps)} gaps > 5 minutes (max: {big_gaps.max():.0f} min)")

    # Weekend gaps are expected
    weekday_gaps = []
    for ts, gap in big_gaps.items():
        if ts.weekday() < 5:  # Mon-Fri
            weekday_gaps.append((ts, gap))
    if weekday_gaps:
        issues.append(f"  {len(weekday_gaps)} weekday gaps > 5 min")

    if issues:
        for i in issues:
            print(i)
    else:
        print("  All checks passed!")
    return len(issues) == 0

def combine(old_df, new_df):
    print("\n[Combine] Merging datasets...")
    # Remove overlapping dates (keep new data for overlap)
    if not old_df.empty and not new_df.empty:
        overlap_start = new_df.index.min()
        old_df = old_df[old_df.index < overlap_start]

    combined = pd.concat([old_df, new_df]).sort_index()

    # Remove exact duplicates
    combined = combined[~combined.index.duplicated(keep="last")]

    # Forward-fill small gaps (up to 2 minutes)
    combined = combined.asfreq("1min")
    combined[["open", "high", "low", "close"]] = combined[["open", "high", "low", "close"]].ffill(limit=2)
    combined["volume"] = combined["volume"].fillna(0)
    combined = combined.dropna(subset=["open", "high", "low", "close"])

    print(f"  Combined: {len(combined):,} rows")
    print(f"  Range: {combined.index.min()} to {combined.index.max()}")
    return combined

def main():
    print("=" * 60)
    print("Data Preprocessing & Combination")
    print("=" * 60)

    old_df = load_old()
    new_df = load_new()

    if new_df.empty:
        print("\n[ERROR] Dukascopy data not available yet. Exiting.")
        return

    valid_old = validate(old_df, "Old CSV")
    valid_new = validate(new_df, "Dukascopy M1")

    if not (valid_old and valid_new):
        print("\n[WARNING] Validation issues found. Proceeding with caution.")

    combined = combine(old_df, new_df)
    combined.to_csv(OUTPUT_CSV)
    print(f"\n[SAVED] {OUTPUT_CSV} ({len(combined):,} rows)")

if __name__ == "__main__":
    main()
