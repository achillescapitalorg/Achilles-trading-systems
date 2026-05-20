"""
Gold Data Validator
===================
Comprehensive validation for 1-minute gold OHLCV data.
Based on the quality checklist from the data-sources report.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

from .config import (
    EXPECTED_BAR_MINUTES,
    MAX_GAP_MINUTES,
    WEEKEND_GAP_HOURS,
    GOLD_PRICE_MIN,
    GOLD_PRICE_MAX,
)


def validate_gold_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run full validation suite on 1-minute gold data.

    Expects df.index to be DatetimeIndex and columns to include
    ['open', 'high', 'low', 'close', 'volume'] (case-insensitive).
    """
    report: Dict[str, Any] = {}

    # Standardise column names
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        report["error"] = f"Missing columns: {missing}"
        return report

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
        else:
            report["error"] = "No DatetimeIndex and no 'date'/'datetime' column found."
            return report

    report["total_rows"] = len(df)
    report["date_range"] = f"{df.index.min()} to {df.index.max()}"
    report["trading_days"] = int(df.index.normalize().nunique())

    # 1. NaN values
    nan_counts = df[list(required)].isna().sum()
    report["nan_counts"] = nan_counts.to_dict()
    report["has_nan"] = bool(nan_counts.sum() > 0)

    # 2. Duplicate timestamps
    report["duplicate_timestamps"] = int(df.index.duplicated().sum())

    # 3. OHLC logic errors
    ohlc_errors = (
        (df["high"] < df["low"])
        | (df["close"] > df["high"])
        | (df["close"] < df["low"])
        | (df["open"] > df["high"])
        | (df["open"] < df["low"])
    ).sum()
    report["ohlc_logic_errors"] = int(ohlc_errors)

    # 4. Zero-volume bars
    zero_vol = (df["volume"] == 0).sum()
    report["zero_volume_bars"] = int(zero_vol)
    report["zero_volume_pct"] = round(100 * zero_vol / len(df), 2)

    # 5. Trading gaps (>2 min, excluding weekends)
    time_diffs = df.index.to_series().diff().dropna()
    trading_gaps = time_diffs[
        (time_diffs > pd.Timedelta(minutes=MAX_GAP_MINUTES))
        & (time_diffs < pd.Timedelta(hours=WEEKEND_GAP_HOURS))
    ]
    report["trading_gaps_count"] = len(trading_gaps)
    report["trading_gaps_max_minutes"] = (
        trading_gaps.max().total_seconds() / 60 if len(trading_gaps) > 0 else 0.0
    )

    # 6. Weekend gaps (expected)
    weekend_gaps = time_diffs[time_diffs >= pd.Timedelta(hours=WEEKEND_GAP_HOURS)]
    report["weekend_gaps_count"] = len(weekend_gaps)

    # 7. Price range sanity
    report["price_min"] = float(df["low"].min())
    report["price_max"] = float(df["high"].max())
    report["price_range_reasonable"] = bool(
        GOLD_PRICE_MIN < report["price_min"] < report["price_max"] < GOLD_PRICE_MAX
    )

    # 8. Yearly coverage
    yearly_counts = df.groupby(df.index.year).size()
    report["yearly_coverage"] = {
        str(year): int(count) for year, count in yearly_counts.items()
    }

    # Quality score
    score = 100
    if report["has_nan"]:
        score -= 20
    if report["duplicate_timestamps"] > 0:
        score -= 10
    if report["ohlc_logic_errors"] > 0:
        score -= 15
    if report["trading_gaps_count"] > 10:
        score -= 15
    if not report["price_range_reasonable"]:
        score -= 20

    report["quality_score"] = max(0, score)
    report["quality_rating"] = (
        "EXCELLENT"
        if score >= 90
        else "GOOD"
        if score >= 70
        else "FAIR"
        if score >= 50
        else "POOR"
    )

    return report


def print_validation_report(report: Dict[str, Any]) -> None:
    """Pretty-print validation report to console."""
    print("=" * 60)
    print("Gold Data Validation Report")
    print("=" * 60)
    for key, value in report.items():
        if key == "yearly_coverage":
            print(f"{key}:")
            for yr, cnt in value.items():
                print(f"  {yr}: {cnt:,} bars")
        else:
            print(f"{key}: {value}")
    print("=" * 60)
