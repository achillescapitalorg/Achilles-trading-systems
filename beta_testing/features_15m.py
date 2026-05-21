"""
15-Minute Gold Feature Engineering
====================================
Resamples 1m data to 15m and computes features adapted for higher-timeframe
direction prediction. Adds session/daily/weekly context on top of the base
1m feature set.

Target: 4-bar (1h) directional move with 0.2% threshold.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from beta_testing.features import compute_1m_features


def resample_to_15m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-minute OHLCV to 15-minute bars.

    Args:
        df_1m: DataFrame with columns [open, high, low, close, volume]
               and DatetimeIndex.

    Returns:
        15m OHLCV DataFrame.
    """
    df = df_1m.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'date' column")

    # Drop metadata columns before resampling
    df = df.drop(columns=[c for c in ['is_original', 'minutes_since_last_bar'] if c in df.columns], errors='ignore')

    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

    df_15m = df.resample('15min').agg(agg_dict)
    df_15m = df_15m.dropna(subset=['open', 'high', 'low', 'close'])
    return df_15m


def compute_15m_features(df_15m: pd.DataFrame) -> pd.DataFrame:
    """
    Generate complete 15m feature set.

    Strategy:
      1. Run base 1m feature engine on 15m bars (proven features scale)
      2. Add 15m-specific session / daily / weekly context
      3. Engineer 4-bar (1h) target with 0.2% threshold

    Returns:
        DataFrame with features + targets.
    """
    # =====================================================================
    # 1. Base features (reuse 1m engine on 15m data)
    # =====================================================================
    features = compute_1m_features(df_15m)

    # =====================================================================
    # 2. 15m-Specific Session / Daily / Weekly Features
    # =====================================================================
    close = df_15m['close']
    high = df_15m['high']
    low = df_15m['low']
    idx = df_15m.index

    # ---- Session Kill Zones ----
    if isinstance(idx, pd.DatetimeIndex):
        hour = idx.hour
        features['is_london_kill'] = ((hour >= 7) & (hour < 9)).astype(float)
        features['is_ny_kill'] = ((hour >= 13) & (hour < 15)).astype(float)
        features['is_asian_range'] = ((hour >= 0) & (hour < 8)).astype(float)

        # Day of week context (0=Monday … 4=Friday)
        dow = idx.dayofweek
        features['day_of_week'] = dow
        features['is_monday'] = (dow == 0).astype(float)
        features['is_friday'] = (dow == 4).astype(float)
        features['is_midweek'] = ((dow >= 1) & (dow <= 3)).astype(float)

    # ---- Daily Context ----
    # Daily high / low / range
    daily_high = high.groupby(idx.date).transform('max')
    daily_low = low.groupby(idx.date).transform('min')
    features['daily_range'] = (daily_high - daily_low) / (close + 1e-10)
    features['daily_position'] = (close - daily_low) / (daily_high - daily_low + 1e-10)
    features['daily_range_ma20'] = features['daily_range'].rolling(window=20, min_periods=1).mean()

    # Distance from daily high / low
    features['dist_daily_high'] = (daily_high - close) / (close + 1e-10)
    features['dist_daily_low'] = (close - daily_low) / (close + 1e-10)

    # ---- Weekly Context ----
    # Weekly high / low tracking
    week_start = idx.to_period('W').start_time
    weekly_high = high.groupby(week_start).transform('max')
    weekly_low = low.groupby(week_start).transform('min')
    features['weekly_position'] = (close - weekly_low) / (weekly_high - weekly_low + 1e-10)

    # Distance from weekly high / low
    features['dist_weekly_high'] = (weekly_high - close) / (close + 1e-10)
    features['dist_weekly_low'] = (close - weekly_low) / (close + 1e-10)

    # ---- Opening Range (first 4 bars of session) ----
    # Identify session open: 00:00, 08:00 (London), 13:00 (NY) UTC
    if isinstance(idx, pd.DatetimeIndex):
        session_open = pd.Series(
            ((idx.hour == 0) & (idx.minute == 0)) |
            ((idx.hour == 8) & (idx.minute == 0)) |
            ((idx.hour == 13) & (idx.minute == 0)),
            index=idx
        )
    else:
        session_open = pd.Series(False, index=idx)

    # Opening range = high-low of first 4 bars after each session open
    session_id = session_open.cumsum()
    bar_in_session = session_open.groupby(session_id).cumcount()
    in_or = bar_in_session < 4

    or_high = high.where(in_or).groupby(session_id).transform('max')
    or_low = low.where(in_or).groupby(session_id).transform('min')
    features['opening_range'] = (or_high - or_low) / (close + 1e-10)
    features['opening_range_breakout'] = (
        (close > or_high) | (close < or_low)
    ).astype(float)

    # ---- 15m Momentum (aligned with 4-bar target) ----
    features['momentum_4bar'] = close.pct_change(4)
    features['momentum_8bar'] = close.pct_change(8)
    features['momentum_16bar'] = close.pct_change(16)

    # Rate of change acceleration
    features['momentum_accel'] = features['momentum_4bar'] - features['momentum_4bar'].shift(4)

    # =====================================================================
    # 3. Target Engineering (4-bar / 1-hour ahead)
    # =====================================================================
    features['target_ret_4'] = close.pct_change(4).shift(-4)
    features['target_dir_4'] = np.sign(features['target_ret_4'])

    # 3-class: strong up (>0.2%), noise, strong down (<-0.2%)
    features['target_3class_4'] = np.where(
        features['target_ret_4'] > 0.002, 2,
        np.where(features['target_ret_4'] < -0.002, 0, 1)
    )

    # Clean
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().fillna(0)
    return features
