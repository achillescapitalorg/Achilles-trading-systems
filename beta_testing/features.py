"""
1-Minute Gold Feature Engineering
==================================
~50 price features + ~20 cross-asset features.
"""
import pandas as pd
import numpy as np
from scipy import stats
import sys
from pathlib import Path

# Add project root for cross-asset import
sys.path.insert(0, str(Path(__file__).parent.parent))
from features.cross_asset_features import compute_cross_asset_features
from features.features_microstructure import MicrostructureFeatureEngine


def compute_1m_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete 1-minute gold feature engineering.
    Produces ~50 features + targets.
    """
    f = pd.DataFrame(index=df.index)
    close = df['close']
    high = df['high']
    low = df['low']
    open_p = df['open']
    volume = df.get('volume', pd.Series(1, index=df.index))

    # ==================== PRICE RETURNS ====================
    returns = close.pct_change()
    f['ret_1'] = returns
    f['ret_5'] = close.pct_change(5)
    f['ret_10'] = close.pct_change(10)
    f['ret_20'] = close.pct_change(20)
    f['sign_ret_1'] = np.sign(returns)
    f['abs_ret_1'] = returns.abs()

    # ==================== MOMENTUM (TREND) ====================
    for span in [5, 10, 20, 50]:
        ema = close.ewm(span=span).mean()
        f[f'ema_{span}'] = (close - ema) / (ema + 1e-10)
        f[f'ema_slope_{span}'] = ema.diff(span) / (ema + 1e-10)

    f['ema_5_10_cross'] = f['ema_5'] - f['ema_10']
    f['ema_10_20_cross'] = f['ema_10'] - f['ema_20']

    # RSI
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/period).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period).mean()
        rs = gain / (loss + 1e-10)
        f[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # MACD
    ema_8 = close.ewm(span=8).mean()
    ema_17 = close.ewm(span=17).mean()
    macd = ema_8 - ema_17
    signal = macd.ewm(span=5).mean()
    f['macd'] = macd / (close * 0.001 + 1e-10)
    f['macd_signal'] = signal / (close * 0.001 + 1e-10)
    f['macd_hist'] = f['macd'] - f['macd_signal']
    f['macd_hist_slope'] = f['macd_hist'].diff(3)

    # ==================== VOLATILITY ====================
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    for period in [7, 14, 30, 50]:
        atr = tr.rolling(period).mean()
        f[f'atr_{period}'] = atr / (close + 1e-10)

    f['atr_ratio'] = f['atr_14'] / (f['atr_50'] + 1e-10)

    # Bollinger Bands
    bb_ma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    f['bb_position'] = (close - bb_ma) / (2 * bb_std + 1e-10)
    f['bb_width'] = bb_std / (bb_ma + 1e-10)
    f['bb_squeeze'] = f['bb_width'] / (f['bb_width'].rolling(50).max() + 1e-10)

    # Realized volatility
    log_ret = np.log(close / close.shift())
    f['realized_vol_10'] = log_ret.rolling(10).std()
    f['realized_vol_30'] = log_ret.rolling(30).std()
    f['vol_ratio'] = f['realized_vol_10'] / (f['realized_vol_30'] + 1e-10)

    # ==================== MEAN REVERSION ====================
    for period in [20, 50, 100]:
        ma = close.rolling(period).mean()
        f[f'dist_ma_{period}'] = (close - ma) / (tr.rolling(14).mean() + 1e-10)

    lowest_14 = low.rolling(14).min()
    highest_14 = high.rolling(14).max()
    f['stoch_k'] = 100 * (close - lowest_14) / (highest_14 - lowest_14 + 1e-10)
    f['stoch_d'] = f['stoch_k'].rolling(3).mean()

    tp = (high + low + close) / 3
    cci_ma = tp.rolling(20).mean()
    cci_std = tp.rolling(20).std()
    f['cci'] = (tp - cci_ma) / (0.015 * cci_std + 1e-10)

    # ==================== VOLUME ====================
    vol_ma_20 = volume.rolling(20).mean()
    f['volume_ratio'] = volume / (vol_ma_20 + 1e-10)
    f['volume_ratio_ma'] = f['volume_ratio'].rolling(5).mean()

    obv = (np.sign(close.diff()) * volume).cumsum()
    f['obv_slope'] = obv.diff(10) / (obv.abs() + 1e-10)

    # ==================== MARKET MICROSTRUCTURE ====================
    f['bar_range'] = (high - low) / (close + 1e-10)
    f['bar_body'] = (close - open_p).abs() / (high - low + 1e-10)
    f['bar_direction'] = np.sign(close - open_p)

    f['consecutive_up'] = (returns > 0).astype(int).groupby(
        ((returns <= 0)).astype(int).cumsum()
    ).cumsum()
    f['consecutive_down'] = (returns < 0).astype(int).groupby(
        ((returns >= 0)).astype(int).cumsum()
    ).cumsum()

    # ==================== TEMPORAL ====================
    if isinstance(df.index, pd.DatetimeIndex):
        f['hour'] = df.index.hour
        f['minute'] = df.index.minute
        f['is_london'] = ((df.index.hour >= 8) & (df.index.hour < 17)).astype(float)
        f['is_ny'] = ((df.index.hour >= 13) & (df.index.hour < 22)).astype(float)
        f['is_overlap'] = ((df.index.hour >= 13) & (df.index.hour < 17)).astype(float)

    # ==================== TARGET ENGINEERING ====================
    for horizon in [3, 5, 10, 20, 60]:
        f[f'target_ret_{horizon}'] = close.pct_change(horizon).shift(-horizon)
        f[f'target_dir_{horizon}'] = np.sign(f[f'target_ret_{horizon}'])
        # 3-class: strong up (>0.1%), noise (-0.1% to +0.1%), strong down (<-0.1%)
        f[f'target_3class_{horizon}'] = np.where(
            f[f'target_ret_{horizon}'] > 0.001, 2,
            np.where(f[f'target_ret_{horizon}'] < -0.001, 0, 1)
        )

    # ==================== CROSS-ASSET FEATURES ====================
    try:
        cross = compute_cross_asset_features(df)
        if not cross.empty:
            f = pd.concat([f, cross], axis=1)
    except Exception as e:
        print(f"[Features] Cross-asset features failed: {e}")

    # ==================== MICROSTRUCTURE FEATURES ====================
    try:
        micro_engine = MicrostructureFeatureEngine(tick_size=0.01)
        micro_features = micro_engine.generate_all_features(df)
        if not micro_features.empty:
            f = pd.concat([f, micro_features], axis=1)
    except Exception as e:
        print(f"[Features] Microstructure features failed: {e}")

    # Clean
    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.ffill().fillna(0)
    return f
