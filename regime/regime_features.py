"""
Regime Feature Engineering
==========================
Compute features specifically designed for regime detection.
"""
import numpy as np
import pandas as pd
from scipy import stats


class RegimeFeatureEngineer:
    """
    Compute features specifically designed for regime detection.
    These capture the statistical signature of each market state.
    """

    def __init__(self, lookback_short: int = 20, lookback_medium: int = 50,
                 lookback_long: int = 100):
        self.lookback_short = lookback_short
        self.lookback_medium = lookback_medium
        self.lookback_long = lookback_long

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute complete regime feature set from 1-minute OHLCV data.
        Returns DataFrame with regime-indicative features.
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('volume', pd.Series(1, index=df.index))

        # --- 1. VOLATILITY REGIME FEATURES ---
        returns = close.pct_change().dropna()
        # ATR at multiple periods
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        features['atr_14'] = true_range.rolling(14).mean()
        features['atr_50'] = true_range.rolling(50).mean()
        features['atr_ratio'] = features['atr_14'] / (features['atr_50'] + 1e-10)
        # Realized volatility (annualized)
        features['realized_vol_20'] = returns.rolling(20).std() * np.sqrt(252 * 24 * 60)
        features['realized_vol_50'] = returns.rolling(50).std() * np.sqrt(252 * 24 * 60)
        features['vol_ratio'] = features['realized_vol_20'] / (features['realized_vol_50'] + 1e-10)
        # Volatility regime classification
        vol_percentile = features['realized_vol_20'].rolling(500).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) > 50 else 50,
            raw=False
        )
        features['vol_percentile'] = vol_percentile

        # --- 2. TREND STRENGTH FEATURES ---
        features['adx_14'] = self._compute_adx(high, low, close, 14)
        features['adx_50'] = self._compute_adx(high, low, close, 50)
        # EMA slopes (degrees of trend)
        ema_20 = close.ewm(span=20).mean()
        ema_50 = close.ewm(span=50).mean()
        ema_200 = close.ewm(span=200).mean()
        features['ema20_slope'] = np.degrees(np.arctan(ema_20.diff(20) / 20))
        features['ema50_slope'] = np.degrees(np.arctan(ema_50.diff(50) / 50))
        features['ema200_slope'] = np.degrees(np.arctan(ema_200.diff(200) / 200))
        # Trend alignment
        features['trend_alignment'] = np.sign(features['ema20_slope']) * np.sign(features['ema50_slope'])
        # Distance from long-term EMA
        features['dist_from_ema200'] = (close - ema_200) / ema_200

        # --- 3. MEAN-REVERSION FEATURES ---
        bb_ma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        features['bb_position'] = (close - bb_ma) / (2 * bb_std + 1e-10)
        features['bb_width'] = bb_std / (bb_ma + 1e-10)
        # Distance from rolling VWAP (1-day lookback, not expanding)
        tp = (high + low + close) / 3
        vwap = (tp * volume).rolling(1440).sum() / (volume.rolling(1440).sum() + 1e-10)
        features['vwap_deviation'] = (close - vwap) / (features['atr_14'] + 1e-10)
        # Return autocorrelation (mean-reversion signature)
        features['autocorr_1'] = returns.rolling(50).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 10 else 0
        )
        features['autocorr_5'] = returns.rolling(50).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 10 else 0
        )

        # --- 4. VOLUME REGIME FEATURES ---
        vol_ma = volume.rolling(20).mean()
        features['volume_ratio'] = volume / (vol_ma + 1e-10)
        features['volume_trend'] = volume.rolling(20).apply(
            lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 10 else 0
        )

        # --- 5. FRACTAL/CHAOS FEATURES ---
        features['hurst_50'] = self._compute_hurst(returns, 50)
        features['return_skew'] = returns.rolling(50).skew()
        features['return_kurt'] = returns.rolling(50).kurt()
        features['consecutive_up'] = self._consecutive_count(returns > 0)
        features['consecutive_down'] = self._consecutive_count(returns < 0)

        # --- 6. MARKET MICROSTRUCTURE FEATURES ---
        features['eff_range'] = (high - low) / close
        features['intraday_drift'] = (close - df['open']) / (features['atr_14'] + 1e-10)
        features['overnight_gap'] = (df['open'] - close.shift(1)) / close.shift(1)

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().fillna(0)
        return features

    def _compute_adx(self, high, low, close, period=14):
        """Compute ADX."""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * plus_dm.rolling(period).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.rolling(period).mean() / (atr + 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        return adx

    def _compute_hurst(self, series, window):
        """Compute Hurst exponent using R/S analysis."""
        def hurst_rs(x):
            if len(x) < 20:
                return 0.5
            lags = range(2, min(20, len(x) // 4))
            tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
            if any(t <= 0 for t in tau):
                return 0.5
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return reg[0]
        return series.rolling(window).apply(hurst_rs, raw=True)

    def _consecutive_count(self, series):
        """Count consecutive True values."""
        s = series.astype(int)
        groups = (s != s.shift()).cumsum()
        return s * groups.groupby(groups).cumcount()
