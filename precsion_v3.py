
# ============================================================
# ENHANCED 1-MINUTE PRECISION TRADING SYSTEM v3.0
# Research-Backed Architecture - Selective Surgical Enhancements
# Based on: Claude triage + 2025 market microstructure literature
# ============================================================
# SELECTIVE ENHANCEMENTS (8 surgical additions, no bloat):
# 1. CVD + CVD-Price divergence (institutional flow edge)
# 2. Volume Profile (POC, Value Area, absorption) 
# 3. Lightweight FVG detection (for stop placement only)
# 4. MTF Confluence (1m/5m/15m weighted alignment)
# 5. Fundamental Context (news impact + DXY + session regime)
# 6. Kelly Criterion sizing (Quarter-Kelly with floor/cap)
# 7. Isotonic probability calibration (fixes XGBoost bias)
# 8. Alternative Bar Sampling (Tick/Volume Imbalance Bars)
# ============================================================

enhanced_v3 = '''
"""
================================================================================
ENHANCED 1-MINUTE PRECISION TRADING SYSTEM v3.0
Asset Coverage: XAU/USD, BTC/USD, EUR/USD, GBP/USD
================================================================================
Research-Backed Selective Enhancements (8 surgical additions):

    1. Cumulative Volume Delta (CVD) + Price Divergence
    2. Volume Profile (POC, Value Area, Absorption Detection)
    3. Fair Value Gap Detection (lightweight, for stops only)
    4. Multi-Timeframe Confluence (1m/5m/15m)
    5. Fundamental Context (News + DXY + Session Regime)
    6. Kelly Criterion Position Sizing (Quarter-Kelly)
    7. Isotonic Probability Calibration
    8. Alternative Bar Sampling (Tick/Volume Imbalance Bars)

EXCLUDED (per triage - marginal gain or too heavy):
    - Stacked LSTM/GRU/Transformer ensemble (TF dependency)
    - Per-bar ARIMA baseline (too slow for 1m)
    - Order Blocks / BOS / CHoCH (buggy, no peer review)
    - Residual XGBoost correction (needs base ensemble)
================================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Literal, Any
from enum import Enum
import warnings
from collections import deque, defaultdict
import json
from datetime import datetime, timedelta
import pickle
import os

# -- Optional deps; fall back gracefully -------------------------------------
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not available.")

try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False

try:
    from pykalman import KalmanFilter
    PYKALMAN_AVAILABLE = True
except ImportError:
    PYKALMAN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIG
# =============================================================================

class Asset(Enum):
    XAUUSD = "XAUUSD"
    BTCUSD = "BTCUSD"
    EURUSD = "EURUSD"
    GBPUSD = "GBPUSD"

class Signal(Enum):
    STRONG_BUY = 2; WEAK_BUY = 1; NEUTRAL = 0; WEAK_SELL = -1; STRONG_SELL = -2

class Regime(Enum):
    TRENDING = "trending"; MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"; LOW_VOLATILITY = "low_volatility"

class TradeDirection(Enum):
    LONG = 1; SHORT = -1; FLAT = 0


@dataclass
class AssetConfig:
    asset: Asset
    pip_value: float
    spread_avg: float
    tick_size: float
    contract_size: float = 1.0
    leverage: float = 100.0
    
    london_start: int = 8; london_end: int = 17
    ny_start: int = 13; ny_end: int = 22
    asia_start: int = 0; asia_end: int = 8
    
    kalman_observation_covariance: float = 1.0
    kalman_transition_covariance: float = 0.01
    
    vpin_buckets: int = 50
    vpin_window: int = 50
    vpin_threshold: float = 0.85
    
    max_risk_per_trade_pct: float = 0.01
    kelly_fraction: float = 0.25
    atr_multiplier_stop: float = 1.5
    atr_multiplier_tp1: float = 1.0
    atr_multiplier_tp2: float = 2.0
    atr_multiplier_tp3: float = 3.0
    
    # FVG
    fvg_lookback: int = 10
    fvg_body_multiplier: float = 1.5
    
    # Volume Profile
    vp_lookback: int = 50
    vp_value_area_pct: float = 0.70
    
    # News
    high_impact_news_window_minutes: int = 30
    dxy_correlation_threshold: float = 0.7
    
    # Calibration
    calibration_method: str = "isotonic"


ASSET_CONFIGS: Dict[Asset, AssetConfig] = {
    Asset.XAUUSD: AssetConfig(
        asset=Asset.XAUUSD, pip_value=0.01, spread_avg=0.05, tick_size=0.01,
        contract_size=100.0, leverage=100.0,
        kalman_observation_covariance=0.5, kalman_transition_covariance=0.005,
        vpin_buckets=50, vpin_window=50, vpin_threshold=0.90,
        max_risk_per_trade_pct=0.01, kelly_fraction=0.25,
        atr_multiplier_stop=2.0,
        atr_multiplier_tp1=1.5, atr_multiplier_tp2=2.5, atr_multiplier_tp3=4.0,
        fvg_lookback=10, fvg_body_multiplier=1.5,
        vp_lookback=50, vp_value_area_pct=0.70,
        high_impact_news_window_minutes=45,
        dxy_correlation_threshold=0.75,
        calibration_method="isotonic",
    ),
    Asset.BTCUSD: AssetConfig(
        asset=Asset.BTCUSD, pip_value=1.0, spread_avg=20.0, tick_size=0.01,
        contract_size=1.0, leverage=50.0,
        kalman_observation_covariance=100.0, kalman_transition_covariance=1.0,
        vpin_buckets=100, vpin_window=100, vpin_threshold=0.80,
        max_risk_per_trade_pct=0.01, kelly_fraction=0.20,
        atr_multiplier_stop=2.5,
        atr_multiplier_tp1=2.0, atr_multiplier_tp2=3.5, atr_multiplier_tp3=5.0,
        fvg_lookback=10, fvg_body_multiplier=2.0,
        vp_lookback=100, vp_value_area_pct=0.70,
        high_impact_news_window_minutes=30,
        dxy_correlation_threshold=0.5,
        calibration_method="isotonic",
    ),
    Asset.EURUSD: AssetConfig(
        asset=Asset.EURUSD, pip_value=0.0001, spread_avg=0.0001, tick_size=0.00001,
        contract_size=100000.0, leverage=100.0,
        kalman_observation_covariance=0.0001, kalman_transition_covariance=0.000001,
        vpin_buckets=50, vpin_window=50, vpin_threshold=0.90,
        max_risk_per_trade_pct=0.01, kelly_fraction=0.25,
        atr_multiplier_stop=1.5,
        atr_multiplier_tp1=1.0, atr_multiplier_tp2=2.0, atr_multiplier_tp3=3.0,
        fvg_lookback=10, fvg_body_multiplier=1.2,
        vp_lookback=50, vp_value_area_pct=0.70,
        high_impact_news_window_minutes=30,
        dxy_correlation_threshold=0.80,
        calibration_method="isotonic",
    ),
    Asset.GBPUSD: AssetConfig(
        asset=Asset.GBPUSD, pip_value=0.0001, spread_avg=0.0002, tick_size=0.00001,
        contract_size=100000.0, leverage=100.0,
        kalman_observation_covariance=0.0002, kalman_transition_covariance=0.000002,
        vpin_buckets=50, vpin_window=50, vpin_threshold=0.85,
        max_risk_per_trade_pct=0.01, kelly_fraction=0.25,
        atr_multiplier_stop=1.5,
        atr_multiplier_tp1=1.0, atr_multiplier_tp2=2.0, atr_multiplier_tp3=3.0,
        fvg_lookback=10, fvg_body_multiplier=1.3,
        vp_lookback=50, vp_value_area_pct=0.70,
        high_impact_news_window_minutes=30,
        dxy_correlation_threshold=0.75,
        calibration_method="isotonic",
    ),
}


# =============================================================================
# LAYER 0 -- FUNDAMENTAL CONTEXT (News + DXY + Session)
# =============================================================================

class FundamentalContext:
    """
    Layer 0: Fundamental and macro context.
    - Economic calendar impact scoring
    - Timezone-based regime detection  
    - DXY correlation proxy
    """
    
    def __init__(self, config: AssetConfig):
        self.config = config
        self.high_impact_events = {
            Asset.XAUUSD: ["NFP", "CPI", "FOMC", "PPI", "GDP", "Retail Sales", 
                          "ISM Manufacturing", "Fed Chair Speech", "Geopolitical"],
            Asset.BTCUSD: ["FOMC", "CPI", "SEC Decision", "ETF Approval", 
                          "Exchange Hack", "Regulatory News"],
            Asset.EURUSD: ["ECB Rate Decision", "NFP", "CPI", "FOMC", 
                          "Eurozone GDP", "PMI", "Fed Chair Speech", "Lagarde Speech"],
            Asset.GBPUSD: ["BOE Rate Decision", "NFP", "CPI", "FOMC", 
                          "UK GDP", "UK PMI", "Fed Chair Speech", "Bailey Speech"],
        }
    
    def get_timezone_regime(self, timestamp: datetime) -> str:
        hour = timestamp.hour
        if self.config.london_start <= hour < self.config.ny_start:
            return "london"
        elif self.config.ny_start <= hour <= self.config.ny_end:
            return "ny"
        elif self.config.asia_start <= hour < self.config.asia_end:
            return "asia"
        else:
            return "overlap" if (self.config.london_start <= hour <= self.config.london_end and 
                                 self.config.ny_start <= hour <= self.config.ny_end) else "transition"
    
    def compute_news_impact_score(self, timestamp: datetime, 
                                   news_events: Optional[List[Dict]] = None) -> float:
        if news_events is None:
            return 0.0
        score = 0.0
        window = timedelta(minutes=self.config.high_impact_news_window_minutes)
        for event in news_events:
            event_time = pd.to_datetime(event['time'])
            if abs((timestamp - event_time).total_seconds()) < window.total_seconds():
                if event.get('asset', self.config.asset.value) == self.config.asset.value:
                    impact = event.get('impact', 'low')
                    if impact == 'high': score += 0.5
                    elif impact == 'medium': score += 0.25
                    elif impact == 'low': score += 0.1
        return min(score, 1.0)
    
    def compute_dxy_correlation_proxy(self, df: pd.DataFrame,
                                       dxy_series: Optional[pd.Series] = None) -> pd.Series:
        if dxy_series is not None and len(dxy_series) == len(df):
            corr = df['close'].pct_change().rolling(50).corr(dxy_series.pct_change())
            return corr.fillna(0)
        else:
            returns = df['close'].pct_change()
            proxy = -returns.rolling(20).mean() * 10
            return proxy.fillna(0)
    
    def generate_context_features(self, df: pd.DataFrame,
                                   news_events: Optional[List[Dict]] = None,
                                   dxy_series: Optional[pd.Series] = None) -> pd.DataFrame:
        result = df.copy()
        result['timezone_regime'] = [self.get_timezone_regime(ts) for ts in result.index]
        result['is_london'] = (result['timezone_regime'] == 'london').astype(int)
        result['is_ny'] = (result['timezone_regime'] == 'ny').astype(int)
        result['is_overlap'] = (result['timezone_regime'] == 'overlap').astype(int)
        
        if news_events:
            result['news_impact'] = [self.compute_news_impact_score(ts, news_events) 
                                     for ts in result.index]
        else:
            result['news_impact'] = 0.0
        
        result['dxy_correlation'] = self.compute_dxy_correlation_proxy(result, dxy_series)
        result['dxy_aligned'] = (result['dxy_correlation'].abs() > self.config.dxy_correlation_threshold).astype(int)
        
        result['session_vol'] = result.groupby('timezone_regime')['close'].transform(
            lambda x: x.pct_change().rolling(20).std()
        ).fillna(0)
        return result


# =============================================================================
# LAYER 1 -- MICROSTRUCTURE CLEANING
# =============================================================================

class MicrostructureCleaner:
    def __init__(self, config: AssetConfig):
        self.config = config
        self.kalman = None
        self._init_kalman()

    def _init_kalman(self):
        if not PYKALMAN_AVAILABLE:
            return
        self.kalman = KalmanFilter(
            transition_matrices=[1], observation_matrices=[1],
            initial_state_mean=0, initial_state_covariance=1,
            observation_covariance=self.config.kalman_observation_covariance,
            transition_covariance=self.config.kalman_transition_covariance,
        )

    def compute_smart_price(self, bid, ask, bid_vol=1.0, ask_vol=1.0):
        total = np.asarray(bid_vol) + np.asarray(ask_vol)
        bid = np.asarray(bid); ask = np.asarray(ask)
        bv = np.asarray(bid_vol); av = np.asarray(ask_vol)
        out = np.where(total == 0, (bid + ask) / 2, 
                       (bid * av + ask * bv) / np.where(total == 0, 1, total))
        return out

    def apply_kalman_filter(self, prices: np.ndarray) -> np.ndarray:
        if not PYKALMAN_AVAILABLE or self.kalman is None:
            return pd.Series(prices).ewm(span=5).mean().values
        s = pd.Series(prices).ffill().bfill()
        if s.empty:
            return prices
        self.kalman.initial_state_mean = float(s.iloc[0])
        try:
            means, _ = self.kalman.filter(s.values)
            return means.flatten()
        except Exception:
            return pd.Series(prices).ewm(span=5).mean().values

    def compute_spread_filter(self, spread, atr):
        if atr is None or (isinstance(atr, float) and atr <= 0):
            return False
        return spread > (atr * 0.3)

    def clean_ohlcv(self, df: pd.DataFrame, has_l2_data: bool = False) -> pd.DataFrame:
        result = df.copy()
        if has_l2_data and "bid" in df.columns and "ask" in df.columns:
            bid_vol = df.get("bid_vol", pd.Series(1.0, index=df.index))
            ask_vol = df.get("ask_vol", pd.Series(1.0, index=df.index))
            result["smart_price"] = self.compute_smart_price(
                df["bid"].values, df["ask"].values, bid_vol.values, ask_vol.values)
            result["spread"] = df["ask"] - df["bid"]
        else:
            result["smart_price"] = (df["high"] + df["low"]) / 2
            typical = (df["high"] + df["low"] + df["close"]) / 3
            result["spread"] = (df["close"] - typical).abs() * 2
            result["spread"] = result["spread"].clip(lower=self.config.spread_avg)
        
        result["kalman_price"] = self.apply_kalman_filter(result["smart_price"].values)
        result["atr_14"] = self._compute_atr(result, 14)
        result["spread_filter_active"] = result.apply(
            lambda r: self.compute_spread_filter(r["spread"], r["atr_14"]), axis=1)
        return result

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()


# =============================================================================
# LAYER 2 -- MARKET STRUCTURE (FVG only, lightweight)
# =============================================================================

class MarketStructureAnalyzer:
    """Lightweight FVG detection for stop placement only."""
    
    def __init__(self, config: AssetConfig):
        self.config = config
    
    def detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect Fair Value Gaps using 3-candle pattern."""
        result = df.copy()
        n = len(result)
        
        fvg_bullish = np.zeros(n, dtype=bool)
        fvg_bearish = np.zeros(n, dtype=bool)
        fvg_bull_start = np.full(n, np.nan)
        fvg_bull_end = np.full(n, np.nan)
        fvg_bear_start = np.full(n, np.nan)
        fvg_bear_end = np.full(n, np.nan)
        
        highs = result['high'].values
        lows = result['low'].values
        opens = result['open'].values
        closes = result['close'].values
        
        for i in range(2, n):
            start_idx = max(0, i - self.config.fvg_lookback)
            bodies = np.abs(closes[start_idx:i] - opens[start_idx:i])
            avg_body = np.mean(bodies) if len(bodies) > 0 else 0.001
            if avg_body == 0:
                avg_body = 0.001
            
            middle_body = abs(closes[i-1] - opens[i-1])
            
            if lows[i] > highs[i-2] and middle_body > avg_body * self.config.fvg_body_multiplier:
                fvg_bullish[i] = True
                fvg_bull_start[i] = highs[i-2]
                fvg_bull_end[i] = lows[i]
            elif highs[i] < lows[i-2] and middle_body > avg_body * self.config.fvg_body_multiplier:
                fvg_bearish[i] = True
                fvg_bear_start[i] = lows[i-2]
                fvg_bear_end[i] = highs[i]
        
        result['fvg_bullish'] = fvg_bullish
        result['fvg_bearish'] = fvg_bearish
        result['fvg_bull_start'] = fvg_bull_start
        result['fvg_bull_end'] = fvg_bull_end
        result['fvg_bear_start'] = fvg_bear_start
        result['fvg_bear_end'] = fvg_bear_end
        return result
    
    def get_nearest_fvg(self, df: pd.DataFrame, direction: TradeDirection) -> Optional[float]:
        """Get nearest FVG level for stop placement."""
        latest = df.iloc[-1]
        if direction == TradeDirection.LONG and latest.get('fvg_bullish', False):
            return latest.get('fvg_bull_start')
        elif direction == TradeDirection.SHORT and latest.get('fvg_bearish', False):
            return latest.get('fvg_bear_start')
        return None
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.detect_fvg(df)


# =============================================================================
# LAYER 3 -- FLOW & TOXICITY (Enhanced with CVD + Volume Profile)
# =============================================================================

class FlowAnalyzer:
    """
    Enhanced Layer 3:
    - VPIN (existing)
    - Cumulative Volume Delta (CVD) + Divergence
    - Volume Profile (POC, Value Area, Absorption)
    - Signed Volume (existing)
    - Realized Volatility (existing)
    """
    
    def __init__(self, config: AssetConfig):
        self.config = config
    
    def compute_vpin(self, df: pd.DataFrame) -> pd.Series:
        """Volume-Synchronized PIN -- 1-50-50 standard."""
        price_change = df["close"].diff().fillna(0).values
        volume = df["volume"].values
        
        buy_vol = np.where(price_change > 0, volume, 0.0)
        sell_vol = np.where(price_change < 0, volume, 0.0)
        zero = price_change == 0
        buy_vol[zero] = volume[zero] * 0.5
        sell_vol[zero] = volume[zero] * 0.5
        
        total_v = (buy_vol + sell_vol).sum()
        n_buckets = self.config.vpin_buckets
        if total_v <= 0 or n_buckets <= 0:
            return pd.Series(0.0, index=df.index)
        
        bucket_size = total_v / n_buckets
        bucket_buy = bucket_sell = bucket_total = 0.0
        bucket_at_bar = []
        
        for i in range(len(df)):
            bucket_buy += buy_vol[i]
            bucket_sell += sell_vol[i]
            bucket_total += buy_vol[i] + sell_vol[i]
            if bucket_total >= bucket_size:
                vpin_val = abs(bucket_buy - bucket_sell) / bucket_total if bucket_total > 0 else 0.0
                bucket_at_bar.append((i, vpin_val))
                bucket_buy = bucket_sell = bucket_total = 0.0
        
        out = pd.Series(np.nan, index=df.index, dtype=float)
        for bar_idx, val in bucket_at_bar:
            out.iloc[bar_idx] = val
        out = out.ffill().fillna(0.0)
        return out.rolling(window=self.config.vpin_window, min_periods=1).mean().clip(0, 1)
    
    def compute_cvd(self, df: pd.DataFrame) -> pd.Series:
        """Cumulative Volume Delta."""
        price_change = df["close"].diff().fillna(0)
        volume = df["volume"]
        
        buy_vol = pd.Series(np.where(price_change > 0, volume, 0.0), index=df.index)
        sell_vol = pd.Series(np.where(price_change < 0, volume, 0.0), index=df.index)
        zero = price_change == 0
        buy_vol[zero] = volume[zero] * 0.5
        sell_vol[zero] = volume[zero] * 0.5
        
        delta = buy_vol - sell_vol
        return delta.cumsum()
    
    def compute_cvd_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect CVD-Price divergence -- key reversal signal."""
        result = df.copy()
        cvd = result['cvd'] if 'cvd' in result.columns else self.compute_cvd(result)
        
        price_low = result['low'].rolling(10).min()
        price_high = result['high'].rolling(10).max()
        cvd_low = cvd.rolling(10).min()
        cvd_high = cvd.rolling(10).max()
        
        result['cvd_bull_div'] = (
            (result['low'] < price_low.shift(1)) & 
            (cvd > cvd_low.shift(1))
        ).astype(int)
        
        result['cvd_bear_div'] = (
            (result['high'] > price_high.shift(1)) & 
            (cvd < cvd_high.shift(1))
        ).astype(int)
        
        result['cvd'] = cvd
        result['cvd_slope'] = cvd.diff(5).fillna(0)
        return result
    
    def compute_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling Volume Profile: POC and Value Area.
        Uses histogram approach over lookback window.
        """
        result = df.copy()
        n = len(result)
        lookback = self.config.vp_lookback
        
        poc = np.full(n, np.nan)
        value_area_high = np.full(n, np.nan)
        value_area_low = np.full(n, np.nan)
        
        for i in range(lookback, n):
            window = result.iloc[i-lookback:i]
            prices = window['close'].values
            volumes = window['volume'].values
            
            if len(prices) < 10:
                continue
            
            n_bins = min(20, len(prices) // 2)
            hist, bin_edges = np.histogram(prices, bins=n_bins, weights=volumes)
            
            # POC: highest volume bin
            poc_idx = np.argmax(hist)
            poc[i] = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2
            
            # Value Area: 70% of volume around POC
            total_vol = np.sum(hist)
            target_vol = total_vol * self.config.vp_value_area_pct
            
            poc_bin = poc_idx
            current_vol = hist[poc_bin]
            low_bin = high_bin = poc_bin
            
            while current_vol < target_vol and (low_bin > 0 or high_bin < len(hist) - 1):
                low_vol = hist[low_bin - 1] if low_bin > 0 else 0
                high_vol = hist[high_bin + 1] if high_bin < len(hist) - 1 else 0
                
                if low_vol > high_vol and low_bin > 0:
                    low_bin -= 1
                    current_vol += low_vol
                elif high_bin < len(hist) - 1:
                    high_bin += 1
                    current_vol += high_vol
                else:
                    break
            
            value_area_low[i] = bin_edges[low_bin]
            value_area_high[i] = bin_edges[high_bin + 1]
        
        result['poc'] = pd.Series(poc, index=result.index).ffill().bfill()
        result['value_area_high'] = pd.Series(value_area_high, index=result.index).ffill().bfill()
        result['value_area_low'] = pd.Series(value_area_low, index=result.index).ffill().bfill()
        result['in_value_area'] = (
            (result['close'] >= result['value_area_low']) & 
            (result['close'] <= result['value_area_high'])
        ).astype(int)
        result['distance_to_poc'] = (result['close'] - result['poc']) / result['close']
        return result
    
    def detect_absorption(self, df: pd.DataFrame) -> pd.Series:
        """Detect absorption -- large volume with minimal price movement."""
        volume = df['volume']
        price_range = df['high'] - df['low']
        
        vol_ma = volume.rolling(20).mean()
        range_ma = price_range.rolling(20).mean()
        
        absorption = ((volume > vol_ma * 2) & (price_range < range_ma * 0.5)).astype(int)
        return absorption
    
    def compute_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all flow and toxicity features."""
        result = df.copy()
        result["vpin"] = self.compute_vpin(df)
        result["cvd"] = self.compute_cvd(df)
        result = self.compute_cvd_divergence(result)
        result = self.compute_volume_profile(result)
        result["absorption"] = self.detect_absorption(result)
        
        result["signed_volume"] = self.compute_signed_volume(df, "bulk_volume")
        result["signed_volume_momentum"] = result["signed_volume"].rolling(5).mean()
        result["realized_vol_5m"] = self.compute_realized_volatility(df["close"], 5)
        result["realized_vol_20m"] = self.compute_realized_volatility(df["close"], 20)
        result["volume_ma_20"] = df["volume"].rolling(20).mean()
        result["volume_ma_ratio"] = (
            df["volume"] / result["volume_ma_20"].replace(0, np.nan)
        ).fillna(1)
        
        if "bid_vol" in df.columns and "ask_vol" in df.columns:
            result["ob_imbalance"] = self.compute_order_book_imbalance(
                df["bid_vol"], df["ask_vol"])
        else:
            denom = df["volume"].rolling(5).mean().replace(0, np.nan)
            result["ob_imbalance"] = (
                result["signed_volume_momentum"] / denom
            ).fillna(0).clip(-1, 1)
        return result
    
    @staticmethod
    def compute_order_book_imbalance(bid_vol, ask_vol):
        total = bid_vol + ask_vol
        return ((bid_vol - ask_vol) / total.replace(0, np.nan)).fillna(0).clip(-1, 1)
    
    @staticmethod
    def compute_signed_volume(df: pd.DataFrame, method: str = "bulk_volume") -> pd.Series:
        if method == "tick_rule":
            change = df["close"].diff()
            signed = pd.Series(0.0, index=df.index)
            signed[change > 0] = df.loc[change > 0, "volume"]
            signed[change < 0] = -df.loc[change < 0, "volume"]
            return signed
        bar_range = (df["high"] - df["low"]).replace(0, np.nan)
        position = ((df["close"] - df["low"]) / bar_range).fillna(0.5).clip(0, 1)
        return df["volume"] * (2 * position - 1)
    
    @staticmethod
    def compute_realized_volatility(prices: pd.Series, window: int = 5) -> pd.Series:
        rets = prices.pct_change().fillna(0)
        return (rets.rolling(window).std() * np.sqrt(525_600)).fillna(0)


# =============================================================================
# LAYER 4 -- MULTI-TIMEFRAME CONFLUENCE
# =============================================================================

class MTFConfluenceEngine:
    """Multi-Timeframe Confluence: 1m/5m/15m signal alignment."""
    
    def __init__(self):
        self.mtf_weights = {'1m': 0.5, '5m': 0.3, '15m': 0.2}
    
    def aggregate_structure(self, df_1m: pd.DataFrame, 
                           df_5m: Optional[pd.DataFrame] = None,
                           df_15m: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        result = df_1m.copy()
        
        if df_5m is None or df_15m is None:
            result['mtf_confluence'] = 'neutral'
            result['mtf_score'] = 0.0
            return result
        
        result['tf5_trend'] = self._align_tf(result, df_5m, 'close')
        result['tf15_trend'] = self._align_tf(result, df_15m, 'close')
        
        result['trend_1m'] = np.where(
            result['close'] > result['close'].ewm(span=20).mean(), 1, -1)
        result['trend_5m'] = np.where(
            result['tf5_trend'] > pd.Series(result['tf5_trend']).ewm(span=20).mean(), 1, -1)
        result['trend_15m'] = np.where(
            result['tf15_trend'] > pd.Series(result['tf15_trend']).ewm(span=20).mean(), 1, -1)
        
        confluence = (result['trend_1m'] * self.mtf_weights['1m'] + 
                      result['trend_5m'] * self.mtf_weights['5m'] + 
                      result['trend_15m'] * self.mtf_weights['15m'])
        
        result['mtf_score'] = confluence
        
        conditions = [
            confluence > 0.6, confluence > 0.2,
            confluence < -0.2, confluence < -0.6,
        ]
        choices = ['strong_bullish', 'bullish', 'bearish', 'strong_bearish']
        result['mtf_confluence'] = np.select(conditions, choices, default='neutral')
        return result
    
    @staticmethod
    def _align_tf(df_target: pd.DataFrame, df_source: pd.DataFrame, col: str) -> pd.Series:
        source = df_source[[col]].copy()
        source = source.reindex(df_target.index, method='ffill')
        return source[col]


# =============================================================================
# LAYER 5 -- ML SIGNAL MODELS (Enhanced with Calibration)
# =============================================================================

class LorentzianClassifier:
    """Vectorized Lorentzian KNN."""
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.train_data = None
        self.train_labels = None

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
        rs = gain / loss.replace(0, np.nan)
        feats["rsi_7"] = (100 - 100 / (1 + rs)).fillna(50)
        
        typical = (df["high"] + df["low"] + df["close"]) / 3
        flow = typical * df["volume"]
        sign = np.where(typical > typical.shift(1), 1, -1)
        signed = pd.Series(flow * sign, index=df.index)
        pos = signed.where(signed > 0, 0).rolling(7).sum()
        neg = (-signed.where(signed < 0, 0)).rolling(7).sum()
        ratio = pos / neg.replace(0, np.nan)
        feats["mfi_7"] = (100 - 100 / (1 + ratio)).fillna(50)
        
        feats["roc_3"] = ((df["close"] - df["close"].shift(3)) / df["close"].shift(3) * 100).fillna(0)
        
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift()).abs()
        tr3 = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        feats["volatility_gate"] = (tr / atr.replace(0, np.nan)).fillna(1).clip(0, 5)
        return feats

    def fit(self, df: pd.DataFrame, labels: Optional[np.ndarray] = None, target_horizon: int = 3):
        feats = self._compute_features(df).dropna()
        if labels is None:
            fwd = df["close"].shift(-target_horizon) / df["close"] - 1
            labels = pd.Series(0, index=df.index, dtype=int)
            labels[fwd > 0.001] = 2
            labels[(fwd <= 0.001) & (fwd > 0.0003)] = 1
            labels[(fwd >= -0.001) & (fwd < -0.0003)] = -1
            labels[fwd < -0.001] = -2
        else:
            labels = pd.Series(labels, index=df.index)
        aligned = labels.loc[feats.index].values
        X = feats.values
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        self.train_data = X.astype(np.float32)
        self.train_labels = aligned.astype(np.int8)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.train_data is None:
            return pd.Series(0, index=df.index)
        feats = self._compute_features(df).dropna()
        if feats.empty:
            return pd.Series(0, index=df.index)
        X = feats.values
        if self.scaler is not None:
            X = self.scaler.transform(X)
        X32 = X.astype(np.float32)
        M, F = X32.shape
        N = self.train_data.shape[0]
        chunk = max(1, 1_000_000 // max(N, 1))
        preds = np.zeros(M, dtype=np.int8)
        for start in range(0, M, chunk):
            stop = min(start + chunk, M)
            diff = np.abs(X32[start:stop, None, :] - self.train_data[None, :, :])
            dist = np.log1p(diff).sum(axis=2)
            k = min(self.n_neighbors, N)
            idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
            votes = self.train_labels[idx]
            avg = votes.mean(axis=1)
            local = np.zeros(stop - start, dtype=np.int8)
            local[avg > 0.5] = 2
            local[(avg <= 0.5) & (avg > 0.1)] = 1
            local[(avg >= -0.5) & (avg < -0.1)] = -1
            local[avg < -0.5] = -2
            preds[start:stop] = local
        return pd.Series(preds, index=feats.index)


class XGBoostSignalModel:
    """XGBoost with Isotonic Probability Calibration."""
    def __init__(self, n_estimators: int = 200, max_depth: int = 5,
                 learning_rate: float = 0.05, calibration_method: str = "isotonic"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.calibration_method = calibration_method
        self.model = None
        self.calibrated_model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_cols: Optional[List[str]] = None

    @staticmethod
    def _rsi(p: pd.Series, period: int = 14) -> pd.Series:
        delta = p.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - 100 / (1 + rs)).fillna(50)

    @staticmethod
    def _macd(p: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        return p.ewm(span=fast).mean() - p.ewm(span=slow).mean()

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        f["returns_1"] = df["close"].pct_change()
        f["returns_3"] = df["close"].pct_change(3)
        f["returns_5"] = df["close"].pct_change(5)
        f["smart_return"] = (df["smart_price"].pct_change() if "smart_price" in df.columns else f["returns_1"])
        f["smart_vs_close"] = ((df["smart_price"] - df["close"]) / df["close"] if "smart_price" in df.columns else 0)
        f["kalman_return"] = (df["kalman_price"].pct_change() if "kalman_price" in df.columns else f["returns_1"])
        f["kalman_vs_close"] = ((df["kalman_price"] - df["close"]) / df["close"] if "kalman_price" in df.columns else 0)
        f["rsi_7"] = self._rsi(df["close"], 7)
        f["rsi_14"] = self._rsi(df["close"], 14)
        f["macd"] = self._macd(df["close"])
        f["macd_signal"] = f["macd"].ewm(span=9).mean()
        f["macd_hist"] = f["macd"] - f["macd_signal"]
        
        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        f["bb_position"] = ((df["close"] - bb_mid) / bb_std.replace(0, np.nan)).fillna(0)
        f["bb_width"] = (bb_std / bb_mid.replace(0, np.nan)).fillna(0)
        
        f["signed_vol_ma5"] = df["signed_volume"].rolling(5).mean() if "signed_volume" in df.columns else 0
        f["signed_vol_ma10"] = df["signed_volume"].rolling(10).mean() if "signed_volume" in df.columns else 0
        f["vol_intensity"] = df["volume_ma_ratio"] if "volume_ma_ratio" in df.columns else (
            df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)).fillna(1)
        f["ob_imbalance"] = df["ob_imbalance"] if "ob_imbalance" in df.columns else 0
        f["vpin"] = df["vpin"] if "vpin" in df.columns else 0
        f["vpin_high"] = ((df["vpin"] > df["vpin"].rolling(100).quantile(0.9)).astype(int) if "vpin" in df.columns else 0)
        
        # NEW: CVD and Volume Profile features
        f["cvd_slope"] = df["cvd_slope"] if "cvd_slope" in df.columns else 0
        f["cvd_bull_div"] = df["cvd_bull_div"] if "cvd_bull_div" in df.columns else 0
        f["cvd_bear_div"] = df["cvd_bear_div"] if "cvd_bear_div" in df.columns else 0
        f["distance_to_poc"] = df["distance_to_poc"] if "distance_to_poc" in df.columns else 0
        f["in_value_area"] = df["in_value_area"] if "in_value_area" in df.columns else 1
        f["absorption"] = df["absorption"] if "absorption" in df.columns else 0
        
        if "realized_vol_5m" in df.columns:
            f["rv_5m"] = df["realized_vol_5m"]
            f["rv_20m"] = df["realized_vol_20m"]
        else:
            r = df["close"].pct_change().fillna(0)
            f["rv_5m"] = r.rolling(5).std() * np.sqrt(525_600)
            f["rv_20m"] = r.rolling(20).std() * np.sqrt(525_600)
        f["vol_regime"] = (f["rv_5m"] > f["rv_5m"].rolling(50).mean()).astype(int)
        
        for lag in (1, 2, 3):
            f[f"return_lag_{lag}"] = f["returns_1"].shift(lag)
        
        f["rsi_x_vol"] = f["rsi_7"] * f["vol_intensity"]
        f["ob_x_vol"] = f["ob_imbalance"] * f["vol_intensity"]
        f["cvd_x_vol"] = f["cvd_slope"] * f["vol_intensity"]
        
        return f.replace([np.inf, -np.inf], 0).fillna(0)

    def fit(self, df: pd.DataFrame, target_horizon: int = 3):
        feats = self._engineer(df)
        fwd = df["close"].shift(-target_horizon) / df["close"] - 1
        target = pd.Series(0, index=df.index, dtype=int)
        target[fwd > 0.001] = 2
        target[(fwd <= 0.001) & (fwd > 0.0003)] = 1
        target[(fwd >= -0.001) & (fwd < -0.0003)] = -1
        target[fwd < -0.001] = -2
        
        af = feats.loc[target.index].dropna()
        ay = target.loc[af.index]
        self.feature_cols = af.columns.tolist()
        X = self.scaler.fit_transform(af) if self.scaler is not None else af.values
        y = ay.values + 2
        
        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, objective="multi:softprob",
                num_class=5, eval_metric="mlogloss", random_state=42,
                tree_method="hist", reg_lambda=1.0, min_child_weight=10,
            )
            self.model.fit(X, y)
            
            # Probability calibration with Isotonic Regression
            if SKLEARN_AVAILABLE and self.calibration_method in ["isotonic", "sigmoid"]:
                self.calibrated_model = CalibratedClassifierCV(
                    self.model, method=self.calibration_method, cv=3
                )
                self.calibrated_model.fit(X, y)
        elif SKLEARN_AVAILABLE:
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, random_state=42)
            self.model.fit(X, y)
        else:
            raise ImportError("Need xgboost or scikit-learn")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None or self.feature_cols is None:
            return pd.Series(0, index=df.index)
        feats = self._engineer(df)
        for c in self.feature_cols:
            if c not in feats.columns:
                feats[c] = 0
        feats = feats[self.feature_cols]
        X = self.scaler.transform(feats) if self.scaler is not None else feats.values
        return pd.Series(self.model.predict(X) - 2, index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get calibrated probabilities if available."""
        if self.calibrated_model is not None:
            feats = self._engineer(df)
            for c in self.feature_cols:
                if c not in feats.columns:
                    feats[c] = 0
            feats = feats[self.feature_cols]
            X = self.scaler.transform(feats) if self.scaler is not None else feats.values
            proba = self.calibrated_model.predict_proba(X)
            return pd.DataFrame(proba, index=df.index, columns=[-2, -1, 0, 1, 2])
        elif self.model is not None:
            feats = self._engineer(df)
            for c in self.feature_cols:
                if c not in feats.columns:
                    feats[c] = 0
            feats = feats[self.feature_cols]
            X = self.scaler.transform(feats) if self.scaler is not None else feats.values
            proba = self.model.predict_proba(X)
            return pd.DataFrame(proba, index=df.index, columns=[-2, -1, 0, 1, 2])
        else:
            return pd.DataFrame(0.2, index=df.index, columns=[-2, -1, 0, 1, 2])


class HMMRegimeDetector:
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None

    def fit(self, df: pd.DataFrame):
        if not HMMLEARN_AVAILABLE or self.scaler is None:
            return
        rets = df["close"].pct_change().fillna(0).values.reshape(-1, 1)
        vol = pd.Series(rets.flatten()).rolling(10).std().fillna(0).values.reshape(-1, 1)
        rv = df.get('realized_vol_5m', pd.Series(0, index=df.index)).values.reshape(-1, 1)
        feats = np.hstack([rets, vol, rv])
        feats = feats[~np.isnan(feats).any(axis=1)]
        if len(feats) < 100:
            return
        Xs = self.scaler.fit_transform(feats)
        self.model = GaussianHMM(
            n_components=self.n_regimes, covariance_type="full",
            n_iter=100, random_state=42)
        try:
            self.model.fit(Xs)
        except Exception as e:
            print(f"[HMM] fit failed: {e}")
            self.model = None

    def predict_regime(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None or self.scaler is None:
            return pd.Series(0, index=df.index)
        rets = df["close"].pct_change().fillna(0).values.reshape(-1, 1)
        vol = pd.Series(rets.flatten()).rolling(10).std().fillna(0).values.reshape(-1, 1)
        rv = df.get('realized_vol_5m', pd.Series(0, index=df.index)).values.reshape(-1, 1)
        feats = np.hstack([rets, vol, rv])
        feats = self.scaler.transform(feats)
        try:
            return pd.Series(self.model.predict(feats), index=df.index)
        except Exception:
            return pd.Series(0, index=df.index)


# =============================================================================
# LAYER 6 -- RISK + EXECUTION (Enhanced with Kelly + FVG Stops)
# =============================================================================

@dataclass
class Trade:
    entry_time: datetime
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    position_size: float
    asset: Asset
    fvg_stop: Optional[float] = None
    mae: float = 0.0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "open"
    tp_level_hit: int = 0


class RiskManager:
    """Enhanced Risk Manager with Kelly Criterion + FVG-based stops."""
    
    def __init__(self, config: AssetConfig):
        self.config = config
        self.equity = 10_000.0
        self.initial_equity = 10_000.0
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[float] = [10_000.0]
        self.trade_history_pnls: List[float] = []

    def set_equity(self, equity: float):
        self.equity = equity
        self.initial_equity = equity

    def kelly_position_size(self, win_rate: float, avg_win: float, avg_loss: float,
                            entry: float, stop: float) -> float:
        """Quarter-Kelly position sizing for optimal growth."""
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return self.calculate_position_size(entry, stop)
        
        b = avg_win / avg_loss
        kelly = (win_rate * b - (1 - win_rate)) / b
        kelly = max(0, min(kelly, 0.5))
        fraction = kelly * self.config.kelly_fraction
        
        risk_amt = self.equity * self.config.max_risk_per_trade_pct * (1 + fraction * 4)
        price_risk = abs(entry - stop)
        if price_risk == 0:
            return 0
        return risk_amt / (price_risk * self.config.contract_size / self.config.leverage)

    def calculate_position_size(self, entry: float, stop: float) -> float:
        risk_amt = self.equity * self.config.max_risk_per_trade_pct
        price_risk = abs(entry - stop)
        if price_risk == 0:
            return 0
        return risk_amt / (price_risk * self.config.contract_size / self.config.leverage)

    def calculate_levels(self, entry: float, direction: TradeDirection, atr: float,
                         fvg_level: Optional[float] = None) -> Tuple[float, float, float, float, Optional[float]]:
        if direction == TradeDirection.LONG:
            sl = entry - atr * self.config.atr_multiplier_stop
            tp1 = entry + atr * self.config.atr_multiplier_tp1
            tp2 = entry + atr * self.config.atr_multiplier_tp2
            tp3 = entry + atr * self.config.atr_multiplier_tp3
            fvg_sl = fvg_level if fvg_level and fvg_level < entry else None
        else:
            sl = entry + atr * self.config.atr_multiplier_stop
            tp1 = entry - atr * self.config.atr_multiplier_tp1
            tp2 = entry - atr * self.config.atr_multiplier_tp2
            tp3 = entry - atr * self.config.atr_multiplier_tp3
            fvg_sl = fvg_level if fvg_level and fvg_level > entry else None
        return sl, tp1, tp2, tp3, fvg_sl

    def open_trade(self, ts: datetime, direction: TradeDirection,
                    entry: float, atr: float,
                    fvg_level: Optional[float] = None,
                    use_kelly: bool = True) -> Optional[Trade]:
        if direction == TradeDirection.FLAT:
            return None
        
        sl, tp1, tp2, tp3, fvg_sl = self.calculate_levels(entry, direction, atr, fvg_level)
        
        # Use tighter of ATR stop or FVG stop
        final_sl = sl
        if fvg_sl is not None:
            if direction == TradeDirection.LONG and fvg_sl > sl:
                final_sl = fvg_sl
            elif direction == TradeDirection.SHORT and fvg_sl < sl:
                final_sl = fvg_sl
        
        if use_kelly and len(self.trade_history_pnls) >= 20:
            wins = [p for p in self.trade_history_pnls if p > 0]
            losses = [p for p in self.trade_history_pnls if p <= 0]
            win_rate = len(wins) / len(self.trade_history_pnls) if self.trade_history_pnls else 0.5
            avg_win = np.mean(wins) if wins else 1
            avg_loss = abs(np.mean(losses)) if losses else 1
            size = self.kelly_position_size(win_rate, avg_win, avg_loss, entry, final_sl)
        else:
            size = self.calculate_position_size(entry, final_sl)
        
        if size <= 0:
            return None
        
        t = Trade(
            entry_time=ts, direction=direction, entry_price=entry,
            stop_loss=final_sl, take_profit_1=tp1, take_profit_2=tp2,
            take_profit_3=tp3, position_size=size, asset=self.config.asset,
            fvg_stop=fvg_sl)
        self.open_trades.append(t)
        return t

    def update_trades(self, ts: datetime, high: float, low: float, close: float):
        for t in self.open_trades[:]:
            if t.status != "open":
                continue
            if t.direction == TradeDirection.LONG:
                t.mae = max(t.mae, t.entry_price - low)
                if low <= t.stop_loss:
                    self._close(t, ts, t.stop_loss, "stopped"); continue
                if t.tp_level_hit == 0 and high >= t.take_profit_1: t.tp_level_hit = 1
                if t.tp_level_hit == 1 and high >= t.take_profit_2: t.tp_level_hit = 2
                if t.tp_level_hit == 2 and high >= t.take_profit_3:
                    self._close(t, ts, t.take_profit_3, "closed_tp3"); continue
            else:
                t.mae = max(t.mae, high - t.entry_price)
                if high >= t.stop_loss:
                    self._close(t, ts, t.stop_loss, "stopped"); continue
                if t.tp_level_hit == 0 and low <= t.take_profit_1: t.tp_level_hit = 1
                if t.tp_level_hit == 1 and low <= t.take_profit_2: t.tp_level_hit = 2
                if t.tp_level_hit == 2 and low <= t.take_profit_3:
                    self._close(t, ts, t.take_profit_3, "closed_tp3"); continue

    def _close(self, t: Trade, ts: datetime, exit_price: float, status: str):
        t.exit_time = ts; t.exit_price = exit_price
        if t.direction == TradeDirection.LONG:
            t.pnl = (exit_price - t.entry_price) * t.position_size * self.config.contract_size
        else:
            t.pnl = (t.entry_price - exit_price) * t.position_size * self.config.contract_size
        t.status = status
        self.closed_trades.append(t)
        self.trade_history_pnls.append(t.pnl)
        if t in self.open_trades:
            self.open_trades.remove(t)
        self.equity += t.pnl
        self.equity_curve.append(self.equity)

    def close_all(self, ts: datetime, price: float):
        for t in self.open_trades[:]:
            self._close(t, ts, price, "closed_manual")

    def get_stats(self) -> Dict:
        if not self.closed_trades:
            return {"total_trades": 0, "equity_curve": self.equity_curve}
        pnls = [t.pnl for t in self.closed_trades if t.pnl is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        total = len(pnls)
        win_rate = len(wins) / total if total else 0
        
        true_positives = len([t for t in self.closed_trades if t.pnl and t.pnl > 0 and t.direction == TradeDirection.LONG])
        false_positives = len([t for t in self.closed_trades if t.pnl and t.pnl <= 0 and t.direction == TradeDirection.LONG])
        precision_long = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        true_negatives = len([t for t in self.closed_trades if t.pnl and t.pnl > 0 and t.direction == TradeDirection.SHORT])
        false_negatives = len([t for t in self.closed_trades if t.pnl and t.pnl <= 0 and t.direction == TradeDirection.SHORT])
        precision_short = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0
        
        return {
            "total_trades": len(self.closed_trades),
            "win_rate": float(win_rate),
            "precision_long": float(precision_long),
            "precision_short": float(precision_short),
            "avg_win": float(np.mean(wins)) if wins else 0.0,
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "total_pnl": float(sum(pnls)),
            "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf"),
            "max_drawdown": self._max_dd(pnls),
            "sharpe": self._sharpe(pnls),
            "avg_mae": float(np.mean([t.mae for t in self.closed_trades])),
            "total_return": float((self.equity - self.initial_equity) / self.initial_equity),
            "equity_curve": self.equity_curve,
        }

    @staticmethod
    def _max_dd(pnls: List[float]) -> float:
        cum = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        return float(abs(dd.min())) if len(dd) else 0.0

    @staticmethod
    def _sharpe(pnls: List[float]) -> float:
        if len(pnls) < 2: return 0.0
        a = np.array(pnls)
        if a.std() == 0: return 0.0
        return float(a.mean() / a.std() * np.sqrt(252 * 24 * 60))


# =============================================================================
# ALTERNATIVE BAR SAMPLING (Tick/Volume Imbalance Bars)
# =============================================================================

class AlternativeBarSampler:
    """
    Information-driven bar sampling per Lopez de Prado AFML.
    Produces bars with more uniform information content than time bars.
    """
    
    @staticmethod
    def tick_imbalance_bars(df: pd.DataFrame, threshold: int = 100) -> pd.DataFrame:
        """
        Fixed-threshold tick imbalance bars.
        Bar forms when |Sigma b_t| >= threshold, where b_t = +1 (buy) / -1 (sell).
        """
        if len(df) < threshold:
            return df.copy()
        
        price_change = df['close'].diff().fillna(0).values
        tick_signs = np.sign(price_change)
        tick_signs[tick_signs == 0] = 1  # Neutral ticks count as buys
        
        bars = []
        cum_imbalance = 0
        bar_start = 0
        
        for i in range(len(df)):
            cum_imbalance += tick_signs[i]
            
            if abs(cum_imbalance) >= threshold:
                bar_df = df.iloc[bar_start:i+1]
                bars.append({
                    'timestamp': bar_df.index[-1],
                    'open': bar_df['open'].iloc[0],
                    'high': bar_df['high'].max(),
                    'low': bar_df['low'].min(),
                    'close': bar_df['close'].iloc[-1],
                    'volume': bar_df['volume'].sum(),
                    'tick_count': len(bar_df),
                    'imbalance': cum_imbalance,
                })
                cum_imbalance = 0
                bar_start = i + 1
        
        if not bars:
            return df.copy()
        
        return pd.DataFrame(bars).set_index('timestamp')
    
    @staticmethod
    def volume_imbalance_bars(df: pd.DataFrame, threshold: float = 50000) -> pd.DataFrame:
        """
        Fixed-threshold volume imbalance bars.
        Bar forms when |Sigma b_t * v_t| >= threshold.
        """
        if len(df) < 10:
            return df.copy()
        
        price_change = df['close'].diff().fillna(0).values
        volume = df['volume'].values
        
        buy_vol = np.where(price_change > 0, volume, 0.0)
        sell_vol = np.where(price_change < 0, volume, 0.0)
        zero = price_change == 0
        buy_vol[zero] = volume[zero] * 0.5
        sell_vol[zero] = volume[zero] * 0.5
        
        signed_vol = buy_vol - sell_vol
        
        bars = []
        cum_imbalance = 0.0
        bar_start = 0
        
        for i in range(len(df)):
            cum_imbalance += signed_vol[i]
            
            if abs(cum_imbalance) >= threshold:
                bar_df = df.iloc[bar_start:i+1]
                bars.append({
                    'timestamp': bar_df.index[-1],
                    'open': bar_df['open'].iloc[0],
                    'high': bar_df['high'].max(),
                    'low': bar_df['low'].min(),
                    'close': bar_df['close'].iloc[-1],
                    'volume': bar_df['volume'].sum(),
                    'buy_volume': buy_vol[bar_start:i+1].sum(),
                    'sell_volume': sell_vol[bar_start:i+1].sum(),
                    'imbalance': cum_imbalance,
                })
                cum_imbalance = 0.0
                bar_start = i + 1
        
        if not bars:
            return df.copy()
        
        return pd.DataFrame(bars).set_index('timestamp')


# =============================================================================
# MAIN ORCHESTRATOR -- Enhanced Precision Trading System v3.0
# =============================================================================

class EnhancedPrecisionTradingSystem:
    """
    Enhanced 1-Minute Precision Trading System v3.0.
    8 surgical enhancements, no bloat.
    """
    
    def __init__(self, asset: Asset, use_hmm: bool = True):
        self.asset = asset
        self.config = ASSET_CONFIGS[asset]
        self.use_hmm = use_hmm
        
        # Layers
        self.fundamental = FundamentalContext(self.config)
        self.cleaner = MicrostructureCleaner(self.config)
        self.structure = MarketStructureAnalyzer(self.config)
        self.flow = FlowAnalyzer(self.config)
        self.mtf = MTFConfluenceEngine()
        self.signal_model = XGBoostSignalModel(calibration_method=self.config.calibration_method)
        self.hmm = HMMRegimeDetector(n_regimes=3) if use_hmm else None
        self.risk_manager = RiskManager(self.config)
        
        self.is_trained = False
        self.data_buffer = pd.DataFrame()
        self.news_events: List[Dict] = []
        self.dxy_series: Optional[pd.Series] = None
    
    def add_news_event(self, time: datetime, event: str, impact: str, asset: Optional[str] = None):
        """Add macroeconomic news event."""
        self.news_events.append({
                        'time': time,
                        'event': event,
                        'impact': impact,
                        'asset': asset if asset else self.asset.value,
        })

    def set_dxy_data(self, dxy_series: pd.Series):
        """Set DXY reference series for correlation features."""
        self.dxy_series = dxy_series

    def _aggregate_to_higher_tf(self, df: pd.DataFrame, minutes: int) -> pd.DataFrame:
        """Aggregate 1‑minute data to a higher timeframe."""
        if df.empty:
            return df
        rule = f"{minutes}min"
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }
        # keep only OHLCV columns
        cols = ['open', 'high', 'low', 'close', 'volume']
        available = [c for c in cols if c in df.columns]
        return df[available].resample(rule).apply(agg_dict).dropna()

    def train(self, df: pd.DataFrame, target_horizon: int = 3):
        """
        Full training on historical 1‑minute data.
        Expects columns: open, high, low, close, volume, (optional bid/ask).
        """
        if len(df) < 500:
            print("Not enough data for training.")
            return

        # Store a copy for future inference references
        self.data_buffer = df.copy()

        # Microstructure cleaning
        cleaned = self.cleaner.clean_ohlcv(df)
        # Add smart_price, kalman_price, atr_14, spread to the buffer for feature engineering
        for col in ['smart_price', 'kalman_price', 'atr_14', 'spread']:
            if col in cleaned.columns:
                self.data_buffer[col] = cleaned[col]

        # Fundamental context (use internal news list and DXY)
        ctx = self.fundamental.generate_context_features(
            cleaned,
            news_events=self.news_events,
            dxy_series=self.dxy_series
        )
        # Inject context features back into the main buffer
        for col in ['news_impact', 'dxy_correlation', 'dxy_aligned',
                    'session_vol', 'timezone_regime']:
            if col in ctx.columns:
                self.data_buffer[col] = ctx[col]

        # Flow features (VPIN, CVD, Volume Profile, etc.)
        flow_feats = self.flow.compute_flow_features(self.data_buffer)
        for col in flow_feats.columns:
            self.data_buffer[col] = flow_feats[col]

        # Market structure (lightweight FVG)
        structure = self.structure.analyze(self.data_buffer)
        for col in ['fvg_bullish', 'fvg_bearish',
                    'fvg_bull_start', 'fvg_bull_end',
                    'fvg_bear_start', 'fvg_bear_end']:
            if col in structure.columns:
                self.data_buffer[col] = structure[col]

        # Multi‑timeframe confluence (5m / 15m)
        df_5m = self._aggregate_to_higher_tf(self.data_buffer, 5)
        df_15m = self._aggregate_to_higher_tf(self.data_buffer, 15)
        mtf_feats = self.mtf.aggregate_structure(self.data_buffer, df_5m, df_15m)
        for col in ['mtf_score', 'mtf_confluence']:
            if col in mtf_feats.columns:
                self.data_buffer[col] = mtf_feats[col]

        # HMM regime fitting
        if self.hmm is not None:
            self.hmm.fit(self.data_buffer)

        # ML signal training
        self.signal_model.fit(self.data_buffer, target_horizon)
        self.is_trained = True

    def process_bar(self, bar: Dict):
        """
        Feed a single 1‑minute bar into the system.
        Returns (signal, open_trades, stats) after processing.
        """
        # Convert bar to DataFrame row
        if isinstance(bar, dict):
            new_row = pd.DataFrame([bar])
            new_row.index = pd.to_datetime([bar['timestamp']])
            new_row = new_row.drop(columns=['timestamp'], errors='ignore')
        elif isinstance(bar, pd.Series):
            new_row = bar.to_frame().T
        else:
            raise ValueError("bar must be dict or Series with OHLCV")

        # Append to buffer and keep only the last 5000 bars (or configurable)
        self.data_buffer = pd.concat([self.data_buffer, new_row])
        if len(self.data_buffer) > 5000:
            self.data_buffer = self.data_buffer.iloc[-5000:]

        # Run all layers on the updated buffer
        # 1. Microstructure
        cleaned = self.cleaner.clean_ohlcv(self.data_buffer)
        self.data_buffer['smart_price'] = cleaned['smart_price']
        self.data_buffer['kalman_price'] = cleaned['kalman_price']
        self.data_buffer['atr_14'] = cleaned['atr_14']
        self.data_buffer['spread'] = cleaned['spread']

        # 2. Fundamental
        ctx = self.fundamental.generate_context_features(
            self.data_buffer, self.news_events, self.dxy_series)
        for col in ['news_impact', 'dxy_correlation', 'dxy_aligned',
                    'session_vol', 'timezone_regime']:
            if col in ctx.columns:
                self.data_buffer[col] = ctx[col]

        # 3. Flow
        flow_feats = self.flow.compute_flow_features(self.data_buffer)
        for col in flow_feats.columns:
            self.data_buffer[col] = flow_feats[col]

        # 4. Structure (FVG)
        struct = self.structure.analyze(self.data_buffer)
        for col in ['fvg_bullish', 'fvg_bearish']:
            if col in struct.columns:
                self.data_buffer[col] = struct[col]
        # nearest FVG levels will be read directly from the last row

        # 5. MTF
        df_5m = self._aggregate_to_higher_tf(self.data_buffer, 5)
        df_15m = self._aggregate_to_higher_tf(self.data_buffer, 15)
        mtf_feats = self.mtf.aggregate_structure(self.data_buffer, df_5m, df_15m)
        self.data_buffer['mtf_score'] = mtf_feats['mtf_score']
        self.data_buffer['mtf_confluence'] = mtf_feats['mtf_confluence']

        # 6. HMM regime
        if self.hmm is not None:
            regime = self.hmm.predict_regime(self.data_buffer)
            self.data_buffer['hmm_regime'] = regime

        # 7. ML signal (with calibrated probabilities)
        signal_raw = self.signal_model.predict(self.data_buffer)
        proba_df = self.signal_model.predict_proba(self.data_buffer)
        self.data_buffer['ml_signal'] = signal_raw
        # Store probabilities columns
        for c in proba_df.columns:
            self.data_buffer[f'prob_{c}'] = proba_df[c]

        # Generate final trade signal
        final_signal = self._generate_final_signal()

        # Risk management – update open trades with new high/low/close
        last_bar = self.data_buffer.iloc[-1]
        self.risk_manager.update_trades(
            last_bar.name, last_bar['high'], last_bar['low'], last_bar['close'])

        # Check if we should open a new trade
        self._execute_signal(final_signal, last_bar)

        stats = self.risk_manager.get_stats()
        return final_signal, self.risk_manager.open_trades, stats

    def _generate_final_signal(self) -> Signal:
        """Combine ML, MTF, regime, and flow information into a discrete signal."""
        if not self.is_trained or len(self.data_buffer) < 5:
            return Signal.NEUTRAL

        last = self.data_buffer.iloc[-1]
        ml_signal = int(last.get('ml_signal', 0))
        prob_long = last.get('prob_2', 0) + last.get('prob_1', 0)
        prob_short = last.get('prob_-2', 0) + last.get('prob_-1', 0)
        mtf_score = last.get('mtf_score', 0)
        mtf_confluence = last.get('mtf_confluence', 'neutral')
        cvd_div = last.get('cvd_bull_div', 0) - last.get('cvd_bear_div', 0)
        vpin = last.get('vpin', 0.5)
        absorption = last.get('absorption', 0)
        news_impact = last.get('news_impact', 0)

        # Base confidence from calibrated probabilities
        confidence_long = prob_long
        confidence_short = prob_short

        # Adjust with confluence
        if mtf_confluence in ('strong_bullish',):
            confidence_long += 0.15
        elif mtf_confluence in ('bullish',):
            confidence_long += 0.05
        elif mtf_confluence in ('strong_bearish',):
            confidence_short += 0.15
        elif mtf_confluence in ('bearish',):
            confidence_short += 0.05

        # Adjust with CVD divergence
        if cvd_div > 0:   # bullish divergence
            confidence_long += 0.1
        elif cvd_div < 0: # bearish divergence
            confidence_short += 0.1

        # High VPIN → reduce confidence (toxicity)
        if vpin > self.config.vpin_threshold:
            confidence_long *= 0.8
            confidence_short *= 0.8

        # Absorption adds weight to reversal
        if absorption:
            # if we're overbought/oversold, increase reversal confidence
            pass  # simplistic: can be refined

        # News impact – avoid trading shortly before/after high impact news
        if news_impact > 0.5:
            confidence_long *= 0.3
            confidence_short *= 0.3

        # Final decision
        if confidence_long > 0.55 and confidence_long > confidence_short + 0.1:
            return Signal.STRONG_BUY if confidence_long > 0.7 else Signal.WEAK_BUY
        elif confidence_short > 0.55 and confidence_short > confidence_long + 0.1:
            return Signal.STRONG_SELL if confidence_short > 0.7 else Signal.WEAK_SELL
        else:
            return Signal.NEUTRAL

    def _execute_signal(self, signal: Signal, bar: pd.Series):
        """Convert a signal into a trade, using FVG for stop if available."""
        if signal == Signal.NEUTRAL or signal is None:
            return

        direction = None
        if signal in (Signal.STRONG_BUY, Signal.WEAK_BUY):
            direction = TradeDirection.LONG
        elif signal in (Signal.STRONG_SELL, Signal.WEAK_SELL):
            direction = TradeDirection.SHORT
        else:
            return

        # Check if we already have an open trade in this direction (avoid duplicates)
        for t in self.risk_manager.open_trades:
            if t.direction == direction and t.status == 'open':
                return

        entry = bar['close']
        atr = bar.get('atr_14', 0.0)
        if atr <= 0:
            atr = (bar['high'] - bar['low']) * 1.5

        # Get nearest FVG level for potential tighter stop
        fvg = self.structure.get_nearest_fvg(self.data_buffer, direction)

        # Use Kelly sizing if enough history, else standard
        self.risk_manager.open_trade(
            ts=bar.name,
            direction=direction,
            entry=entry,
            atr=atr,
            fvg_level=fvg,
            use_kelly=len(self.risk_manager.trade_history_pnls) >= 20
        )

    def get_equity_curve(self) -> List[float]:
        return self.risk_manager.equity_curve

    def get_stats(self) -> Dict:
        return self.risk_manager.get_stats()

    def reset(self):
        self.risk_manager = RiskManager(self.config)
        self.data_buffer = pd.DataFrame()
        self.is_trained = False
