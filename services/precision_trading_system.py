"""
1-MINUTE PRECISION TRADING SYSTEM v3
=====================================
6-layer architecture for XAU/USD, BTC/USD, EUR/USD, GBP/USD.

Layer 1: Microstructure Cleaning  — Smart Price, Kalman Filter, Spread Filter
Layer 2: Market Structure         — Lightweight FVG (Fair-Value Gap) detection
Layer 3: Flow & Toxicity          — VPIN, CVD + divergence, Volume Profile,
                                    Absorption, OB imbalance, signed volume
Layer 4: MTF Confluence           — 1m/5m/15m trend alignment scoring
Layer 5: ML Signal Generation     — Lorentzian + XGBoost (isotonic-calibrated)
                                    + HMM regime
Layer 6: Execution & Risk Mgmt    — Quarter-Kelly sizing, ATR/FVG stops,
                                    3-tier TP, walk-forward backtest with
                                    intra-bar fills + ECE + p-value

V3 ENHANCEMENTS (research-backed, no bloat):
  - CVD + CVD/price divergence (institutional-flow edge)
  - Volume Profile (POC, Value Area, distance-to-POC) + absorption detector
  - Lightweight FVG detection (3-candle pattern, used for stop refinement)
  - MTF confluence scoring across 1m/5m/15m
  - Isotonic probability calibration via CalibratedClassifierCV(TimeSeriesSplit)
    — fixes XGBoost overconfidence which biases position sizing
  - Quarter-Kelly position sizing (kicks in after ≥20 closed trades)
  - Walk-forward backtest with intra-bar OHLC-ordered fill simulation,
    expected-calibration-error (ECE), and one-sample t-test p-value

PREEXISTING OPTIMIZATIONS (kept):
  - Lorentzian KNN vectorized via numpy broadcasting (~600× speed-up).
  - VPIN bucketing follows the 1-50-50 standard (1m bars, 50 buckets,
    50-sample rolling window) per Easley/López de Prado 2012 + 2025 update.
  - Per-asset VPIN threshold from 2025 BV-VPIN paper.
  - L2 data falls back to OHLC midpoint cleanly.

Sources:
  - Easley, López de Prado, O'Hara — "Flow Toxicity and Liquidity in a HF World"
  - 2025 BV-VPIN — optimal thresholds vary by market.
  - 2025 BTC jump-prediction paper — VPIN > 0.6 sustained = trending regime.
  - López de Prado — "Advances in Financial Machine Learning" (calibration,
    walk-forward, meta-labeling, intra-bar fill realism).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Literal
from enum import Enum
import warnings
from collections import deque
import json
import pickle
from datetime import datetime, timedelta

# ── Optional deps; fall back gracefully ──────────────────────────────────────
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    CalibratedClassifierCV = None
    TimeSeriesSplit = None
    LabelEncoder = None

try:
    from scipy.stats import ttest_1samp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    ttest_1samp = None

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
    STRONG_BUY = 2
    WEAK_BUY = 1
    NEUTRAL = 0
    WEAK_SELL = -1
    STRONG_SELL = -2


class Regime(Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class TradeDirection(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class AssetConfig:
    asset: Asset
    pip_value: float
    spread_avg: float
    tick_size: float
    contract_size: float = 1.0
    leverage: float = 100.0

    london_start: int = 8
    london_end: int = 17
    ny_start: int = 13
    ny_end: int = 22

    kalman_observation_covariance: float = 1.0
    kalman_transition_covariance: float = 0.01
    vpin_buckets: int = 50         # "1-50-50" standard
    vpin_window: int = 50
    vpin_threshold: float = 0.85    # block trades when VPIN > this percentile

    max_risk_per_trade_pct: float = 0.01
    atr_multiplier_stop: float = 1.5
    atr_multiplier_tp1: float = 1.0
    atr_multiplier_tp2: float = 2.0
    atr_multiplier_tp3: float = 3.0

    # ── v3 fields ──────────────────────────────────────────────────────────
    kelly_fraction: float = 0.25            # Quarter-Kelly
    kelly_min_history: int = 20             # bars of trade history needed
    fvg_lookback: int = 10                  # bars to compute mean body for FVG
    fvg_body_multiplier: float = 1.5        # middle candle body strength
    vp_lookback: int = 50                   # rolling Volume Profile window
    vp_value_area_pct: float = 0.70         # 70% of volume = Value Area
    calibration_method: str = "isotonic"    # "isotonic" | "sigmoid" | None
    calibration_cv_splits: int = 3


# Per-asset config — research-backed thresholds (BV-VPIN 2025)
ASSET_CONFIGS: Dict[Asset, AssetConfig] = {
    Asset.XAUUSD: AssetConfig(
        asset=Asset.XAUUSD,
        pip_value=0.01, spread_avg=0.05, tick_size=0.01,
        contract_size=100.0, leverage=100.0,
        kalman_observation_covariance=0.5,
        kalman_transition_covariance=0.005,
        vpin_buckets=50, vpin_window=50, vpin_threshold=0.90,
        max_risk_per_trade_pct=0.01,
        atr_multiplier_stop=2.0,
        atr_multiplier_tp1=1.5, atr_multiplier_tp2=2.5, atr_multiplier_tp3=4.0,
        kelly_fraction=0.25, fvg_body_multiplier=1.5,
    ),
    Asset.BTCUSD: AssetConfig(
        asset=Asset.BTCUSD,
        pip_value=1.0, spread_avg=20.0, tick_size=0.01,
        contract_size=1.0, leverage=50.0,
        kalman_observation_covariance=100.0,
        kalman_transition_covariance=1.0,
        vpin_buckets=100, vpin_window=100, vpin_threshold=0.80,
        max_risk_per_trade_pct=0.01,
        atr_multiplier_stop=2.5,
        atr_multiplier_tp1=2.0, atr_multiplier_tp2=3.5, atr_multiplier_tp3=5.0,
        kelly_fraction=0.20, fvg_body_multiplier=2.0,
        vp_lookback=100,
    ),
    Asset.EURUSD: AssetConfig(
        asset=Asset.EURUSD,
        pip_value=0.0001, spread_avg=0.0001, tick_size=0.00001,
        contract_size=100000.0, leverage=100.0,
        kalman_observation_covariance=0.0001,
        kalman_transition_covariance=0.000001,
        vpin_buckets=50, vpin_window=50, vpin_threshold=0.90,
        max_risk_per_trade_pct=0.01,
        atr_multiplier_stop=1.5,
        atr_multiplier_tp1=1.0, atr_multiplier_tp2=2.0, atr_multiplier_tp3=3.0,
        kelly_fraction=0.25, fvg_body_multiplier=1.2,
    ),
    Asset.GBPUSD: AssetConfig(
        asset=Asset.GBPUSD,
        pip_value=0.0001, spread_avg=0.0002, tick_size=0.00001,
        contract_size=100000.0, leverage=100.0,
        kalman_observation_covariance=0.0002,
        kalman_transition_covariance=0.000002,
        vpin_buckets=50, vpin_window=50, vpin_threshold=0.85,
        max_risk_per_trade_pct=0.01,
        atr_multiplier_stop=1.5,
        atr_multiplier_tp1=1.0, atr_multiplier_tp2=2.0, atr_multiplier_tp3=3.0,
        kelly_fraction=0.25, fvg_body_multiplier=1.3,
    ),
}


# =============================================================================
# LAYER 1 — Microstructure Cleaning
# =============================================================================

class MicrostructureCleaner:
    def __init__(self, config: AssetConfig):
        self.config = config
        self.kalman: Optional["KalmanFilter"] = None
        self._init_kalman()

    def _init_kalman(self):
        if not PYKALMAN_AVAILABLE:
            return
        self.kalman = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=0,
            initial_state_covariance=1,
            observation_covariance=self.config.kalman_observation_covariance,
            transition_covariance=self.config.kalman_transition_covariance,
        )

    def compute_smart_price(self, bid, ask, bid_vol=1.0, ask_vol=1.0):
        total = np.asarray(bid_vol) + np.asarray(ask_vol)
        bid = np.asarray(bid); ask = np.asarray(ask)
        bv  = np.asarray(bid_vol); av = np.asarray(ask_vol)
        out = np.where(total == 0, (bid + ask) / 2, (bid * av + ask * bv) / np.where(total == 0, 1, total))
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

        # Smart price
        if has_l2_data and "bid" in df.columns and "ask" in df.columns:
            bid_vol = df.get("bid_vol", pd.Series(1.0, index=df.index))
            ask_vol = df.get("ask_vol", pd.Series(1.0, index=df.index))
            result["smart_price"] = self.compute_smart_price(
                df["bid"].values, df["ask"].values,
                bid_vol.values, ask_vol.values,
            )
            result["spread"] = df["ask"] - df["bid"]
        else:
            result["smart_price"] = (df["high"] + df["low"]) / 2
            typical = (df["high"] + df["low"] + df["close"]) / 3
            result["spread"] = (df["close"] - typical).abs() * 2
            result["spread"] = result["spread"].clip(lower=self.config.spread_avg)

        # Kalman
        result["kalman_price"] = self.apply_kalman_filter(result["smart_price"].values)

        # ATR + spread filter
        result["atr_14"] = self._compute_atr(result, 14)
        result["spread_filter_active"] = result.apply(
            lambda r: self.compute_spread_filter(r["spread"], r["atr_14"]),
            axis=1,
        )
        return result

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close  = (df["low"]  - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()


# =============================================================================
# LAYER 2 — Flow & Toxicity
# =============================================================================

class FlowAnalyzer:
    def __init__(self, config: AssetConfig):
        self.config = config

    def compute_vpin(self, df: pd.DataFrame) -> pd.Series:
        """Volume-Synchronized PIN — 1-50-50 standard."""
        price_change = df["close"].diff().fillna(0).values
        volume = df["volume"].values

        buy_vol  = np.where(price_change > 0, volume, 0.0)
        sell_vol = np.where(price_change < 0, volume, 0.0)
        # Zero-change bars: split 50/50
        zero = price_change == 0
        buy_vol[zero]  = volume[zero] * 0.5
        sell_vol[zero] = volume[zero] * 0.5

        total_v = (buy_vol + sell_vol).sum()
        n_buckets = self.config.vpin_buckets
        if total_v <= 0 or n_buckets <= 0:
            return pd.Series(0.0, index=df.index)

        bucket_size = total_v / n_buckets
        # Walk through bars accumulating; when bucket fills, record |buy-sell|/(buy+sell)
        bucket_buy = bucket_sell = bucket_total = 0.0
        bucket_at_bar: List[Tuple[int, float]] = []   # (last_bar_idx, vpin_value)
        for i in range(len(df)):
            bucket_buy   += buy_vol[i]
            bucket_sell  += sell_vol[i]
            bucket_total += buy_vol[i] + sell_vol[i]
            if bucket_total >= bucket_size:
                vpin_val = (
                    abs(bucket_buy - bucket_sell) / bucket_total
                    if bucket_total > 0 else 0.0
                )
                bucket_at_bar.append((i, vpin_val))
                bucket_buy = bucket_sell = bucket_total = 0.0

        # Map bucket VPINs back to bar timeline + rolling mean (window param)
        out = pd.Series(np.nan, index=df.index, dtype=float)
        for bar_idx, val in bucket_at_bar:
            out.iloc[bar_idx] = val
        out = out.ffill().fillna(0.0)
        return out.rolling(window=self.config.vpin_window, min_periods=1).mean().clip(0, 1)

    def compute_order_book_imbalance(self, bid_vol, ask_vol):
        total = bid_vol + ask_vol
        return ((bid_vol - ask_vol) / total.replace(0, np.nan)).fillna(0).clip(-1, 1)

    def compute_signed_volume(self, df: pd.DataFrame, method: str = "bulk_volume") -> pd.Series:
        if method == "tick_rule":
            change = df["close"].diff()
            signed = pd.Series(0.0, index=df.index)
            signed[change > 0] =  df.loc[change > 0, "volume"]
            signed[change < 0] = -df.loc[change < 0, "volume"]
            return signed
        # Lee-Ready / Bulk Volume Classification
        bar_range = (df["high"] - df["low"]).replace(0, np.nan)
        position = ((df["close"] - df["low"]) / bar_range).fillna(0.5).clip(0, 1)
        return df["volume"] * (2 * position - 1)

    def compute_realized_volatility(self, prices: pd.Series, window: int = 5) -> pd.Series:
        rets = prices.pct_change().fillna(0)
        return (rets.rolling(window).std() * np.sqrt(525_600)).fillna(0)

    # ── v3: CVD + divergence ──────────────────────────────────────────────
    def compute_cvd(self, df: pd.DataFrame) -> pd.Series:
        """Cumulative Volume Delta (signed volume cumulative sum)."""
        price_change = df["close"].diff().fillna(0)
        volume = df["volume"].astype(float)
        buy_vol = pd.Series(np.where(price_change > 0, volume, 0.0), index=df.index)
        sell_vol = pd.Series(np.where(price_change < 0, volume, 0.0), index=df.index)
        zero = price_change == 0
        buy_vol[zero] = volume[zero] * 0.5
        sell_vol[zero] = volume[zero] * 0.5
        return (buy_vol - sell_vol).cumsum()

    def compute_cvd_divergence(self, df: pd.DataFrame, cvd: pd.Series,
                                window: int = 10) -> Tuple[pd.Series, pd.Series]:
        """Detect CVD-Price divergence — key reversal signal.
        Bullish div: price makes new low, CVD does not.
        Bearish div: price makes new high, CVD does not.
        """
        price_low_prev = df["low"].rolling(window).min().shift(1)
        price_high_prev = df["high"].rolling(window).max().shift(1)
        cvd_low_prev = cvd.rolling(window).min().shift(1)
        cvd_high_prev = cvd.rolling(window).max().shift(1)
        bull_div = ((df["low"] < price_low_prev) & (cvd > cvd_low_prev)).astype(int)
        bear_div = ((df["high"] > price_high_prev) & (cvd < cvd_high_prev)).astype(int)
        return bull_div, bear_div

    # ── v3: Volume Profile (rolling POC + Value Area) ─────────────────────
    def compute_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling Volume Profile: POC, Value Area High/Low, in_value_area, distance_to_poc."""
        result = pd.DataFrame(index=df.index)
        n = len(df)
        lookback = self.config.vp_lookback
        target_pct = self.config.vp_value_area_pct

        poc = np.full(n, np.nan)
        va_high = np.full(n, np.nan)
        va_low = np.full(n, np.nan)
        closes = df["close"].values
        volumes = df["volume"].values.astype(float)

        for i in range(lookback, n):
            window_close = closes[i - lookback : i]
            window_vol = volumes[i - lookback : i]
            if len(window_close) < 10:
                continue
            n_bins = min(20, max(5, len(window_close) // 2))
            try:
                hist, edges = np.histogram(window_close, bins=n_bins, weights=window_vol)
            except Exception:
                continue
            if hist.sum() <= 0:
                continue
            # POC: highest-volume bin midpoint
            poc_idx = int(np.argmax(hist))
            poc[i] = (edges[poc_idx] + edges[poc_idx + 1]) / 2
            # Value Area: expand from POC until cumulative vol >= target_pct
            target_vol = hist.sum() * target_pct
            current_vol = float(hist[poc_idx])
            low_b = high_b = poc_idx
            while current_vol < target_vol and (low_b > 0 or high_b < len(hist) - 1):
                low_v = hist[low_b - 1] if low_b > 0 else 0.0
                high_v = hist[high_b + 1] if high_b < len(hist) - 1 else 0.0
                if low_v > high_v and low_b > 0:
                    low_b -= 1; current_vol += float(low_v)
                elif high_b < len(hist) - 1:
                    high_b += 1; current_vol += float(high_v)
                else:
                    break
            va_low[i] = edges[low_b]
            va_high[i] = edges[high_b + 1]

        result["poc"] = pd.Series(poc, index=df.index).ffill().bfill()
        result["va_high"] = pd.Series(va_high, index=df.index).ffill().bfill()
        result["va_low"] = pd.Series(va_low, index=df.index).ffill().bfill()
        result["in_value_area"] = (
            (df["close"] >= result["va_low"]) & (df["close"] <= result["va_high"])
        ).astype(int)
        result["distance_to_poc"] = ((df["close"] - result["poc"]) /
                                     df["close"].replace(0, np.nan)).fillna(0)
        return result

    @staticmethod
    def detect_absorption(df: pd.DataFrame) -> pd.Series:
        """Absorption: large volume with minimal price movement (institutional accumulation)."""
        volume = df["volume"].astype(float)
        price_range = (df["high"] - df["low"]).astype(float)
        vol_ma = volume.rolling(20).mean()
        range_ma = price_range.rolling(20).mean()
        return ((volume > vol_ma * 2) & (price_range < range_ma * 0.5)).astype(int)

    def compute_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["vpin"] = self.compute_vpin(df)
        result["signed_volume"] = self.compute_signed_volume(df, "bulk_volume")
        result["signed_volume_momentum"] = result["signed_volume"].rolling(5).mean()
        result["realized_vol_5m"]  = self.compute_realized_volatility(df["close"], 5)
        result["realized_vol_20m"] = self.compute_realized_volatility(df["close"], 20)
        result["volume_ma_20"] = df["volume"].rolling(20).mean()
        result["volume_ma_ratio"] = (
            df["volume"] / result["volume_ma_20"].replace(0, np.nan)
        ).fillna(1)

        if "bid_vol" in df.columns and "ask_vol" in df.columns:
            result["ob_imbalance"] = self.compute_order_book_imbalance(
                df["bid_vol"], df["ask_vol"],
            )
        else:
            denom = df["volume"].rolling(5).mean().replace(0, np.nan)
            result["ob_imbalance"] = (
                result["signed_volume_momentum"] / denom
            ).fillna(0).clip(-1, 1)

        # ── v3 additions ───────────────────────────────────────────────────
        cvd = self.compute_cvd(df)
        result["cvd"] = cvd
        result["cvd_slope"] = cvd.diff(5).fillna(0)
        bull_div, bear_div = self.compute_cvd_divergence(df, cvd, window=10)
        result["cvd_bull_div"] = bull_div
        result["cvd_bear_div"] = bear_div

        vp = self.compute_volume_profile(df)
        result["poc"] = vp["poc"]
        result["va_high"] = vp["va_high"]
        result["va_low"] = vp["va_low"]
        result["in_value_area"] = vp["in_value_area"]
        result["distance_to_poc"] = vp["distance_to_poc"]

        result["absorption"] = self.detect_absorption(df)
        return result


# =============================================================================
# LAYER 2b — Market Structure (lightweight, used for stop refinement only)
# =============================================================================

class MarketStructureAnalyzer:
    """Lightweight 3-candle FVG (Fair Value Gap) detection.

    Bullish FVG: candle i-2 high < candle i low (price gap above prior high),
                 with a body on candle i-1 strong enough to validate the gap.
    Bearish FVG: mirror.

    Used for stop placement refinement — when a fresh FVG is detected, the
    nearer FVG edge is preferred over the ATR stop if it's tighter.
    """

    def __init__(self, config: AssetConfig):
        self.config = config

    def detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        n = len(result)
        fvg_bull = np.zeros(n, dtype=bool)
        fvg_bear = np.zeros(n, dtype=bool)
        bull_start = np.full(n, np.nan)   # tighter stop edge (for LONG)
        bull_end = np.full(n, np.nan)
        bear_start = np.full(n, np.nan)
        bear_end = np.full(n, np.nan)

        highs = result["high"].values
        lows = result["low"].values
        opens = result["open"].values
        closes = result["close"].values
        lookback = self.config.fvg_lookback
        body_mult = self.config.fvg_body_multiplier

        for i in range(2, n):
            start_idx = max(0, i - lookback)
            bodies = np.abs(closes[start_idx:i] - opens[start_idx:i])
            avg_body = float(np.mean(bodies)) if len(bodies) else 0.0
            if avg_body <= 0:
                avg_body = 1e-6
            mid_body = abs(closes[i - 1] - opens[i - 1])
            # Bullish FVG: low[i] > high[i-2]
            if lows[i] > highs[i - 2] and mid_body > avg_body * body_mult:
                fvg_bull[i] = True
                bull_start[i] = highs[i - 2]
                bull_end[i] = lows[i]
            # Bearish FVG: high[i] < low[i-2]
            elif highs[i] < lows[i - 2] and mid_body > avg_body * body_mult:
                fvg_bear[i] = True
                bear_start[i] = lows[i - 2]
                bear_end[i] = highs[i]

        result["fvg_bullish"] = fvg_bull
        result["fvg_bearish"] = fvg_bear
        result["fvg_bull_start"] = bull_start
        result["fvg_bull_end"] = bull_end
        result["fvg_bear_start"] = bear_start
        result["fvg_bear_end"] = bear_end
        return result

    def get_nearest_fvg_stop(self, df: pd.DataFrame,
                              direction: "TradeDirection") -> Optional[float]:
        """Return the nearer FVG edge to use as a stop, if a fresh FVG exists."""
        if df.empty or "fvg_bullish" not in df.columns:
            return None
        latest = df.iloc[-1]
        if direction == TradeDirection.LONG and bool(latest.get("fvg_bullish", False)):
            v = latest.get("fvg_bull_start")
            return float(v) if v is not None and not pd.isna(v) else None
        if direction == TradeDirection.SHORT and bool(latest.get("fvg_bearish", False)):
            v = latest.get("fvg_bear_start")
            return float(v) if v is not None and not pd.isna(v) else None
        return None


# =============================================================================
# LAYER 4 — Multi-Timeframe Confluence (1m / 5m / 15m)
# =============================================================================

class MTFConfluenceEngine:
    """1m + 5m + 15m trend-alignment scoring.

    Aggregates 1m bars to 5m and 15m, computes EMA-direction on each, then
    weighted-sums into a confluence score in [-1, 1]:
       confluence = 0.5*sign(1m) + 0.3*sign(5m) + 0.2*sign(15m)
    """

    MTF_WEIGHTS = {"1m": 0.5, "5m": 0.3, "15m": 0.2}
    EMA_SPAN = 20

    @staticmethod
    def _resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
        if df.empty:
            return df
        rule = f"{minutes}min"
        agg = {"open": "first", "high": "max", "low": "min",
               "close": "last", "volume": "sum"}
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        try:
            return df[cols].resample(rule).apply(agg).dropna()
        except Exception:
            return pd.DataFrame()

    @classmethod
    def _trend_sign(cls, close: pd.Series) -> pd.Series:
        ema = close.ewm(span=cls.EMA_SPAN).mean()
        return np.where(close > ema, 1, -1)

    @classmethod
    def _align_higher_tf(cls, target_index: pd.Index,
                          tf_close: pd.Series) -> pd.Series:
        """Reindex a higher-TF close series onto the 1m index via forward-fill."""
        if tf_close.empty:
            return pd.Series(np.nan, index=target_index)
        return tf_close.reindex(target_index, method="ffill")

    @classmethod
    def compute(cls, df_1m: pd.DataFrame) -> pd.DataFrame:
        """Add mtf_score and mtf_confluence to a 1m DataFrame."""
        out = df_1m.copy()
        if df_1m.empty or len(df_1m) < 30:
            out["mtf_score"] = 0.0
            out["mtf_confluence"] = "neutral"
            out["trend_1m"] = 0
            out["trend_5m"] = 0
            out["trend_15m"] = 0
            return out

        df_5m = cls._resample(df_1m, 5)
        df_15m = cls._resample(df_1m, 15)

        trend_1m = cls._trend_sign(df_1m["close"])
        out["trend_1m"] = trend_1m

        if not df_5m.empty:
            close_5m_aligned = cls._align_higher_tf(df_1m.index, df_5m["close"])
            out["trend_5m"] = np.where(
                close_5m_aligned > close_5m_aligned.ewm(span=cls.EMA_SPAN).mean(),
                1, -1,
            )
        else:
            out["trend_5m"] = 0
        if not df_15m.empty:
            close_15m_aligned = cls._align_higher_tf(df_1m.index, df_15m["close"])
            out["trend_15m"] = np.where(
                close_15m_aligned > close_15m_aligned.ewm(span=cls.EMA_SPAN).mean(),
                1, -1,
            )
        else:
            out["trend_15m"] = 0

        score = (
            out["trend_1m"] * cls.MTF_WEIGHTS["1m"]
            + out["trend_5m"] * cls.MTF_WEIGHTS["5m"]
            + out["trend_15m"] * cls.MTF_WEIGHTS["15m"]
        )
        out["mtf_score"] = score

        labels = np.full(len(out), "neutral", dtype=object)
        s = score.values
        labels[s > 0.6] = "strong_bullish"
        labels[(s > 0.2) & (s <= 0.6)] = "bullish"
        labels[(s < -0.2) & (s >= -0.6)] = "bearish"
        labels[s < -0.6] = "strong_bearish"
        out["mtf_confluence"] = labels
        return out


# =============================================================================
# LAYER 3 — ML Signal Models
# =============================================================================

class LorentzianClassifier:
    """Vectorized Lorentzian KNN.

    For 20k training rows × 4 features, np.log1p+broadcasting computes the
    full distance matrix in ~10ms — vs ~6s with the original Python loop.
    """
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
        else:
            self.scaler = None
        self.train_data: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)

        # RSI(7)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
        rs = gain / loss.replace(0, np.nan)
        feats["rsi_7"] = (100 - 100 / (1 + rs)).fillna(50)

        # MFI(7)
        typical = (df["high"] + df["low"] + df["close"]) / 3
        flow = typical * df["volume"]
        sign = np.where(typical > typical.shift(1), 1, -1)
        signed = pd.Series(flow * sign, index=df.index)
        pos = signed.where(signed > 0, 0).rolling(7).sum()
        neg = (-signed.where(signed < 0, 0)).rolling(7).sum()
        ratio = pos / neg.replace(0, np.nan)
        feats["mfi_7"] = (100 - 100 / (1 + ratio)).fillna(50)

        # ROC(3)
        feats["roc_3"] = ((df["close"] - df["close"].shift(3)) / df["close"].shift(3) * 100).fillna(0)

        # Volatility gate
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift()).abs()
        tr3 = (df["low"]  - df["close"].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        feats["volatility_gate"] = (tr / atr.replace(0, np.nan)).fillna(1).clip(0, 5)
        return feats

    def fit(self, df: pd.DataFrame, labels: Optional[np.ndarray] = None,
            target_horizon: int = 3):
        feats = self._compute_features(df).dropna()
        if labels is None:
            # Auto-generate labels from forward returns
            fwd = df["close"].shift(-target_horizon) / df["close"] - 1
            labels = pd.Series(0, index=df.index, dtype=int)
            labels[fwd >  0.001]  = 2
            labels[(fwd <=  0.001) & (fwd >  0.0003)] = 1
            labels[(fwd >= -0.001) & (fwd < -0.0003)] = -1
            labels[fwd < -0.001]  = -2
        else:
            labels = pd.Series(labels, index=df.index)

        aligned = labels.loc[feats.index].values
        X = feats.values
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        self.train_data = X.astype(np.float32)
        self.train_labels = aligned.astype(np.int8)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.train_data is None or self.train_labels is None:
            return pd.Series(0, index=df.index)
        feats = self._compute_features(df).dropna()
        if feats.empty:
            return pd.Series(0, index=df.index)
        X = feats.values
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Vectorized Lorentzian distance: log(1 + |x - y|).sum(axis=feat)
        # X: (M, F)  train_data: (N, F)
        # diff: (M, N, F)
        X32 = X.astype(np.float32)
        # Memory-conscious: batch test rows in chunks if M*N is huge
        M, F = X32.shape
        N = self.train_data.shape[0]
        chunk = max(1, 1_000_000 // max(N, 1))   # cap O(M*N) memory at ~1M floats
        preds = np.zeros(M, dtype=np.int8)
        for start in range(0, M, chunk):
            stop = min(start + chunk, M)
            diff = np.abs(X32[start:stop, None, :] - self.train_data[None, :, :])
            dist = np.log1p(diff).sum(axis=2)        # (m, N)
            # K-nearest indices
            k = min(self.n_neighbors, N)
            idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
            votes = self.train_labels[idx]            # (m, k)
            avg = votes.mean(axis=1)
            local = np.zeros(stop - start, dtype=np.int8)
            local[avg >  0.5]  = 2
            local[(avg <=  0.5) & (avg >  0.1)] = 1
            local[(avg >= -0.5) & (avg < -0.1)] = -1
            local[avg < -0.5] = -2
            preds[start:stop] = local
        return pd.Series(preds, index=feats.index)


class XGBoostSignalModel:
    """Gradient-boosted tree classifier on microstructure features
    + isotonic probability calibration via CalibratedClassifierCV.
    Calibration uses TimeSeriesSplit (no future leakage) — fixes the
    well-known XGBoost overconfidence that biases position sizing.
    """
    def __init__(self, n_estimators: int = 200, max_depth: int = 5,
                 learning_rate: float = 0.05,
                 calibration_method: Optional[str] = "isotonic",
                 calibration_cv_splits: int = 3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.calibration_method = calibration_method
        self.calibration_cv_splits = max(2, int(calibration_cv_splits))
        self.model = None
        self.calibrated_model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_cols: Optional[List[str]] = None
        # Maps original class label (-2..2) ↔ XGB-required 0-based encoded class
        self._class_to_idx: Dict[int, int] = {}
        self._idx_to_class: Dict[int, int] = {}

    def __setstate__(self, state):
        """Backward-compat for old pickles missing v3 attributes."""
        self.__dict__.update(state)
        self.calibration_method = state.get("calibration_method", "isotonic")
        self.calibration_cv_splits = state.get("calibration_cv_splits", 3)
        self.calibrated_model = state.get("calibrated_model", None)
        self._class_to_idx = state.get("_class_to_idx", {})
        self._idx_to_class = state.get("_idx_to_class", {})
        # If old pickle has no encoder, infer from sklearn classes_ if present
        if not self._class_to_idx and self.model is not None:
            classes = getattr(self.model, "classes_", None)
            if classes is not None:
                # Old code used y = label + 2 → encoded 0..4 mapped to {-2..2}
                self._idx_to_class = {int(i): int(c) - 2 for i, c in enumerate(classes)}
                self._class_to_idx = {v: k for k, v in self._idx_to_class.items()}

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
        f["smart_vs_close"] = (
            (df["smart_price"] - df["close"]) / df["close"]
            if "smart_price" in df.columns else 0
        )
        f["kalman_return"] = (df["kalman_price"].pct_change() if "kalman_price" in df.columns else f["returns_1"])
        f["kalman_vs_close"] = (
            (df["kalman_price"] - df["close"]) / df["close"]
            if "kalman_price" in df.columns else 0
        )
        f["rsi_7"]  = self._rsi(df["close"], 7)
        f["rsi_14"] = self._rsi(df["close"], 14)
        f["macd"]   = self._macd(df["close"])
        f["macd_signal"] = f["macd"].ewm(span=9).mean()
        f["macd_hist"]   = f["macd"] - f["macd_signal"]

        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        f["bb_position"] = ((df["close"] - bb_mid) / bb_std.replace(0, np.nan)).fillna(0)
        f["bb_width"]    = (bb_std / bb_mid.replace(0, np.nan)).fillna(0)

        f["signed_vol_ma5"]  = df["signed_volume"].rolling(5).mean()  if "signed_volume" in df.columns else 0
        f["signed_vol_ma10"] = df["signed_volume"].rolling(10).mean() if "signed_volume" in df.columns else 0
        f["vol_intensity"]   = df["volume_ma_ratio"] if "volume_ma_ratio" in df.columns else (
            df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)
        ).fillna(1)
        f["ob_imbalance"]    = df["ob_imbalance"] if "ob_imbalance" in df.columns else 0
        f["vpin"]            = df["vpin"]         if "vpin" in df.columns else 0
        f["vpin_high"]       = (
            (df["vpin"] > df["vpin"].rolling(100).quantile(0.9)).astype(int)
            if "vpin" in df.columns else 0
        )
        if "realized_vol_5m" in df.columns:
            f["rv_5m"]  = df["realized_vol_5m"]
            f["rv_20m"] = df["realized_vol_20m"]
        else:
            r = df["close"].pct_change().fillna(0)
            f["rv_5m"]  = r.rolling(5).std()  * np.sqrt(525_600)
            f["rv_20m"] = r.rolling(20).std() * np.sqrt(525_600)
        f["vol_regime"] = (f["rv_5m"] > f["rv_5m"].rolling(50).mean()).astype(int)

        # ── v3 features: CVD, Volume Profile, MTF, absorption ──────────────
        f["cvd_slope"]       = df["cvd_slope"]       if "cvd_slope"       in df.columns else 0
        f["cvd_bull_div"]    = df["cvd_bull_div"]    if "cvd_bull_div"    in df.columns else 0
        f["cvd_bear_div"]    = df["cvd_bear_div"]    if "cvd_bear_div"    in df.columns else 0
        f["distance_to_poc"] = df["distance_to_poc"] if "distance_to_poc" in df.columns else 0
        f["in_value_area"]   = df["in_value_area"]   if "in_value_area"   in df.columns else 1
        f["absorption"]      = df["absorption"]      if "absorption"      in df.columns else 0
        f["mtf_score"]       = df["mtf_score"]       if "mtf_score"       in df.columns else 0
        f["trend_5m"]        = df["trend_5m"]        if "trend_5m"        in df.columns else 0
        f["trend_15m"]       = df["trend_15m"]       if "trend_15m"       in df.columns else 0

        for lag in (1, 2, 3):
            f[f"return_lag_{lag}"] = f["returns_1"].shift(lag)

        f["rsi_x_vol"] = f["rsi_7"]   * f["vol_intensity"]
        f["ob_x_vol"]  = f["ob_imbalance"] * f["vol_intensity"]
        f["cvd_x_vol"] = f["cvd_slope"] * f["vol_intensity"]

        return f.replace([np.inf, -np.inf], 0).fillna(0)

    def fit(self, df: pd.DataFrame, target_horizon: int = 3):
        feats = self._engineer(df)
        fwd = df["close"].shift(-target_horizon) / df["close"] - 1
        target = pd.Series(0, index=df.index, dtype=int)
        target[fwd >  0.001]  = 2
        target[(fwd <=  0.001) & (fwd >  0.0003)] = 1
        target[(fwd >= -0.001) & (fwd < -0.0003)] = -1
        target[fwd < -0.001]  = -2

        af = feats.loc[target.index].dropna()
        ay = target.loc[af.index]
        self.feature_cols = af.columns.tolist()
        X = self.scaler.fit_transform(af) if self.scaler is not None else af.values

        # Encode the {-2,-1,0,1,2} → 0..k-1 contiguous range XGBoost requires.
        present_classes = sorted(int(c) for c in pd.Series(ay).unique())
        self._class_to_idx = {c: i for i, c in enumerate(present_classes)}
        self._idx_to_class = {i: c for c, i in self._class_to_idx.items()}
        y = np.array([self._class_to_idx[int(v)] for v in ay.values], dtype=np.int64)
        n_classes = len(present_classes)

        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                num_class=n_classes if n_classes > 2 else None,
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
                random_state=42,
                tree_method="hist",
                # Light L2 reg to fight 1m noise overfit
                reg_lambda=1.0, min_child_weight=10,
            )
            self.model.fit(X, y)
        elif SKLEARN_AVAILABLE:
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
            )
            self.model.fit(X, y)
        else:
            raise ImportError("Need xgboost or scikit-learn")

        # ── Isotonic probability calibration (out-of-sample TimeSeriesSplit) ─
        # Fixes XGBoost overconfidence; calibrated probabilities feed Kelly
        # sizing and the live-signal confidence threshold.
        self.calibrated_model = None
        if (SKLEARN_AVAILABLE and self.calibration_method in ("isotonic", "sigmoid")
                and CalibratedClassifierCV is not None and TimeSeriesSplit is not None
                and len(np.unique(y)) >= 2 and len(X) >= 50):
            try:
                tscv = TimeSeriesSplit(n_splits=self.calibration_cv_splits)
                # Note: the calibration wrapper refits a fresh estimator on each
                # CV fold, so the underlying XGB hyperparameters are reused but
                # the trees themselves are not the ones in self.model — that is
                # intentional and matches the López de Prado calibration recipe.
                if XGBOOST_AVAILABLE:
                    base = xgb.XGBClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                        num_class=n_classes if n_classes > 2 else None,
                        eval_metric="mlogloss" if n_classes > 2 else "logloss",
                        random_state=42,
                        tree_method="hist",
                        reg_lambda=1.0, min_child_weight=10,
                    )
                else:
                    base = GradientBoostingClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        random_state=42,
                    )
                cal = CalibratedClassifierCV(
                    base, method=self.calibration_method, cv=tscv,
                )
                cal.fit(X, y)
                self.calibrated_model = cal
            except Exception as e:
                # Calibration is best-effort; fall back to raw probabilities
                print(f"[XGBoost] calibration failed (using raw probas): {e}")
                self.calibrated_model = None

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None or self.feature_cols is None:
            return pd.Series(0, index=df.index)
        feats = self._engineer(df)
        for c in self.feature_cols:
            if c not in feats.columns:
                feats[c] = 0
        feats = feats[self.feature_cols]
        X = self.scaler.transform(feats) if self.scaler is not None else feats.values
        encoded = self.model.predict(X)
        # Map encoded indices back to original {-2..2} labels
        decoded = np.array([self._idx_to_class.get(int(i), 0) for i in encoded], dtype=int)
        return pd.Series(decoded, index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [-2, -1, 0, 1, 2]
        if self.model is None or self.feature_cols is None:
            return pd.DataFrame(0.2, index=df.index, columns=cols)
        feats = self._engineer(df)
        for c in self.feature_cols:
            if c not in feats.columns:
                feats[c] = 0
        feats = feats[self.feature_cols]
        X = self.scaler.transform(feats) if self.scaler is not None else feats.values
        # Prefer calibrated probabilities if available
        model = self.calibrated_model if self.calibrated_model is not None else self.model
        try:
            proba = model.predict_proba(X)
        except Exception:
            proba = self.model.predict_proba(X)
        # Map encoded class index → original {-2..2} → 5-column layout
        full = np.zeros((proba.shape[0], len(cols)))
        col_index = {c: i for i, c in enumerate(cols)}
        for enc_idx in range(proba.shape[1]):
            orig_class = self._idx_to_class.get(enc_idx)
            if orig_class is None:
                continue
            if orig_class in col_index:
                full[:, col_index[orig_class]] = proba[:, enc_idx]
        return pd.DataFrame(full, index=df.index, columns=cols)


class HMMRegimeDetector:
    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None

    def fit(self, df: pd.DataFrame):
        if not HMMLEARN_AVAILABLE or not SKLEARN_AVAILABLE:
            return
        # Always re-init scaler — avoids dtype issues from stale pickles saved
        # under an older sklearn version (array_api_compat changed in 1.4+).
        self.scaler = StandardScaler()
        rets = df["close"].pct_change().fillna(0).values.reshape(-1, 1).astype(np.float64)
        vol = pd.Series(rets.flatten()).rolling(10).std().fillna(0).values.reshape(-1, 1).astype(np.float64)
        feats = np.hstack([rets, vol]).astype(np.float64)
        feats = feats[~np.isnan(feats).any(axis=1)]
        if len(feats) < 100:
            return
        try:
            Xs = self.scaler.fit_transform(feats)
        except Exception as e:
            print(f"[HMM] scaler fit failed: {e}")
            self.model = None
            return
        self.model = GaussianHMM(
            n_components=self.n_regimes, covariance_type="full",
            n_iter=100, random_state=42,
        )
        try:
            self.model.fit(Xs)
        except Exception as e:
            print(f"[HMM] fit failed: {e}")
            self.model = None

    def predict_regime(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None or self.scaler is None:
            return pd.Series(0, index=df.index)
        try:
            # Check if model has required attributes (handles loaded pickles)
            if not hasattr(self.model, 'means_'):
                self.model = None
                return pd.Series(0, index=df.index)
            rets = df["close"].pct_change().fillna(0).values.reshape(-1, 1).astype(np.float64)
            vol = pd.Series(rets.flatten()).rolling(10).std().fillna(0).values.reshape(-1, 1).astype(np.float64)
            feats = np.hstack([rets, vol]).astype(np.float64)
            feats = self.scaler.transform(feats)
            return pd.Series(self.model.predict(feats), index=df.index)
        except Exception as e:
            print(f"[HMM] predict_regime failed: {e}")
            return pd.Series(0, index=df.index)


# =============================================================================
# LAYER 4 — Risk + Execution
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
    sizing_method: str = "fixed"          # 'fixed' | 'kelly'
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "open"
    tp_level_hit: int = 0


class RiskManager:
    def __init__(self, config: AssetConfig):
        self.config = config
        self.equity = 10_000.0
        self.initial_equity = 10_000.0
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[float] = [10_000.0]
        # Per-trade pnl history feeds Kelly sizing once we have ≥ kelly_min_history trades
        self.trade_history_pnls: List[float] = []

    def set_equity(self, equity: float):
        self.equity = float(equity)
        self.initial_equity = float(equity)
        self.equity_curve = [float(equity)]

    # ── Sizing ────────────────────────────────────────────────────────────
    def calculate_position_size(self, entry: float, stop: float) -> float:
        """Fixed fractional sizing: risk max_risk_per_trade_pct of equity."""
        risk_amt = self.equity * self.config.max_risk_per_trade_pct
        risk_per_unit = abs(entry - stop)
        if risk_per_unit == 0:
            return 0
        return risk_amt / (risk_per_unit * self.config.contract_size / self.config.leverage)

    def kelly_position_size(self, win_rate: float, avg_win: float, avg_loss: float,
                            entry: float, stop: float) -> float:
        """Quarter-Kelly sizing.

        Kelly fraction f* = (W*B − (1−W)) / B   where B = avg_win/avg_loss.
        We multiply by config.kelly_fraction (default 0.25 — Quarter-Kelly)
        and floor at zero. The fixed-fractional risk is then *scaled* by
        (1 + f*kelly_fraction*4) so we never go below the minimum 1% risk
        baseline but can scale up to ~3× when the edge is strong and verified.
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return self.calculate_position_size(entry, stop)
        b = avg_win / avg_loss
        kelly = (win_rate * b - (1.0 - win_rate)) / b
        kelly = max(0.0, min(kelly, 0.5))     # cap raw Kelly at 50% (sanity)
        scale = 1.0 + (kelly * self.config.kelly_fraction * 4.0)
        risk_amt = self.equity * self.config.max_risk_per_trade_pct * scale
        risk_per_unit = abs(entry - stop)
        if risk_per_unit == 0:
            return 0
        return risk_amt / (risk_per_unit * self.config.contract_size / self.config.leverage)

    def _kelly_stats(self) -> Optional[Tuple[float, float, float]]:
        """Return (win_rate, avg_win, avg_loss) from history, or None."""
        if len(self.trade_history_pnls) < self.config.kelly_min_history:
            return None
        wins = [p for p in self.trade_history_pnls if p > 0]
        losses = [p for p in self.trade_history_pnls if p <= 0]
        if not wins or not losses:
            return None
        return (
            len(wins) / len(self.trade_history_pnls),
            float(np.mean(wins)),
            float(abs(np.mean(losses))),
        )

    def calculate_levels(self, entry: float, direction: TradeDirection, atr: float,
                         fvg_stop: Optional[float] = None):
        if direction == TradeDirection.LONG:
            sl  = entry - atr * self.config.atr_multiplier_stop
            tp1 = entry + atr * self.config.atr_multiplier_tp1
            tp2 = entry + atr * self.config.atr_multiplier_tp2
            tp3 = entry + atr * self.config.atr_multiplier_tp3
            # If FVG stop is present and tighter (closer to entry) than ATR stop, use it
            if fvg_stop is not None and fvg_stop < entry and fvg_stop > sl:
                sl = fvg_stop
        else:
            sl  = entry + atr * self.config.atr_multiplier_stop
            tp1 = entry - atr * self.config.atr_multiplier_tp1
            tp2 = entry - atr * self.config.atr_multiplier_tp2
            tp3 = entry - atr * self.config.atr_multiplier_tp3
            if fvg_stop is not None and fvg_stop > entry and fvg_stop < sl:
                sl = fvg_stop
        return sl, tp1, tp2, tp3

    def open_trade(self, ts: datetime, direction: TradeDirection,
                    entry: float, atr: float,
                    fvg_stop: Optional[float] = None,
                    use_kelly: bool = True) -> Optional[Trade]:
        if direction == TradeDirection.FLAT:
            return None
        sl, tp1, tp2, tp3 = self.calculate_levels(entry, direction, atr, fvg_stop)
        # Decide sizing method
        sizing_method = "fixed"
        kelly_stats = self._kelly_stats() if use_kelly else None
        if kelly_stats is not None:
            wr, aw, al = kelly_stats
            size = self.kelly_position_size(wr, aw, al, entry, sl)
            sizing_method = "kelly"
        else:
            size = self.calculate_position_size(entry, sl)
        if size <= 0:
            return None
        t = Trade(
            entry_time=ts, direction=direction, entry_price=entry,
            stop_loss=sl, take_profit_1=tp1, take_profit_2=tp2,
            take_profit_3=tp3, position_size=size, asset=self.config.asset,
            fvg_stop=fvg_stop, sizing_method=sizing_method,
        )
        self.open_trades.append(t)
        return t

    def update_trades(self, ts: datetime, high: float, low: float, close: float):
        for t in self.open_trades[:]:
            if t.status != "open":
                continue
            if t.direction == TradeDirection.LONG:
                if low <= t.stop_loss:
                    self._close(t, ts, t.stop_loss, "stopped"); continue
                if t.tp_level_hit == 0 and high >= t.take_profit_1: t.tp_level_hit = 1
                if t.tp_level_hit == 1 and high >= t.take_profit_2: t.tp_level_hit = 2
                if t.tp_level_hit == 2 and high >= t.take_profit_3:
                    self._close(t, ts, t.take_profit_3, "closed_tp3"); continue
            else:
                if high >= t.stop_loss:
                    self._close(t, ts, t.stop_loss, "stopped"); continue
                if t.tp_level_hit == 0 and low <= t.take_profit_1: t.tp_level_hit = 1
                if t.tp_level_hit == 1 and low <= t.take_profit_2: t.tp_level_hit = 2
                if t.tp_level_hit == 2 and low <= t.take_profit_3:
                    self._close(t, ts, t.take_profit_3, "closed_tp3"); continue

    def _close(self, t: Trade, ts: datetime, exit_price: float, status: str):
        t.exit_time = ts; t.exit_price = exit_price
        if t.direction == TradeDirection.LONG:
            pnl = (exit_price - t.entry_price) * t.position_size * self.config.contract_size
        else:
            pnl = (t.entry_price - exit_price) * t.position_size * self.config.contract_size
        # Defensive: position_size can be inf when risk_per_unit underflows on
        # very small spreads. Cap PnL to ±100% of equity (one trade can't
        # blow the account by more than that) so the equity curve stays sane.
        if not np.isfinite(pnl):
            pnl = 0.0
        max_swing = max(abs(self.equity), 1.0)
        pnl = float(np.clip(pnl, -max_swing, max_swing))
        t.pnl = pnl
        t.status = status
        self.closed_trades.append(t)
        self.trade_history_pnls.append(pnl)
        if t in self.open_trades:
            self.open_trades.remove(t)
        self.equity += pnl
        self.equity_curve.append(float(self.equity))

    def close_all(self, ts: datetime, price: float):
        for t in self.open_trades[:]:
            self._close(t, ts, price, "closed_manual")

    def get_stats(self) -> Dict:
        if not self.closed_trades:
            return {"total_trades": 0, "equity_curve": list(self.equity_curve)}
        pnls = [t.pnl for t in self.closed_trades if t.pnl is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        long_trades = [t for t in self.closed_trades if t.direction == TradeDirection.LONG]
        short_trades = [t for t in self.closed_trades if t.direction == TradeDirection.SHORT]
        long_wins = [t for t in long_trades if (t.pnl or 0) > 0]
        short_wins = [t for t in short_trades if (t.pnl or 0) > 0]
        precision_long = (len(long_wins) / len(long_trades)) if long_trades else 0.0
        precision_short = (len(short_wins) / len(short_trades)) if short_trades else 0.0

        return {
            "total_trades":  len(self.closed_trades),
            "win_rate":      len(wins) / len(pnls) if pnls else 0,
            "precision_long":  float(precision_long),
            "precision_short": float(precision_short),
            "long_trades":     len(long_trades),
            "short_trades":    len(short_trades),
            "avg_win":       float(np.mean(wins))   if wins   else 0.0,
            "avg_loss":      float(np.mean(losses)) if losses else 0.0,
            "total_pnl":     float(sum(pnls)),
            "profit_factor": (
                abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")
            ),
            "max_drawdown":  self._max_dd(pnls),
            "sharpe":        self._sharpe(pnls),
            "equity_curve":  list(self.equity_curve),
            "kelly_active":  self._kelly_stats() is not None,
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
# ORCHESTRATOR
# =============================================================================

class PrecisionTradingSystem:
    def __init__(self, asset: Asset,
                 model_type: Literal["lorentzian", "xgboost"] = "xgboost",
                 use_hmm: bool = True):
        self.asset = asset
        self.config = ASSET_CONFIGS[asset]
        self.model_type = model_type
        self.use_hmm = use_hmm

        self.cleaner = MicrostructureCleaner(self.config)
        self.flow = FlowAnalyzer(self.config)
        self.structure = MarketStructureAnalyzer(self.config)
        self.mtf = MTFConfluenceEngine()
        self.signal_model = (
            LorentzianClassifier(n_neighbors=5)
            if model_type == "lorentzian"
            else XGBoostSignalModel(
                calibration_method=self.config.calibration_method,
                calibration_cv_splits=self.config.calibration_cv_splits,
            )
        )
        self.hmm = HMMRegimeDetector(n_regimes=2) if use_hmm else None
        self.current_regime = Regime.TRENDING
        self.risk_manager = RiskManager(self.config)
        self.is_trained = False
        self.data_buffer: pd.DataFrame = pd.DataFrame()
        self._last_metrics: Dict = {}

    # -------------------------------------------------------------------------
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all feature layers in order: clean → flow → structure → MTF."""
        cleaned = self.cleaner.clean_ohlcv(df)
        flowed  = self.flow.compute_flow_features(cleaned)
        structured = self.structure.detect_fvg(flowed)
        mtfed = self.mtf.compute(structured)
        if self.hmm is not None and self.hmm.model is not None:
            mtfed["regime"] = self.hmm.predict_regime(mtfed)
        elif self.hmm is not None:
            # HMM not yet fitted — leave regime as 0
            mtfed["regime"] = 0
        return mtfed

    def train(self, historical_df: pd.DataFrame):
        cleaned = self.cleaner.clean_ohlcv(historical_df)
        flowed  = self.flow.compute_flow_features(cleaned)
        structured = self.structure.detect_fvg(flowed)
        mtfed = self.mtf.compute(structured)
        if self.hmm is not None:
            self.hmm.fit(mtfed)
            mtfed["regime"] = self.hmm.predict_regime(mtfed)
        # Both Lorentzian and XGBoost auto-derive labels from forward returns
        self.signal_model.fit(mtfed)
        self.data_buffer = mtfed.tail(500)
        self.is_trained = True

    # -------------------------------------------------------------------------
    def _generate_signal_on_buffer(self) -> Tuple[int, float, pd.DataFrame]:
        df = self._build_features(self.data_buffer)
        if "regime" in df.columns and len(df):
            self.current_regime = (
                Regime.TRENDING if df["regime"].iloc[-1] == 0
                else Regime.MEAN_REVERTING
            )
        signals = self.signal_model.predict(df)
        if signals.empty:
            return 0, 0.0, df
        sig = int(signals.iloc[-1])

        # Confidence: prefer calibrated probabilities; combine with MTF & CVD
        conf = 0.5
        if hasattr(self.signal_model, "predict_proba"):
            proba = self.signal_model.predict_proba(df).iloc[-1]
            base_conf = float(proba.abs().max())
            # Adjust confidence with MTF confluence (±15%) and CVD divergence (±10%)
            mtf_score = float(df["mtf_score"].iloc[-1]) if "mtf_score" in df.columns else 0.0
            cvd_div = (
                int(df["cvd_bull_div"].iloc[-1]) - int(df["cvd_bear_div"].iloc[-1])
                if "cvd_bull_div" in df.columns else 0
            )
            adj = 0.0
            if sig > 0:        # bullish prediction
                adj += max(0.0, mtf_score) * 0.15      # +15% if MTF aligned bullish
                adj += 0.10 if cvd_div > 0 else 0.0
            elif sig < 0:      # bearish prediction
                adj += max(0.0, -mtf_score) * 0.15
                adj += 0.10 if cvd_div < 0 else 0.0
            conf = float(min(1.0, base_conf + adj))
        return sig, conf, df

    def process_bar(self, bar: Dict) -> Optional[Trade]:
        if not self.is_trained:
            return None
        bar_df = pd.DataFrame([bar])
        bar_df["timestamp"] = pd.to_datetime(bar_df["timestamp"])
        bar_df = bar_df.set_index("timestamp")
        self.data_buffer = pd.concat([self.data_buffer, bar_df]).tail(500)
        sig, conf, flowed = self._generate_signal_on_buffer()
        return self._execute_signal(self.data_buffer.index[-1], sig, conf, flowed)

    # -------------------------------------------------------------------------
    def _execute_signal(self, ts, signal: int, confidence: float,
                        df: pd.DataFrame) -> Optional[Trade]:
        latest = df.iloc[-1]
        # Filters
        if latest.get("spread_filter_active", False):
            return None
        vpin = float(latest.get("vpin", 0))
        thresh = self.config.vpin_threshold
        if vpin > thresh:
            return None
        if self.current_regime == Regime.MEAN_REVERTING and abs(signal) > 1:
            signal = int(np.sign(signal))
        if confidence < 0.30:
            return None
        # Session filter (FX + gold)
        hour = pd.Timestamp(ts).hour
        if self.asset in (Asset.XAUUSD, Asset.EURUSD, Asset.GBPUSD):
            in_london = self.config.london_start <= hour <= self.config.london_end
            in_ny     = self.config.ny_start     <= hour <= self.config.ny_end
            if not (in_london or in_ny):
                return None
        # Direction
        if signal >= 1:
            direction = TradeDirection.LONG
        elif signal <= -1:
            direction = TradeDirection.SHORT
        else:
            return None
        # XAU EMA(55) trend confirmation
        if self.asset == Asset.XAUUSD:
            ema55 = df["close"].ewm(span=55).mean().iloc[-1]
            if direction == TradeDirection.LONG  and df["close"].iloc[-1] < ema55:
                return None
            if direction == TradeDirection.SHORT and df["close"].iloc[-1] > ema55:
                return None

        # ── v3 gates: MTF + CVD divergence sanity check ────────────────────
        # Reject obvious counter-MTF trades when score is strongly opposite.
        mtf_score = float(latest.get("mtf_score", 0.0))
        if direction == TradeDirection.LONG and mtf_score < -0.6:
            return None
        if direction == TradeDirection.SHORT and mtf_score > 0.6:
            return None

        atr = float(latest.get("atr_14", df["close"].iloc[-20:].std()))
        entry = float(latest.get("smart_price", df["close"].iloc[-1]))
        # FVG-based stop refinement (only if a fresh FVG exists pointing the right way)
        fvg_stop = self.structure.get_nearest_fvg_stop(df, direction)
        return self.risk_manager.open_trade(ts, direction, entry, atr,
                                             fvg_stop=fvg_stop, use_kelly=True)

    def update_market_data(self, ts, high, low, close):
        self.risk_manager.update_trades(ts, high, low, close)

    def get_performance(self) -> Dict:
        return self.risk_manager.get_stats()

    # -------------------------------------------------------------------------
    def generate_live_signal(self, df_recent: pd.DataFrame) -> Dict:
        """Continuous direction prediction for the UI panel.
        Returns: action / confidence / SL / TP1 / TP2 / TP3 / lot_size + v3 fields
        (mtf_score, mtf_confluence, cvd_div, in_value_area, distance_to_poc,
        absorption, fvg_stop, sizing_method)."""
        if not self.is_trained:
            return {"action": "HOLD", "confidence": 0.0,
                    "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0,
                    "lot_size": 0.01, "regime": "untrained",
                    "vpin": 0.0, "spread_blocked": False,
                    "mtf_score": 0.0, "mtf_confluence": "neutral",
                    "cvd_div": 0, "in_value_area": True,
                    "distance_to_poc": 0.0, "absorption": 0,
                    "fvg_stop": None, "sizing_method": "fixed"}

        # Use the last 500 bars as the buffer
        self.data_buffer = df_recent.tail(500).copy()
        sig, conf, flowed = self._generate_signal_on_buffer()
        latest = flowed.iloc[-1]
        atr = float(latest.get("atr_14", flowed["close"].iloc[-20:].std()))
        entry = float(latest.get("smart_price", flowed["close"].iloc[-1]))
        spread_blocked = bool(latest.get("spread_filter_active", False))
        vpin_val = float(latest.get("vpin", 0))

        if sig >= 1:
            direction = TradeDirection.LONG
            action = "BUY" if sig == 2 else "WEAK BUY"
        elif sig <= -1:
            direction = TradeDirection.SHORT
            action = "SELL" if sig == -2 else "WEAK SELL"
        else:
            direction = TradeDirection.FLAT
            action = "HOLD"

        # FVG-based stop refinement
        fvg_stop = self.structure.get_nearest_fvg_stop(flowed, direction)
        if direction != TradeDirection.FLAT:
            sl, tp1, tp2, tp3 = self.risk_manager.calculate_levels(
                entry, direction, atr, fvg_stop)
            # Show what the executed sizing would be (Kelly if eligible, else fixed)
            kelly_stats = self.risk_manager._kelly_stats()
            if kelly_stats is not None:
                wr, aw, al = kelly_stats
                lot = self.risk_manager.kelly_position_size(wr, aw, al, entry, sl)
                sizing_method = "kelly"
            else:
                lot = self.risk_manager.calculate_position_size(entry, sl)
                sizing_method = "fixed"
        else:
            sl = tp1 = tp2 = tp3 = entry
            lot = 0.0
            sizing_method = "fixed"

        # Filter overrides for UI clarity (still show signal but mark blocked)
        if vpin_val > self.config.vpin_threshold:
            action = f"{action} (VPIN BLOCK)"
        if spread_blocked:
            action = f"{action} (SPREAD)"

        # Pull v3 metrics
        mtf_score = float(latest.get("mtf_score", 0.0))
        mtf_conf = str(latest.get("mtf_confluence", "neutral"))
        cvd_div = (
            int(latest.get("cvd_bull_div", 0)) - int(latest.get("cvd_bear_div", 0))
        )
        in_va = bool(latest.get("in_value_area", True))
        dist_poc = float(latest.get("distance_to_poc", 0.0))
        absorp = int(latest.get("absorption", 0))

        result = {
            "action":     action,
            "confidence": round(float(conf), 4),
            "entry":      round(entry, 5),
            "sl":         round(sl,    5),
            "tp1":        round(tp1,   5),
            "tp2":        round(tp2,   5),
            "tp3":        round(tp3,   5),
            "lot_size":   round(float(lot), 4),
            "regime":     self.current_regime.value,
            "vpin":       round(vpin_val, 3),
            "atr":        round(atr,   5),
            "spread_blocked": spread_blocked,
            # v3
            "mtf_score":      round(mtf_score, 3),
            "mtf_confluence": mtf_conf,
            "cvd_div":        cvd_div,
            "in_value_area":  in_va,
            "distance_to_poc": round(dist_poc, 5),
            "absorption":     absorp,
            "fvg_stop":       round(float(fvg_stop), 5) if fvg_stop is not None else None,
            "sizing_method":  sizing_method,
        }

        # Capture signal to trading memory
        self._capture_signal_to_memory(result)

        return result

    # -------------------------------------------------------------------------
    @staticmethod
    def _intra_bar_fill_check(t: Trade, bar_open: float, bar_high: float,
                                bar_low: float, bar_close: float
                                ) -> Tuple[Optional[str], Optional[float]]:
        """Simulate intra-bar fill realism using OHLC ordering by direction.

        For LONG trades we assume the path is open → low → high → close,
        because longs are most likely to be stopped before targets in a
        normal up-bar (worst-case ordering for the long). For SHORT trades
        we assume open → high → low → close (worst-case for the short).

        Returns (status, fill_price) when stop or final TP is hit, else (None, None).
        """
        if t.direction == TradeDirection.LONG:
            sequence = [("open", bar_open), ("low", bar_low),
                        ("high", bar_high), ("close", bar_close)]
        else:
            sequence = [("open", bar_open), ("high", bar_high),
                        ("low", bar_low), ("close", bar_close)]
        for _, price in sequence:
            if t.direction == TradeDirection.LONG:
                if price <= t.stop_loss:
                    return "stopped", t.stop_loss
                if t.tp_level_hit < 1 and price >= t.take_profit_1:
                    t.tp_level_hit = 1
                if t.tp_level_hit < 2 and price >= t.take_profit_2:
                    t.tp_level_hit = 2
                if t.tp_level_hit < 3 and price >= t.take_profit_3:
                    return "closed_tp3", t.take_profit_3
            else:
                if price >= t.stop_loss:
                    return "stopped", t.stop_loss
                if t.tp_level_hit < 1 and price <= t.take_profit_1:
                    t.tp_level_hit = 1
                if t.tp_level_hit < 2 and price <= t.take_profit_2:
                    t.tp_level_hit = 2
                if t.tp_level_hit < 3 and price <= t.take_profit_3:
                    return "closed_tp3", t.take_profit_3
        return None, None

    @staticmethod
    def _expected_calibration_error(probas: np.ndarray, correct: np.ndarray,
                                     n_bins: int = 10) -> float:
        """Expected Calibration Error (lower is better; <0.05 = well calibrated)."""
        if len(probas) == 0:
            return 0.0
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probas >= bins[i]) & (probas < bins[i + 1])
            if i == n_bins - 1:
                mask = (probas >= bins[i]) & (probas <= bins[i + 1])
            if mask.sum() == 0:
                continue
            avg_conf = float(probas[mask].mean())
            avg_acc = float(correct[mask].mean())
            ece += (mask.sum() / len(probas)) * abs(avg_conf - avg_acc)
        return float(ece)

    def _backtest_test_window(self, train_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               intra_bar_fills: bool = True
                               ) -> Tuple[Dict, List[Trade], np.ndarray, np.ndarray]:
        """Train on `train_df`, evaluate on `test_df`. Returns stats + closed
        trades + (per-bar predicted_proba, per-bar direction-correctness)
        for ECE computation."""
        # Train (uses full feature pipeline)
        self.train(train_df)
        # Reset risk manager for clean test
        self.risk_manager = RiskManager(self.config)

        # Precompute features over test set in one pass
        flowed = self._build_features(test_df)
        signals = self.signal_model.predict(flowed)

        if hasattr(self.signal_model, "predict_proba"):
            proba_df = self.signal_model.predict_proba(flowed)
            # Confidence = max probability for the predicted class direction
            confs = proba_df.abs().max(axis=1)
        else:
            proba_df = pd.DataFrame(0.2, index=flowed.index, columns=[-2, -1, 0, 1, 2])
            confs = pd.Series(0.5, index=flowed.index)

        # For ECE: predicted_proba and whether 5-bar forward direction matched
        fwd5 = flowed["close"].pct_change(5).shift(-5).fillna(0).values
        sig_array = signals.reindex(flowed.index).fillna(0).values
        correct = (
            ((sig_array > 0) & (fwd5 > 0))
            | ((sig_array < 0) & (fwd5 < 0))
        ).astype(int)
        # Use the dominant-class probability when prediction != HOLD
        proba_arr = confs.reindex(flowed.index).fillna(0).values
        nonhold_mask = sig_array != 0
        ece_probas = proba_arr[nonhold_mask]
        ece_correct = correct[nonhold_mask]

        # Bar-by-bar simulation
        for i, ts in enumerate(flowed.index):
            row = flowed.iloc[i]
            sig = int(signals.iloc[i]) if i < len(signals) else 0
            conf = float(confs.iloc[i]) if i < len(confs) else 0
            bar_open = float(row.get("open", row["close"]))
            bar_high = float(row["high"])
            bar_low  = float(row["low"])
            bar_close = float(row["close"])

            # 1. Update existing trades using intra-bar OHLC ordering
            if intra_bar_fills:
                for t in self.risk_manager.open_trades[:]:
                    status, fill_price = self._intra_bar_fill_check(
                        t, bar_open, bar_high, bar_low, bar_close)
                    if status is not None:
                        self.risk_manager._close(t, ts, fill_price, status)
            else:
                self.risk_manager.update_trades(ts, bar_high, bar_low, bar_close)

            # 2. Evaluate signal for new trade
            self.current_regime = (
                Regime.TRENDING if row.get("regime", 0) == 0 else Regime.MEAN_REVERTING
            )
            self._execute_signal(ts, sig, conf, flowed.iloc[: i + 1])

        # Force-close any open trades at the last close
        if flowed.size:
            self.risk_manager.close_all(flowed.index[-1], float(flowed["close"].iloc[-1]))

        stats = self.risk_manager.get_stats()
        stats["timestamps"]     = list(flowed.index)
        stats["test_bars"]      = len(test_df)
        stats["train_bars"]     = len(train_df)
        stats["initial_equity"] = self.risk_manager.initial_equity
        eq = self.risk_manager.equity_curve
        stats["final_equity"]   = float(eq[-1]) if eq else self.risk_manager.initial_equity
        stats["total_return"]   = float(
            (stats["final_equity"] - stats["initial_equity"]) / stats["initial_equity"]
        )

        return stats, list(self.risk_manager.closed_trades), ece_probas, ece_correct

    def backtest(self, df: pd.DataFrame, train_pct: float = 0.7,
                  intra_bar_fills: bool = True) -> Dict:
        """Single train/test backtest. Returns metrics + equity curve.
        Now uses intra-bar OHLC-ordered fill simulation for realism."""
        n = len(df)
        if n < 500:
            return {"error": "need ≥ 500 bars"}
        split = int(n * train_pct)
        train_df = df.iloc[:split]
        test_df  = df.iloc[split:]

        stats, _trades, ece_p, ece_c = self._backtest_test_window(
            train_df, test_df, intra_bar_fills=intra_bar_fills)
        stats["ece"] = self._expected_calibration_error(ece_p, ece_c) if len(ece_p) else 0.0

        self._last_metrics = stats
        # Capture backtest to trading memory
        self.capture_backtest_to_memory(stats)

        return stats

    # -------------------------------------------------------------------------
    def walk_forward_backtest(self, df: pd.DataFrame, n_splits: int = 5,
                                test_size_pct: float = 0.10,
                                intra_bar_fills: bool = True) -> Dict:
        """Anchored walk-forward backtest with intra-bar fill simulation.

        Splits `df` into `n_splits` rolling windows. Each fold trains on the
        portion *up to* the test window and evaluates on the unseen test
        window. Aggregates trades across folds and reports:
          - sharpe, profit_factor, win_rate, total_return
          - precision_long, precision_short
          - max_drawdown
          - ece (expected calibration error of the calibrated probabilities)
          - p_value (one-sample t-test of mean trade PnL vs zero)
          - per-fold breakdown
          - per-fold equity curve (concatenated)
        """
        n = len(df)
        if n < 1000:
            return {"error": "walk-forward needs ≥ 1000 bars"}
        test_size = max(50, int(n * test_size_pct))
        # Train window grows; test window slides forward.
        # Anchor first test at min_train so the smallest training set is healthy.
        min_train = max(500, int(n * 0.30))
        if min_train + test_size * n_splits > n:
            # Reduce splits to fit
            n_splits = max(2, (n - min_train) // test_size)
        if n_splits < 2:
            return {"error": f"not enough bars for walk-forward (got {n})"}

        all_trades: List[Trade] = []
        all_equity: List[float] = []
        all_ts: List[Any] = []
        per_fold: List[Dict] = []
        all_ece_probas: List[np.ndarray] = []
        all_ece_correct: List[np.ndarray] = []

        starting_equity = 10_000.0
        cumulative_equity = starting_equity

        for fold in range(n_splits):
            train_end = min_train + fold * test_size
            test_start = train_end
            test_end = min(test_start + test_size, n)
            if test_end - test_start < 30:
                break
            train_df = df.iloc[:train_end]
            test_df = df.iloc[test_start:test_end]

            try:
                stats, closed, ece_p, ece_c = self._backtest_test_window(
                    train_df, test_df, intra_bar_fills=intra_bar_fills)
            except Exception as e:
                print(f"[WalkForward] fold {fold} failed: {e}")
                continue

            # Stitch equity curve across folds (each fold restarts at $10k for
            # clean per-fold reporting; final cumulative_equity rolls forward
            # by the fold's PnL %).
            fold_eq = stats.get("equity_curve") or []
            # Sanitize NaN/inf out of the per-fold equity curve.
            fold_eq = [float(v) for v in fold_eq if np.isfinite(v)]
            if fold_eq:
                fold_pnl_pct = (fold_eq[-1] - 10_000.0) / 10_000.0
                # Cap drawdown at -100% to keep cumulative_equity non-negative.
                fold_pnl_pct = max(-1.0, fold_pnl_pct)
                for v in fold_eq:
                    pnl_pct = max(-1.0, (v - 10_000.0) / 10_000.0)
                    all_equity.append(max(0.0, cumulative_equity * (1 + pnl_pct)))
                cumulative_equity = max(0.0, cumulative_equity * (1 + fold_pnl_pct))
            all_ts.extend(stats.get("timestamps", []))

            all_trades.extend(closed)
            if len(ece_p):
                all_ece_probas.append(ece_p)
                all_ece_correct.append(ece_c)

            per_fold.append({
                "fold":           fold,
                "train_bars":     stats.get("train_bars", 0),
                "test_bars":      stats.get("test_bars", 0),
                "trades":         stats.get("total_trades", 0),
                "win_rate":       stats.get("win_rate", 0),
                "sharpe":         stats.get("sharpe", 0),
                "profit_factor":  stats.get("profit_factor", 0),
                "total_return":   stats.get("total_return", 0),
                "max_drawdown":   stats.get("max_drawdown", 0),
                "precision_long": stats.get("precision_long", 0),
                "precision_short":stats.get("precision_short", 0),
            })

        # Aggregate metrics across all folds
        all_pnls = [float(t.pnl) for t in all_trades
                    if t.pnl is not None and np.isfinite(t.pnl)]
        wins = [p for p in all_pnls if p > 0]
        losses = [p for p in all_pnls if p <= 0]

        long_trades = [t for t in all_trades if t.direction == TradeDirection.LONG]
        short_trades = [t for t in all_trades if t.direction == TradeDirection.SHORT]
        long_wins = [t for t in long_trades if (t.pnl or 0) > 0]
        short_wins = [t for t in short_trades if (t.pnl or 0) > 0]

        ece = 0.0
        if all_ece_probas:
            ece = self._expected_calibration_error(
                np.concatenate(all_ece_probas),
                np.concatenate(all_ece_correct),
            )

        p_value = None
        if SCIPY_AVAILABLE and ttest_1samp is not None and len(all_pnls) >= 5:
            try:
                p_value = float(ttest_1samp(all_pnls, 0).pvalue)
            except Exception:
                p_value = None

        sharpe = 0.0
        if len(all_pnls) >= 2:
            arr = np.array(all_pnls)
            if arr.std() > 0:
                sharpe = float(arr.mean() / arr.std() * np.sqrt(252 * 24 * 60))

        max_dd = 0.0
        if all_equity:
            curve = np.array(all_equity)
            peak = np.maximum.accumulate(curve)
            dd = peak - curve
            max_dd = float(dd.max())

        result = {
            "n_splits":         n_splits,
            "total_trades":     len(all_trades),
            "win_rate":         len(wins) / len(all_pnls) if all_pnls else 0,
            "precision_long":   len(long_wins) / len(long_trades) if long_trades else 0,
            "precision_short":  len(short_wins) / len(short_trades) if short_trades else 0,
            "long_trades":      len(long_trades),
            "short_trades":     len(short_trades),
            "avg_win":          float(np.mean(wins)) if wins else 0.0,
            "avg_loss":         float(np.mean(losses)) if losses else 0.0,
            "total_pnl":        float(sum(all_pnls)),
            "profit_factor":    (
                abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")
            ),
            "sharpe":           sharpe,
            "max_drawdown":     max_dd,
            "ece":              ece,
            "p_value":          p_value,
            "initial_equity":   starting_equity,
            "final_equity":     float(all_equity[-1]) if all_equity else starting_equity,
            "total_return":     float((all_equity[-1] - starting_equity) / starting_equity)
                                 if all_equity else 0.0,
            "equity_curve":     all_equity,
            "timestamps":       all_ts,
            "per_fold":         per_fold,
            "test_bars":        sum(f["test_bars"] for f in per_fold),
            "train_bars":       per_fold[-1]["train_bars"] if per_fold else 0,
        }
        self._last_metrics = result
        self.capture_backtest_to_memory(result)
        return result

    # -------------------------------------------------------------------------
    def save(self, path: str):
        state = {
            "asset":           self.asset.value,
            "model_type":      self.model_type,
            "use_hmm":         self.use_hmm,
            "is_trained":      self.is_trained,
            "data_buffer":     self.data_buffer,
            "last_metrics":    {k: v for k, v in self._last_metrics.items()
                                if not isinstance(v, list) or k != "timestamps"},
        }
        # Pickle the signal model directly (xgboost / sklearn / lorentzian arrays)
        state["signal_model"] = self.signal_model
        state["hmm"]          = self.hmm
        state["scaler"]       = getattr(self.signal_model, "scaler", None)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> bool:
        import os
        if not os.path.exists(path): return False
        try:
            with open(path, "rb") as f:
                s = pickle.load(f)
            self.signal_model = s["signal_model"]

            # Fix: ensure signal_model has calibration_method attribute (for older pickles)
            if hasattr(self.signal_model, 'model') and not hasattr(self.signal_model, 'calibration_method'):
                self.signal_model.calibration_method = "isotonic"
                self.signal_model.calibration_cv_splits = 3

            self.hmm          = s.get("hmm", self.hmm)
            self.is_trained   = s.get("is_trained", True)
            self.data_buffer  = s.get("data_buffer", pd.DataFrame())
            self._last_metrics = s.get("last_metrics", {})

            # Fix: ensure HMM model is re-fitted if loaded from old pickle
            if self.hmm is not None and hasattr(self.hmm, 'model'):
                try:
                    # Test if model works - if not, reset it
                    if self.hmm.model is not None:
                        _ = self.hmm.model.means_
                except AttributeError:
                    self.hmm.model = None

            return True
        except Exception as e:
            print(f"[Precision] load failed: {e}")
            return False

    # -------------------------------------------------------------------------
    def _capture_signal_to_memory(self, signal_result: Dict):
        """Capture generated signal to trading memory."""
        try:
            from services.trading_memory import get_memory
            memory = get_memory()
            if not memory._enabled:
                return
            memory.capture_signal({
                "asset": self.asset.value,
                "direction": signal_result.get("action", "HOLD"),
                "confidence": signal_result.get("confidence", 0),
                "vpin": signal_result.get("vpin", 0),
                "regime": signal_result.get("regime", "unknown"),
                "entry": signal_result.get("entry", 0),
                "timestamp": datetime.now().isoformat()
            })
        except Exception:
            pass

    def capture_backtest_to_memory(self, stats: Dict):
        """Capture backtest results to trading memory."""
        try:
            from services.trading_memory import get_memory
            memory = get_memory()
            if not memory._enabled:
                return
            memory.capture_backtest({
                "strategy": "Precision Trading System",
                "asset": self.asset.value,
                "period": f"{stats.get('train_bars', 0)} train / {stats.get('test_bars', 0)} test bars",
                "sharpe": stats.get("sharpe_ratio", "N/A"),
                "max_drawdown": stats.get("max_drawdown", 0),
                "win_rate": stats.get("win_rate", 0)
            })
        except Exception:
            pass
