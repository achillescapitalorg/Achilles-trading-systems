
"""
================================================================================
ENHANCED 1-MINUTE PRECISION TRADING SYSTEM v2.0
Asset Coverage: XAU/USD, BTC/USD, EUR/USD, GBP/USD
================================================================================
Research-Backed Enhancements:
    1. Stacked Heterogeneous Ensemble (Lorentzian + LSTM + GRU + Transformer 
       + ARIMA baseline → XGBoost meta-learner + residual correction)
    2. Probability Calibration (Platt/Isotonic scaling) for bias correction
    3. ICT Market Structure (FVG, Order Blocks, Liquidity Sweeps, BOS/CHoCH)
    4. Cumulative Volume Delta (CVD) + Volume Profile (POC, Value Area)
    5. Fundamental Layer (Macro calendar, Timezone regime, DXY correlation)
    6. Kelly Criterion Position Sizing + Dynamic FVG-Based Stops
    7. FinBERT Sentiment Integration
    8. Multi-Timeframe Confluence Engine (1m/5m/15m alignment)
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

# ── Optional deps; fall back gracefully ──────────────────────────────────────
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve, brier_score_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not available. ML models will use fallback logic.")

try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    print("WARNING: hmmlearn not available. HMM regime detection disabled.")

try:
    from pykalman import KalmanFilter
    PYKALMAN_AVAILABLE = True
except ImportError:
    PYKALMAN_AVAILABLE = False
    print("WARNING: pykalman not available. Using EWMA fallback for price cleaning.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: xgboost not available. Using sklearn GradientBoosting fallback.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Attention, Concatenate
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not available. Deep learning models disabled.")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("WARNING: transformers not available. FinBERT sentiment disabled.")

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("WARNING: statsmodels not available. ARIMA baseline disabled.")

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION & ENUMS
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
    NEWS_IMPACT = "news_impact"

class TradeDirection(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0

class MTFConfluence(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


@dataclass
class AssetConfig:
    """Per-asset configuration with research-backed parameters."""
    asset: Asset
    pip_value: float
    spread_avg: float
    tick_size: float
    contract_size: float = 1.0
    leverage: float = 100.0

    # Session filters (UTC)
    london_start: int = 8
    london_end: int = 17
    ny_start: int = 13
    ny_end: int = 22
    asia_start: int = 0
    asia_end: int = 8

    # Kalman
    kalman_observation_covariance: float = 1.0
    kalman_transition_covariance: float = 0.01

    # VPIN (1-50-50 standard per Easley/López de Prado)
    vpin_buckets: int = 50
    vpin_window: int = 50
    vpin_threshold: float = 0.85

    # Risk
    max_risk_per_trade_pct: float = 0.01
    kelly_fraction: float = 0.25  # Quarter-Kelly for safety
    atr_multiplier_stop: float = 1.5
    atr_multiplier_tp1: float = 1.0
    atr_multiplier_tp2: float = 2.0
    atr_multiplier_tp3: float = 3.0

    # Market Structure
    fvg_lookback: int = 10
    fvg_body_multiplier: float = 1.5
    ob_lookback: int = 5
    liquidity_sweep_lookback: int = 20

    # Ensemble
    use_lstm: bool = True
    use_gru: bool = True
    use_transformer: bool = False  # Computationally expensive
    use_arima: bool = True

    # Macro
    high_impact_news_window_minutes: int = 30
    dxy_correlation_threshold: float = 0.7


ASSET_CONFIGS: Dict[Asset, AssetConfig] = {
    Asset.XAUUSD: AssetConfig(
        asset=Asset.XAUUSD,
        pip_value=0.01, spread_avg=0.05, tick_size=0.01,
        contract_size=100.0, leverage=100.0,
        kalman_observation_covariance=0.5,
        kalman_transition_covariance=0.005,
        vpin_buckets=50, vpin_window=50, vpin_threshold=0.90,
        max_risk_per_trade_pct=0.01, kelly_fraction=0.25,
        atr_multiplier_stop=2.0,
        atr_multiplier_tp1=1.5, atr_multiplier_tp2=2.5, atr_multiplier_tp3=4.0,
        fvg_lookback=10, fvg_body_multiplier=1.5,
        use_lstm=True, use_gru=True, use_transformer=False, use_arima=True,
        high_impact_news_window_minutes=45,
        dxy_correlation_threshold=0.75,
    ),
    Asset.BTCUSD: AssetConfig(
        asset=Asset.BTCUSD,
        pip_value=1.0, spread_avg=20.0, tick_size=0.01,
        contract_size=1.0, leverage=50.0,
        kalman_observation_covariance=100.0,
        kalman_transition_covariance=1.0,
        vpin_buckets=100, vpin_window=100, vpin_threshold=0.80,
        max_risk_per_trade_pct=0.01, kelly_fraction=0.20,
        atr_multiplier_stop=2.5,
        atr_multiplier_tp1=2.0, atr_multiplier_tp2=3.5, atr_multiplier_tp3=5.0,
        fvg_lookback=10, fvg_body_multiplier=2.0,
        use_lstm=True, use_gru=True, use_transformer=False, use_arima=False,
        high_impact_news_window_minutes=30,
        dxy_correlation_threshold=0.5,
    ),
    Asset.EURUSD: AssetConfig(
        asset=Asset.EURUSD,
        pip_value=0.0001, spread_avg=0.0001, tick_size=0.00001,
        contract_size=100000.0, leverage=100.0,
        kalman_observation_covariance=0.0001,
        kalman_transition_covariance=0.000001,
        vpin_buckets=50, vpin_window=50, vpin_threshold=0.90,
        max_risk_per_trade_pct=0.01, kelly_fraction=0.25,
        atr_multiplier_stop=1.5,
        atr_multiplier_tp1=1.0, atr_multiplier_tp2=2.0, atr_multiplier_tp3=3.0,
        fvg_lookback=10, fvg_body_multiplier=1.2,
        use_lstm=True, use_gru=True, use_transformer=False, use_arima=True,
        high_impact_news_window_minutes=30,
        dxy_correlation_threshold=0.80,
    ),
    Asset.GBPUSD: AssetConfig(
        asset=Asset.GBPUSD,
        pip_value=0.0001, spread_avg=0.0002, tick_size=0.00001,
        contract_size=100000.0, leverage=100.0,
        kalman_observation_covariance=0.0002,
        kalman_transition_covariance=0.000002,
        vpin_buckets=50, vpin_window=50, vpin_threshold=0.85,
        max_risk_per_trade_pct=0.01, kelly_fraction=0.25,
        atr_multiplier_stop=1.5,
        atr_multiplier_tp1=1.0, atr_multiplier_tp2=2.0, atr_multiplier_tp3=3.0,
        fvg_lookback=10, fvg_body_multiplier=1.3,
        use_lstm=True, use_gru=True, use_transformer=False, use_arima=True,
        high_impact_news_window_minutes=30,
        dxy_correlation_threshold=0.75,
    ),
}


# =============================================================================
# LAYER 0 — DATA FUSION & MACRO CONTEXT
# =============================================================================

class FundamentalContext:
    """
    Layer 0: Fundamental and macro context integration.

    Components:
    - Economic calendar impact scoring (NFP, CPI, FOMC, ECB, BOE)
    - Timezone-based regime detection (Asia, London, NY)
    - DXY correlation proxy
    - FinBERT sentiment scoring (if transformers available)
    """

    def __init__(self, config: AssetConfig):
        self.config = config
        self.sentiment_pipeline = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="yiyanghkust/finbert-tone",
                    tokenizer="yiyanghkust/finbert-tone"
                )
            except Exception:
                pass

        # High-impact events by asset
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
        """Identify trading session regime."""
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

    def compute_news_impact_score(
        self, 
        timestamp: datetime,
        news_events: Optional[List[Dict]] = None
    ) -> float:
        """
        Compute news impact score (0-1, higher = more dangerous).

        Args:
            news_events: List of dicts with keys: 'time', 'event', 'impact' (high/medium/low), 'asset'
        """
        if news_events is None:
            return 0.0

        score = 0.0
        window = timedelta(minutes=self.config.high_impact_news_window_minutes)

        for event in news_events:
            event_time = pd.to_datetime(event['time'])
            if abs((timestamp - event_time).total_seconds()) < window.total_seconds():
                if event.get('asset', self.config.asset.value) == self.config.asset.value:
                    impact = event.get('impact', 'low')
                    if impact == 'high':
                        score += 0.5
                    elif impact == 'medium':
                        score += 0.25
                    elif impact == 'low':
                        score += 0.1

        return min(score, 1.0)

    def compute_dxy_correlation_proxy(
        self, 
        df: pd.DataFrame,
        dxy_series: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Compute rolling correlation with DXY as a feature.
        If DXY not provided, use inverse price momentum as proxy.
        """
        if dxy_series is not None and len(dxy_series) == len(df):
            corr = df['close'].pct_change().rolling(50).corr(dxy_series.pct_change())
            return corr.fillna(0)
        else:
            # Proxy: momentum divergence from typical session behavior
            returns = df['close'].pct_change()
            proxy = -returns.rolling(20).mean() * 10  # Inverted momentum proxy
            return proxy.fillna(0)

    def analyze_sentiment(self, news_texts: List[str]) -> float:
        """
        Analyze sentiment using FinBERT.
        Returns score: -1 (very negative) to +1 (very positive).
        """
        if self.sentiment_pipeline is None or not news_texts:
            return 0.0

        try:
            results = self.sentiment_pipeline(news_texts)
            scores = []
            for r in results:
                label = r['label']
                score = r['score']
                if label == 'Positive':
                    scores.append(score)
                elif label == 'Negative':
                    scores.append(-score)
                else:
                    scores.append(0)
            return np.mean(scores) if scores else 0.0
        except Exception:
            return 0.0

    def generate_context_features(
        self, 
        df: pd.DataFrame,
        news_events: Optional[List[Dict]] = None,
        dxy_series: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Generate all fundamental context features."""
        result = df.copy()

        # Timezone regime
        result['timezone_regime'] = [self.get_timezone_regime(ts) for ts in result.index]
        result['is_london'] = (result['timezone_regime'] == 'london').astype(int)
        result['is_ny'] = (result['timezone_regime'] == 'ny').astype(int)
        result['is_overlap'] = (result['timezone_regime'] == 'overlap').astype(int)

        # News impact
        if news_events:
            result['news_impact'] = [self.compute_news_impact_score(ts, news_events) 
                                     for ts in result.index]
        else:
            result['news_impact'] = 0.0

        # DXY correlation proxy
        result['dxy_correlation'] = self.compute_dxy_correlation_proxy(result, dxy_series)
        result['dxy_aligned'] = (result['dxy_correlation'].abs() > self.config.dxy_correlation_threshold).astype(int)

        # Session volatility patterns
        result['session_vol'] = result.groupby('timezone_regime')['close'].transform(
            lambda x: x.pct_change().rolling(20).std()
        ).fillna(0)

        return result


# =============================================================================
# LAYER 1 — MICROSTRUCTURE CLEANING (Enhanced)
# =============================================================================

class MicrostructureCleaner:
    """Enhanced Layer 1 with multiple cleaning methods."""

    def __init__(self, config: AssetConfig):
        self.config = config
        self.kalman = None
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
        bv = np.asarray(bid_vol); av = np.asarray(ask_vol)
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

        result["kalman_price"] = self.apply_kalman_filter(result["smart_price"].values)
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
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()


# =============================================================================
# LAYER 2 — MARKET STRUCTURE (ICT / Smart Money Concepts)
# =============================================================================

class MarketStructureAnalyzer:
    """
    Layer 2: ICT Market Structure Detection.

    Detects:
    - Fair Value Gaps (FVG)
    - Order Blocks (OB)
    - Liquidity Sweeps
    - Break of Structure (BOS) / Change of Character (CHoCH)
    - Demand/Supply Zones with strength scoring
    """

    def __init__(self, config: AssetConfig):
        self.config = config
        self.active_fvgs: List[Dict] = []
        self.active_obs: List[Dict] = []
        self.liquidity_levels: Dict[str, List[float]] = {"highs": [], "lows": []}

    def detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Fair Value Gaps using 3-candle pattern with volatility scaling.

        Bullish FVG: Candle 3 low > Candle 1 high, with strong middle candle body.
        Bearish FVG: Candle 3 high < Candle 1 low, with strong middle candle body.
        """
        result = df.copy()
        n = len(result)

        fvg_bullish = np.zeros(n, dtype=bool)
        fvg_bearish = np.zeros(n, dtype=bool)
        fvg_bull_start = np.full(n, np.nan)
        fvg_bull_end = np.full(n, np.nan)
        fvg_bear_start = np.full(n, np.nan)
        fvg_bear_end = np.full(n, np.nan)
        fvg_strength = np.zeros(n)

        highs = result['high'].values
        lows = result['low'].values
        opens = result['open'].values
        closes = result['close'].values

        for i in range(2, n):
            # Average body size over lookback
            start_idx = max(0, i - self.config.fvg_lookback)
            bodies = np.abs(closes[start_idx:i] - opens[start_idx:i])
            avg_body = np.mean(bodies) if len(bodies) > 0 else 0.001
            if avg_body == 0:
                avg_body = 0.001

            middle_body = abs(closes[i-1] - opens[i-1])

            # Bullish FVG
            if lows[i] > highs[i-2] and middle_body > avg_body * self.config.fvg_body_multiplier:
                fvg_bullish[i] = True
                fvg_bull_start[i] = highs[i-2]
                fvg_bull_end[i] = lows[i]
                fvg_strength[i] = middle_body / avg_body

            # Bearish FVG
            elif highs[i] < lows[i-2] and middle_body > avg_body * self.config.fvg_body_multiplier:
                fvg_bearish[i] = True
                fvg_bear_start[i] = lows[i-2]
                fvg_bear_end[i] = highs[i]
                fvg_strength[i] = middle_body / avg_body

        result['fvg_bullish'] = fvg_bullish
        result['fvg_bearish'] = fvg_bearish
        result['fvg_bull_start'] = fvg_bull_start
        result['fvg_bull_end'] = fvg_bull_end
        result['fvg_bear_start'] = fvg_bear_start
        result['fvg_bear_end'] = fvg_bear_end
        result['fvg_strength'] = fvg_strength

        # Track active FVGs and compute nearest FVG distance
        result['nearest_fvg_dist'] = self._compute_nearest_fvg_distance(result)
        result['in_fvg_zone'] = self._check_in_fvg_zone(result)

        return result

    def _compute_nearest_fvg_distance(self, df: pd.DataFrame) -> pd.Series:
        """Compute distance to nearest active FVG zone."""
        distances = []
        active_bull = []
        active_bear = []

        for i, row in df.iterrows():
            price = row['close']

            # Update active FVGs (mitigation check)
            if row['fvg_bullish']:
                active_bull.append({'start': row['fvg_bull_start'], 'end': row['fvg_bull_end'], 'idx': i})
            if row['fvg_bearish']:
                active_bear.append({'start': row['fvg_bear_start'], 'end': row['fvg_bear_end'], 'idx': i})

            # Remove mitigated FVGs
            active_bull = [f for f in active_bull if price > f['start']]
            active_bear = [f for f in active_bear if price < f['start']]

            # Compute min distance
            min_dist = float('inf')
            for f in active_bull:
                if f['start'] <= price <= f['end']:
                    min_dist = 0
                    break
                min_dist = min(min_dist, abs(price - f['start']))
            for f in active_bear:
                if f['end'] <= price <= f['start']:
                    min_dist = 0
                    break
                min_dist = min(min_dist, abs(price - f['start']))

            distances.append(min_dist if min_dist != float('inf') else np.nan)

        return pd.Series(distances, index=df.index).ffill().fillna(0)

    def _check_in_fvg_zone(self, df: pd.DataFrame) -> pd.Series:
        """Check if current price is inside any FVG zone."""
        in_zone = np.zeros(len(df), dtype=int)
        active_bull = []
        active_bear = []

        for idx, (i, row) in enumerate(df.iterrows()):
            price = row['close']

            if row['fvg_bullish']:
                active_bull.append((row['fvg_bull_start'], row['fvg_bull_end']))
            if row['fvg_bearish']:
                active_bear.append((row['fvg_bear_start'], row['fvg_bear_end']))

            active_bull = [(s, e) for s, e in active_bull if price > s]
            active_bear = [(s, e) for s, e in active_bear if price < s]

            for s, e in active_bull:
                if s <= price <= e:
                    in_zone[idx] = 1
            for s, e in active_bear:
                if e <= price <= s:
                    in_zone[idx] = -1

        return pd.Series(in_zone, index=df.index)

    def detect_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Order Blocks (OB) — final candle before aggressive move.

        Bullish OB: Last down candle before strong up move (3+ candles).
        Bearish OB: Last up candle before strong down move (3+ candles).
        """
        result = df.copy()
        n = len(result)

        ob_bull = np.zeros(n, dtype=bool)
        ob_bear = np.zeros(n, dtype=bool)
        ob_strength = np.zeros(n)

        closes = result['close'].values
        opens = result['open'].values
        highs = result['high'].values
        lows = result['low'].values

        for i in range(self.config.ob_lookback + 1, n):
            # Look for bullish OB
            if closes[i] > opens[i] and closes[i] > closes[i-1] * 1.001:
                # Find last bearish candle before this move
                for j in range(i-1, max(i-self.config.ob_lookback, -1), -1):
                    if closes[j] < opens[j]:
                        # Check if subsequent move is strong
                        move_pct = (closes[i] - closes[j]) / closes[j]
                        if move_pct > 0.002:
                            ob_bull[j] = True
                            ob_strength[j] = move_pct * 100
                        break

            # Look for bearish OB
            if closes[i] < opens[i] and closes[i] < closes[i-1] * 0.999:
                for j in range(i-1, max(i-self.config.ob_lookback, -1), -1):
                    if closes[j] > opens[j]:
                        move_pct = (closes[j] - closes[i]) / closes[j]
                        if move_pct > 0.002:
                            ob_bear[j] = True
                            ob_strength[j] = move_pct * 100
                        break

        result['ob_bullish'] = ob_bull
        result['ob_bearish'] = ob_bear
        result['ob_strength'] = ob_strength
        return result

    def detect_liquidity_sweeps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect liquidity sweeps — price briefly breaks a recent high/low 
        then immediately reverses.

        Returns sweep signals: +1 for bullish sweep (swept lows, reversed up),
        -1 for bearish sweep (swept highs, reversed down).
        """
        result = df.copy()
        n = len(result)

        sweep_signal = np.zeros(n, dtype=int)
        sweep_strength = np.zeros(n)

        highs = result['high'].values
        lows = result['low'].values
        closes = result['close'].values

        lookback = self.config.liquidity_sweep_lookback

        for i in range(lookback + 2, n):
            # Recent range
            recent_highs = highs[i-lookback:i]
            recent_lows = lows[i-lookback:i]

            prev_high = np.max(recent_highs[:-1])
            prev_low = np.min(recent_lows[:-1])

            # Bullish sweep: wick below prev low, close back above
            if lows[i] < prev_low and closes[i] > prev_low and closes[i] > opens[i]:
                sweep_signal[i] = 1
                sweep_strength[i] = (closes[i] - lows[i]) / (prev_low * 0.01)

            # Bearish sweep: wick above prev high, close back below
            elif highs[i] > prev_high and closes[i] < prev_high and closes[i] < opens[i]:
                sweep_signal[i] = -1
                sweep_strength[i] = (highs[i] - closes[i]) / (prev_high * 0.01)

        result['sweep_signal'] = sweep_signal
        result['sweep_strength'] = sweep_strength
        return result

    def detect_structure_breaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Break of Structure (BOS) and Change of Character (CHoCH).

        BOS: Price breaks previous high/low in direction of trend.
        CHoCH: Price breaks previous high/low against trend (trend change signal).
        """
        result = df.copy()
        n = len(result)

        bos_bull = np.zeros(n, dtype=bool)
        bos_bear = np.zeros(n, dtype=bool)
        choch_bull = np.zeros(n, dtype=bool)
        choch_bear = np.zeros(n, dtype=bool)

        highs = result['high'].values
        lows = result['low'].values
        closes = result['close'].values

        # Track swing highs/lows
        swing_highs = []
        swing_lows = []
        trend = 0  # 1 = up, -1 = down

        for i in range(2, n):
            # Simple swing detection
            if highs[i-1] > highs[i-2] and highs[i-1] > highs[i]:
                swing_highs.append((i-1, highs[i-1]))
            if lows[i-1] < lows[i-2] and lows[i-1] < lows[i]:
                swing_lows.append((i-1, lows[i-1]))

            # Determine trend from recent swings
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                last_sh = swing_highs[-1][1]
                prev_sh = swing_highs[-2][1]
                last_sl = swing_lows[-1][1]
                prev_sl = swing_lows[-2][1]

                if last_sh > prev_sh and last_sl > prev_sl:
                    trend = 1
                elif last_sh < prev_sh and last_sl < prev_sl:
                    trend = -1

                # BOS: break in trend direction
                if trend == 1 and highs[i] > last_sh:
                    bos_bull[i] = True
                elif trend == -1 and lows[i] < last_sl:
                    bos_bear[i] = True

                # CHoCH: break against trend
                if trend == 1 and lows[i] < last_sl:
                    choch_bear[i] = True
                    trend = -1
                elif trend == -1 and highs[i] > last_sh:
                    choch_bull[i] = True
                    trend = 1

        result['bos_bull'] = bos_bull
        result['bos_bear'] = bos_bear
        result['choch_bull'] = choch_bull
        result['choch_bear'] = choch_bear
        result['market_structure_trend'] = trend
        return result

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full market structure analysis."""
        result = self.detect_fvg(df)
        result = self.detect_order_blocks(result)
        result = self.detect_liquidity_sweeps(result)
        result = self.detect_structure_breaks(result)
        return result


# =============================================================================
# LAYER 3 — FLOW & TOXICITY (Enhanced with CVD + Volume Profile)
# =============================================================================

class FlowAnalyzer:
    """
    Enhanced Layer 3 with:
    - Cumulative Volume Delta (CVD)
    - Volume Profile / Market Profile (POC, Value Area)
    - Absorption detection
    - Enhanced VPIN
    """

    def __init__(self, config: AssetConfig):
        self.config = config

    def compute_vpin(self, df: pd.DataFrame) -> pd.Series:
        """Volume-Synchronized PIN — 1-50-50 standard."""
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
        """
        Cumulative Volume Delta.
        CVD = cumsum(Buy Volume - Sell Volume)
        """
        price_change = df["close"].diff().fillna(0)
        volume = df["volume"]

        buy_vol = pd.Series(np.where(price_change > 0, volume, 0.0), index=df.index)
        sell_vol = pd.Series(np.where(price_change < 0, volume, 0.0), index=df.index)
        zero = price_change == 0
        buy_vol[zero] = volume[zero] * 0.5
        sell_vol[zero] = volume[zero] * 0.5

        delta = buy_vol - sell_vol
        cvd = delta.cumsum()
        return cvd

    def compute_cvd_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect CVD-Price divergence — key reversal signal.
        Bullish divergence: Price makes lower low, CVD makes higher low.
        Bearish divergence: Price makes higher high, CVD makes lower high.
        """
        result = df.copy()
        cvd = result['cvd'] if 'cvd' in result.columns else self.compute_cvd(result)

        # Rolling lows/highs
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

    def compute_volume_profile(self, df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
        """
        Compute rolling Volume Profile metrics: POC and Value Area.
        """
        result = df.copy()
        n = len(result)

        poc = np.full(n, np.nan)
        value_area_high = np.full(n, np.nan)
        value_area_low = np.full(n, np.nan)

        for i in range(lookback, n):
            window = result.iloc[i-lookback:i]

            # Create price-volume distribution
            # Approximate by using close prices binned
            prices = window['close'].values
            volumes = window['volume'].values

            if len(prices) < 10:
                continue

            # Simple histogram approach
            n_bins = min(20, len(prices) // 2)
            hist, bin_edges = np.histogram(prices, bins=n_bins, weights=volumes)

            # POC: price level with highest volume
            poc_idx = np.argmax(hist)
            poc[i] = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2

            # Value Area: 70% of volume around POC
            total_vol = np.sum(hist)
            target_vol = total_vol * 0.70

            # Expand from POC until 70% volume captured
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
        """
        Detect absorption — large volume with minimal price movement.
        Indicates strong hands absorbing retail flow.
        """
        volume = df['volume']
        price_range = df['high'] - df['low']

        vol_ma = volume.rolling(20).mean()
        range_ma = price_range.rolling(20).mean()

        # Absorption: volume > 2× average but range < 0.5× average
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
                df["bid_vol"], df["ask_vol"],
            )
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
# LAYER 4 — MULTI-TIMEFRAME CONFLUENCE
# =============================================================================

class MTFConfluenceEngine:
    """
    Multi-Timeframe Confluence Engine.

    Aggregates signals from 1m, 5m, and 15m to determine
    higher-probability setups.
    """

    def __init__(self):
        self.mtf_weights = {'1m': 0.5, '5m': 0.3, '15m': 0.2}

    def aggregate_structure(self, df_1m: pd.DataFrame, 
                           df_5m: Optional[pd.DataFrame] = None,
                           df_15m: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Add MTF confluence scores to 1m data.

        Scoring:
        - If 1m, 5m, and 15m all align bullish → STRONG_BULLISH
        - If 1m and 5m align but 15m neutral → BULLISH
        - Mixed signals → NEUTRAL
        """
        result = df_1m.copy()

        # Default: use only 1m if higher TFs not provided
        if df_5m is None or df_15m is None:
            result['mtf_confluence'] = MTFConfluence.NEUTRAL.value
            result['mtf_score'] = 0.0
            return result

        # Align higher TF signals to 1m timestamps
        result['tf5_trend'] = self._align_tf(result, df_5m, 'close')
        result['tf15_trend'] = self._align_tf(result, df_15m, 'close')

        # Compute trend direction for each TF
        result['trend_1m'] = np.where(
            result['close'] > result['close'].ewm(span=20).mean(), 1, -1
        )
        result['trend_5m'] = np.where(
            result['tf5_trend'] > pd.Series(result['tf5_trend']).ewm(span=20).mean(), 1, -1
        )
        result['trend_15m'] = np.where(
            result['tf15_trend'] > pd.Series(result['tf15_trend']).ewm(span=20).mean(), 1, -1
        )

        # Confluence score
        confluence = result['trend_1m'] * self.mtf_weights['1m'] +                      result['trend_5m'] * self.mtf_weights['5m'] +                      result['trend_15m'] * self.mtf_weights['15m']

        result['mtf_score'] = confluence

        # Classify
        conditions = [
            confluence > 0.6,
            confluence > 0.2,
            confluence < -0.2,
            confluence < -0.6,
        ]
        choices = [
            MTFConfluence.STRONG_BULLISH.value,
            MTFConfluence.BULLISH.value,
            MTFConfluence.BEARISH.value,
            MTFConfluence.STRONG_BEARISH.value,
        ]
        result['mtf_confluence'] = np.select(conditions, choices, default=MTFConfluence.NEUTRAL.value)

        return result

    @staticmethod
    def _align_tf(df_target: pd.DataFrame, df_source: pd.DataFrame, col: str) -> pd.Series:
        """Align higher timeframe data to lower timeframe index."""
        source = df_source[[col]].copy()
        source = source.reindex(df_target.index, method='ffill')
        return source[col]


# =============================================================================
# LAYER 5 — ML ENSEMBLE (Stacked Heterogeneous Ensemble)
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


class DeepLearningModel:
    """
    LSTM/GRU/Transformer base learner for the ensemble.
    """
    def __init__(self, model_type: str = 'lstm', sequence_length: int = 20):
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False

    def _build_model(self, input_dim: int):
        if not TF_AVAILABLE:
            return None

        inputs = Input(shape=(self.sequence_length, input_dim))

        if self.model_type == 'lstm':
            x = LSTM(64, return_sequences=False)(inputs)
        elif self.model_type == 'gru':
            x = GRU(64, return_sequences=False)(inputs)
        elif self.model_type == 'transformer':
            x = LSTM(32, return_sequences=True)(inputs)
            x = Attention()([x, x])
            x = tf.reduce_mean(x, axis=1)
        else:
            x = LSTM(64, return_sequences=False)(inputs)

        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(5, activation='softmax')(x)  # 5 classes

        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def _create_sequences(self, X: np.ndarray, y: np.ndarray):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i+self.sequence_length])
            y_seq.append(y[i+self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def fit(self, df: pd.DataFrame, feature_cols: List[str], target_horizon: int = 3):
        if not TF_AVAILABLE:
            return

        # Prepare features
        feats = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X = self.scaler.fit_transform(feats) if self.scaler else feats.values

        # Target
        fwd = df["close"].shift(-target_horizon) / df["close"] - 1
        y = pd.Series(0, index=df.index, dtype=int)
        y[fwd > 0.001] = 2
        y[(fwd <= 0.001) & (fwd > 0.0003)] = 1
        y[(fwd >= -0.001) & (fwd < -0.0003)] = -1
        y[fwd < -0.001] = -2
        y = y + 2  # 0-4

        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y.loc[feats.index].values)
        if len(X_seq) < 100:
            return

        # One-hot encode
        y_cat = tf.keras.utils.to_categorical(y_seq, num_classes=5)

        self.model = self._build_model(X.shape[1])
        if self.model is None:
            return

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model.fit(X_seq, y_cat, epochs=50, batch_size=32, 
                      validation_split=0.2, callbacks=[early_stop], verbose=0)
        self.is_trained = True

    def predict(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
        if not self.is_trained or self.model is None:
            return pd.Series(0, index=df.index)

        feats = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X = self.scaler.transform(feats) if self.scaler else feats.values

        # Create sequences with padding for initial rows
        predictions = []
        for i in range(len(X)):
            if i < self.sequence_length:
                predictions.append(0)
            else:
                seq = X[i-self.sequence_length:i].reshape(1, self.sequence_length, -1)
                pred = self.model.predict(seq, verbose=0)
                predictions.append(np.argmax(pred) - 2)

        return pd.Series(predictions, index=df.index)


class ARIMABaseline:
    """ARIMA statistical baseline for the ensemble."""
    def __init__(self, order: Tuple[int, int, int] = (2, 0, 2)):
        self.order = order
        self.model = None
        self.is_trained = False

    def fit(self, series: pd.Series):
        if not STATSMODELS_AVAILABLE:
            return
        try:
            self.model = ARIMA(series, order=self.order)
            self.model = self.model.fit()
            self.is_trained = True
        except Exception:
            self.is_trained = False

    def predict(self, series: pd.Series, steps: int = 3) -> pd.Series:
        if not self.is_trained or self.model is None:
            return pd.Series(0, index=series.index)

        predictions = []
        for i in range(len(series)):
            if i < 50:
                predictions.append(0)
                continue
            try:
                forecast = self.model.forecast(steps=steps)
                change = (forecast.iloc[-1] - series.iloc[i]) / series.iloc[i]
                if change > 0.001:
                    predictions.append(2)
                elif change > 0.0003:
                    predictions.append(1)
                elif change < -0.001:
                    predictions.append(-2)
                elif change < -0.0003:
                    predictions.append(-1)
                else:
                    predictions.append(0)
            except Exception:
                predictions.append(0)

        return pd.Series(predictions, index=series.index)


class StackedEnsemble:
    """
    Stacked Heterogeneous Ensemble.

    Base Learners: Lorentzian, LSTM, GRU, ARIMA
    Meta-Learner: XGBoost (with probability calibration)
    Residual Correction: XGBoost on meta-learner residuals
    """

    def __init__(self, config: AssetConfig):
        self.config = config
        self.lorentzian = LorentzianClassifier(n_neighbors=5)
        self.lstm = DeepLearningModel('lstm', sequence_length=20) if config.use_lstm else None
        self.gru = DeepLearningModel('gru', sequence_length=20) if config.use_gru else None
        self.transformer = DeepLearningModel('transformer', sequence_length=20) if config.use_transformer else None
        self.arima = ARIMABaseline() if config.use_arima else None

        self.meta_learner = None
        self.calibrator = None
        self.residual_model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_cols = None
        self.is_trained = False

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Extract relevant feature columns for deep learning models."""
        cols = []
        for c in ['returns_1', 'returns_3', 'returns_5', 'rsi_7', 'rsi_14', 
                  'macd_hist', 'bb_position', 'bb_width', 'ob_imbalance', 
                  'vpin', 'rv_5m', 'vol_intensity', 'signed_vol_ma5']:
            if c in df.columns:
                cols.append(c)
        return cols if cols else ['close', 'volume']

    def fit(self, df: pd.DataFrame, target_horizon: int = 3):
        """Train all base learners and meta-learner."""
        print("[Ensemble] Training base learners...")

        # Fit Lorentzian
        self.lorentzian.fit(df, target_horizon=target_horizon)

        # Fit deep learning models
        self.feature_cols = self._get_feature_cols(df)
        if self.lstm:
            print("[Ensemble] Training LSTM...")
            self.lstm.fit(df, self.feature_cols, target_horizon)
        if self.gru:
            print("[Ensemble] Training GRU...")
            self.gru.fit(df, self.feature_cols, target_horizon)
        if self.transformer:
            print("[Ensemble] Training Transformer...")
            self.transformer.fit(df, self.feature_cols, target_horizon)
        if self.arima:
            print("[Ensemble] Training ARIMA...")
            self.arima.fit(df['close'])

        # Generate base predictions for meta-learner training
        print("[Ensemble] Generating base predictions for meta-learner...")
        base_preds = self._generate_base_predictions(df)

        # Target
        fwd = df["close"].shift(-target_horizon) / df["close"] - 1
        target = pd.Series(0, index=df.index, dtype=int)
        target[fwd > 0.001] = 2
        target[(fwd <= 0.001) & (fwd > 0.0003)] = 1
        target[(fwd >= -0.001) & (fwd < -0.0003)] = -1
        target[fwd < -0.001] = -2

        # Align
        aligned_preds = base_preds.loc[target.index].dropna()
        aligned_target = target.loc[aligned_preds.index]

        # Meta-learner: XGBoost
        print("[Ensemble] Training meta-learner...")
        X_meta = aligned_preds.values
        y_meta = aligned_target.values + 2  # 0-4

        if XGBOOST_AVAILABLE:
            self.meta_learner = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                objective="multi:softprob", num_class=5,
                eval_metric="mlogloss", random_state=42,
                tree_method="hist", reg_lambda=1.0, min_child_weight=10,
            )
            self.meta_learner.fit(X_meta, y_meta)
        elif SKLEARN_AVAILABLE:
            self.meta_learner = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, random_state=42
            )
            self.meta_learner.fit(X_meta, y_meta)

        # Probability Calibration (Platt Scaling / Isotonic)
        print("[Ensemble] Calibrating probabilities...")
        if SKLEARN_AVAILABLE and self.meta_learner is not None:
            self.calibrator = CalibratedClassifierCV(
                self.meta_learner, method='isotonic', cv=3
            )
            self.calibrator.fit(X_meta, y_meta)

        # Residual Correction
        print("[Ensemble] Training residual correction...")
        meta_preds = self.meta_learner.predict(X_meta)
        residuals = y_meta - meta_preds

        if XGBOOST_AVAILABLE:
            self.residual_model = xgb.XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
            )
            self.residual_model.fit(X_meta, residuals)
        elif SKLEARN_AVAILABLE:
            self.residual_model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=42
            )
            self.residual_model.fit(X_meta, residuals)

        self.is_trained = True
        print("[Ensemble] Training complete.")

    def _generate_base_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions from all base learners."""
        preds = pd.DataFrame(index=df.index)

        preds['lorentzian'] = self.lorentzian.predict(df).reindex(df.index, fill_value=0)

        if self.lstm and self.lstm.is_trained:
            preds['lstm'] = self.lstm.predict(df, self.feature_cols).reindex(df.index, fill_value=0)
        else:
            preds['lstm'] = 0

        if self.gru and self.gru.is_trained:
            preds['gru'] = self.gru.predict(df, self.feature_cols).reindex(df.index, fill_value=0)
        else:
            preds['gru'] = 0

        if self.transformer and self.transformer.is_trained:
            preds['transformer'] = self.transformer.predict(df, self.feature_cols).reindex(df.index, fill_value=0)
        else:
            preds['transformer'] = 0

        if self.arima and self.arima.is_trained:
            preds['arima'] = self.arima.predict(df['close']).reindex(df.index, fill_value=0)
        else:
            preds['arima'] = 0

        # Add structural features as meta-features
        for col in ['fvg_strength', 'sweep_strength', 'ob_strength', 
                    'cvd_slope', 'absorption', 'news_impact']:
            if col in df.columns:
                preds[col] = df[col].fillna(0)

        return preds.fillna(0)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Generate final ensemble prediction."""
        if not self.is_trained:
            return pd.Series(0, index=df.index)

        base_preds = self._generate_base_predictions(df)
        X_meta = base_preds.values

        # Meta-learner prediction
        if self.calibrator is not None:
            proba = self.calibrator.predict_proba(X_meta)
            pred = np.argmax(proba, axis=1)
        elif self.meta_learner is not None:
            pred = self.meta_learner.predict(X_meta)
        else:
            return pd.Series(0, index=df.index)

        # Residual correction
        if self.residual_model is not None:
            try:
                residual_pred = self.residual_model.predict(X_meta)
                pred = np.clip(pred + np.round(residual_pred).astype(int), 0, 4)
            except Exception:
                pass

        return pd.Series(pred - 2, index=df.index)  # Map back to -2..2

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get calibrated probabilities."""
        if not self.is_trained or self.calibrator is None:
            return pd.DataFrame(0.2, index=df.index, columns=[-2, -1, 0, 1, 2])

        base_preds = self._generate_base_predictions(df)
        X_meta = base_preds.values
        proba = self.calibrator.predict_proba(X_meta)
        return pd.DataFrame(proba, index=df.index, columns=[-2, -1, 0, 1, 2])


class HMMRegimeDetector:
    def __init__(self, n_regimes: int = 3):  # 3 regimes: trend up, range, trend down
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
# LAYER 6 — EXECUTION & RISK (Enhanced with Kelly + FVG Stops)
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
    fvg_stop: Optional[float] = None  # Dynamic FVG-based stop
    mae: float = 0.0  # Maximum adverse excursion tracking
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "open"
    tp_level_hit: int = 0


class RiskManager:
    """
    Enhanced Risk Manager with:
    - Kelly Criterion position sizing
    - FVG-based dynamic stops
    - MAE tracking
    - Correlation risk across positions
    """

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
        """
        Quarter-Kelly position sizing for safety.
        f* = (p*b - q) / b, where b = avg_win/avg_loss
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return self.calculate_position_size(entry, stop)

        b = avg_win / avg_loss
        kelly = (win_rate * b - (1 - win_rate)) / b
        kelly = max(0, min(kelly, 0.5))  # Cap at 50%
        fraction = kelly * self.config.kelly_fraction

        risk_amt = self.equity * self.config.max_risk_per_trade_pct * (1 + fraction * 4)
        price_risk = abs(entry - stop)
        if price_risk == 0:
            return 0
        return risk_amt / (price_risk * self.config.contract_size / self.config.leverage)

    def calculate_position_size(self, entry: float, stop: float) -> float:
        """Standard risk-based sizing."""
        risk_amt = self.equity * self.config.max_risk_per_trade_pct
        price_risk = abs(entry - stop)
        if price_risk == 0:
            return 0
        return risk_amt / (price_risk * self.config.contract_size / self.config.leverage)

    def calculate_levels(self, entry: float, direction: TradeDirection, atr: float,
                         fvg_level: Optional[float] = None) -> Tuple[float, float, float, float, Optional[float]]:
        """
        Calculate stop, TP levels, and optional FVG-based stop.
        """
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

        # Position sizing
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
            fvg_stop=fvg_sl,
        )
        self.open_trades.append(t)
        return t

    def update_trades(self, ts: datetime, high: float, low: float, close: float):
        for t in self.open_trades[:]:
            if t.status != "open":
                continue

            # Update MAE
            if t.direction == TradeDirection.LONG:
                t.mae = max(t.mae, t.entry_price - low)
                if low <= t.stop_loss:
                    self._close(t, ts, t.stop_loss, "stopped")
                    continue
                if t.tp_level_hit == 0 and high >= t.take_profit_1: t.tp_level_hit = 1
                if t.tp_level_hit == 1 and high >= t.take_profit_2: t.tp_level_hit = 2
                if t.tp_level_hit == 2 and high >= t.take_profit_3:
                    self._close(t, ts, t.take_profit_3, "closed_tp3")
                    continue
            else:
                t.mae = max(t.mae, high - t.entry_price)
                if high >= t.stop_loss:
                    self._close(t, ts, t.stop_loss, "stopped")
                    continue
                if t.tp_level_hit == 0 and low <= t.take_profit_1: t.tp_level_hit = 1
                if t.tp_level_hit == 1 and low <= t.take_profit_2: t.tp_level_hit = 2
                if t.tp_level_hit == 2 and low <= t.take_profit_3:
                    self._close(t, ts, t.take_profit_3, "closed_tp3")
                    continue

    def _close(self, t: Trade, ts: datetime, exit_price: float, status: str):
        t.exit_time = ts
        t.exit_price = exit_price
        if t.direction == TradeDirection.LONG:
            t.pnl = (exit_price - t.entry_price) * t.position_size * self.config.contract_size
        else:
            t.pnl = (t.entry_price - exit_price) * t.position_size * self.config.contract_size
        t.status = status
        self.closed_trades.append(t)
        self.trade_history_pnls.append(t.pnl)
        if t in self.open_trades:
            self.open_trades.remove(t)

        # Update equity curve
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

        # Precision / Recall style metrics
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
        if len(pnls) < 2:
            return 0.0
        a = np.array(pnls)
        if a.std() == 0:
            return 0.0
        return float(a.mean() / a.std() * np.sqrt(252 * 24 * 60))


# =============================================================================
# MAIN ORCHESTRATOR — Enhanced Precision Trading System v2.0
# =============================================================================

class EnhancedPrecisionTradingSystem:
    """
    Enhanced 1-Minute Precision Trading System v2.0.

    Integrates all 8 layers for maximum accuracy and risk-adjusted returns.
    """

    def __init__(self, asset: Asset, use_hmm: bool = True, use_sentiment: bool = False):
        self.asset = asset
        self.config = ASSET_CONFIGS[asset]
        self.use_hmm = use_hmm
        self.use_sentiment = use_sentiment

        # Layers
        self.fundamental = FundamentalContext(self.config)
        self.cleaner = MicrostructureCleaner(self.config)
        self.structure = MarketStructureAnalyzer(self.config)
        self.flow = FlowAnalyzer(self.config)
        self.mtf = MTFConfluenceEngine()
        self.ensemble = StackedEnsemble(self.config)
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
            'asset': asset or self.asset.value
        })

    def set_dxy_series(self, dxy: pd.Series):
        """Set DXY series for correlation analysis."""
        self.dxy_series = dxy

    def train(self, historical_df: pd.DataFrame, 
              df_5m: Optional[pd.DataFrame] = None,
              df_15m: Optional[pd.DataFrame] = None):
        """Train all models on historical data."""
        print(f"[EnhancedSystem] Training {self.asset.value}...")

        # Layer 0: Fundamental context
        df = self.fundamental.generate_context_features(
            historical_df, self.news_events, self.dxy_series
        )
        print("  Layer 0 (Fundamental): Complete")

        # Layer 1: Microstructure
        df = self.cleaner.clean_ohlcv(df)
        print("  Layer 1 (Microstructure): Complete")

        # Layer 2: Market Structure
        df = self.structure.analyze(df)
        print("  Layer 2 (Market Structure): Complete")

        # Layer 3: Flow & Toxicity
        df = self.flow.compute_flow_features(df)
        print("  Layer 3 (Flow): Complete")

        # Layer 4: MTF Confluence
        df = self.mtf.aggregate_structure(df, df_5m, df_15m)
        print("  Layer 4 (MTF Confluence): Complete")

        # Layer 5: HMM Regime
        if self.hmm:
            self.hmm.fit(df)
            df['regime'] = self.hmm.predict_regime(df)
            print("  Layer 5 (HMM Regime): Complete")

        # Layer 6: Ensemble
        self.ensemble.fit(df, target_horizon=3)
        print("  Layer 6 (Ensemble): Complete")

        self.data_buffer = df.tail(500)
        self.is_trained = True
        print(f"[EnhancedSystem] Training complete for {self.asset.value}")

    def process_bar(self, bar: Dict) -> Optional[Trade]:
        """Process a single 1-minute bar."""
        if not self.is_trained:
            return None

        bar_df = pd.DataFrame([bar])
        bar_df['timestamp'] = pd.to_datetime(bar_df['timestamp'])
        bar_df = bar_df.set_index('timestamp')

        self.data_buffer = pd.concat([self.data_buffer, bar_df]).tail(500)

        # Regenerate all features
        df = self.fundamental.generate_context_features(
            self.data_buffer, self.news_events, self.dxy_series
        )
        df = self.cleaner.clean_ohlcv(df)
        df = self.structure.analyze(df)
        df = self.flow.compute_flow_features(df)
        df = self.mtf.aggregate_structure(df)

        if self.hmm:
            df['regime'] = self.hmm.predict_regime(df)
            regime_val = df['regime'].iloc[-1]
        else:
            regime_val = 0

        # Ensemble prediction
        signals = self.ensemble.predict(df)
        proba = self.ensemble.predict_proba(df)

        latest_signal = int(signals.iloc[-1]) if len(signals) > 0 else 0
        confidence = float(proba.iloc[-1].abs().max()) if len(proba) > 0 else 0.5

        return self._execute_signal(
            df.index[-1], latest_signal, confidence, df, regime_val
        )

    def _execute_signal(self, ts: datetime, signal: int, confidence: float,
                        df: pd.DataFrame, regime_val: int) -> Optional[Trade]:
        """Execute signal with all filters."""
        latest = df.iloc[-1]

        # Filter 1: Spread
        if latest.get("spread_filter_active", False):
            return None

        # Filter 2: VPIN toxicity
        vpin = float(latest.get("vpin", 0))
        if vpin > self.config.vpin_threshold:
            return None

        # Filter 3: News impact
        news_impact = float(latest.get("news_impact", 0))
        if news_impact > 0.5:
            return None

        # Filter 4: Regime alignment
        if regime_val == 1 and abs(signal) > 1:  # Range regime
            signal = int(np.sign(signal))

        # Filter 5: MTF Confluence
        mtf = latest.get('mtf_confluence', MTFConfluence.NEUTRAL.value)
        if mtf in [MTFConfluence.STRONG_BEARISH.value, MTFConfluence.STRONG_BULLISH.value]:
            if (mtf == MTFConfluence.STRONG_BULLISH.value and signal < 0) or                (mtf == MTFConfluence.STRONG_BEARISH.value and signal > 0):
                return None

        # Filter 6: Confidence
        if confidence < 0.35:
            return None

        # Filter 7: Session
        hour = pd.Timestamp(ts).hour
        if self.asset in (Asset.XAUUSD, Asset.EURUSD, Asset.GBPUSD):
            in_london = self.config.london_start <= hour <= self.config.london_end
            in_ny = self.config.ny_start <= hour <= self.config.ny_end
            if not (in_london or in_ny):
                return None

        # Direction
        if signal >= 1:
            direction = TradeDirection.LONG
        elif signal <= -1:
            direction = TradeDirection.SHORT
        else:
            return None

        # Filter 8: Structure confirmation
        if direction == TradeDirection.LONG:
            if latest.get('sweep_signal', 0) != 1 and latest.get('bos_bull', False) == False:
                if latest.get('fvg_bullish', False) == False:
                    pass  # Optional: require structure

        # Entry & Stop calculation
        atr = float(latest.get("atr_14", df["close"].iloc[-20:].std()))
        entry = float(latest.get("smart_price", df["close"].iloc[-1]))

        # FVG-based stop
        fvg_stop = None
        if direction == TradeDirection.LONG and not np.isnan(latest.get('fvg_bull_start', np.nan)):
            fvg_stop = latest['fvg_bull_start']
        elif direction == TradeDirection.SHORT and not np.isnan(latest.get('fvg_bear_start', np.nan)):
            fvg_stop = latest['fvg_bear_start']

        return self.risk_manager.open_trade(
            ts, direction, entry, atr, fvg_stop, use_kelly=True
        )

    def update_market_data(self, ts: datetime, high: float, low: float, close: float):
        self.risk_manager.update_trades(ts, high, low, close)

    def get_performance(self) -> Dict:
        return self.risk_manager.get_stats()

    def generate_live_signal(self, df_recent: pd.DataFrame) -> Dict:
        """Generate live signal for UI/dashboard."""
        if not self.is_trained:
            return {"action": "HOLD", "confidence": 0.0}

        self.data_buffer = df_recent.tail(500).copy()
        df = self.fundamental.generate_context_features(self.data_buffer, self.news_events, self.dxy_series)
        df = self.cleaner.clean_ohlcv(df)
        df = self.structure.analyze(df)
        df = self.flow.compute_flow_features(df)
        df = self.mtf.aggregate_structure(df)

        if self.hmm:
            df['regime'] = self.hmm.predict_regime(df)

        signals = self.ensemble.predict(df)
        proba = self.ensemble.predict_proba(df)

        sig = int(signals.iloc[-1]) if len(signals) > 0 else 0
        conf = float(proba.iloc[-1].abs().max()) if len(proba) > 0 else 0

        latest = df.iloc[-1]
        atr = float(latest.get("atr_14", df["close"].iloc[-20:].std()))
        entry = float(latest.get("smart_price", df["close"].iloc[-1]))

        if sig >= 1:
            direction = TradeDirection.LONG
            action = "BUY" if sig == 2 else "WEAK BUY"
        elif sig <= -1:
            direction = TradeDirection.SHORT
            action = "SELL" if sig == -2 else "WEAK SELL"
        else:
            direction = TradeDirection.FLAT
            action = "HOLD"

        if direction != TradeDirection.FLAT:
            sl, tp1, tp2, tp3, fvg_sl = self.risk_manager.calculate_levels(entry, direction, atr)
            lot = self.risk_manager.calculate_position_size(entry, sl)
        else:
            sl = tp1 = tp2 = tp3 = entry
            lot = 0.0

        return {
            "action": action,
            "confidence": round(conf, 4),
            "entry": round(entry, 5),
            "sl": round(sl, 5),
            "tp1": round(tp1, 5),
            "tp2": round(tp2, 5),
            "tp3": round(tp3, 5),
            "lot_size": round(lot, 4),
            "regime": int(latest.get('regime', 0)),
            "vpin": round(float(latest.get('vpin', 0)), 3),
            "atr": round(atr, 5),
            "mtf": latest.get('mtf_confluence', 'neutral'),
        }

    def backtest(self, df: pd.DataFrame, train_pct: float = 0.7,
                 df_5m: Optional[pd.DataFrame] = None,
                 df_15m: Optional[pd.DataFrame] = None) -> Dict:
        """Walk-forward backtest."""
        n = len(df)
        if n < 500:
            return {"error": "need ≥ 500 bars"}

        split = int(n * train_pct)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]

        self.train(train_df, df_5m, df_15m)
        self.risk_manager = RiskManager(self.config)

        # Precompute
        cleaned = self.cleaner.clean_ohlcv(test_df)
        flowed = self.flow.compute_flow_features(cleaned)
        structured = self.structure.analyze(cleaned)

        for i, ts in enumerate(test_df.index):
            bar = test_df.iloc[i].to_dict()
            bar['timestamp'] = ts

            trade = self.process_bar(bar)
            if i + 1 < len(test_df):
                next_bar = test_df.iloc[i + 1]
                self.update_market_data(ts, next_bar['high'], next_bar['low'], next_bar['close'])

        if len(test_df) > 0:
            self.risk_manager.close_all(test_df.index[-1], float(test_df['close'].iloc[-1]))

        return self.get_performance()

    def save(self, path: str):
        state = {
            "asset": self.asset.value,
            "is_trained": self.is_trained,
            "data_buffer": self.data_buffer,
            "ensemble": self.ensemble,
            "hmm": self.hmm,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                s = pickle.load(f)
            self.ensemble = s["ensemble"]
            self.hmm = s.get("hmm", self.hmm)
            self.is_trained = s.get("is_trained", True)
            self.data_buffer = s.get("data_buffer", pd.DataFrame())
            return True
        except Exception as e:
            print(f"[EnhancedSystem] load failed: {e}")
            return False


# =============================================================================
# SYNTHETIC DATA GENERATOR (Enhanced with structure)
# =============================================================================

def generate_synthetic_data(n_bars: int = 10000, asset: Asset = Asset.XAUUSD) -> pd.DataFrame:
    """Generate realistic synthetic data with volatility regimes and structure."""
    np.random.seed(42)
    timestamps = pd.date_range(start='2024-01-01', periods=n_bars, freq='1min')

    if asset == Asset.XAUUSD:
        base_price = 2000.0
        daily_vol = 0.001
    elif asset == Asset.BTCUSD:
        base_price = 50000.0
        daily_vol = 0.005
    elif asset == Asset.EURUSD:
        base_price = 1.0800
        daily_vol = 0.0003
    else:
        base_price = 1.2600
        daily_vol = 0.0004

    returns = []
    vol = daily_vol / np.sqrt(1440)
    regime_multipliers = [1.0, 2.5, 0.5]
    segments = np.array_split(range(n_bars), 4)

    for i in range(n_bars):
        for idx, segment in enumerate(segments):
            if i in segment:
                current_regime = idx % len(regime_multipliers)
                break
        if i > 0:
            vol = 0.94 * vol + 0.06 * returns[-1]**2
        vol = max(vol, daily_vol / np.sqrt(1440) * 0.1)
        trend = 0.3 * np.sin(2 * np.pi * i / 1000) * vol
        ret = np.random.normal(trend, vol * regime_multipliers[current_regime])
        returns.append(ret)

    prices = base_price * np.exp(np.cumsum(returns))
    df = pd.DataFrame(index=timestamps)
    df['close'] = prices
    df['open'] = df['close'].shift(1).fillna(base_price)
    intrabar_vol = np.abs(returns) * 0.5 + daily_vol / np.sqrt(1440) * 0.3
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + intrabar_vol)
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - intrabar_vol)
    df['volume'] = np.random.lognormal(10, 1) * (1 + np.abs(returns) * 100)
    df['bid'] = df['close'] - ASSET_CONFIGS[asset].spread_avg / 2
    df['ask'] = df['close'] + ASSET_CONFIGS[asset].spread_avg / 2
    df['bid_vol'] = df['volume'] * np.random.uniform(0.3, 0.7)
    df['ask_vol'] = df['volume'] - df['bid_vol']
    return df


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def main():
    print("=" * 80)
    print("ENHANCED 1-MINUTE PRECISION TRADING SYSTEM v2.0")
    print("=" * 80)

    data = generate_synthetic_data(n_bars=8000, asset=Asset.XAUUSD)
    print(f"\n[1] Data shape: {data.shape}")

    system = EnhancedPrecisionTradingSystem(
        asset=Asset.XAUUSD,
        use_hmm=True,
        use_sentiment=False
    )

    # Add sample news event
    system.add_news_event(
        time=pd.Timestamp('2024-01-15 08:30:00'),
        event='NFP',
        impact='high',
        asset='XAUUSD'
    )

    print("\n[2] Training...")
    system.train(data.iloc[:5000])

    print("\n[3] Simulating live trading...")
    test_data = data.iloc[5000:]
    for i in range(0, len(test_data), 10):
        bar = test_data.iloc[i].to_dict()
        bar['timestamp'] = test_data.index[i]
        trade = system.process_bar(bar)
        if i + 1 < len(test_data):
            next_bar = test_data.iloc[i + 1]
            system.update_market_data(
                test_data.index[i], next_bar['high'], next_bar['low'], next_bar['close']
            )
        if trade:
            print(f"    Trade: {trade.direction.name} @ {trade.entry_price:.2f}")

    print("\n[4] Performance:")
    stats = system.get_performance()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        elif isinstance(v, int):
            print(f"    {k}: {v}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
