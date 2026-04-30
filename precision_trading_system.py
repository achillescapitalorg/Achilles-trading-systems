
"""
================================================================================
1-MINUTE PRECISION TRADING SYSTEM
Asset Coverage: XAU/USD, BTC/USD, EUR/USD, GBP/USD
================================================================================
Architecture:
    Layer 1: Microstructure Cleaning (Smart Price, Kalman Filter, Spread Filter)
    Layer 2: Flow & Toxicity (VPIN, Order Book Imbalance, Signed Volume)
    Layer 3: ML Signal Generation (Lorentzian, XGBoost, HMM Regime Detection)
    Layer 4: Execution & Risk Management (Dynamic Stops, Position Sizing)
================================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Literal
from enum import Enum
import warnings
from collections import deque
import json
from datetime import datetime, timedelta

# ML / Statistical Libraries
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
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

# For Kalman Filter
try:
    from pykalman import KalmanFilter
    PYKALMAN_AVAILABLE = True
except ImportError:
    PYKALMAN_AVAILABLE = False
    print("WARNING: pykalman not available. Using EWMA fallback for price cleaning.")

# For XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: xgboost not available. Using sklearn GradientBoosting fallback.")

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

class TradeDirection(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class AssetConfig:
    """Per-asset configuration parameters."""
    asset: Asset
    pip_value: float  # For forex: 0.0001, for gold: 0.01, for BTC: 1.0
    spread_avg: float  # Average spread in price terms
    tick_size: float
    contract_size: float = 1.0
    leverage: float = 100.0

    # Session filters (UTC hours)
    london_start: int = 8
    london_end: int = 17
    ny_start: int = 13
    ny_end: int = 22

    # Model-specific params
    kalman_observation_covariance: float = 1.0
    kalman_transition_covariance: float = 0.01
    vpin_buckets: int = 50
    vpin_window: int = 50

    # Risk params
    max_risk_per_trade_pct: float = 0.01  # 1% of equity
    atr_multiplier_stop: float = 1.5
    atr_multiplier_tp1: float = 1.0
    atr_multiplier_tp2: float = 2.0
    atr_multiplier_tp3: float = 3.0


# Asset-specific configurations
ASSET_CONFIGS = {
    Asset.XAUUSD: AssetConfig(
        asset=Asset.XAUUSD,
        pip_value=0.01,
        spread_avg=0.05,
        tick_size=0.01,
        contract_size=100.0,
        leverage=100.0,
        kalman_observation_covariance=0.5,
        kalman_transition_covariance=0.005,
        vpin_buckets=50,
        vpin_window=50,
        max_risk_per_trade_pct=0.01,
        atr_multiplier_stop=2.0,
        atr_multiplier_tp1=1.5,
        atr_multiplier_tp2=2.5,
        atr_multiplier_tp3=4.0,
    ),
    Asset.BTCUSD: AssetConfig(
        asset=Asset.BTCUSD,
        pip_value=1.0,
        spread_avg=20.0,
        tick_size=0.01,
        contract_size=1.0,
        leverage=50.0,
        kalman_observation_covariance=100.0,
        kalman_transition_covariance=1.0,
        vpin_buckets=100,
        vpin_window=100,
        max_risk_per_trade_pct=0.01,
        atr_multiplier_stop=2.5,
        atr_multiplier_tp1=2.0,
        atr_multiplier_tp2=3.5,
        atr_multiplier_tp3=5.0,
    ),
    Asset.EURUSD: AssetConfig(
        asset=Asset.EURUSD,
        pip_value=0.0001,
        spread_avg=0.0001,
        tick_size=0.00001,
        contract_size=100000.0,
        leverage=100.0,
        kalman_observation_covariance=0.0001,
        kalman_transition_covariance=0.000001,
        vpin_buckets=50,
        vpin_window=50,
        max_risk_per_trade_pct=0.01,
        atr_multiplier_stop=1.5,
        atr_multiplier_tp1=1.0,
        atr_multiplier_tp2=2.0,
        atr_multiplier_tp3=3.0,
    ),
    Asset.GBPUSD: AssetConfig(
        asset=Asset.GBPUSD,
        pip_value=0.0001,
        spread_avg=0.0002,
        tick_size=0.00001,
        contract_size=100000.0,
        leverage=100.0,
        kalman_observation_covariance=0.0002,
        kalman_transition_covariance=0.000002,
        vpin_buckets=50,
        vpin_window=50,
        max_risk_per_trade_pct=0.01,
        atr_multiplier_stop=1.5,
        atr_multiplier_tp1=1.0,
        atr_multiplier_tp2=2.0,
        atr_multiplier_tp3=3.0,
    ),
}


# =============================================================================
# LAYER 1: MICROSTRUCTURE CLEANING
# =============================================================================

class MicrostructureCleaner:
    """
    Layer 1: Clean raw price data to remove microstructure noise.

    Components:
    - Smart Price: Volume-weighted mid using L2 depth
    - Kalman Filter: Latent price extraction from noisy observations
    - Spread Filter: Block trades when noise dominates signal
    """

    def __init__(self, config: AssetConfig):
        self.config = config
        self.kalman = None
        self._init_kalman()

    def _init_kalman(self):
        """Initialize Kalman filter for price cleaning."""
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

    def compute_smart_price(
        self, 
        bid: float, 
        ask: float, 
        bid_vol: float = 1.0, 
        ask_vol: float = 1.0
    ) -> float:
        """
        Compute Smart Price using volume-weighted mid.

        Formula: (bid * ask_vol + ask * bid_vol) / (bid_vol + ask_vol)

        If no L2 data available, bid_vol and ask_vol default to 1.0 
        (degrades to simple mid-price).
        """
        total_vol = bid_vol + ask_vol
        if total_vol == 0:
            return (bid + ask) / 2
        return (bid * ask_vol + ask * bid_vol) / total_vol

    def apply_kalman_filter(self, prices: np.ndarray) -> np.ndarray:
        """
        Apply Kalman filter to extract latent price from noisy observations.

        Args:
            prices: Array of observed prices (can be smart prices or closes)

        Returns:
            Filtered prices with microstructure noise reduced
        """
        if not PYKALMAN_AVAILABLE or self.kalman is None:
            # Fallback: EWMA with span=5
            return pd.Series(prices).ewm(span=5).mean().values

        # Handle NaN values
        prices_clean = pd.Series(prices).fillna(method='ffill').fillna(method='bfill').values

        # Run Kalman filter
        filtered_state_means, _ = self.kalman.filter(prices_clean)
        return filtered_state_means.flatten()

    def compute_spread_filter(
        self, 
        spread: float, 
        atr: float
    ) -> bool:
        """
        Determine if spread is too wide relative to volatility.

        Returns True if trading should be blocked (spread > 30% of ATR).
        """
        if atr == 0:
            return False
        return spread > (atr * 0.3)

    def clean_ohlcv(
        self, 
        df: pd.DataFrame,
        has_l2_data: bool = False
    ) -> pd.DataFrame:
        """
        Full cleaning pipeline for OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
                Optional: [bid, ask, bid_vol, ask_vol]
            has_l2_data: Whether L2 order book data is available

        Returns:
            DataFrame with added columns:
                - smart_price
                - kalman_price
                - spread
                - spread_filter_active
        """
        result = df.copy()

        # Compute Smart Price
        if has_l2_data and all(col in df.columns for col in ['bid', 'ask']):
            bid_vol = df.get('bid_vol', pd.Series(1.0, index=df.index))
            ask_vol = df.get('ask_vol', pd.Series(1.0, index=df.index))
            result['smart_price'] = self.compute_smart_price(
                df['bid'].values, df['ask'].values, 
                bid_vol.values, ask_vol.values
            )
            result['spread'] = df['ask'] - df['bid']
        else:
            # Use OHLC midpoint as proxy
            result['smart_price'] = (df['high'] + df['low']) / 2
            # Estimate spread from close vs typical price
            typical = (df['high'] + df['low'] + df['close']) / 3
            result['spread'] = abs(df['close'] - typical) * 2
            result['spread'] = result['spread'].clip(lower=self.config.spread_avg)

        # Apply Kalman Filter
        result['kalman_price'] = self.apply_kalman_filter(result['smart_price'].values)

        # Compute ATR for spread filter
        result['atr_14'] = self._compute_atr(result, period=14)

        # Spread filter
        result['spread_filter_active'] = result.apply(
            lambda row: self.compute_spread_filter(row['spread'], row['atr_14']),
            axis=1
        )

        return result

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range."""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()


# =============================================================================
# LAYER 2: FLOW & TOXICITY
# =============================================================================

class FlowAnalyzer:
    """
    Layer 2: Measure order flow toxicity and market microstructure.

    Components:
    - VPIN: Volume-Synchronized Probability of Informed Trading
    - Order Book Imbalance: Top-of-book depth asymmetry
    - Signed Transaction Volume: Buyer vs seller initiated volume
    - Realized Volatility: Short-term volatility measurement
    """

    def __init__(self, config: AssetConfig):
        self.config = config
        self.vpin_buckets = config.vpin_buckets
        self.vpin_window = config.vpin_window

    def compute_vpin(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Volume-Synchronized PIN (VPIN).

        VPIN measures order flow toxicity. High VPIN indicates informed
        traders are active, suggesting imminent volatility.

        Algorithm:
        1. Divide volume into equal-sized buckets
        2. For each bucket, compute |buy_vol - sell_vol| / total_vol
        3. VPIN = rolling average of these values

        Args:
            df: DataFrame with [close, volume] and optionally [bid, ask]

        Returns:
            VPIN series (0 to 1, higher = more toxic)
        """
        # Classify volume as buy or sell initiated
        # Using tick rule: if price up -> buy, if down -> sell
        price_change = df['close'].diff()

        buy_volume = pd.Series(0.0, index=df.index)
        sell_volume = pd.Series(0.0, index=df.index)

        buy_volume[price_change > 0] = df.loc[price_change > 0, 'volume']
        sell_volume[price_change < 0] = df.loc[price_change < 0, 'volume']

        # For zero price change, split volume 50/50
        zero_change = price_change == 0
        buy_volume[zero_change] = df.loc[zero_change, 'volume'] * 0.5
        sell_volume[zero_change] = df.loc[zero_change, 'volume'] * 0.5

        # Volume bucketing
        cumulative_vol = (buy_volume + sell_volume).cumsum()
        bucket_size = cumulative_vol.iloc[-1] / self.vpin_buckets if len(cumulative_vol) > 0 else 1

        vpin_values = []
        bucket_buy = 0
        bucket_sell = 0
        current_bucket_vol = 0

        for i in range(len(df)):
            bucket_buy += buy_volume.iloc[i]
            bucket_sell += sell_volume.iloc[i]
            current_bucket_vol += (buy_volume.iloc[i] + sell_volume.iloc[i])

            if current_bucket_vol >= bucket_size:
                vpin_bucket = abs(bucket_buy - bucket_sell) / current_bucket_vol if current_bucket_vol > 0 else 0
                vpin_values.append(vpin_bucket)
                bucket_buy = 0
                bucket_sell = 0
                current_bucket_vol = 0

        # Create VPIN series aligned to original index
        vpin_series = pd.Series(np.nan, index=df.index)

        # Map VPIN values back to approximate timestamps
        if len(vpin_values) > 0:
            step = max(1, len(df) // len(vpin_values))
            for i, v in enumerate(vpin_values):
                idx = min(i * step, len(df) - 1)
                vpin_series.iloc[idx] = v

        # Forward fill and rolling mean
        vpin_series = vpin_series.fillna(method='ffill').fillna(0)
        vpin_series = vpin_series.rolling(window=self.vpin_window, min_periods=1).mean()

        return vpin_series.clip(0, 1)

    def compute_order_book_imbalance(
        self, 
        bid_vol: pd.Series, 
        ask_vol: pd.Series
    ) -> pd.Series:
        """
        Compute order book imbalance from top-of-book depth.

        Formula: (bid_vol - ask_vol) / (bid_vol + ask_vol)

        Range: -1 (all asks) to +1 (all bids)
        Positive values suggest buying pressure.
        """
        total = bid_vol + ask_vol
        total = total.replace(0, np.nan)
        imbalance = (bid_vol - ask_vol) / total
        return imbalance.fillna(0)

    def compute_signed_volume(
        self, 
        df: pd.DataFrame,
        method: str = 'tick_rule'
    ) -> pd.Series:
        """
        Classify volume as buyer-initiated or seller-initiated.

        Methods:
        - 'tick_rule': Price up = buy, down = sell
        - 'bulk_volume': Proportional classification based on position within bar
        """
        if method == 'tick_rule':
            price_change = df['close'].diff()
            signed = pd.Series(0.0, index=df.index)
            signed[price_change > 0] = df.loc[price_change > 0, 'volume']
            signed[price_change < 0] = -df.loc[price_change < 0, 'volume']
            return signed

        elif method == 'bulk_volume':
            # Lee-Ready / Bulk Volume Classification
            typical = (df['high'] + df['low'] + df['close']) / 3
            prev_typical = typical.shift(1)

            # Position of close within the bar
            bar_range = df['high'] - df['low']
            bar_range = bar_range.replace(0, np.nan)

            position = (df['close'] - df['low']) / bar_range
            position = position.fillna(0.5)

            # Signed volume
            signed = df['volume'] * (2 * position - 1)
            return signed

        else:
            raise ValueError(f"Unknown method: {method}")

    def compute_realized_volatility(
        self, 
        prices: pd.Series, 
        window: int = 5
    ) -> pd.Series:
        """
        Compute annualized realized volatility from 1-minute returns.

        Formula: std(returns) * sqrt(minutes_per_year)
        """
        returns = prices.pct_change().fillna(0)
        rv = returns.rolling(window=window).std() * np.sqrt(525600)  # Minutes in a year
        return rv.fillna(0)

    def compute_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all flow and toxicity features.

        Returns DataFrame with added columns:
            - vpin
            - signed_volume
            - signed_volume_momentum (5-bar MA of signed volume)
            - realized_vol_5m
            - realized_vol_20m
            - volume_ma_ratio (volume / 20-bar MA)
        """
        result = df.copy()

        # VPIN
        result['vpin'] = self.compute_vpin(df)

        # Signed Volume
        result['signed_volume'] = self.compute_signed_volume(df, method='bulk_volume')
        result['signed_volume_momentum'] = result['signed_volume'].rolling(5).mean()

        # Realized Volatility
        result['realized_vol_5m'] = self.compute_realized_volatility(df['close'], 5)
        result['realized_vol_20m'] = self.compute_realized_volatility(df['close'], 20)

        # Volume intensity
        result['volume_ma_20'] = df['volume'].rolling(20).mean()
        result['volume_ma_ratio'] = df['volume'] / result['volume_ma_20'].replace(0, np.nan)
        result['volume_ma_ratio'] = result['volume_ma_ratio'].fillna(1)

        # Order book imbalance (if L2 data available)
        if all(col in df.columns for col in ['bid_vol', 'ask_vol']):
            result['ob_imbalance'] = self.compute_order_book_imbalance(
                df['bid_vol'], df['ask_vol']
            )
        else:
            # Proxy: use signed volume as imbalance proxy
            result['ob_imbalance'] = result['signed_volume_momentum'] / (
                df['volume'].rolling(5).mean().replace(0, np.nan)
            )
            result['ob_imbalance'] = result['ob_imbalance'].fillna(0).clip(-1, 1)

        return result


# =============================================================================
# LAYER 3: ML SIGNAL GENERATION
# =============================================================================

class LorentzianClassifier:
    """
    Lorentian Distance Classifier optimized for XAU/USD.

    Uses Lorentzian distance metric which is less sensitive to outliers
    than Euclidean distance - critical for gold's spike behavior.
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.feature_cols = ['rsi_7', 'mfi_7', 'roc_3', 'volatility_gate']
        self.scaler = StandardScaler()
        self.train_data = None
        self.train_labels = None

    def _lorentzian_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Lorentzian distance between two vectors."""
        diff = x - y
        return np.sum(np.log(1 + np.abs(diff)))

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Lorentzian-specific features."""
        features = pd.DataFrame(index=df.index)

        # RSI(7)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
        rs = gain / loss.replace(0, np.nan)
        features['rsi_7'] = (100 - (100 / (1 + rs))).fillna(50)

        # MFI(7) - Money Flow Index
        typical = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical * df['volume']
        money_flow_sign = np.where(typical > typical.shift(1), 1, -1)
        signed_money_flow = raw_money_flow * money_flow_sign

        positive_flow = pd.Series(signed_money_flow, index=df.index).where(
            signed_money_flow > 0, 0
        ).rolling(7).sum()
        negative_flow = pd.Series(-signed_money_flow, index=df.index).where(
            signed_money_flow < 0, 0
        ).rolling(7).sum()

        mfi_ratio = positive_flow / negative_flow.replace(0, np.nan)
        features['mfi_7'] = (100 - (100 / (1 + mfi_ratio))).fillna(50)

        # ROC(3)
        features['roc_3'] = ((df['close'] - df['close'].shift(3)) / 
                            df['close'].shift(3) * 100).fillna(0)

        # Volatility gate (ATR-based)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        features['volatility_gate'] = (tr / atr.replace(0, np.nan)).fillna(1).clip(0, 5)

        return features

    def fit(self, df: pd.DataFrame, labels: np.ndarray):
        """Fit the classifier on historical data."""
        features = self._compute_features(df).dropna()

        # Align labels
        aligned_labels = pd.Series(labels, index=df.index).loc[features.index].values

        self.train_data = self.scaler.fit_transform(features)
        self.train_labels = aligned_labels

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict signal for each bar.

        Returns series with values:
            2: Strong Buy, 1: Weak Buy, 0: Neutral, -1: Weak Sell, -2: Strong Sell
        """
        features = self._compute_features(df).dropna()
        if len(features) == 0:
            return pd.Series(0, index=df.index)

        scaled = self.scaler.transform(features)

        predictions = []
        for i in range(len(scaled)):
            # Compute Lorentzian distance to all training points
            distances = []
            for j in range(len(self.train_data)):
                dist = self._lorentzian_distance(scaled[i], self.train_data[j])
                distances.append((dist, self.train_labels[j]))

            # Find k nearest neighbors
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.n_neighbors]

            # Vote
            votes = [label for _, label in k_nearest]
            prediction = np.sign(np.mean(votes)) if votes else 0

            # Map to 5-class signal
            avg_vote = np.mean(votes) if votes else 0
            if avg_vote > 0.5:
                predictions.append(2)
            elif avg_vote > 0.1:
                predictions.append(1)
            elif avg_vote < -0.5:
                predictions.append(-2)
            elif avg_vote < -0.1:
                predictions.append(-1)
            else:
                predictions.append(0)

        return pd.Series(predictions, index=features.index)


class XGBoostSignalModel:
    """
    XGBoost-based signal model for BTC/USD and EUR/USD.

    Uses microstructure-aware features for 1-minute prediction.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 6):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive feature set."""
        features = pd.DataFrame(index=df.index)

        # Price-based features
        features['returns_1'] = df['close'].pct_change()
        features['returns_3'] = df['close'].pct_change(3)
        features['returns_5'] = df['close'].pct_change(5)

        # Smart price features (if available)
        if 'smart_price' in df.columns:
            features['smart_return'] = df['smart_price'].pct_change()
            features['smart_vs_close'] = (df['smart_price'] - df['close']) / df['close']
        else:
            features['smart_return'] = features['returns_1']
            features['smart_vs_close'] = 0

        # Kalman price features
        if 'kalman_price' in df.columns:
            features['kalman_return'] = df['kalman_price'].pct_change()
            features['kalman_vs_close'] = (df['kalman_price'] - df['close']) / df['close']
        else:
            features['kalman_return'] = features['returns_1']
            features['kalman_vs_close'] = 0

        # Momentum features
        features['rsi_7'] = self._compute_rsi(df['close'], 7)
        features['rsi_14'] = self._compute_rsi(df['close'], 14)
        features['macd'] = self._compute_macd(df['close'])
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']

        # Bollinger Bands
        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        features['bb_position'] = (df['close'] - bb_mid) / bb_std.replace(0, np.nan)
        features['bb_width'] = bb_std / bb_mid.replace(0, np.nan)

        # Volume features
        if 'signed_volume' in df.columns:
            features['signed_vol_ma5'] = df['signed_volume'].rolling(5).mean()
            features['signed_vol_ma10'] = df['signed_volume'].rolling(10).mean()
        else:
            features['signed_vol_ma5'] = 0
            features['signed_vol_ma10'] = 0

        if 'volume_ma_ratio' in df.columns:
            features['vol_intensity'] = df['volume_ma_ratio']
        else:
            features['vol_intensity'] = df['volume'] / df['volume'].rolling(20).mean().replace(0, np.nan)

        # Flow features
        if 'ob_imbalance' in df.columns:
            features['ob_imbalance'] = df['ob_imbalance']
        else:
            features['ob_imbalance'] = 0

        if 'vpin' in df.columns:
            features['vpin'] = df['vpin']
            features['vpin_high'] = (df['vpin'] > df['vpin'].rolling(100).quantile(0.9)).astype(int)
        else:
            features['vpin'] = 0
            features['vpin_high'] = 0

        # Volatility features
        if 'realized_vol_5m' in df.columns:
            features['rv_5m'] = df['realized_vol_5m']
            features['rv_20m'] = df['realized_vol_20m']
        else:
            returns = df['close'].pct_change().fillna(0)
            features['rv_5m'] = returns.rolling(5).std() * np.sqrt(525600)
            features['rv_20m'] = returns.rolling(20).std() * np.sqrt(525600)

        features['vol_regime'] = (features['rv_5m'] > features['rv_5m'].rolling(50).mean()).astype(int)

        # Lagged features
        for lag in [1, 2, 3]:
            features[f'return_lag_{lag}'] = features['returns_1'].shift(lag)

        # Interaction features
        features['rsi_x_vol'] = features['rsi_7'] * features['vol_intensity']
        features['ob_x_vol'] = features['ob_imbalance'] * features['vol_intensity']

        return features.replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)

    @staticmethod
    def _compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow

    def fit(self, df: pd.DataFrame, target_horizon: int = 3):
        """
        Fit the model.

        Args:
            df: DataFrame with OHLCV + microstructure features
            target_horizon: Bars ahead to predict (1, 3, 5, etc.)
        """
        features = self._engineer_features(df)

        # Target: Direction of future return
        future_returns = df['close'].shift(-target_horizon) / df['close'] - 1

        # 5-class classification
        target = pd.Series(0, index=df.index)
        target[future_returns > 0.001] = 2      # Strong up
        target[future_returns > 0.0003] = 1     # Weak up
        target[future_returns < -0.001] = -2   # Strong down
        target[future_returns < -0.0003] = -1    # Weak down

        # Align and clean
        aligned_features = features.loc[target.index].dropna()
        aligned_target = target.loc[aligned_features.index]

        self.feature_cols = aligned_features.columns.tolist()

        X = self.scaler.fit_transform(aligned_features)
        y = aligned_target.values

        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.1,
                objective='multi:softprob',
                num_class=5,
                eval_metric='mlogloss',
                random_state=42
            )
            # Map labels to 0-4 for XGBoost
            y_mapped = y + 2
            self.model.fit(X, y_mapped)
        elif SKLEARN_AVAILABLE:
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42
            )
            y_mapped = y + 2
            self.model.fit(X, y_mapped)
        else:
            raise ImportError("No ML library available. Install xgboost or scikit-learn.")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict signals for new data."""
        features = self._engineer_features(df)

        if self.feature_cols:
            # Ensure same columns
            for col in self.feature_cols:
                if col not in features.columns:
                    features[col] = 0
            features = features[self.feature_cols]

        X = self.scaler.transform(features)

        if self.model is None:
            return pd.Series(0, index=df.index)

        predictions = self.model.predict(X) - 2  # Map back to -2 to 2
        return pd.Series(predictions, index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get prediction probabilities for each class."""
        features = self._engineer_features(df)

        if self.feature_cols:
            for col in self.feature_cols:
                if col not in features.columns:
                    features[col] = 0
            features = features[self.feature_cols]

        X = self.scaler.transform(features)

        if self.model is None:
            return pd.DataFrame(0.2, index=df.index, columns=[-2, -1, 0, 1, 2])

        proba = self.model.predict_proba(X)
        return pd.DataFrame(
            proba, 
            index=df.index, 
            columns=[-2, -1, 0, 1, 2]
        )


class HMMRegimeDetector:
    """
    Hidden Markov Model for regime detection.

    Detects whether market is in trending, mean-reverting, or high/low
    volatility regime. Critical for adapting strategy parameters.
    """

    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame):
        """Fit HMM on return and volatility features."""
        if not HMMLEARN_AVAILABLE:
            print("WARNING: hmmlearn not available. Regime detection disabled.")
            return

        returns = df['close'].pct_change().fillna(0).values.reshape(-1, 1)

        # Add volatility as second feature
        vol = pd.Series(returns.flatten()).rolling(10).std().fillna(0).values.reshape(-1, 1)

        features = np.hstack([returns, vol])
        features = features[~np.isnan(features).any(axis=1)]

        if len(features) < 100:
            print("WARNING: Insufficient data for HMM fitting.")
            return

        features_scaled = self.scaler.fit_transform(features)

        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.model.fit(features_scaled)

    def predict_regime(self, df: pd.DataFrame) -> pd.Series:
        """Predict regime for each time step."""
        if self.model is None or not HMMLEARN_AVAILABLE:
            return pd.Series(0, index=df.index)

        returns = df['close'].pct_change().fillna(0).values.reshape(-1, 1)
        vol = pd.Series(returns.flatten()).rolling(10).std().fillna(0).values.reshape(-1, 1)
        features = np.hstack([returns, vol])
        features = self.scaler.transform(features)

        regimes = self.model.predict(features)
        return pd.Series(regimes, index=df.index)

    def get_regime_characteristics(self) -> Dict[int, Dict]:
        """Get mean return and volatility for each regime."""
        if self.model is None:
            return {}

        characteristics = {}
        for i in range(self.n_regimes):
            characteristics[i] = {
                'mean_return': self.model.means_[i][0],
                'volatility': np.sqrt(self.model.covars_[i][0][0]),
                'stationary_prob': self.model.transmat_.sum(axis=0)[i] / self.n_regimes
            }
        return characteristics


# =============================================================================
# LAYER 4: EXECUTION & RISK MANAGEMENT
# =============================================================================

@dataclass
class Trade:
    """Represents an active or completed trade."""
    entry_time: datetime
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    position_size: float
    asset: Asset

    # Tracking
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "open"
    tp_level_hit: int = 0  # 0=none, 1=TP1, 2=TP2, 3=TP3


class RiskManager:
    """
    Layer 4: Position sizing, stop management, and trade execution.
    """

    def __init__(self, config: AssetConfig):
        self.config = config
        self.equity = 10000.0  # Default starting equity
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []

    def set_equity(self, equity: float):
        self.equity = equity

    def calculate_position_size(
        self, 
        entry_price: float, 
        stop_loss: float
    ) -> float:
        """
        Calculate position size based on risk per trade.

        Risk = |entry - stop| * position_size * contract_size / (entry_price * leverage)
        """
        risk_amount = self.equity * self.config.max_risk_per_trade_pct
        price_risk = abs(entry_price - stop_loss)

        if price_risk == 0:
            return 0

        # For forex: position_size = risk_amount / (price_risk * contract_size / leverage)
        position_size = risk_amount / (price_risk * self.config.contract_size / self.config.leverage)

        return position_size

    def calculate_levels(
        self, 
        entry_price: float, 
        direction: TradeDirection, 
        atr: float
    ) -> Tuple[float, float, float, float]:
        """
        Calculate stop loss and take profit levels.

        Returns: (stop_loss, tp1, tp2, tp3)
        """
        if direction == TradeDirection.LONG:
            stop = entry_price - atr * self.config.atr_multiplier_stop
            tp1 = entry_price + atr * self.config.atr_multiplier_tp1
            tp2 = entry_price + atr * self.config.atr_multiplier_tp2
            tp3 = entry_price + atr * self.config.atr_multiplier_tp3
        else:
            stop = entry_price + atr * self.config.atr_multiplier_stop
            tp1 = entry_price - atr * self.config.atr_multiplier_tp1
            tp2 = entry_price - atr * self.config.atr_multiplier_tp2
            tp3 = entry_price - atr * self.config.atr_multiplier_tp3

        return stop, tp1, tp2, tp3

    def open_trade(
        self, 
        timestamp: datetime,
        direction: TradeDirection,
        entry_price: float,
        atr: float
    ) -> Optional[Trade]:
        """Open a new trade with calculated risk parameters."""
        if direction == TradeDirection.FLAT:
            return None

        stop, tp1, tp2, tp3 = self.calculate_levels(entry_price, direction, atr)
        position_size = self.calculate_position_size(entry_price, stop)

        if position_size <= 0:
            return None

        trade = Trade(
            entry_time=timestamp,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            position_size=position_size,
            asset=self.config.asset
        )

        self.open_trades.append(trade)
        return trade

    def update_trades(self, timestamp: datetime, high: float, low: float, close: float):
        """Update all open trades with new price data."""
        for trade in self.open_trades[:]:
            if trade.status != "open":
                continue

            # Check stop loss
            if trade.direction == TradeDirection.LONG:
                if low <= trade.stop_loss:
                    trade.exit_time = timestamp
                    trade.exit_price = trade.stop_loss
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.position_size * self.config.contract_size
                    trade.status = "stopped"
                    self.closed_trades.append(trade)
                    self.open_trades.remove(trade)
                    continue

                # Check take profits (partial close logic)
                if trade.tp_level_hit == 0 and high >= trade.take_profit_1:
                    trade.tp_level_hit = 1
                if trade.tp_level_hit == 1 and high >= trade.take_profit_2:
                    trade.tp_level_hit = 2
                if trade.tp_level_hit == 2 and high >= trade.take_profit_3:
                    trade.exit_time = timestamp
                    trade.exit_price = trade.take_profit_3
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.position_size * self.config.contract_size
                    trade.status = "closed_tp3"
                    self.closed_trades.append(trade)
                    self.open_trades.remove(trade)
                    continue

            else:  # SHORT
                if high >= trade.stop_loss:
                    trade.exit_time = timestamp
                    trade.exit_price = trade.stop_loss
                    trade.pnl = (trade.entry_price - trade.exit_price) * trade.position_size * self.config.contract_size
                    trade.status = "stopped"
                    self.closed_trades.append(trade)
                    self.open_trades.remove(trade)
                    continue

                if trade.tp_level_hit == 0 and low <= trade.take_profit_1:
                    trade.tp_level_hit = 1
                if trade.tp_level_hit == 1 and low <= trade.take_profit_2:
                    trade.tp_level_hit = 2
                if trade.tp_level_hit == 2 and low <= trade.take_profit_3:
                    trade.exit_time = timestamp
                    trade.exit_price = trade.take_profit_3
                    trade.pnl = (trade.entry_price - trade.exit_price) * trade.position_size * self.config.contract_size
                    trade.status = "closed_tp3"
                    self.closed_trades.append(trade)
                    self.open_trades.remove(trade)
                    continue

    def close_all_trades(self, timestamp: datetime, price: float):
        """Close all open trades at current price (e.g., end of session)."""
        for trade in self.open_trades[:]:
            trade.exit_time = timestamp
            trade.exit_price = price

            if trade.direction == TradeDirection.LONG:
                trade.pnl = (price - trade.entry_price) * trade.position_size * self.config.contract_size
            else:
                trade.pnl = (trade.entry_price - price) * trade.position_size * self.config.contract_size

            trade.status = "closed_manual"
            self.closed_trades.append(trade)
            self.open_trades.remove(trade)

    def get_stats(self) -> Dict:
        """Get trading performance statistics."""
        if not self.closed_trades:
            return {"total_trades": 0}

        pnls = [t.pnl for t in self.closed_trades if t.pnl is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            "total_trades": len(self.closed_trades),
            "win_rate": len(wins) / len(pnls) if pnls else 0,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "total_pnl": sum(pnls),
            "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
            "max_drawdown": self._calculate_max_drawdown(pnls),
            "sharpe": self._calculate_sharpe(pnls)
        }

    @staticmethod
    def _calculate_max_drawdown(pnls: List[float]) -> float:
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return abs(min(drawdown)) if len(drawdown) > 0 else 0

    @staticmethod
    def _calculate_sharpe(pnls: List[float]) -> float:
        if len(pnls) < 2:
            return 0
        returns = np.array(pnls)
        if returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(252 * 24 * 60)  # Annualized for 1-min


# =============================================================================
# MAIN TRADING SYSTEM ORCHESTRATOR
# =============================================================================

class PrecisionTradingSystem:
    """
    Full 1-Minute Precision Trading System.

    Orchestrates all four layers:
    1. Microstructure Cleaning
    2. Flow & Toxicity Analysis
    3. ML Signal Generation
    4. Execution & Risk Management
    """

    def __init__(
        self, 
        asset: Asset,
        model_type: Literal['lorentzian', 'xgboost'] = 'xgboost',
        use_hmm: bool = True
    ):
        self.asset = asset
        self.config = ASSET_CONFIGS[asset]
        self.model_type = model_type
        self.use_hmm = use_hmm

        # Layer 1
        self.cleaner = MicrostructureCleaner(self.config)

        # Layer 2
        self.flow_analyzer = FlowAnalyzer(self.config)

        # Layer 3
        if model_type == 'lorentzian':
            self.signal_model = LorentzianClassifier(n_neighbors=5)
        else:
            self.signal_model = XGBoostSignalModel(n_estimators=100, max_depth=6)

        self.hmm = HMMRegimeDetector(n_regimes=2) if use_hmm else None
        self.current_regime = Regime.TRENDING

        # Layer 4
        self.risk_manager = RiskManager(self.config)

        # State
        self.is_trained = False
        self.data_buffer = pd.DataFrame()

    def train(self, historical_df: pd.DataFrame):
        """
        Train all models on historical data.

        Args:
            historical_df: DataFrame with columns:
                [timestamp, open, high, low, close, volume]
                Optional: [bid, ask, bid_vol, ask_vol]
        """
        print(f"Training {self.asset.value} system...")

        # Layer 1: Clean data
        cleaned = self.cleaner.clean_ohlcv(historical_df)
        print("  Layer 1 (Microstructure Cleaning): Complete")

        # Layer 2: Flow features
        flowed = self.flow_analyzer.compute_flow_features(cleaned)
        print("  Layer 2 (Flow Analysis): Complete")

        # Layer 3a: HMM Regime Detection
        if self.hmm is not None:
            self.hmm.fit(flowed)
            regimes = self.hmm.predict_regime(flowed)
            flowed['regime'] = regimes
            print("  Layer 3a (HMM Regime Detection): Complete")

        # Layer 3b: Signal Model
        self.signal_model.fit(flowed)
        print("  Layer 3b (Signal Model): Complete")

        self.data_buffer = flowed
        self.is_trained = True
        print(f"Training complete for {self.asset.value}")

    def process_bar(self, bar: Dict) -> Optional[Trade]:
        """
        Process a single 1-minute bar and generate trade signal.

        Args:
            bar: Dictionary with keys:
                timestamp, open, high, low, close, volume
                Optional: bid, ask, bid_vol, ask_vol

        Returns:
            Trade object if signal generated, None otherwise
        """
        if not self.is_trained:
            print("System not trained. Call train() first.")
            return None

        # Convert to DataFrame
        bar_df = pd.DataFrame([bar])
        bar_df['timestamp'] = pd.to_datetime(bar_df['timestamp'])
        bar_df.set_index('timestamp', inplace=True)

        # Append to buffer and keep last 200 bars for context
        self.data_buffer = pd.concat([self.data_buffer, bar_df]).tail(200)

        # Layer 1: Clean
        cleaned = self.cleaner.clean_ohlcv(self.data_buffer)

        # Layer 2: Flow
        flowed = self.flow_analyzer.compute_flow_features(cleaned)

        # Layer 3a: Regime
        if self.hmm is not None:
            regimes = self.hmm.predict_regime(flowed)
            flowed['regime'] = regimes
            latest_regime = regimes.iloc[-1]
            self.current_regime = Regime.TRENDING if latest_regime == 0 else Regime.MEAN_REVERTING

        # Layer 3b: Signal
        signals = self.signal_model.predict(flowed)
        latest_signal = signals.iloc[-1]

        # Get probabilities if available
        if hasattr(self.signal_model, 'predict_proba'):
            proba = self.signal_model.predict_proba(flowed).iloc[-1]
            confidence = proba.abs().max()
        else:
            confidence = 0.5

        # Layer 4: Execution logic
        return self._execute_signal(
            timestamp=bar_df.index[-1],
            signal=latest_signal,
            confidence=confidence,
            df=flowed
        )

    def _execute_signal(
        self, 
        timestamp: datetime,
        signal: int,
        confidence: float,
        df: pd.DataFrame
    ) -> Optional[Trade]:
        """Convert signal to trade with all filters applied."""

        latest = df.iloc[-1]

        # Filter 1: Spread filter
        if latest.get('spread_filter_active', False):
            return None

        # Filter 2: VPIN toxicity
        vpin = latest.get('vpin', 0)
        vpin_threshold = df['vpin'].rolling(100).quantile(0.9).iloc[-1] if 'vpin' in df.columns else 0.7
        if vpin > vpin_threshold:
            return None  # Toxic flow, avoid

        # Filter 3: Regime alignment
        if self.current_regime == Regime.MEAN_REVERTING and abs(signal) > 1:
            # In mean-reverting regime, only take weak signals
            signal = np.sign(signal) * 1

        # Filter 4: Session filter (for XAU/USD and FX)
        hour = timestamp.hour
        if self.asset in [Asset.XAUUSD, Asset.EURUSD, Asset.GBPUSD]:
            if not (self.config.london_start <= hour <= self.config.london_end or 
                    self.config.ny_start <= hour <= self.config.ny_end):
                return None

        # Filter 5: Confidence threshold
        if confidence < 0.3:
            return None

        # Determine direction
        if signal >= 1:
            direction = TradeDirection.LONG
        elif signal <= -1:
            direction = TradeDirection.SHORT
        else:
            return None

        # Filter 6: Trend filter (EMA alignment for XAU/USD)
        if self.asset == Asset.XAUUSD:
            ema_55 = df['close'].ewm(span=55).mean().iloc[-1]
            if direction == TradeDirection.LONG and df['close'].iloc[-1] < ema_55:
                return None
            if direction == TradeDirection.SHORT and df['close'].iloc[-1] > ema_55:
                return None

        # Get ATR for position sizing
        atr = latest.get('atr_14', df['close'].iloc[-20:].std())

        # Open trade
        entry_price = latest.get('smart_price', df['close'].iloc[-1])

        trade = self.risk_manager.open_trade(
            timestamp=timestamp,
            direction=direction,
            entry_price=entry_price,
            atr=atr
        )

        return trade

    def update_market_data(self, timestamp: datetime, high: float, low: float, close: float):
        """Update open trades with new price data."""
        self.risk_manager.update_trades(timestamp, high, low, close)

    def get_performance(self) -> Dict:
        """Get system performance statistics."""
        return self.risk_manager.get_stats()


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

class BacktestEngine:
    """
    Walk-forward backtesting engine for the precision trading system.
    """

    def __init__(self, system: PrecisionTradingSystem):
        self.system = system

    def run(
        self, 
        df: pd.DataFrame, 
        train_size: int = 5000,
        step_size: int = 1000
    ) -> pd.DataFrame:
        """
        Run walk-forward backtest.

        Args:
            df: Full historical DataFrame
            train_size: Initial training window size
            step_size: Bars between retraining

        Returns:
            DataFrame with trade log and equity curve
        """
        results = []
        equity_curve = [self.system.risk_manager.equity]

        for i in range(train_size, len(df), step_size):
            # Training window
            train_df = df.iloc[max(0, i - train_size):i]

            # Test window
            test_end = min(i + step_size, len(df))
            test_df = df.iloc[i:test_end]

            # Train system
            self.system.train(train_df)

            # Test
            for idx, row in test_df.iterrows():
                bar = row.to_dict()
                bar['timestamp'] = idx

                # Process bar
                trade = self.system.process_bar(bar)

                # Update trades with next bar's data (simulating execution)
                next_idx = test_df.index.get_loc(idx) + 1
                if next_idx < len(test_df):
                    next_row = test_df.iloc[next_idx]
                    self.system.update_market_data(
                        idx, next_row['high'], next_row['low'], next_row['close']
                    )

                # Track equity
                current_pnl = sum(t.pnl for t in self.system.risk_manager.closed_trades if t.pnl)
                equity_curve.append(self.system.risk_manager.equity + current_pnl)

                results.append({
                    'timestamp': idx,
                    'signal_generated': trade is not None,
                    'trade_direction': trade.direction.value if trade else 0,
                    'equity': equity_curve[-1]
                })

        return pd.DataFrame(results)


# =============================================================================
# DATA INGESTION HELPERS
# =============================================================================

def generate_synthetic_data(
    n_bars: int = 10000,
    asset: Asset = Asset.XAUUSD,
    trend_strength: float = 0.3,
    volatility_regime_changes: int = 3
) -> pd.DataFrame:
    """
    Generate synthetic 1-minute OHLCV data for testing.

    Simulates:
    - Volatility clustering (GARCH-like)
    - Occasional trend regimes
    - Microstructure noise
    """
    np.random.seed(42)

    timestamps = pd.date_range(start='2024-01-01', periods=n_bars, freq='1min')

    # Base price
    if asset == Asset.XAUUSD:
        base_price = 2000.0
        daily_vol = 0.001
    elif asset == Asset.BTCUSD:
        base_price = 50000.0
        daily_vol = 0.005
    elif asset == Asset.EURUSD:
        base_price = 1.0800
        daily_vol = 0.0003
    else:  # GBPUSD
        base_price = 1.2600
        daily_vol = 0.0004

    # Generate returns with volatility clustering
    returns = []
    vol = daily_vol / np.sqrt(1440)  # Per-minute vol

    regime_vol_multipliers = [1.0, 2.5, 0.5]
    regime_lengths = np.array_split(range(n_bars), volatility_regime_changes + 1)

    current_regime = 0
    for i in range(n_bars):
        # Check regime
        for idx, segment in enumerate(regime_lengths):
            if i in segment:
                current_regime = idx % len(regime_vol_multipliers)
                break

        # GARCH-like volatility clustering
        if i > 0:
            vol = 0.94 * vol + 0.06 * returns[-1]**2

        vol = max(vol, daily_vol / np.sqrt(1440) * 0.1)

        # Trend component
        trend = trend_strength * np.sin(2 * np.pi * i / 1000) * vol

        # Generate return
        ret = np.random.normal(trend, vol * regime_vol_multipliers[current_regime])
        returns.append(ret)

    returns = np.array(returns)

    # Build prices
    prices = base_price * np.exp(np.cumsum(returns))

    # Build OHLCV from close prices
    df = pd.DataFrame(index=timestamps)
    df['close'] = prices
    df['open'] = df['close'].shift(1).fillna(base_price)

    # Generate realistic high/low from close
    intrabar_vol = abs(returns) * 0.5 + daily_vol / np.sqrt(1440) * 0.3
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + intrabar_vol)
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - intrabar_vol)

    # Volume (higher during volatile periods)
    df['volume'] = np.random.lognormal(10, 1) * (1 + abs(returns) * 100)

    # Add synthetic L2 data
    df['bid'] = df['close'] - ASSET_CONFIGS[asset].spread_avg / 2
    df['ask'] = df['close'] + ASSET_CONFIGS[asset].spread_avg / 2
    df['bid_vol'] = df['volume'] * np.random.uniform(0.3, 0.7)
    df['ask_vol'] = df['volume'] - df['bid_vol']

    return df


def load_mt5_csv(filepath: str) -> pd.DataFrame:
    """
    Load MT5 exported 1-minute CSV data.

    Expected columns: Time, Open, High, Low, Close, Volume
    """
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['Time'])
    df.set_index('timestamp', inplace=True)
    df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    return df


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def main():
    """Example usage of the complete trading system."""

    print("=" * 80)
    print("1-MINUTE PRECISION TRADING SYSTEM")
    print("=" * 80)

    # 1. Generate or load data
    print("\n[1] Generating synthetic test data...")
    data = generate_synthetic_data(n_bars=8000, asset=Asset.XAUUSD)
    print(f"    Data shape: {data.shape}")
    print(f"    Date range: {data.index[0]} to {data.index[-1]}")

    # 2. Initialize system
    print("\n[2] Initializing trading system for XAU/USD...")
    system = PrecisionTradingSystem(
        asset=Asset.XAUUSD,
        model_type='xgboost',  # Use 'lorentzian' for gold-specific
        use_hmm=True
    )

    # 3. Train
    print("\n[3] Training models...")
    train_data = data.iloc[:5000]
    system.train(train_data)

    # 4. Simulate live trading
    print("\n[4] Simulating live trading on out-of-sample data...")
    test_data = data.iloc[5000:]

    for i in range(0, len(test_data), 10):  # Process every 10th bar for demo
        bar = test_data.iloc[i].to_dict()
        bar['timestamp'] = test_data.index[i]

        trade = system.process_bar(bar)

        # Update with next bar
        if i + 1 < len(test_data):
            next_bar = test_data.iloc[i + 1]
            system.update_market_data(
                test_data.index[i],
                next_bar['high'],
                next_bar['low'],
                next_bar['close']
            )

        if trade:
            print(f"    Trade opened: {trade.direction.name} at {trade.entry_price:.2f}")

    # 5. Results
    print("\n[5] Performance Summary:")
    stats = system.get_performance()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    print("\n" + "=" * 80)
    print("System ready for live deployment.")
    print("Connect your broker API to process_bar() for live trading.")
    print("=" * 80)


if __name__ == "__main__":
    main()
