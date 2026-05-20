"""
features_microstructure.py
==========================
Research-backed microstructure feature engineering for 1m Gold (XAU/USD) trading.

Sources:
- Gould et al. (2013): Queue Imbalance as a One-Tick-Ahead Predictor
- Easley, Lopez de Prado, O'Hara (2012): VPIN / Flow Toxicity
- Cont (2001): Stochastic Volatility & Order Flow
- Khalifa, Miao, Ramchander (2011): Realized Volatility in Metal Futures
- Stoikov (2018): Micro-price & CQW Midpoint
- Brogaard et al. (2014): HFT Predictability & Information
- ECB (2020): Fast Trading and the Virtue of Entropy

This module proxies institutional microstructure signals using only 1m OHLCV data
(retail-accessible) to create a hardened execution filter.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MicrostructureFeatureEngine:
    """
    Generates microstructure-aware features from 1m OHLCV bars.

    All features are designed to be computable from standard retail data feeds
    (no Level 2 order book required).
    """

    def __init__(self, tick_size: float = 0.01, volume_bucket_size: float = 1000.0):
        self.tick_size = tick_size
        self.volume_bucket_size = volume_bucket_size

    # =========================================================================
    # 1. ORDER FLOW IMBALANCE PROXY (OFI-proxy)
    # =========================================================================
    def compute_ofi_proxy(self, df: pd.DataFrame) -> pd.Series:
        """
        Proxies Order Flow Imbalance using intrabar price-volume dynamics.

        Logic: Buy pressure concentrates volume near the high; sell pressure
        concentrates near the low. We proxy this via close position, VWAP
        position, bar delta, and volume intensity.

        Returns:
            pd.Series: OFI-proxy (positive = buy pressure, negative = sell pressure)
        """
        # Volume-weighted average price position within bar (0=low, 1=high)
        vwap = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        # Actually compute rolling VWAP properly
        typical = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical * df['volume']).rolling(window=20, min_periods=1).sum() / df['volume'].rolling(window=20, min_periods=1).sum()

        vwap_position = (vwap - df['low']) / (df['high'] - df['low'] + 1e-9)

        # Close position (buying pressure if close near high)
        close_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)

        # Bar delta: signed directional pressure normalized by range
        bar_delta = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)

        # Volume intensity relative to recent average
        volume_ma = df['volume'].rolling(window=20, min_periods=1).mean()
        volume_intensity = df['volume'] / (volume_ma + 1e-9)

        # Composite OFI-proxy
        ofi_proxy = (
            0.40 * (2 * close_position - 1) +
            0.30 * (2 * vwap_position - 1) +
            0.20 * bar_delta +
            0.10 * np.log(volume_intensity + 1e-9)
        ) * np.sqrt(df['volume'] + 1e-9)

        return ofi_proxy

    # =========================================================================
    # 2. VPIN PROXY (Volume-Synchronized Probability of Informed Trading)
    # =========================================================================
    def compute_vpin_proxy(self, df: pd.DataFrame, window: int = 50) -> pd.Series:
        """
        VPIN proxy using volume-bucket approach.

        High VPIN = toxic flow (informed traders active) = avoid or expect
        large moves. Low VPIN = benign flow = safer to trade.

        Uses Lee-Ready algorithm proxy: close above typical price = buyer-initiated.
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        buy_volume = df['volume'] * (df['close'] > typical_price).astype(float)
        sell_volume = df['volume'] * (df['close'] <= typical_price).astype(float)

        vpin_values = []

        for i in range(len(df)):
            start = max(0, i - window + 1)
            window_slice = df.iloc[start:i+1]

            if len(window_slice) < 10:
                vpin_values.append(0.5)
                continue

            total_vol = window_slice['volume'].sum()
            if total_vol < self.volume_bucket_size:
                vpin_values.append(0.5)
                continue

            buy_vol_win = buy_volume.iloc[start:i+1].sum()
            sell_vol_win = sell_volume.iloc[start:i+1].sum()

            # Volume imbalance normalized
            vol_imbalance = np.abs(buy_vol_win - sell_vol_win) / (total_vol + 1e-9)

            # Return component: sum of absolute returns in window
            returns = np.abs(window_slice['close'].pct_change().fillna(0))
            return_component = returns.sum() * 100  # Scale to percentage

            # VPIN = vol_imbalance * return_component, capped at 1.0
            vpin = min(vol_imbalance * return_component, 1.0)
            vpin_values.append(vpin)

        return pd.Series(vpin_values, index=df.index)

    # =========================================================================
    # 3. REALIZED VOLATILITY ESTIMATORS (Gold-Specific)
    # =========================================================================
    def compute_realized_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Multiple realized volatility estimators proven effective for gold futures.

        Parkinson (1980): Uses high-low, 5.2x more efficient than close-to-close.
        Garman-Klass (1980): Uses OHLC, 7.4x more efficient.
        Yang-Zhang (2000): Uses OHLC + overnight, most robust.
        Bipower Variation: Jump-robust estimator.
        """
        features = pd.DataFrame(index=df.index)

        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])

        # 1. Parkinson Estimator
        parkinson = (log_hl ** 2) / (4 * np.log(2))
        features['rv_parkinson'] = parkinson.rolling(window=window, min_periods=1).mean()

        # 2. Garman-Klass Estimator
        gk = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        features['rv_garman_klass'] = gk.rolling(window=window, min_periods=1).mean()

        # 3. Yang-Zhang Estimator
        log_oc = np.log(df['open'] / df['close'].shift(1))
        log_cc = np.log(df['close'] / df['open'])

        rs = (
            np.log(df['high'] / df['close']) * np.log(df['high'] / df['open']) +
            np.log(df['low'] / df['close']) * np.log(df['low'] / df['open'])
        )

        k = 0.34 / (1.34 + (window + 1) / (window - 1)) if window > 1 else 0.34

        features['rv_yang_zhang'] = (
            log_oc.rolling(window=window, min_periods=1).var() +
            k * parkinson.rolling(window=window, min_periods=1).mean() +
            (1 - k) * rs.rolling(window=window, min_periods=1).mean()
        )

        # 4. Bipower Variation (jump-robust)
        returns = np.log(df['close'] / df['close'].shift(1))
        abs_returns = np.abs(returns)
        features['rv_bipower'] = (
            (abs_returns * abs_returns.shift(1)).rolling(window=window, min_periods=2).sum() * (np.pi / 2)
        )

        # 5. Volatility of Volatility (VoV)
        rv = features['rv_parkinson'].fillna(0)
        features['vov'] = rv.rolling(window=window, min_periods=1).std()

        # 6. Standard close-to-close RV (baseline)
        features['rv_standard'] = (returns ** 2).rolling(window=window, min_periods=1).sum()

        return features

    # =========================================================================
    # 4. MICRO-PRICE & CQW MIDPOINT PROXIES
    # =========================================================================
    def compute_microstructure_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Proxies for CQW (Constrained Quantity-Weighted) midpoint and micro-price.

        These are more efficient estimators of true asset value than transaction
        price (close). At 1m sampling, CQW shows 96%+ non-zero return ratios
        vs 67% for raw close price.
        """
        features = pd.DataFrame(index=df.index)

        typical_price = (df['high'] + df['low'] + df['close']) / 3

        # Volume imbalance proxy: where did volume concentrate within the bar?
        volume_imbalance = (df['close'] - typical_price) / (df['high'] - df['low'] + 1e-9)
        volume_imbalance = np.clip(volume_imbalance, -1, 1)

        mid_price = (df['high'] + df['low']) / 2

        # CQW Midpoint proxy
        features['cqw_midpoint'] = mid_price + 0.5 * self.tick_size * volume_imbalance

        # Micro-price proxy (Stoikov 2018 simplified)
        spread = df['high'] - df['low']
        micro_adjustment = spread * np.tanh(volume_imbalance * 3) * 0.3
        features['micro_price'] = mid_price + micro_adjustment

        # Returns on efficient prices (more predictable than close-to-close)
        features['cqw_return'] = features['cqw_midpoint'].pct_change()
        features['micro_return'] = features['micro_price'].pct_change()

        # Bid-ask bounce detection (reversal probability after large moves)
        features['reversal_prob'] = (
            ((df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1))).astype(float)
            .rolling(window=10, min_periods=1).mean()
        )

        # Spread-to-range ratio (microstructure dominance indicator)
        features['spread_range_ratio'] = spread / (df['close'].rolling(window=20, min_periods=1).std() + 1e-9)

        return features

    # =========================================================================
    # 5. ENTROPY & MARKET STRUCTURE FEATURES
    # =========================================================================
    def compute_entropy_features(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Shannon entropy proxies for market structure predictability.

        Low entropy = structured flow (HFT market making, predictable).
        High entropy = chaotic flow (spoofing, news events, avoid).

        From ECB (2020): "Fast Trading and the Virtue of Entropy"
        """
        features = pd.DataFrame(index=df.index)
        returns = df['close'].pct_change()

        # 1. Return sign entropy (run-length distribution)
        def _sign_entropy(x: pd.Series) -> float:
            if len(x) < 5:
                return 0.5
            signs = np.sign(x)
            runs = []
            current_run = 1
            for i in range(1, len(signs)):
                if signs.iloc[i] == signs.iloc[i-1]:
                    current_run += 1
                else:
                    runs.append(current_run)
                    current_run = 1
            runs.append(current_run)

            if len(runs) <= 1:
                return 0.5

            probs = np.array(runs) / sum(runs)
            entropy = -np.sum(probs * np.log2(probs + 1e-9))
            max_entropy = np.log2(len(runs))
            return entropy / max_entropy if max_entropy > 0 else 0

        features['sign_entropy'] = returns.rolling(window=window, min_periods=5).apply(
            _sign_entropy, raw=False
        )

        # 2. Price path efficiency (direct vs meandering)
        bar_range = df['high'] - df['low']
        bar_move = np.abs(df['close'] - df['open'])
        features['path_efficiency'] = bar_move / (bar_range + 1e-9)

        # 3. Volume entropy (concentrated vs dispersed)
        vol_ma = df['volume'].rolling(window=window, min_periods=1).mean()
        vol_pct = df['volume'] / (vol_ma * window + 1e-9)
        vol_pct = np.clip(vol_pct, 1e-9, 1.0)
        features['volume_entropy'] = -(vol_pct * np.log2(vol_pct)).rolling(window=window, min_periods=1).sum()

        # 4. HFT activity proxy (Brogaard et al. 2014)
        # High trade intensity + low return per volume = HFT market making
        trade_intensity = df['volume'] / (vol_ma + 1e-9)
        return_per_volume = np.abs(returns) / (df['volume'] + 1e-9)
        features['hft_activity_proxy'] = trade_intensity / (return_per_volume * 1000 + 1)

        # 5. Bar coherence (open-to-close consistency with high-low range)
        features['bar_coherence'] = np.abs(df['close'] - df['open']) / (bar_range + 1e-9)

        return features

    # =========================================================================
    # 6. CROSS-FEATURE INTERACTIONS (Microstructure + Technical)
    # =========================================================================
    def compute_interactions(self, df: pd.DataFrame, micro_features: pd.DataFrame) -> pd.DataFrame:
        """
        Critical interactions between microstructure and traditional features.
        These capture regime-dependent microstructure effects.
        """
        features = pd.DataFrame(index=df.index)

        # RSI × OFI alignment (momentum confirmed by flow)
        if 'rsi_14' in df.columns and 'ofi_proxy' in micro_features.columns:
            features['rsi_ofi_aligned'] = (
                ((df['rsi_14'] > 50) & (micro_features['ofi_proxy'] > 0)) |
                ((df['rsi_14'] < 50) & (micro_features['ofi_proxy'] < 0))
            ).astype(float)

        # ATR × VPIN (volatility expansion with toxic flow = avoid)
        if 'atr_14' in df.columns and 'vpin_proxy' in micro_features.columns:
            features['atr_vpin_danger'] = (
                (df['atr_14'] > df['atr_14'].rolling(20).mean()) &
                (micro_features['vpin_proxy'] > 0.6)
            ).astype(float)

        # Volume × Micro-price deviation (volume confirming efficient price move)
        if 'micro_price' in micro_features.columns:
            micro_deviation = np.abs(df['close'] - micro_features['micro_price']) / df['close']
            features['volume_micro_confirm'] = df['volume'] * (1 - micro_deviation)

        # Entropy × Trend (structured flow in trend direction = strong)
        if 'ema_20' in df.columns and 'sign_entropy' in micro_features.columns:
            trend_direction = np.sign(df['close'] - df['ema_20'])
            ofi_sign = np.sign(micro_features.get('ofi_proxy', 0))
            features['trend_flow_aligned'] = (trend_direction == ofi_sign).astype(float) * (1 - micro_features['sign_entropy'].fillna(0.5))

        return features

    # =========================================================================
    # 7. MASTER FEATURE GENERATION
    # =========================================================================
    def generate_all_features(self, df: pd.DataFrame, include_interactions: bool = True) -> pd.DataFrame:
        """
        Generate complete microstructure feature set.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            include_interactions: Whether to compute cross-feature interactions

        Returns:
            pd.DataFrame: All microstructure features
        """
        print("[Microstructure] Generating OFI-proxy...")
        ofi = self.compute_ofi_proxy(df)

        print("[Microstructure] Generating VPIN-proxy...")
        vpin = self.compute_vpin_proxy(df)

        print("[Microstructure] Generating Realized Volatility estimators...")
        rv = self.compute_realized_volatility(df)

        print("[Microstructure] Generating Micro-price & CQW features...")
        micro_price = self.compute_microstructure_price(df)

        print("[Microstructure] Generating Entropy & Market Structure features...")
        entropy = self.compute_entropy_features(df)

        # Combine all
        micro_features = pd.DataFrame({
            'ofi_proxy': ofi,
            'vpin_proxy': vpin,
        }, index=df.index)

        micro_features = pd.concat([micro_features, rv, micro_price, entropy], axis=1)

        # Interactions
        if include_interactions:
            print("[Microstructure] Generating cross-feature interactions...")
            interactions = self.compute_interactions(df, micro_features)
            micro_features = pd.concat([micro_features, interactions], axis=1)

        # Fill NaN
        micro_features = micro_features.ffill().fillna(0)

        print(f"[Microstructure] Generated {len(micro_features.columns)} features.")
        return micro_features


# =============================================================================
# STANDALONE HELPER FUNCTIONS (for quick use)
# =============================================================================

def add_microstructure_features(df: pd.DataFrame, tick_size: float = 0.01) -> pd.DataFrame:
    """
    One-shot function to add all microstructure features to an existing dataframe.

    Usage:
        df_enhanced = add_microstructure_features(df)
    """
    engine = MicrostructureFeatureEngine(tick_size=tick_size)
    micro = engine.generate_all_features(df)
    return pd.concat([df, micro], axis=1)


def get_microstructure_filter_mask(
    df: pd.DataFrame,
    ensemble_confidence: pd.Series,
    ensemble_direction: pd.Series,
    ofi_proxy: pd.Series,
    vpin_proxy: pd.Series,
    sign_entropy: pd.Series,
    hft_activity: pd.Series,
    regime_allows_trading: pd.Series,
    conf_threshold: float = 0.65,
    vpin_threshold: float = 0.6,
    entropy_threshold: float = 0.7
) -> pd.Series:
    """
    Returns boolean mask of bars that pass ALL microstructure filters.

    This is the hardened execution filter based on research findings.
    """
    # 1. High confidence
    mask_conf = ensemble_confidence >= conf_threshold

    # 2. OFI aligns with ensemble direction (microstructure confirms ML)
    mask_ofi = (ensemble_direction > 0) & (ofi_proxy > 0) | (ensemble_direction < 0) & (ofi_proxy < 0)

    # 3. VPIN below toxicity threshold (avoid adverse selection)
    mask_vpin = vpin_proxy < vpin_threshold

    # 4. Market structure is predictable (not chaotic)
    mask_entropy = sign_entropy < entropy_threshold

    # 5. HFT activity in normal range (not spoofing/layering)
    hft_median = hft_activity.rolling(100, min_periods=10).median()
    mask_hft = hft_activity < (hft_median * 3)  # Allow 3x normal, reject extreme

    # 6. Regime permits trading
    mask_regime = regime_allows_trading

    return mask_conf & mask_ofi & mask_vpin & mask_entropy & mask_hft & mask_regime
