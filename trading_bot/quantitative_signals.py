"""
Quantitative Trading Signals Module
====================================
Institutional-grade quantitative signals used by hedge funds and prop trading firms.
Includes mean reversion, momentum, volatility breakout, and statistical arbitrage signals.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Type of quantitative signal."""
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    PAIRS_TRADING = "pairs_trading"
    MARKET_REGIME = "market_regime"


@dataclass
class QuantSignal:
    """Container for quantitative trading signals."""
    signal_type: SignalType
    signal: str  # 'BUY', 'SELL', 'NEUTRAL'
    confidence: float  # 0-1
    z_score: Optional[float] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = None


class QuantitativeSignals:
    """
    Advanced Quantitative Signal Generator
    
    Implements strategies used by quantitative hedge funds:
    - Mean Reversion (Bollinger, RSI, Statistical)
    - Momentum (Time-series, Cross-sectional)
    - Volatility Breakout
    - Statistical Arbitrage
    - Market Regime Detection
    """
    
    def __init__(self, df: pd.DataFrame, lookback: int = 252):
        """
        Initialize quantitative signal generator.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with 'timestamp', 'open', 'high', 'low', 'close', 'volume'
        lookback : int
            Lookback period for statistical calculations (default: 252 trading days)
        """
        self.df = df.copy()
        self.lookback = lookback
        self.signals: List[QuantSignal] = []
        
    # =========================================================================
    # Mean Reversion Signals
    # =========================================================================
    
    def mean_reversion_bollinger(self, std_threshold: float = 2.0) -> QuantSignal:
        """
        Mean reversion signal based on Bollinger Bands.
        
        Entry when price exceeds 2+ standard deviations from mean.
        """
        close = self.df['close'].iloc[-self.lookback:]
        current_price = self.df['close'].iloc[-1]
        
        mean = close.mean()
        std = close.std()
        z_score = (current_price - mean) / std
        
        # Calculate position sizing based on z-score
        confidence = min(abs(z_score) / std_threshold, 1.0)
        
        if z_score < -std_threshold:
            signal = 'BUY'
            entry_price = current_price
            stop_loss = current_price * (1 - 0.02)  # 2% stop
            take_profit = mean  # Target mean reversion
        elif z_score > std_threshold:
            signal = 'SELL'
            entry_price = current_price
            stop_loss = current_price * (1 + 0.02)
            take_profit = mean
        else:
            signal = 'NEUTRAL'
            entry_price = stop_loss = take_profit = None
            
        return QuantSignal(
            signal_type=SignalType.MEAN_REVERSION,
            signal=signal,
            confidence=confidence * 0.7,  # Weight factor
            z_score=z_score,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={'mean': mean, 'std': std, 'method': 'bollinger'}
        )
    
    def mean_reversion_rsi(self, oversold: float = 30, overbought: float = 70) -> QuantSignal:
        """
        Mean reversion using RSI extremes.
        """
        close = self.df['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        if current_rsi < oversold:
            signal = 'BUY'
            confidence = (oversold - current_rsi) / oversold
        elif current_rsi > overbought:
            signal = 'SELL'
            confidence = (current_rsi - overbought) / (100 - overbought)
        else:
            signal = 'NEUTRAL'
            confidence = 0.0
            
        return QuantSignal(
            signal_type=SignalType.MEAN_REVERSION,
            signal=signal,
            confidence=min(confidence, 1.0) * 0.6,
            z_score=(current_rsi - 50) / 25,
            metadata={'rsi': current_rsi, 'method': 'rsi'}
        )
    
    def mean_reversion_statistical(self, window: int = 60) -> QuantSignal:
        """
        Statistical mean reversion using z-score and half-life.
        
        Calculates:
        - Z-score of current price
        - Half-life of mean reversion (via Ornstein-Uhlenbeck)
        """
        close = self.df['close'].iloc[-self.lookback:]
        current_price = self.df['close'].iloc[-1]
        
        # Z-score
        mean = close.mean()
        std = close.std()
        z_score = (current_price - mean) / std
        
        # Half-life estimation via Ornstein-Uhlenbeck
        diff_close = close.diff().dropna()
        lag_close = close.shift(1).dropna()
        
        if len(diff_close) > 10:
            # OU process: dX = theta * (mu - X) * dt + sigma * dW
            # Regression: delta_y = alpha + beta * y_lagged
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    lag_close.values, diff_close.values
                )
                
                if slope < 0:
                    half_life = -np.log(2) / slope
                    is_mean_reverting = True
                else:
                    half_life = np.inf
                    is_mean_reverting = False
            except:
                half_life = np.inf
                is_mean_reverting = False
        else:
            half_life = np.inf
            is_mean_reverting = False
        
        # Generate signal
        threshold = 1.5
        if z_score < -threshold and is_mean_reverting:
            signal = 'BUY'
            confidence = min(abs(z_score) / 3, 1.0)
        elif z_score > threshold and is_mean_reverting:
            signal = 'SELL'
            confidence = min(abs(z_score) / 3, 1.0)
        else:
            signal = 'NEUTRAL'
            confidence = 0.0
            
        return QuantSignal(
            signal_type=SignalType.MEAN_REVERSION,
            signal=signal,
            confidence=confidence * 0.8,
            z_score=z_score,
            metadata={
                'half_life': half_life,
                'is_mean_reverting': is_mean_reverting,
                'mean': mean,
                'method': 'statistical'
            }
        )
    
    # =========================================================================
    # Momentum Signals
    # =========================================================================
    
    def momentum_time_series(self, periods: List[int] = [5, 10, 20, 60]) -> QuantSignal:
        """
        Time-series momentum (trend following).
        
        Aggregates momentum across multiple time horizons.
        """
        close = self.df['close']
        current_price = close.iloc[-1]
        
        momentum_scores = []
        for period in periods:
            if len(close) > period:
                past_price = close.iloc[-period]
                momentum = (current_price - past_price) / past_price
                # Normalize momentum by volatility
                vol = close.pct_change().rolling(window=period).std().iloc[-1]
                if vol > 0:
                    risk_adjusted_momentum = momentum / vol
                else:
                    risk_adjusted_momentum = momentum
                momentum_scores.append(risk_adjusted_momentum)
        
        if not momentum_scores:
            return QuantSignal(
                signal_type=SignalType.MOMENTUM,
                signal='NEUTRAL',
                confidence=0.0,
                metadata={'method': 'time_series'}
            )
        
        avg_momentum = np.mean(momentum_scores)
        momentum_strength = np.abs(avg_momentum)
        
        # Signal generation
        if avg_momentum > 0.5:
            signal = 'BUY'
            confidence = min(momentum_strength / 2, 1.0)
        elif avg_momentum < -0.5:
            signal = 'SELL'
            confidence = min(momentum_strength / 2, 1.0)
        else:
            signal = 'NEUTRAL'
            confidence = 0.0
            
        return QuantSignal(
            signal_type=SignalType.MOMENTUM,
            signal=signal,
            confidence=confidence * 0.7,
            z_score=avg_momentum,
            metadata={
                'momentum_scores': momentum_scores,
                'avg_momentum': avg_momentum,
                'method': 'time_series'
            }
        )
    
    def momentum_rsi_divergence(self) -> QuantSignal:
        """
        Detect RSI divergence with price (hidden momentum signal).
        
        Bullish divergence: Price makes lower low, RSI makes higher low
        Bearish divergence: Price makes higher high, RSI makes lower high
        """
        close = self.df['close']
        
        # Calculate RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        
        # Find local extrema (simplified)
        window = 20
        local_min_price = close.rolling(window=window).min()
        local_max_price = close.rolling(window=window).max()
        local_min_rsi = rsi.rolling(window=window).min()
        local_max_rsi = rsi.rolling(window=window).max()
        
        current_price = close.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Check for divergence
        price_trend = close.iloc[-5:].corr(pd.Series(np.arange(5)))
        rsi_trend = rsi.iloc[-5:].corr(pd.Series(np.arange(5)))
        
        if price_trend < -0.5 and rsi_trend > 0.5:
            signal = 'BUY'  # Bullish divergence
            confidence = min(abs(price_trend - rsi_trend) / 2, 1.0)
        elif price_trend > 0.5 and rsi_trend < -0.5:
            signal = 'SELL'  # Bearish divergence
            confidence = min(abs(price_trend - rsi_trend) / 2, 1.0)
        else:
            signal = 'NEUTRAL'
            confidence = 0.0
            
        return QuantSignal(
            signal_type=SignalType.MOMENTUM,
            signal=signal,
            confidence=confidence * 0.6,
            metadata={
                'price_trend': price_trend,
                'rsi_trend': rsi_trend,
                'method': 'divergence'
            }
        )
    
    # =========================================================================
    # Volatility Breakout Signals
    # =========================================================================
    
    def volatility_breakout_atr(self, multiplier: float = 2.0) -> QuantSignal:
        """
        Volatility breakout using ATR bands.
        
        Entry when price breaks out of ATR-based channel.
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()
        
        # ATR bands
        middle = close.ewm(span=20, adjust=False).mean()
        upper = middle + multiplier * atr
        lower = middle - multiplier * atr
        
        current_price = close.iloc[-1]
        current_atr = atr.iloc[-1]
        
        # Volatility regime
        atr_ratio = current_atr / atr.rolling(window=60).mean().iloc[-1]
        is_low_vol = atr_ratio < 0.8  # Low volatility = potential breakout
        
        if current_price > upper.iloc[-1]:
            signal = 'BUY'
            confidence = min((current_price - upper.iloc[-1]) / current_atr, 1.0)
        elif current_price < lower.iloc[-1]:
            signal = 'SELL'
            confidence = min((lower.iloc[-1] - current_price) / current_atr, 1.0)
        else:
            signal = 'NEUTRAL'
            confidence = 0.0
            
        # Boost confidence if coming from low volatility
        if is_low_vol and signal != 'NEUTRAL':
            confidence = min(confidence * 1.3, 1.0)
            
        return QuantSignal(
            signal_type=SignalType.VOLATILITY_BREAKOUT,
            signal=signal,
            confidence=confidence * 0.7,
            z_score=atr_ratio,
            metadata={
                'atr': current_atr,
                'atr_ratio': atr_ratio,
                'is_low_vol': is_low_vol,
                'method': 'atr_breakout'
            }
        )
    
    def volatility_squeeze(self) -> QuantSignal:
        """
        Bollinger Band inside Keltner Channel = Squeeze.
        
        Indicates consolidation before explosive move.
        Direction determined by momentum.
        """
        close = self.df['close']
        
        # Bollinger Bands
        bb_middle = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        
        # Keltner Channel
        hl2 = (self.df['high'] + self.df['low']) / 2
        atr = (self.df['high'] - self.df['low']).ewm(span=20, adjust=False).mean()
        kc_upper = hl2.ewm(span=20, adjust=False).mean() + 1.5 * atr
        kc_lower = hl2.ewm(span=20, adjust=False).mean() - 1.5 * atr
        
        # Check for squeeze
        is_squeeze = (bb_lower.iloc[-1] > kc_lower.iloc[-1]) and \
                     (bb_upper.iloc[-1] < kc_upper.iloc[-1])
        
        # Check squeeze history
        squeeze_count = 0
        for i in range(min(10, len(self.df))):
            if (bb_lower.iloc[-i] > kc_lower.iloc[-i]) and \
               (bb_upper.iloc[-i] < kc_upper.iloc[-i]):
                squeeze_count += 1
        
        # Momentum for direction
        momentum = close.iloc[-1] / close.iloc[-20] - 1
        
        if is_squeeze and squeeze_count >= 3:
            if momentum > 0.02:
                signal = 'BUY'
                confidence = min(squeeze_count / 10, 1.0)
            elif momentum < -0.02:
                signal = 'SELL'
                confidence = min(squeeze_count / 10, 1.0)
            else:
                signal = 'NEUTRAL'
                confidence = squeeze_count / 10 * 0.5
        else:
            signal = 'NEUTRAL'
            confidence = 0.0
            
        return QuantSignal(
            signal_type=SignalType.VOLATILITY_BREAKOUT,
            signal=signal,
            confidence=confidence * 0.8,
            metadata={
                'is_squeeze': is_squeeze,
                'squeeze_count': squeeze_count,
                'momentum': momentum,
                'method': 'squeeze'
            }
        )
    
    # =========================================================================
    # Market Regime Detection (Improved)
    # =========================================================================
    
    def detect_market_regime(self) -> Dict[str, any]:
        """
        Detect current market regime using statistical analysis.
        
        Regimes:
        - BULL_LOW_VOL
        - BULL_HIGH_VOL
        - BEAR_LOW_VOL
        - BEAR_HIGH_VOL
        - SIDEWAYS
        
        Uses:
        - Trend strength (linear regression slope)
        - Volatility regime (historical percentile)
        - Autocorrelation (Hurst-like)
        - Statistical thresholds based on data distribution
        """
        close = self.df['close'].iloc[-self.lookback:].values
        returns = np.diff(close) / close[:-1]
        
        if len(returns) < 20:
            return self._default_regime_result()
        
        n = len(returns)
        
        trend = self._calculate_trend_strength(close)
        volatility = np.std(returns) * np.sqrt(252)
        
        skewness = float(pd.Series(returns).skew()) if len(returns) > 2 else 0.0
        kurtosis = float(pd.Series(returns).kurtosis()) if len(returns) > 3 else 0.0
        
        vol_history = pd.Series(returns).rolling(window=min(60, n//4)).std() * np.sqrt(252)
        if len(vol_history) > 20:
            vol_percentile = float(np.sum(vol_history.dropna() < volatility) / len(vol_history.dropna()))
        else:
            vol_percentile = 0.5
        
        vol_threshold = 0.6 if vol_percentile > 0.6 else (0.4 if vol_percentile < 0.4 else 0.5)
        
        autocorr = self._calculate_autocorrelation(returns)
        
        abs_trend_threshold = self._calculate_trend_threshold(returns)
        
        if abs(trend) < abs_trend_threshold:
            regime = 'SIDEWAYS'
            regime_confidence = 1.0 - min(abs(trend) / abs_trend_threshold, 1.0)
        elif trend > 0:
            if vol_percentile < vol_threshold:
                regime = 'BULL_LOW_VOL'
            else:
                regime = 'BULL_HIGH_VOL'
            regime_confidence = min(abs(trend) / (abs_trend_threshold * 2), 1.0)
        else:
            if vol_percentile < vol_threshold:
                regime = 'BEAR_LOW_VOL'
            else:
                regime = 'BEAR_HIGH_VOL'
            regime_confidence = min(abs(trend) / (abs_trend_threshold * 2), 1.0)
        
        if autocorr > 0.3:
            regime_confidence *= 1.2
        
        regime_confidence = min(regime_confidence, 1.0)
        
        return {
            'regime': regime,
            'confidence': regime_confidence,
            'trend': trend,
            'volatility': volatility,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'vol_percentile': vol_percentile,
            'autocorrelation': autocorr,
            'trend_threshold': abs_trend_threshold,
            'vol_threshold': vol_threshold
        }
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength using linear regression."""
        if len(prices) < 10:
            return 0.0
        
        x = np.arange(len(prices))
        x_mean = x.mean()
        
        slope = np.sum((x - x_mean) * (prices - prices.mean())) / np.sum((x - x_mean) ** 2)
        trend = slope * len(prices) / prices.mean()
        
        return float(trend)
    
    def _calculate_autocorrelation(self, returns: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation at given lag."""
        if len(returns) <= lag:
            return 0.0
        
        n = len(returns) - lag
        c0 = np.sum(returns[:n] ** 2) / n
        c1 = np.sum(returns[:n] * returns[lag:lag+n]) / n
        
        if c0 < 1e-10:
            return 0.0
        
        return float(c1 / c0)
    
    def _calculate_trend_threshold(self, returns: np.ndarray) -> float:
        """Calculate adaptive trend threshold based on return distribution."""
        if len(returns) < 10:
            return 0.05
        
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        threshold = max(0.02, min(0.10, 2.0 * std_ret + abs(mean_ret)))
        
        return float(threshold)
    
    def _default_regime_result(self) -> Dict[str, any]:
        """Return default regime result for insufficient data."""
        return {
            'regime': 'SIDEWAYS',
            'confidence': 0.5,
            'trend': 0.0,
            'volatility': 0.20,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'vol_percentile': 0.5,
            'autocorrelation': 0.0,
            'trend_threshold': 0.05,
            'vol_threshold': 0.5
        }
    
    def regime_adjusted_signal(self) -> QuantSignal:
        """
        Generate signal adjusted for current market regime.
        
        Adapts strategy based on detected regime:
        - High volatility regimes: reduce position size, wider stops
        - Trending regimes: momentum strategies favored
        - Sideways: mean reversion strategies favored
        """
        regime_data = self.detect_market_regime()
        regime = regime_data['regime']
        confidence = regime_data['confidence']
        
        vol_multiplier = 1.0
        if 'HIGH_VOL' in regime:
            vol_multiplier = 0.7
        elif 'LOW_VOL' in regime:
            vol_multiplier = 1.0
        
        if regime in ['BULL_LOW_VOL', 'BULL_HIGH_VOL']:
            signal = 'BUY'
            adjusted_confidence = confidence * 0.8 * vol_multiplier
        elif regime in ['BEAR_LOW_VOL', 'BEAR_HIGH_VOL']:
            signal = 'SELL'
            adjusted_confidence = confidence * 0.8 * vol_multiplier
        else:
            signal = 'NEUTRAL'
            adjusted_confidence = confidence * 0.4
        
        adjusted_confidence = min(adjusted_confidence, 1.0)
            
        return QuantSignal(
            signal_type=SignalType.MARKET_REGIME,
            signal=signal,
            confidence=adjusted_confidence,
            metadata=regime_data
        )
    
    # =========================================================================
    # Aggregated Signal
    # =========================================================================
    
    def generate_all_signals(self) -> List[QuantSignal]:
        """Generate all quantitative signals."""
        self.signals = [
            self.mean_reversion_bollinger(),
            self.mean_reversion_rsi(),
            self.mean_reversion_statistical(),
            self.momentum_time_series(),
            self.momentum_rsi_divergence(),
            self.volatility_breakout_atr(),
            self.volatility_squeeze(),
            self.regime_adjusted_signal(),
        ]
        return self.signals
    
    def get_aggregated_signal(self) -> Tuple[str, float, Dict]:
        """
        Get aggregated quantitative signal.
        
        Returns
        -------
        Tuple[str, float, Dict]
            (signal, confidence, metadata)
        """
        if not self.signals:
            self.generate_all_signals()
        
        # Weight by signal type importance
        weights = {
            SignalType.MEAN_REVERSION: 0.25,
            SignalType.MOMENTUM: 0.25,
            SignalType.VOLATILITY_BREAKOUT: 0.25,
            SignalType.MARKET_REGIME: 0.25,
        }
        
        buy_score = 0
        sell_score = 0
        
        for signal in self.signals:
            if signal.signal == 'NEUTRAL':
                continue
                
            weight = weights.get(signal.signal_type, 0.1)
            adjusted_conf = signal.confidence * weight
            
            if signal.signal == 'BUY':
                buy_score += adjusted_conf
            elif signal.signal == 'SELL':
                sell_score += adjusted_conf
        
        total_score = buy_score + sell_score
        
        if total_score == 0:
            return 'HOLD', 0.0, {'buy_score': 0, 'sell_score': 0}
        
        if buy_score > sell_score * 1.2:
            return 'BUY', buy_score / total_score, {
                'buy_score': buy_score,
                'sell_score': sell_score,
                'signal_count': len([s for s in self.signals if s.signal == 'BUY'])
            }
        elif sell_score > buy_score * 1.2:
            return 'SELL', sell_score / total_score, {
                'buy_score': buy_score,
                'sell_score': sell_score,
                'signal_count': len([s for s in self.signals if s.signal == 'SELL'])
            }
        else:
            return 'HOLD', max(buy_score, sell_score) / total_score, {
                'buy_score': buy_score,
                'sell_score': sell_score
            }
