"""
Advanced Technical Indicators Module
=====================================
Professional-grade technical analysis indicators used by quantitative trading firms.
Includes standard indicators and advanced institutional-grade signals.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum


class SignalStrength(Enum):
    """Signal strength classification."""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


@dataclass
class TechnicalSignal:
    """Container for technical analysis signals."""
    indicator: str
    signal: str  # 'BUY', 'SELL', 'NEUTRAL'
    strength: SignalStrength
    value: float
    timestamp: pd.Timestamp
    metadata: Dict = None


class TechnicalIndicators:
    """
    Advanced Technical Indicators Library
    
    Implements institutional-grade technical analysis used by hedge funds
    and proprietary trading firms.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV data.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        self.df = df.copy()
        self.signals: List[TechnicalSignal] = []
        
    # =========================================================================
    # Trend Indicators
    # =========================================================================
    
    def ema(self, period: int, source: str = 'close') -> pd.Series:
        """Exponential Moving Average."""
        return self.df[source].ewm(span=period, adjust=False).mean()
    
    def sma(self, period: int, source: str = 'close') -> pd.Series:
        """Simple Moving Average."""
        return self.df[source].rolling(window=period).mean()
    
    def wma(self, period: int, source: str = 'close') -> pd.Series:
        """Weighted Moving Average."""
        weights = np.arange(1, period + 1)
        return self.df[source].rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    def hull_ma(self, period: int, source: str = 'close') -> pd.Series:
        """
        Hull Moving Average - Reduces lag significantly.
        HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        """
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        wma_half = self.wma(half_period, source)
        wma_full = self.wma(period, source)
        
        raw_hma = 2 * wma_half - wma_full
        return raw_hma.rolling(window=sqrt_period).apply(
            lambda x: np.dot(x, np.arange(1, sqrt_period + 1)) / np.arange(1, sqrt_period + 1).sum(),
            raw=True
        )
    
    def vwap(self) -> pd.Series:
        """Volume Weighted Average Price (intraday)."""
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        return (typical_price * self.df['volume']).cumsum() / self.df['volume'].cumsum()
    
    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence).
        
        Returns
        -------
        Dict with 'macd', 'signal', 'histogram' series
        """
        ema_fast = self.ema(fast)
        ema_slow = self.ema(slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def adx(self, period: int = 14) -> Dict[str, pd.Series]:
        """
        Average Directional Index - Measures trend strength.
        
        Returns
        -------
        Dict with 'adx', '+di', '-di' series
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Smoothed DM
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return {'adx': adx, '+di': plus_di, '-di': minus_di, 'atr': atr}
    
    def supertrend(self, period: int = 10, multiplier: float = 3.0) -> Dict[str, pd.Series]:
        """
        Supertrend Indicator - Trend following based on ATR.
        
        Returns
        -------
        Dict with 'supertrend', 'direction' series
        """
        atr_result = self.adx(period)
        atr = atr_result['atr']
        
        hl2 = (self.df['high'] + self.df['low']) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr
        
        # Initialize
        supertrend = pd.Series(index=self.df.index, dtype=float)
        direction = pd.Series(index=self.df.index, dtype=int)
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(self.df)):
            if direction.iloc[i-1] == 1:
                if self.df['close'].iloc[i] < lower_band.iloc[i]:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = upper_band.iloc[i]
                else:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
            else:
                if self.df['close'].iloc[i] > upper_band.iloc[i]:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = lower_band.iloc[i]
                else:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
        
        return {'supertrend': supertrend, 'direction': direction}
    
    # =========================================================================
    # Momentum Indicators
    # =========================================================================
    
    def rsi(self, period: int = 14, source: str = 'close') -> pd.Series:
        """Relative Strength Index."""
        delta = self.df[source].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def stochastic(self, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = self.df['low'].rolling(window=k_period).min()
        highest_high = self.df['high'].rolling(window=k_period).max()
        
        k = 100 * (self.df['close'] - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        
        return {'k': k, 'd': d}
    
    def williams_r(self, period: int = 14) -> pd.Series:
        """Williams %R."""
        highest_high = self.df['high'].rolling(window=period).max()
        lowest_low = self.df['low'].rolling(window=period).min()
        
        williams_r = -100 * (highest_high - self.df['close']) / (highest_high - lowest_low)
        return williams_r
    
    def roc(self, period: int = 12, source: str = 'close') -> pd.Series:
        """Rate of Change."""
        return ((self.df[source] - self.df[source].shift(period)) / 
                self.df[source].shift(period)) * 100
    
    def cci(self, period: int = 20) -> pd.Series:
        """Commodity Channel Index."""
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_dev = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_dev)
        return cci
    
    def momentum(self, period: int = 10, source: str = 'close') -> pd.Series:
        """Momentum indicator."""
        return self.df[source] - self.df[source].shift(period)
    
    # =========================================================================
    # Volatility Indicators
    # =========================================================================
    
    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Bollinger Bands.
        
        Returns
        -------
        Dict with 'upper', 'middle', 'lower', 'bandwidth', 'percent_b' series
        """
        middle = self.sma(period)
        std = self.df['close'].rolling(window=period).std()
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        bandwidth = (upper - lower) / middle * 100
        percent_b = (self.df['close'] - lower) / (upper - lower)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'bandwidth': bandwidth,
            'percent_b': percent_b
        }
    
    def keltner_channel(self, period: int = 20, multiplier: float = 2.0) -> Dict[str, pd.Series]:
        """Keltner Channel - ATR-based bands."""
        atr_result = self.adx(period)
        atr = atr_result['atr']
        
        middle = self.ema(period)
        upper = middle + multiplier * atr
        lower = middle - multiplier * atr
        
        return {'upper': upper, 'middle': middle, 'lower': lower}
    
    def donchian_channel(self, period: int = 20) -> Dict[str, pd.Series]:
        """Donchian Channel - Highest high and lowest low."""
        upper = self.df['high'].rolling(window=period).max()
        lower = self.df['low'].rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return {'upper': upper, 'middle': middle, 'lower': lower}
    
    def atr(self, period: int = 14) -> pd.Series:
        """Average True Range."""
        return self.adx(period)['atr']
    
    def historical_volatility(self, period: int = 20, annualize: bool = True) -> pd.Series:
        """Historical volatility (annualized by default)."""
        returns = self.df['close'].pct_change()
        hv = returns.rolling(window=period).std()
        if annualize:
            hv = hv * np.sqrt(252)  # Daily data
        return hv
    
    # =========================================================================
    # Volume Indicators
    # =========================================================================
    
    def obv(self) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(self.df['close'].diff())
        direction.iloc[0] = 0
        obv = (direction * self.df['volume']).cumsum()
        return obv
    
    def vwap_deviation(self) -> pd.Series:
        """VWAP deviation percentage."""
        vwap = self.vwap()
        return (self.df['close'] - vwap) / vwap * 100
    
    def mfi(self, period: int = 14) -> pd.Series:
        """Money Flow Index."""
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        raw_money_flow = typical_price * self.df['volume']
        
        delta = typical_price.diff()
        positive_flow = raw_money_flow.where(delta > 0, 0)
        negative_flow = raw_money_flow.where(delta < 0, 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    def accumulation_distribution(self) -> pd.Series:
        """Accumulation/Distribution Line."""
        clv = ((self.df['close'] - self.df['low']) - (self.df['high'] - self.df['close'])) / \
              (self.df['high'] - self.df['low'])
        clv = clv.fillna(0)
        ad = (clv * self.df['volume']).cumsum()
        return ad
    
    def cmf(self, period: int = 20) -> pd.Series:
        """Chaikin Money Flow."""
        mfm = ((self.df['close'] - self.df['low']) - (self.df['high'] - self.df['close'])) / \
              (self.df['high'] - self.df['low'])
        mfm = mfm.fillna(0)
        mfv = mfm * self.df['volume']
        cmf = mfv.rolling(window=period).sum() / self.df['volume'].rolling(window=period).sum()
        return cmf
    
    # =========================================================================
    # Advanced Quantitative Signals
    # =========================================================================
    
    def zscore(self, period: int = 20, source: str = 'close') -> pd.Series:
        """Z-Score of price relative to moving average."""
        sma = self.sma(period, source)
        std = self.df[source].rolling(window=period).std()
        return (self.df[source] - sma) / std
    
    def keltner_breach(self, period: int = 20) -> pd.Series:
        """Keltner Channel breach signal."""
        kc = self.keltner_channel(period)
        breach = pd.Series(0, index=self.df.index)
        breach[self.df['close'] > kc['upper']] = 1
        breach[self.df['close'] < kc['lower']] = -1
        return breach
    
    def squeeze_indicator(self) -> pd.Series:
        """
        Bollinger Band inside Keltner Channel = Squeeze.
        Indicates potential explosive move.
        """
        bb = self.bollinger_bands()
        kc = self.keltner_channel()
        
        squeeze = (bb['lower'] > kc['lower']) & (bb['upper'] < kc['upper'])
        return squeeze.astype(int)
    
    def fisher_transform(self, period: int = 9) -> Dict[str, pd.Series]:
        """
        Fisher Transform - Normalizes prices to Gaussian distribution.
        Better for identifying turning points.
        """
        hl2 = (self.df['high'] + self.df['low']) / 2
        highest = hl2.rolling(window=period).max()
        lowest = hl2.rolling(window=period).min()
        
        # Normalize to 0-1 range
        normalized = (hl2 - lowest) / (highest - lowest + 1e-10)
        normalized = 0.999 * normalized + 0.001  # Clamp to avoid inf
        
        # Fisher transform
        fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
        trigger = fisher.shift(1)
        
        return {'fisher': fisher, 'trigger': trigger}
    
    def ehlers_fisher(self, period: int = 10) -> pd.Series:
        """Ehlers Fisher Transform with adaptive period."""
        # Hilbert Transform - Dominant Cycle Period
        q = (self.df['high'] + self.df['low']) / 2
        
        # Simple adaptive period estimation
        autocorr = q.autocorr(lag=1)
        adaptive_period = max(int(2 * np.pi / np.arccos(autocorr)), period)
        
        return self.fisher_transform(adaptive_period)['fisher']
    
    def hurst_exponent(self, window: int = 100) -> pd.Series:
        """
        Hurst Exponent - Measures mean reversion vs trending.
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        def compute_hurst(x):
            n = len(x)
            if n < 10:
                return 0.5
            
            lags = range(2, min(20, n // 4))
            tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
            
            try:
                reg = np.polyfit(np.log(lags), np.log(tau), 1)
                return reg[0]
            except:
                return 0.5
        
        returns = self.df['close'].pct_change()
        hurst = returns.rolling(window=window).apply(compute_hurst, raw=True)
        return hurst
    
    # =========================================================================
    # Signal Generation
    # =========================================================================
    
    def generate_rsi_signal(self, period: int = 14) -> TechnicalSignal:
        """Generate RSI-based signal."""
        rsi = self.rsi(period).iloc[-1]
        timestamp = self.df['timestamp'].iloc[-1]
        
        if rsi < 30:
            strength = SignalStrength.STRONG if rsi < 20 else SignalStrength.MODERATE
            return TechnicalSignal('RSI', 'BUY', strength, rsi, timestamp)
        elif rsi > 70:
            strength = SignalStrength.STRONG if rsi > 80 else SignalStrength.MODERATE
            return TechnicalSignal('RSI', 'SELL', strength, rsi, timestamp)
        else:
            return TechnicalSignal('RSI', 'NEUTRAL', SignalStrength.WEAK, rsi, timestamp)
    
    def generate_macd_signal(self) -> TechnicalSignal:
        """Generate MACD-based signal."""
        macd_data = self.macd()
        macd = macd_data['macd'].iloc[-1]
        signal_line = macd_data['signal'].iloc[-1]
        histogram = macd_data['histogram'].iloc[-1]
        timestamp = self.df['timestamp'].iloc[-1]
        
        if macd > signal_line and histogram > 0:
            strength = SignalStrength.STRONG if histogram > np.std(macd_data['histogram']) else SignalStrength.MODERATE
            return TechnicalSignal('MACD', 'BUY', strength, macd - signal_line, timestamp)
        elif macd < signal_line and histogram < 0:
            strength = SignalStrength.STRONG if histogram < -np.std(macd_data['histogram']) else SignalStrength.MODERATE
            return TechnicalSignal('MACD', 'SELL', strength, macd - signal_line, timestamp)
        else:
            return TechnicalSignal('MACD', 'NEUTRAL', SignalStrength.WEAK, histogram, timestamp)
    
    def generate_bollinger_signal(self) -> TechnicalSignal:
        """Generate Bollinger Bands-based signal."""
        bb = self.bollinger_bands()
        percent_b = bb['percent_b'].iloc[-1]
        timestamp = self.df['timestamp'].iloc[-1]
        
        if percent_b < 0:
            return TechnicalSignal('Bollinger', 'BUY', SignalStrength.MODERATE, percent_b, timestamp)
        elif percent_b > 1:
            return TechnicalSignal('Bollinger', 'SELL', SignalStrength.MODERATE, percent_b, timestamp)
        else:
            return TechnicalSignal('Bollinger', 'NEUTRAL', SignalStrength.WEAK, percent_b, timestamp)
    
    def generate_supertrend_signal(self) -> TechnicalSignal:
        """Generate Supertrend-based signal."""
        st = self.supertrend()
        direction = st['direction'].iloc[-1]
        timestamp = self.df['timestamp'].iloc[-1]
        
        if direction == 1:
            return TechnicalSignal('Supertrend', 'BUY', SignalStrength.STRONG, 1, timestamp)
        else:
            return TechnicalSignal('Supertrend', 'SELL', SignalStrength.STRONG, -1, timestamp)
    
    def generate_all_signals(self) -> List[TechnicalSignal]:
        """Generate all technical signals."""
        signals = [
            self.generate_rsi_signal(),
            self.generate_macd_signal(),
            self.generate_bollinger_signal(),
            self.generate_supertrend_signal(),
        ]
        self.signals = signals
        return signals
    
    def get_aggregated_signal(self) -> Tuple[str, float]:
        """
        Get aggregated signal from all indicators.
        
        Returns
        -------
        Tuple[str, float]
            (signal: 'BUY'/'SELL'/'HOLD', confidence: 0-1)
        """
        if not self.signals:
            self.generate_all_signals()
        
        buy_score = 0
        sell_score = 0
        total_weight = 0
        
        strength_weights = {
            SignalStrength.VERY_WEAK: 0.2,
            SignalStrength.WEAK: 0.4,
            SignalStrength.MODERATE: 0.6,
            SignalStrength.STRONG: 0.8,
            SignalStrength.VERY_STRONG: 1.0
        }
        
        for signal in self.signals:
            weight = strength_weights[signal.strength]
            total_weight += weight
            
            if signal.signal == 'BUY':
                buy_score += weight
            elif signal.signal == 'SELL':
                sell_score += weight
        
        if total_weight == 0:
            return 'HOLD', 0.0
        
        buy_ratio = buy_score / total_weight
        sell_ratio = sell_score / total_weight
        
        if buy_ratio > 0.6:
            return 'BUY', buy_ratio
        elif sell_ratio > 0.6:
            return 'SELL', sell_ratio
        else:
            return 'HOLD', max(buy_ratio, sell_ratio)
