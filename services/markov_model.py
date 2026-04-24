"""
Markov Regime-Switching Model for Trading
Uses Hidden Markov Models (HMM) to detect market regimes and generate trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not installed. Using fallback regime detection.")


class MarkovRegimeModel:
    """
    Markov Regime-Switching Model for detecting market regimes.
    Supports: XAUUSD, BTCUSD, EURUSD, etc.
    """
    
    def __init__(self, symbol: str, n_regimes: int = 3):
        self.symbol = symbol
        self.n_regimes = n_regimes
        self.model = None
        self.hidden_states = None
        self.transition_matrix = None
        self.regime_labels = {}
        self.data = None
        self.regime_stats = {}
        
    def fit(self, prices: pd.Series, returns: Optional[pd.Series] = None) -> Dict:
        """
        Fit HMM model to price data.
        
        Args:
            prices: Price series (pandas Series)
            returns: Optional returns series (if not provided, calculated from prices)
            
        Returns:
            Dictionary with regime information
        """
        if not HMM_AVAILABLE:
            return self._fit_fallback(prices)
        
        df = pd.DataFrame({'price': prices})
        df['returns'] = returns if returns is not None else df['price'].pct_change()
        
        # Calculate features for regime detection
        df['volatility_5d'] = df['returns'].rolling(5).std()
        df['volatility_20d'] = df['returns'].rolling(20).std()
        df['trend'] = (df['price'] / df['price'].rolling(20).mean() - 1) * 100
        df['momentum'] = df['returns'].rolling(10).sum()
        
        # Drop NaN
        df = df.dropna()
        
        if len(df) < 50:
            return self._fit_fallback(prices)
        
        self.data = df
        
        # Features for HMM: returns, short-term vol, trend
        X = df[['returns', 'volatility_5d', 'trend']].values
        
        try:
            self.model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            self.model.fit(X)
            
            self.hidden_states = self.model.predict(X)
            df['regime'] = self.hidden_states
            
            # Calculate transition matrix
            self._calculate_transition_matrix()
            
            # Label regimes
            self._label_regimes(df)
            
            # Get current state
            current_state = int(self.hidden_states[-1])
            current_regime = self.regime_labels[current_state]
            
            # Predict next regime
            next_probs = self.transition_matrix[current_state] if self.transition_matrix is not None else None
            
            return {
                'current_regime': current_regime,
                'current_state': current_state,
                'regime_stats': self.regime_stats,
                'transition_matrix': self.transition_matrix,
                'next_state_probs': next_probs,
                'hidden_states': self.hidden_states,
                'data': df
            }
            
        except Exception as e:
            print(f"HMM fitting error: {e}")
            return self._fit_fallback(prices)
    
    def _fit_fallback(self, prices: pd.Series) -> Dict:
        """Fallback regime detection without HMM library."""
        df = pd.DataFrame({'price': prices})
        df['returns'] = df['price'].pct_change()
        df['volatility_5d'] = df['returns'].rolling(5).std()
        df['trend'] = (df['price'] / df['price'].rolling(20).mean() - 1) * 100
        df = df.dropna()
        
        self.data = df
        
        # Simple volatility-based regime detection
        vol_percentile = pd.qcut(df['volatility_5d'], q=self.n_regimes, labels=False, duplicates='drop')
        # Fill NaN (can occur at quantile boundaries) and clamp to valid range
        vol_percentile = vol_percentile.fillna(0).astype(int)
        # qcut with duplicates='drop' may yield fewer than n_regimes bins; remap to 0..k-1
        unique_bins = sorted(vol_percentile.unique())
        bin_remap = {old: new for new, old in enumerate(unique_bins)}
        vol_percentile = vol_percentile.map(bin_remap)
        self.hidden_states = vol_percentile.values
        actual_n = len(unique_bins)

        # Label regimes
        self.regime_labels = {}
        for i in range(actual_n):
            mask = self.hidden_states == i
            avg_vol = df.loc[mask, 'volatility_5d'].mean() if mask.sum() > 0 else 0
            avg_return = df.loc[mask, 'returns'].mean() if mask.sum() > 0 else 0
            
            if avg_vol < df['volatility_5d'].mean() * 0.7:
                label = 'LOW_VOL'
                desc = 'Calm market, low volatility'
            elif avg_vol > df['volatility_5d'].mean() * 1.3:
                label = 'HIGH_VOL'
                desc = 'Volatile market, high uncertainty'
            else:
                label = 'NORMAL'
                desc = 'Moderate volatility'
            
            self.regime_labels[i] = label
            self.regime_stats[i] = {
                'label': label,
                'description': desc,
                'avg_return': avg_return,
                'avg_volatility': avg_vol,
                'count': int(mask.sum())
            }
        
        current_state = int(self.hidden_states[-1])
        current_regime = self.regime_labels.get(current_state, 'NORMAL')

        # Compute empirical transition matrix from fallback states
        self._calculate_transition_matrix()
        next_probs = self.transition_matrix[current_state] if self.transition_matrix is not None else None

        return {
            'current_regime': current_regime,
            'current_state': current_state,
            'regime_stats': self.regime_stats,
            'transition_matrix': self.transition_matrix,
            'next_state_probs': next_probs,
            'hidden_states': self.hidden_states,
            'data': df
        }
    
    def _calculate_transition_matrix(self):
        """Calculate empirical transition matrix from state sequence."""
        n_states = int(self.hidden_states.max()) + 1  # handles fallback with fewer bins
        trans_matrix = np.zeros((n_states, n_states))
        
        for i in range(len(self.hidden_states) - 1):
            current_state = self.hidden_states[i]
            next_state = self.hidden_states[i + 1]
            trans_matrix[current_state, next_state] += 1
        
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.transition_matrix = trans_matrix / row_sums
    
    def _label_regimes(self, df: pd.DataFrame):
        """Label regimes based on return/volatility characteristics."""
        regime_stats = []
        
        for i in range(self.n_regimes):
            mask = self.hidden_states == i
            avg_return = df.loc[mask, 'returns'].mean()
            avg_vol = df.loc[mask, 'volatility_5d'].mean()
            avg_trend = df.loc[mask, 'trend'].mean()
            
            regime_stats.append({
                'regime': i,
                'return': avg_return,
                'volatility': avg_vol,
                'trend': avg_trend,
                'count': int(mask.sum())
            })
        
        # Sort by volatility
        regime_stats.sort(key=lambda x: x['volatility'])
        
        # Label based on characteristics
        if self.n_regimes == 3:
            self.regime_labels[regime_stats[0]['regime']] = 'LOW_VOL'
            self.regime_labels[regime_stats[1]['regime']] = 'NORMAL'
            self.regime_labels[regime_stats[2]['regime']] = 'HIGH_VOL'
        elif self.n_regimes == 4:
            self.regime_labels[regime_stats[0]['regime']] = 'LOW_VOL'
            self.regime_labels[regime_stats[1]['regime']] = 'NORMAL_BULL'
            self.regime_labels[regime_stats[2]['regime']] = 'NORMAL_BEAR'
            self.regime_labels[regime_stats[3]['regime']] = 'CRISIS'
        else:
            for i, stat in enumerate(regime_stats):
                self.regime_labels[stat['regime']] = f'REGIME_{i}'
        
        # Store stats
        for stat in regime_stats:
            i = stat['regime']
            self.regime_stats[i] = {
                'label': self.regime_labels[i],
                'avg_return': stat['return'],
                'avg_volatility': stat['volatility'],
                'count': stat['count']
            }
    
    def get_regime_color(self, regime_id: int) -> str:
        """Get color for regime."""
        colors = {
            0: '#00ff88',  # Green - Low Vol
            1: '#00d4ff',  # Cyan - Normal
            2: '#ff4757',  # Red - High Vol
            3: '#ffa502',  # Orange - Crisis
        }
        return colors.get(regime_id, '#ffffff')
    
    def get_regime_description(self, regime: str) -> str:
        """Get description for regime."""
        descriptions = {
            'LOW_VOL': 'Calm market with low volatility. Options are cheap. Mean reversion strategies may work.',
            'NORMAL': 'Typical market conditions. Standard trend-following strategies apply.',
            'HIGH_VOL': 'High volatility regime. Risk management crucial. Consider defensive positioning.',
            'CRISIS': 'Extreme volatility. Market stress detected. Consider hedging or reducing exposure.',
            'NORMAL_BULL': 'Bullish market with normal volatility. Momentum strategies favored.',
            'NORMAL_BEAR': 'Bearish pressure with normal volatility. Consider short positions or hedging.',
        }
        return descriptions.get(regime, 'Unknown regime')


def run_markov_analysis(symbol: str, prices: pd.Series, returns: Optional[pd.Series] = None) -> Dict:
    """
    Run complete Markov regime analysis.
    
    Args:
        symbol: Trading symbol (e.g., 'XAUUSD')
        prices: Price series
        returns: Optional returns series
        
    Returns:
        Dictionary with analysis results
    """
    model = MarkovRegimeModel(symbol, n_regimes=3)
    results = model.fit(prices, returns)
    
    return results
