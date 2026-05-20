"""
Regime-Conditional Trading Strategy
====================================
Adapts position sizing, stops, and signals based on detected regime.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd


@dataclass
class TradingConfig:
    """Configuration for a specific regime."""
    primary_signal: str
    secondary_signal: str
    position_multiplier: float
    max_position: float
    min_position: float
    stop_atr_multiple: float
    takeprofit_atr_multiple: float
    trailing_stop: bool
    require_confirmation: bool
    max_trades_per_day: int
    hold_time_target: str
    allow_new_entries: bool
    reduce_on_drawdown: float
    time_filter: Optional[str]


class RegimeStrategy:
    """
    Trading strategy that adapts completely based on detected regime.
    """

    REGIME_CONFIGS = {
        'STRONG_TREND_UP': TradingConfig(
            primary_signal='momentum_model',
            secondary_signal='trend_following',
            position_multiplier=1.5,
            max_position=0.05,
            min_position=0.005,
            stop_atr_multiple=2.5,
            takeprofit_atr_multiple=5.0,
            trailing_stop=True,
            require_confirmation=False,
            max_trades_per_day=5,
            hold_time_target='hours_to_days',
            allow_new_entries=True,
            reduce_on_drawdown=0.10,
            time_filter=None,
        ),
        'STRONG_TREND_DOWN': TradingConfig(
            primary_signal='momentum_model',
            secondary_signal='trend_following',
            position_multiplier=1.3,
            max_position=0.04,
            min_position=0.005,
            stop_atr_multiple=2.5,
            takeprofit_atr_multiple=4.5,
            trailing_stop=True,
            require_confirmation=False,
            max_trades_per_day=4,
            hold_time_target='hours_to_days',
            allow_new_entries=True,
            reduce_on_drawdown=0.10,
            time_filter=None,
        ),
        'GRIND_UP': TradingConfig(
            primary_signal='momentum_model',
            secondary_signal='mean_reversion_light',
            position_multiplier=1.0,
            max_position=0.03,
            min_position=0.005,
            stop_atr_multiple=2.0,
            takeprofit_atr_multiple=3.0,
            trailing_stop=False,
            require_confirmation=True,
            max_trades_per_day=3,
            hold_time_target='hours',
            allow_new_entries=True,
            reduce_on_drawdown=0.08,
            time_filter=None,
        ),
        'CHOPPY_RANGE': TradingConfig(
            primary_signal='mean_reversion_model',
            secondary_signal='range_bound',
            position_multiplier=0.6,
            max_position=0.02,
            min_position=0.003,
            stop_atr_multiple=1.0,
            takeprofit_atr_multiple=1.5,
            trailing_stop=False,
            require_confirmation=True,
            max_trades_per_day=8,
            hold_time_target='minutes_to_hours',
            allow_new_entries=True,
            reduce_on_drawdown=0.05,
            time_filter=None,
        ),
        'HIGH_VOL_CHAOS': TradingConfig(
            primary_signal='none',
            secondary_signal='volatility_breakout',
            position_multiplier=0.25,
            max_position=0.01,
            min_position=0.001,
            stop_atr_multiple=3.5,
            takeprofit_atr_multiple=2.0,
            trailing_stop=False,
            require_confirmation=True,
            max_trades_per_day=2,
            hold_time_target='minutes',
            allow_new_entries=True,
            reduce_on_drawdown=0.03,
            time_filter='avoid_news',
        ),
        'LOW_VOL_DRIFT': TradingConfig(
            primary_signal='none',
            secondary_signal='none',
            position_multiplier=0.0,
            max_position=0.0,
            min_position=0.0,
            stop_atr_multiple=1.0,
            takeprofit_atr_multiple=1.0,
            trailing_stop=False,
            require_confirmation=True,
            max_trades_per_day=0,
            hold_time_target='none',
            allow_new_entries=False,
            reduce_on_drawdown=0.0,
            time_filter=None,
        ),
    }

    def __init__(self):
        self.current_config = None
        self.current_regime = None
        self.trades_today = 0
        self.last_day = None

    def get_config(self, regime: str) -> TradingConfig:
        """Get trading configuration for a regime."""
        return self.REGIME_CONFIGS.get(regime, self.REGIME_CONFIGS['CHOPPY_RANGE'])

    def evaluate_signal(self, regime: str, signal_confidence: float,
                        current_drawdown: float) -> Dict:
        """
        Evaluate whether a signal should be traded under current regime.
        Returns dict with allow_trade, position_multiplier, reason.
        """
        config = self.get_config(regime)

        # 1. Check if regime allows new entries
        if not config.allow_new_entries:
            return {
                'allow_trade': False,
                'position_multiplier': 0.0,
                'reason': f'Regime {regime}: trading suspended',
            }

        # 2. Check drawdown limit
        if current_drawdown > config.reduce_on_drawdown:
            return {
                'allow_trade': False,
                'position_multiplier': 0.0,
                'reason': f'Drawdown {current_drawdown:.2%} exceeds limit',
            }

        # 3. Check daily trade limit
        today = pd.Timestamp.now().date()
        if today != self.last_day:
            self.trades_today = 0
            self.last_day = today
        if self.trades_today >= config.max_trades_per_day:
            return {
                'allow_trade': False,
                'position_multiplier': 0.0,
                'reason': f'Daily trade limit reached ({config.max_trades_per_day})',
            }

        # 4. Approve trade with regime-adjusted sizing
        adjusted_size = config.position_multiplier
        if current_drawdown > 0:
            drawdown_ratio = current_drawdown / config.reduce_on_drawdown if config.reduce_on_drawdown > 0 else 0
            if drawdown_ratio > 0.5:
                adjusted_size *= (1 - drawdown_ratio * 0.5)

        return {
            'allow_trade': True,
            'position_multiplier': max(adjusted_size, config.min_position),
            'reason': f'Regime {regime}: approved at {adjusted_size:.2f}x sizing',
            'config': config,
        }

    def compute_stops(self, entry_price: float, atr: float,
                      direction: str, regime: str) -> Dict:
        """Compute regime-adjusted stop loss and take profit."""
        config = self.get_config(regime)
        if direction == 'buy':
            stop_loss = entry_price - config.stop_atr_multiple * atr
            take_profit = entry_price + config.takeprofit_atr_multiple * atr
        else:
            stop_loss = entry_price + config.stop_atr_multiple * atr
            take_profit = entry_price - config.takeprofit_atr_multiple * atr
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'stop_atr': config.stop_atr_multiple,
            'tp_atr': config.takeprofit_atr_multiple,
            'risk_reward': config.takeprofit_atr_multiple / config.stop_atr_multiple,
        }

    def on_trade_executed(self):
        """Track trade count."""
        self.trades_today += 1
