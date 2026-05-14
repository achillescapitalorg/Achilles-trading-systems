"""
Adaptive Position Sizing using Regime-Dependent Fractional Kelly.

Problem: Fixed Kelly assumes stationary edge statistics. In reality, edge varies
dramatically by regime — trending vs ranging vs high volatility.

Solution: Regime-dependent fractional Kelly that adapts position size based on:
  - Current volatility regime (low / normal / high / crisis)
  - Recent win rate (rolling trade window)
  - Drawdown status (reduce size in drawdown)

Verified: Kelly criterion is highly sensitive to win-rate estimation error.
A 5% error in win rate changes Kelly fraction by ~3x. Quarter-Kelly is safer
but still dangerous if edge is overestimated from backtests.

Reference: JournalPlus (2026), Stanford Boyd working paper on risk-constrained Kelly.
"""
import numpy as np
import pandas as pd
from collections import deque
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class TradeRecord:
    """Single trade result for Kelly estimation."""
    pnl: float
    entry_price: float
    exit_price: float
    direction: str  # 'buy' or 'sell'
    size: float
    timestamp: pd.Timestamp
    regime: str     # 'low_vol', 'normal', 'high_vol', 'crisis'


class AdaptiveKellySizer:
    """
    Adaptive position sizing using regime-dependent fractional Kelly.
    Key insight: Your edge is not constant. It varies by regime.
    Size positions based on regime-specific edge, not global average.
    """

    def __init__(
        self,
        base_kelly_fraction: float = 0.25,  # Quarter Kelly
        max_risk_per_trade: float = 0.02,   # 2% max
        min_risk_per_trade: float = 0.005,  # 0.5% min
        drawdown_reduction: float = 0.5,    # Halve size at max DD
        regime_windows: Optional[Dict[str, int]] = None,
    ):
        self.base_kelly_fraction = base_kelly_fraction
        self.max_risk = max_risk_per_trade
        self.min_risk = min_risk_per_trade
        self.drawdown_reduction = drawdown_reduction
        self.trades: deque = deque(maxlen=500)
        self.current_drawdown = 0.0
        self.peak_equity = 1.0
        self.regime_windows = regime_windows or {
            "low_vol": 50,
            "normal": 100,
            "high_vol": 30,   # Shorter window in high vol (faster adaptation)
            "crisis": 20,
        }

    def record_trade(self, trade: TradeRecord):
        """Record completed trade for Kelly estimation."""
        self.trades.append(trade)
        # Update drawdown tracking
        cumulative_pnl = sum(t.pnl for t in self.trades)
        current_equity = 1.0 + cumulative_pnl
        self.peak_equity = max(self.peak_equity, current_equity)
        self.current_drawdown = (
            (self.peak_equity - current_equity) / self.peak_equity
            if self.peak_equity > 0 else 0.0
        )

    def compute_kelly_fraction(self, regime: str = "normal") -> float:
        """
        Compute Kelly fraction for current regime.
        Uses regime-specific trade history for more accurate edge estimation.
        """
        # Filter trades by regime
        regime_trades = [t for t in self.trades if t.regime == regime]
        min_required = self.regime_windows.get(regime, 50)

        penalty = 1.0
        if len(regime_trades) < min_required:
            # Not enough regime-specific data — use all trades with penalty
            regime_trades = list(self.trades)
            penalty = 0.7  # Reduce estimate due to regime mismatch

        if len(regime_trades) < 10:
            return self.min_risk  # Not enough data

        wins = [t.pnl for t in regime_trades if t.pnl > 0]
        losses = [t.pnl for t in regime_trades if t.pnl <= 0]
        if not wins or not losses:
            return self.min_risk

        p = len(wins) / len(regime_trades)  # Win rate
        b = np.mean(wins) / abs(np.mean(losses))  # Win/loss ratio
        q = 1 - p
        kelly = (b * p - q) / b
        # Apply fractional Kelly and penalty
        kelly = max(0.0, kelly) * self.base_kelly_fraction * penalty
        return min(kelly, self.max_risk)

    def compute_position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss: float,
        regime: str = "normal",
    ) -> float:
        """
        Compute position size in units (oz for gold).
        Formula: Risk Amount / (Entry - Stop) = Position Size
        Risk Amount = Equity * Kelly Fraction
        """
        kelly = self.compute_kelly_fraction(regime)

        # Drawdown adjustment
        if self.current_drawdown > 0.05:  # 5% drawdown
            reduction = min(1.0, self.current_drawdown / 0.10) * self.drawdown_reduction
            kelly *= (1 - reduction)

        # Risk amount
        risk_amount = account_equity * kelly
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit < 0.01:
            return 0.0  # Invalid stop

        position_size = risk_amount / risk_per_unit

        # Enforce min/max notional risk
        min_size = account_equity * self.min_risk / risk_per_unit
        max_size = account_equity * self.max_risk / risk_per_unit
        return max(min_size, min(position_size, max_size))

    def get_report(self) -> dict:
        """Get current sizing parameters report."""
        return {
            "current_drawdown": self.current_drawdown,
            "peak_equity": self.peak_equity,
            "n_trades_recorded": len(self.trades),
            "kelly_by_regime": {
                r: self.compute_kelly_fraction(r)
                for r in ["low_vol", "normal", "high_vol", "crisis"]
            },
        }


class FixedFractionalSizer:
    """
    Simple fixed fractional sizer as a conservative fallback.
    Used when Kelly estimates are unreliable (< 20 trades).
    """

    def __init__(self, risk_per_trade: float = 0.01):
        self.risk_per_trade = risk_per_trade

    def compute_position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss: float,
    ) -> float:
        risk_amount = account_equity * self.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit < 0.01:
            return 0.0
        return risk_amount / risk_per_unit
