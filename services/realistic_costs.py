"""
Realistic Variable Cost Model for Gold Backtesting.

Problem: Fixed $0.30 spread + $0.10 slippage is 2-5x optimistic for retail
1-minute gold trading. EXNESS Raw spreads widen in Asian session and news.

Verified:
- London/NY overlap: 0.3-0.6 pips ($0.30-$0.60)
- Asian session: $0.50-$2.00
- NFP/FOMC: $2.00-$5.00+
- Commission: $3-7 per lot round-trip (ECN brokers)
- Slippage: 3-10 pips normal, higher during news

Reference: EXNESS/Pepperstone/Fusion Markets spread data (2026).
"""
import numpy as np
import pandas as pd
from typing import Dict


class RealisticCostModel:
    """
    Variable cost model for gold backtesting.
    Costs vary by:
      - Session (Asian = wider, London/NY = tighter)
      - Volatility regime (high vol = wider)
      - Trade size (larger = more slippage)
    """

    # Session-specific base spreads in USD for XAUUSD (1 pip = $0.01 for 1 oz)
    # For a standard lot (100 oz), 1 pip = $1.00. We model per-oz cost.
    SESSION_SPREADS = {
        "asian": 1.50,      # 00:00 - 09:00 UTC
        "london": 0.30,     # 08:00 - 17:00 UTC
        "ny": 0.35,         # 13:00 - 22:00 UTC
        "overlap": 0.25,    # 13:00 - 17:00 UTC
        "closed": 5.00,     # Weekend
    }

    # Volatility multipliers on spread
    VOL_MULTIPLIERS = {
        "low": 1.0,
        "normal": 1.5,
        "high": 3.0,
        "crisis": 5.0,
    }

    def __init__(self, commission_per_lot: float = 7.0):
        """
        commission_per_lot: Round-trip commission in USD per standard lot (100 oz).
                          Pepperstone Razor: ~$7/lot RT. Fusion Markets: ~$4.50.
        """
        self.commission_per_lot = commission_per_lot

    @staticmethod
    def get_session(timestamp: pd.Timestamp) -> str:
        """Determine trading session."""
        hour = timestamp.hour
        weekday = timestamp.dayofweek
        if weekday >= 5:
            return "closed"
        if 13 <= hour < 17:
            return "overlap"
        if 8 <= hour < 17:
            return "london"
        if 13 <= hour < 22:
            return "ny"
        return "asian"

    def compute_spread(
        self,
        timestamp: pd.Timestamp,
        vol_regime: str = "normal",
    ) -> float:
        """Compute realistic spread in USD per ounce for a given bar."""
        session = self.get_session(timestamp)
        base_spread = self.SESSION_SPREADS.get(session, 0.50)
        vol_mult = self.VOL_MULTIPLIERS.get(vol_regime, 1.5)
        return base_spread * vol_mult

    def compute_slippage(
        self,
        bar: pd.Series,
        direction: str,
        size_oz: float,
        is_market_order: bool = True,
    ) -> float:
        """
        Compute slippage in USD per ounce based on bar range and trade size.
        Slippage = (high - low) * size_factor * market_order_factor
        """
        bar_range = bar.get("high", bar["close"]) - bar.get("low", bar["close"])
        bar_range = max(bar_range, 0.01)  # minimum sensible range
        # Size factor: larger trades get more slippage
        # Normalize to standard lot (100 oz), square-root scaling
        size_factor = min(1.0, (size_oz / 100.0) ** 0.5)
        # Market orders get more slippage than limit orders
        order_factor = 0.3 if is_market_order else 0.1
        return bar_range * size_factor * order_factor

    def compute_total_cost(
        self,
        entry_bar: pd.Series,
        exit_bar: pd.Series,
        direction: str,
        size_oz: float,
        vol_regime: str = "normal",
    ) -> Dict[str, float]:
        """
        Compute all trading costs for a round-trip in USD total.
        Returns per-trade totals, not per-ounce.
        """
        entry_spread = self.compute_spread(
            pd.Timestamp(entry_bar.name) if hasattr(entry_bar, "name") else pd.Timestamp.now(),
            vol_regime,
        )
        exit_spread = self.compute_spread(
            pd.Timestamp(exit_bar.name) if hasattr(exit_bar, "name") else pd.Timestamp.now(),
            vol_regime,
        )
        entry_slippage = self.compute_slippage(entry_bar, direction, size_oz)
        exit_slippage = self.compute_slippage(
            exit_bar, "sell" if direction == "buy" else "buy", size_oz
        )

        # Spread cost is half paid on entry, half on exit (per convention)
        # Here we model full spread on each leg for conservatism
        entry_spread_cost = entry_spread * size_oz * 0.01  # pips -> USD (1 pip = $0.01/oz)
        exit_spread_cost = exit_spread * size_oz * 0.01

        # Commission
        lots = size_oz / 100.0
        commission = self.commission_per_lot * lots

        total_cost = (
            entry_spread_cost
            + exit_spread_cost
            + entry_slippage * size_oz
            + exit_slippage * size_oz
            + commission
        )

        return {
            "entry_spread": entry_spread_cost,
            "exit_spread": exit_spread_cost,
            "entry_slippage": entry_slippage * size_oz,
            "exit_slippage": exit_slippage * size_oz,
            "commission": commission,
            "total_cost": total_cost,
            "cost_per_oz": total_cost / size_oz if size_oz > 0 else 0,
        }

    def estimate_cost_for_signal(
        self,
        timestamp: pd.Timestamp,
        size_oz: float = 1.0,
        vol_regime: str = "normal",
        is_market_order: bool = True,
    ) -> float:
        """Quick estimate of round-trip cost without exit bar."""
        spread = self.compute_spread(timestamp, vol_regime)
        lots = size_oz / 100.0
        commission = self.commission_per_lot * lots
        spread_cost = spread * size_oz * 0.01 * 2  # entry + exit
        # Assume slippage = spread * 0.3 for quick estimate
        slippage_cost = spread * 0.3 * size_oz * 0.01 * 2
        return spread_cost + slippage_cost + commission


def detect_volatility_regime(atr: float, atr_ma: float) -> str:
    """
    Simple regime classifier based on ATR vs its moving average.
    """
    if atr_ma <= 0:
        return "normal"
    ratio = atr / atr_ma
    if ratio > 2.0:
        return "crisis"
    if ratio > 1.5:
        return "high"
    if ratio < 0.7:
        return "low"
    return "normal"
