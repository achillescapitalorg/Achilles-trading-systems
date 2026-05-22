"""
risk_manager.py
===============
Production risk management module for VibeTrading Gold System.

Implements:
- Position sizing (Kelly/4, regime-adjusted)
- Dynamic stop loss (ATR-based + microstructure reversal)
- Daily loss limits & consecutive loss cooldowns
- Regime-dependent risk multipliers
- Real-time P&L tracking and drawdown monitoring

All calculations assume account balance in USD and XAU/USD pip values.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timedelta
import json


@dataclass
class Trade:
    """Represents a single trade for risk tracking."""
    entry_time: datetime
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    size_lots: float
    stop_loss: float
    take_profit: float
    regime: str
    confidence: float
    ofi_proxy: float
    vpin_proxy: float

    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None  # 'TP', 'SL', 'MICRO_REVERSE', 'TIMEOUT', 'MANUAL', 'TRAILING_STOP'

    # Trailing stop state
    trailing_stop_active: bool = False
    trailing_stop_price: Optional[float] = None
    highest_price_seen: Optional[float] = None
    lowest_price_seen: Optional[float] = None


class RiskManager:
    """
    Central risk management engine.

    Rules enforced:
    1. Max 0.5% risk per trade (configurable)
    2. Daily max drawdown 2% -> trading halt
    3. Weighted consecutive losses -> dynamic sizing + short cooldown
    4. Regime-based position sizing multiplier
    5. Dynamic stop: tighter in HIGH_VOL_CHAOS, wider in STRONG_TREND
    6. Microstructure-based emergency exit (OFI reversal)
    """

    # Risk Parameters
    RISK_PER_TRADE_PCT: float = 0.005        # 0.5% of account per trade
    DAILY_MAX_DRAWDOWN_PCT: float = 0.02     # 2% daily loss limit
    CONSECUTIVE_LOSS_LIMIT: float = 3.0      # Cooldown threshold (weighted)
    COOLDOWN_MINUTES_BASE: int = 30          # Base cooldown (regime-adjusted)

    # Loss type weights for consecutive loss scoring
    LOSS_WEIGHTS = {
        'SL': 1.0,
        'TP': 0.0,               # Wins don't count
        'TRAILING_STOP': 0.5,
        'MICRO_REVERSE': 0.3,    # Microstructure noise — low weight
        'TOXIC_FLOW': 0.3,
        'TIMEOUT': 0.3,
        '15M_FLIP': 0.5,
        'MANUAL': 1.0,
    }

    # Regime-aware cooldown (minutes)
    COOLDOWN_BY_REGIME = {
        'STRONG_TREND_UP': 15,
        'STRONG_TREND_DOWN': 15,
        'GRIND_UP': 30,
        'GRIND_DOWN': 30,
        'CHOPPY': 60,
        'HIGH_VOL_CHAOS': 60,
        'LOW_VOL_DRIFT': 60,
        'UNKNOWN': 30,
    }

    # Dynamic sizing tiers based on weighted loss score
    SIZE_TIERS = [
        (0.0, 1.00),   # < 1.0: full size
        (1.0, 0.50),   # 1.0-2.0: half size
        (2.0, 0.25),   # 2.0-3.0: quarter size
        (3.0, 0.00),   # >= 3.0: blocked
    ]

    # ATR Multipliers for Stop/Target
    SL_ATR_MULTIPLIER_DEFAULT: float = 1.5
    SL_ATR_MULTIPLIER_CHAOS: float = 2.0     # Wider in chaos
    SL_ATR_MULTIPLIER_TREND: float = 1.2     # Tighter in strong trend

    # ASYMMETRIC: TP multipliers vary by regime
    TP_ATR_MULTIPLIER: float = 2.5           # Legacy fallback
    TP_ATR_MULTIPLIERS = {
        'STRONG_TREND_UP': 6.0,      # Very wide — trailing stop is primary exit
        'STRONG_TREND_DOWN': 6.0,
        'GRIND_UP': 2.5,
        'GRIND_DOWN': 2.5,
        'CHOPPY': 1.5,               # Quick profits in chop
        'HIGH_VOL_CHAOS': 1.2,       # Very tight — chaos reverses fast
        'LOW_VOL_DRIFT': 2.0,
        'UNKNOWN': 2.0
    }

    # Trailing stop settings (only in strong trends)
    TRAILING_ACTIVATION_ATR: float = 1.5   # Activate after 1.5×ATR profit
    TRAILING_DISTANCE_ATR: float = 1.0     # Trail 1.0×ATR behind price

    # Regime Position Size Multipliers
    REGIME_MULTIPLIERS = {
        'STRONG_TREND_UP': 1.0,
        'STRONG_TREND_DOWN': 1.0,
        'GRIND_UP': 0.75,
        'GRIND_DOWN': 0.75,
        'CHOPPY': 0.5,
        'HIGH_VOL_CHAOS': 0.25,
        'LOW_VOL_DRIFT': 0.25,
        'UNKNOWN': 0.0
    }

    # Gold-specific constants
    PIP_VALUE_PER_LOT: float = 10.0          # $10 per pip per standard lot
    TICK_SIZE: float = 0.01                  # 0.01 = 1 pip for XAU/USD

    def __init__(self, account_balance: float = 10000.0):
        self.account_balance = account_balance
        self.initial_balance_today = account_balance
        self.current_balance = account_balance
        self.peak_balance = account_balance

        # Trade history
        self.trades: List[Trade] = []
        self.open_trade: Optional[Trade] = None

        # State tracking
        self.daily_pnl: float = 0.0
        self.consecutive_losses: float = 0.0   # Weighted score (not integer count)
        self.cooldown_until: Optional[datetime] = None
        self.trading_halted: bool = False
        self.halt_reason: Optional[str] = None

        # Performance metrics
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.total_pnl: float = 0.0
        self.max_drawdown_pct: float = 0.0

    # =====================================================================
    # POSITION SIZING
    # =====================================================================
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        regime: str,
        confidence: float,
        microstructure_quality: float = 0.5
    ) -> float:
        """
        Calculate position size in standard lots.

        Formula:
            Risk Amount = Account * RISK_PER_TRADE_PCT
            Risk Pips = |Entry - Stop| / TICK_SIZE
            Pip Value = Risk Amount / Risk Pips
            Lots = Pip Value / PIP_VALUE_PER_LOT

        Then apply:
            - Regime multiplier
            - Confidence scaling (linear 0.65-0.85 -> 0.5x-1.0x)
            - Microstructure quality multiplier
        """
        # Base risk amount
        risk_amount = self.current_balance * self.RISK_PER_TRADE_PCT

        # Risk in pips
        risk_pips = abs(entry_price - stop_loss) / self.TICK_SIZE
        if risk_pips < 1:
            risk_pips = 1  # Minimum 1 pip risk

        # Base lot size
        pip_value_needed = risk_amount / risk_pips
        base_lots = pip_value_needed / self.PIP_VALUE_PER_LOT

        # Apply regime multiplier
        regime_mult = self.REGIME_MULTIPLIERS.get(regime, 0.25)

        # Confidence scaling: conf 0.65 -> 0.5x, conf 0.85 -> 1.0x
        conf_scale = np.clip((confidence - 0.65) / 0.20, 0.5, 1.0)

        # Microstructure quality (0-1, from filter alignment)
        micro_scale = 0.5 + (microstructure_quality * 0.5)

        final_lots = base_lots * regime_mult * conf_scale * micro_scale

        # Dynamic sizing based on weighted consecutive loss score
        loss_score = self.consecutive_losses
        size_mult = 1.0
        for threshold, mult in self.SIZE_TIERS:
            if loss_score >= threshold:
                size_mult = mult
            else:
                break
        final_lots *= size_mult

        # Hard limits
        final_lots = min(final_lots, 5.0)   # Max 5 lots
        final_lots = max(final_lots, 0.01)  # Min 0.01 lots

        return round(final_lots, 2)

    # =====================================================================
    # STOP LOSS & TAKE PROFIT CALCULATION
    # =====================================================================
    def calculate_stops(
        self,
        entry_price: float,
        direction: str,
        atr_14: float,
        regime: str,
        ofi_proxy: Optional[float] = None,
        bias_strength: float = 0.5
    ) -> Tuple[float, float]:
        """
        Calculate dynamic stop loss and take profit.

        ASYMMETRIC: TP varies by regime and bias strength.
        Stronger bias / stronger trend = wider TP (let winners run).
        Choppy / chaotic = tighter TP (take quick profits).

        Returns:
            (stop_loss_price, take_profit_price)
        """
        # ATR multiplier for SL based on regime
        if regime == 'HIGH_VOL_CHAOS':
            sl_mult = self.SL_ATR_MULTIPLIER_CHAOS
        elif regime in ['STRONG_TREND_UP', 'STRONG_TREND_DOWN']:
            sl_mult = self.SL_ATR_MULTIPLIER_TREND
        else:
            sl_mult = self.SL_ATR_MULTIPLIER_DEFAULT

        sl_distance = atr_14 * sl_mult

        # ASYMMETRIC TP: regime-dependent base
        tp_mult = self.TP_ATR_MULTIPLIERS.get(regime, self.TP_ATR_MULTIPLIER)
        tp_distance = atr_14 * tp_mult

        # Scale TP with bias strength (0.5 → 0.7 maps to 0.9× → 1.15×)
        # Stronger conviction = let winners run further
        bias_scale = 0.7 + (bias_strength * 0.65)  # 0.5→1.025, 0.7→1.155
        tp_distance *= bias_scale

        # Microstructure adjustment: if OFI strongly reverses, tighten stop
        if ofi_proxy is not None:
            ofi_strength = abs(ofi_proxy)
            if ofi_strength > 2.0:  # Strong microstructure pressure
                sl_distance *= 0.8  # Tighten by 20%

        if direction == 'BUY':
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # SELL
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        return round(stop_loss, 2), round(take_profit, 2)

    # =====================================================================
    # TRADE EXECUTION & MONITORING
    # =====================================================================
    def can_trade(self, current_time: datetime) -> Tuple[bool, str]:
        """
        Check if trading is allowed at this time.

        Returns:
            (allowed: bool, reason: str)
        """
        # Check daily reset
        if hasattr(self, '_last_check_date'):
            if current_time.date() != self._last_check_date:
                self._reset_daily(current_time)
        self._last_check_date = current_time.date()

        # 1. Already open trade?
        if self.open_trade is not None:
            return False, "Trade already open"

        # 2. Trading halted (daily limit hit)?
        if self.trading_halted:
            return False, f"Trading halted: {self.halt_reason}"

        # 3. Weighted loss score exceeded?
        if self.consecutive_losses >= self.CONSECUTIVE_LOSS_LIMIT:
            if self.cooldown_until is None or current_time >= self.cooldown_until:
                # Auto-trigger cooldown if not already set
                self.cooldown_until = current_time + timedelta(minutes=self.COOLDOWN_MINUTES_BASE)
            remaining = (self.cooldown_until - current_time).total_seconds() / 60
            return False, f"Risk pause: {remaining:.0f}min remaining (score {self.consecutive_losses:.1f})"

        # 3b. Cooldown active (from previous block)?
        if self.cooldown_until is not None and current_time < self.cooldown_until:
            remaining = (self.cooldown_until - current_time).total_seconds() / 60
            return False, f"Risk pause: {remaining:.0f}min remaining (score {self.consecutive_losses:.1f})"

        # 4. Daily drawdown limit?
        daily_dd = (self.initial_balance_today - self.current_balance) / self.initial_balance_today
        if daily_dd >= self.DAILY_MAX_DRAWDOWN_PCT:
            self.trading_halted = True
            self.halt_reason = f"Daily drawdown limit hit: {daily_dd:.2%}"
            return False, self.halt_reason

        return True, "OK"

    def open_position(
        self,
        current_time: datetime,
        direction: str,
        entry_price: float,
        atr_14: float,
        regime: str,
        confidence: float,
        ofi_proxy: float,
        vpin_proxy: float,
        microstructure_quality: float = 0.5
    ) -> Optional[Trade]:
        """
        Attempt to open a new position with full risk management.

        Returns:
            Trade object if opened, None if rejected
        """
        allowed, reason = self.can_trade(current_time)
        if not allowed:
            print(f"[RiskManager] Trade rejected: {reason}")
            return None

        # Calculate stops (asymmetric, regime-dependent)
        stop_loss, take_profit = self.calculate_stops(
            entry_price, direction, atr_14, regime, ofi_proxy, confidence
        )

        # Calculate size
        size = self.calculate_position_size(
            entry_price, stop_loss, regime, confidence, microstructure_quality
        )

        if size < 0.01:
            print(f"[RiskManager] Trade rejected: Size too small ({size})")
            return None

        trade = Trade(
            entry_time=current_time,
            direction=direction,
            entry_price=entry_price,
            size_lots=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            regime=regime,
            confidence=confidence,
            ofi_proxy=ofi_proxy,
            vpin_proxy=vpin_proxy
        )

        self.open_trade = trade
        print(f"[RiskManager] OPEN {direction} | {size} lots @ {entry_price} | "
              f"SL:{stop_loss} TP:{take_profit} | Regime:{regime} | Conf:{confidence:.2f}")

        return trade

    def _update_trailing_stop(self, trade: Trade, current_price: float, atr_14: float):
        """
        Update trailing stop for open trade.
        Only activates in strong trends after price moves favorably.
        """
        if trade.regime not in ['STRONG_TREND_UP', 'STRONG_TREND_DOWN']:
            return

        # Track highest/lowest price seen
        if trade.highest_price_seen is None:
            trade.highest_price_seen = current_price
            trade.lowest_price_seen = current_price
        else:
            trade.highest_price_seen = max(trade.highest_price_seen, current_price)
            trade.lowest_price_seen = min(trade.lowest_price_seen, current_price)

        # Activation threshold: price moved 1.5×ATR in our favor
        if trade.direction == 'BUY':
            profit_distance = trade.highest_price_seen - trade.entry_price
            if profit_distance >= self.TRAILING_ACTIVATION_ATR * atr_14:
                trade.trailing_stop_active = True
                new_trail = trade.highest_price_seen - (self.TRAILING_DISTANCE_ATR * atr_14)
                if trade.trailing_stop_price is None or new_trail > trade.trailing_stop_price:
                    trade.trailing_stop_price = new_trail
        else:  # SELL
            profit_distance = trade.entry_price - trade.lowest_price_seen
            if profit_distance >= self.TRAILING_ACTIVATION_ATR * atr_14:
                trade.trailing_stop_active = True
                new_trail = trade.lowest_price_seen + (self.TRAILING_DISTANCE_ATR * atr_14)
                if trade.trailing_stop_price is None or new_trail < trade.trailing_stop_price:
                    trade.trailing_stop_price = new_trail

    def check_exit_conditions(
        self,
        current_time: datetime,
        current_price: float,
        current_ofi: float,
        current_vpin: float,
        atr_14: float = 1.0
    ) -> Optional[str]:
        """
        Check if open trade should be exited.

        Returns:
            Exit reason string if exit triggered, None otherwise
        """
        if self.open_trade is None:
            return None

        trade = self.open_trade

        # Update trailing stop first
        self._update_trailing_stop(trade, current_price, atr_14)

        # 1. Stop Loss hit (static)
        if trade.direction == 'BUY' and current_price <= trade.stop_loss:
            return 'SL'
        if trade.direction == 'SELL' and current_price >= trade.stop_loss:
            return 'SL'

        # 1b. Trailing Stop hit
        if trade.trailing_stop_active and trade.trailing_stop_price is not None:
            if trade.direction == 'BUY' and current_price <= trade.trailing_stop_price:
                return 'TRAILING_STOP'
            if trade.direction == 'SELL' and current_price >= trade.trailing_stop_price:
                return 'TRAILING_STOP'

        # 2. Take Profit hit
        if trade.direction == 'BUY' and current_price >= trade.take_profit:
            return 'TP'
        if trade.direction == 'SELL' and current_price <= trade.take_profit:
            return 'TP'

        # 3. Microstructure reversal (OFI flips against position)
        if trade.direction == 'BUY' and current_ofi < -1.5:
            return 'MICRO_REVERSE'
        if trade.direction == 'SELL' and current_ofi > 1.5:
            return 'MICRO_REVERSE'

        # 4. VPIN spike (flow became toxic, adverse selection likely)
        if current_vpin > 0.8:
            return 'TOXIC_FLOW'

        # 5. Time-based exit (max hold 2 hours for 1m system)
        hold_time = current_time - trade.entry_time
        if hold_time > timedelta(hours=2):
            return 'TIMEOUT'

        return None

    def close_position(
        self,
        current_time: datetime,
        current_price: float,
        exit_reason: str
    ) -> float:
        """
        Close open position and update risk state.

        Returns:
            P&L in USD
        """
        if self.open_trade is None:
            return 0.0

        trade = self.open_trade
        trade.exit_time = current_time
        trade.exit_price = current_price
        trade.exit_reason = exit_reason

        # Calculate P&L
        if trade.direction == 'BUY':
            pnl = (current_price - trade.entry_price) * trade.size_lots * 100  # 100 oz per lot
        else:
            pnl = (trade.entry_price - current_price) * trade.size_lots * 100

        # Subtract estimated spread cost (0.3 pips = $3 per lot)
        spread_cost = trade.size_lots * 3.0
        pnl -= spread_cost

        trade.pnl = pnl

        # Update account
        self.current_balance += pnl
        self.daily_pnl += pnl
        self.total_pnl += pnl
        self.total_trades += 1

        # Update peak and drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        dd = (self.peak_balance - self.current_balance) / self.peak_balance
        if dd > self.max_drawdown_pct:
            self.max_drawdown_pct = dd

        # Win/Loss tracking with weighted consecutive loss scoring
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            # Weight loss by exit type (micro-exits count less than SL)
            loss_weight = self.LOSS_WEIGHTS.get(exit_reason, 1.0)
            self.consecutive_losses += loss_weight

            # Regime-aware cooldown duration
            cooldown_minutes = self.COOLDOWN_BY_REGIME.get(trade.regime, self.COOLDOWN_MINUTES_BASE)

            # Check if weighted score triggers cooldown
            if self.consecutive_losses >= self.CONSECUTIVE_LOSS_LIMIT:
                self.cooldown_until = current_time + timedelta(minutes=cooldown_minutes)
                print(f"[RiskManager] RISK PAUSE activated: {cooldown_minutes}min "
                      f"(regime={trade.regime}, score={self.consecutive_losses:.1f}, reason={exit_reason})")

        # Check daily halt
        daily_dd = (self.initial_balance_today - self.current_balance) / self.initial_balance_today
        if daily_dd >= self.DAILY_MAX_DRAWDOWN_PCT:
            self.trading_halted = True
            self.halt_reason = f"Daily drawdown: {daily_dd:.2%}"

        # Append to trade history
        self.trades.append(trade)

        # Persist trade to disk
        self._persist_trade(trade)

        print(f"[RiskManager] CLOSE {trade.direction} | PnL: ${pnl:.2f} | "
              f"Reason: {exit_reason} | Balance: ${self.current_balance:.2f}")

        self.open_trade = None
        return pnl

    def _persist_trade(self, trade: Trade):
        """Append closed trade to JSONL log for persistence across restarts."""
        try:
            from pathlib import Path
            log_dir = Path("data/paper_trades")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "trade_log.jsonl"

            record = {
                'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
                'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                'direction': trade.direction,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'size_lots': trade.size_lots,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'regime': trade.regime,
                'confidence': trade.confidence,
                'ofi_proxy': trade.ofi_proxy,
                'vpin_proxy': trade.vpin_proxy,
                'exit_reason': trade.exit_reason,
                'pnl': trade.pnl,
                'hold_time_min': (trade.exit_time - trade.entry_time).total_seconds() / 60
                    if trade.exit_time and trade.entry_time else None,
            }
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            print(f"[RiskManager] Trade persist error: {e}")

    def load_trade_history(self, filepath: str = "data/paper_trades/trade_log.jsonl"):
        """Load trade history from JSONL and restore metrics."""
        from pathlib import Path
        log_file = Path(filepath)
        if not log_file.exists():
            return

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    # We don't fully reconstruct Trade objects, just update metrics
                    pnl = record.get('pnl', 0.0)
                    exit_reason = record.get('exit_reason', 'SL')
                    if pnl is not None:
                        self.total_pnl += pnl
                        self.total_trades += 1
                        if pnl > 0:
                            self.winning_trades += 1
                            self.consecutive_losses = 0
                        else:
                            loss_weight = self.LOSS_WEIGHTS.get(exit_reason, 1.0)
                            self.consecutive_losses += loss_weight
                        self.current_balance += pnl
                        if self.current_balance > self.peak_balance:
                            self.peak_balance = self.current_balance
            print(f"[RiskManager] Loaded {self.total_trades} trades from history. "
                  f"Balance: ${self.current_balance:.2f}")
        except Exception as e:
            print(f"[RiskManager] History load error: {e}")

    # =====================================================================
    # DAILY RESET
    # =====================================================================
    def _reset_daily(self, current_time: datetime):
        """Reset daily tracking at market open (assumed midnight UTC)."""
        self.initial_balance_today = self.current_balance
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.cooldown_until = None
        self.trading_halted = False
        self.halt_reason = None
        print(f"[RiskManager] Daily reset at {current_time}. Balance: ${self.current_balance:.2f}")

    # =====================================================================
    # METRICS & REPORTING
    # =====================================================================
    def get_metrics(self) -> Dict:
        """Return current risk and performance metrics."""
        win_rate = self.winning_trades / max(self.total_trades, 1)

        # Calculate Sharpe-like ratio (simplified)
        if len(self.trades) >= 10:
            pnls = [t.pnl for t in self.trades if t.pnl is not None]
            if len(pnls) > 1:
                mean_pnl = np.mean(pnls)
                std_pnl = np.std(pnls)
                sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0

        return {
            'account_balance': self.current_balance,
            'daily_pnl': self.daily_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'max_drawdown_pct': self.max_drawdown_pct,
            'consecutive_losses': self.consecutive_losses,
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason,
            'sharpe_proxy': sharpe,
            'open_trade': self.open_trade is not None
        }

    def get_trade_log(self) -> pd.DataFrame:
        """Return trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append({
                'entry_time': t.entry_time,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'size_lots': t.size_lots,
                'stop_loss': t.stop_loss,
                'take_profit': t.take_profit,
                'regime': t.regime,
                'confidence': t.confidence,
                'ofi_proxy': t.ofi_proxy,
                'vpin_proxy': t.vpin_proxy,
                'exit_reason': t.exit_reason,
                'pnl': t.pnl,
                'hold_time_minutes': (t.exit_time - t.entry_time).total_seconds() / 60 if t.exit_time else None
            })

        return pd.DataFrame(records)

    def save_state(self, filepath: str):
        """Save risk manager state to JSON."""
        state = {
            'account_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'total_pnl': self.total_pnl,
            'max_drawdown_pct': self.max_drawdown_pct,
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, filepath: str):
        """Load risk manager state from JSON."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        self.current_balance = state['account_balance']
        self.peak_balance = state['peak_balance']
        self.total_trades = state['total_trades']
        self.winning_trades = state['winning_trades']
        self.total_pnl = state['total_pnl']
        self.max_drawdown_pct = state['max_drawdown_pct']
        self.trading_halted = state['trading_halted']
        self.halt_reason = state['halt_reason']
