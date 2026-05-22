"""
execution/engine.py
===================
Manual-first execution layer for MT5 broker integration.

Validates, prepares, and executes orders with full pre-trade checks.
Phase 6: Manual execution (human confirms each trade).
"""
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import json


try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None


@dataclass
class PreparedOrder:
    """Order ready for manual confirmation."""
    symbol: str
    direction: str          # 'BUY' or 'SELL'
    volume: float
    sl: float
    tp: float
    comment: str
    pre_trade_checks: Dict
    validated: bool
    validation_reason: str


@dataclass
class BrokerPosition:
    """Position as reported by broker."""
    ticket: int
    symbol: str
    direction: str
    volume: float
    open_price: float
    current_profit: float
    sl: float
    tp: float
    open_time: datetime


class ExecutionEngine:
    """
    Manual-first execution engine.

    Usage:
        engine = ExecutionEngine()
        order = engine.prepare_order('XAUUSDm', 'BUY', 0.01, 3300, 3400)
        if order.validated:
            result = engine.execute_order(order)  # Manual confirmation step
    """

    # Gold symbol candidates (Exness uses XAUUSDm)
    SYMBOLS = ["XAUUSDm", "XAUUSD", "GOLD"]

    # Pre-trade thresholds
    MAX_SPREAD_PIPS: float = 5.0      # Reject if spread > 50 cents
    MIN_FREE_MARGIN_PCT: float = 0.3   # Need 30% free margin
    MAX_SLIPPAGE_PIPS: float = 3.0    # Warn if slippage > 30 cents

    def __init__(self):
        self._symbol: Optional[str] = None
        self._last_error: Optional[str] = None
        self._execution_log: List[Dict] = []
        self._log_file = Path("data/paper_trades/execution_log.jsonl")
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

    def _resolve_symbol(self) -> Optional[str]:
        """Find available gold symbol in MT5."""
        if not MT5_AVAILABLE:
            return None
        try:
            term = mt5.terminal_info()
            if term is None or not term.connected:
                return None
        except Exception:
            return None
        for sym in self.SYMBOLS:
            if mt5.symbol_select(sym, True):
                self._symbol = sym
                return sym
        return None

    def get_symbol(self) -> Optional[str]:
        """Return cached or resolved symbol."""
        if self._symbol is not None:
            return self._symbol
        return self._resolve_symbol()

    def _get_spread(self, symbol: str) -> float:
        """Get current spread in pips (0.01 = 1 pip for gold)."""
        if not MT5_AVAILABLE:
            return 0.0
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return float('inf')
        return (tick.ask - tick.bid) / 0.01

    def _get_account_info(self) -> Dict:
        """Get account info from MT5."""
        if not MT5_AVAILABLE:
            return {"balance": 0.0, "free_margin": 0.0, "equity": 0.0}
        info = mt5.account_info()
        if info is None:
            return {"balance": 0.0, "free_margin": 0.0, "equity": 0.0}
        return {
            "balance": info.balance,
            "free_margin": info.margin_free,
            "equity": info.equity,
        }

    def pre_trade_checks(
        self, symbol: str, direction: str, volume: float, sl: float, tp: float
    ) -> Tuple[bool, str, Dict]:
        """
        Run all pre-trade validation checks.

        Returns:
            (validated: bool, reason: str, details: dict)
        """
        if not MT5_AVAILABLE:
            return False, "MT5 not available", {}

        if not mt5.terminal_info().connected:
            return False, "MT5 not connected", {}

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return False, f"Symbol {symbol} not found", {}

        if not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
            return False, f"Symbol {symbol} not tradable", {}

        checks = {
            "mt5_connected": True,
            "symbol_available": True,
            "symbol_tradable": True,
        }

        # 1. Spread check
        spread = self._get_spread(symbol)
        checks["spread_pips"] = round(spread, 2)
        checks["spread_ok"] = spread <= self.MAX_SPREAD_PIPS
        if not checks["spread_ok"]:
            return False, f"Spread too high: {spread:.1f} pips", checks

        # 2. Margin check
        acc = self._get_account_info()
        checks["balance"] = acc["balance"]
        checks["free_margin"] = acc["free_margin"]
        checks["free_margin_pct"] = acc["free_margin"] / acc["balance"] if acc["balance"] > 0 else 0
        checks["margin_ok"] = checks["free_margin_pct"] >= self.MIN_FREE_MARGIN_PCT
        if not checks["margin_ok"]:
            return False, f"Insufficient margin: {checks['free_margin_pct']:.1%}", checks

        # 3. Volume check
        checks["volume_min"] = symbol_info.volume_min
        checks["volume_max"] = symbol_info.volume_max
        checks["volume_step"] = symbol_info.volume_step
        checks["volume_ok"] = (
            symbol_info.volume_min <= volume <= symbol_info.volume_max and
            abs(volume - round(volume / symbol_info.volume_step) * symbol_info.volume_step) < 1e-9
        )
        if not checks["volume_ok"]:
            return False, f"Invalid volume: {volume} (min={symbol_info.volume_min}, max={symbol_info.volume_max})", checks

        # 4. SL/TP distance check (must be >= stop level)
        stop_level = symbol_info.trade_stops_level * symbol_info.point
        sl_dist = abs(sl - (mt5.symbol_info_tick(symbol).ask if direction == 'BUY' else mt5.symbol_info_tick(symbol).bid))
        tp_dist = abs(tp - (mt5.symbol_info_tick(symbol).bid if direction == 'BUY' else mt5.symbol_info_tick(symbol).ask))
        checks["stop_level"] = stop_level
        checks["sl_distance"] = sl_dist
        checks["tp_distance"] = tp_dist
        checks["sl_tp_ok"] = sl_dist >= stop_level and tp_dist >= stop_level
        if not checks["sl_tp_ok"]:
            return False, f"SL/TP too close to price (stop level={stop_level})", checks

        return True, "All checks passed", checks

    def prepare_order(
        self,
        direction: str,
        volume: float,
        sl: float,
        tp: float,
        comment: str = "AchillesHybrid"
    ) -> PreparedOrder:
        """
        Prepare an order with full validation.
        Returns a PreparedOrder ready for manual confirmation.
        """
        symbol = self.get_symbol()
        if symbol is None:
            return PreparedOrder(
                symbol="", direction=direction, volume=volume, sl=sl, tp=tp,
                comment=comment, pre_trade_checks={}, validated=False,
                validation_reason="Could not resolve symbol"
            )

        validated, reason, checks = self.pre_trade_checks(symbol, direction, volume, sl, tp)
        return PreparedOrder(
            symbol=symbol,
            direction=direction,
            volume=volume,
            sl=sl,
            tp=tp,
            comment=comment,
            pre_trade_checks=checks,
            validated=validated,
            validation_reason=reason,
        )

    def execute_order(self, order: PreparedOrder) -> Dict:
        """
        Execute a validated order via MT5.
        Returns execution result dict.
        """
        if not order.validated:
            return {"success": False, "error": f"Order not validated: {order.validation_reason}"}

        if not MT5_AVAILABLE:
            return {"success": False, "error": "MT5 not available"}

        order_type = mt5.ORDER_TYPE_BUY if order.direction == 'BUY' else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": order.volume,
            "type": order_type,
            "price": mt5.symbol_info_tick(order.symbol).ask if order.direction == 'BUY' else mt5.symbol_info_tick(order.symbol).bid,
            "sl": order.sl,
            "tp": order.tp,
            "deviation": 10,
            "magic": 234000,
            "comment": order.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        log_entry = {
            "time": datetime.now().isoformat(),
            "symbol": order.symbol,
            "direction": order.direction,
            "volume": order.volume,
            "sl": order.sl,
            "tp": order.tp,
            "validated": order.validated,
            "request": request,
        }

        if result is None:
            err = f"Order send failed: {mt5.last_error()}"
            self._last_error = err
            log_entry["success"] = False
            log_entry["error"] = err
            self._persist_log(log_entry)
            return {"success": False, "error": err}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            err = f"Order failed: {result.retcode} - {result.comment}"
            self._last_error = err
            log_entry["success"] = False
            log_entry["retcode"] = result.retcode
            log_entry["comment"] = result.comment
            self._persist_log(log_entry)
            return {"success": False, "error": err, "retcode": result.retcode}

        log_entry["success"] = True
        log_entry["ticket"] = result.order
        log_entry["price"] = result.price
        self._persist_log(log_entry)

        return {
            "success": True,
            "ticket": result.order,
            "price": result.price,
            "volume": result.volume,
            "symbol": result.symbol,
        }

    def close_position(self, ticket: int) -> Dict:
        """Close an open position by ticket."""
        if not MT5_AVAILABLE:
            return {"success": False, "error": "MT5 not available"}

        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            return {"success": False, "error": f"Position {ticket} not found"}

        pos = position[0]
        price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 10,
            "magic": 234000,
            "comment": "AchillesClose",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "error": f"Close failed: {result.comment if result else mt5.last_error()}"}

        return {"success": True, "ticket": result.order, "price": result.price}

    def get_open_positions(self) -> List[BrokerPosition]:
        """Get all open positions from broker."""
        if not MT5_AVAILABLE:
            return []

        positions = mt5.positions_get()
        if positions is None:
            return []

        result = []
        for pos in positions:
            result.append(BrokerPosition(
                ticket=pos.ticket,
                symbol=pos.symbol,
                direction="BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                volume=pos.volume,
                open_price=pos.price_open,
                current_profit=pos.profit,
                sl=pos.sl,
                tp=pos.tp,
                open_time=datetime.fromtimestamp(pos.time),
            ))
        return result

    def _persist_log(self, entry: Dict):
        """Append execution log entry to JSONL."""
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            print(f"[ExecutionEngine] Log persist error: {e}")

    def get_last_error(self) -> Optional[str]:
        return self._last_error
