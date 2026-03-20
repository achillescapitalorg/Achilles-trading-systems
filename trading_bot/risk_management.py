"""
Risk Management Module
=======================
Professional risk management for algorithmic trading.
Includes position sizing, stop-loss/take-profit calculation,
portfolio risk limits, and drawdown controls.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class RiskModel(Enum):
    """Position sizing models."""
    FIXED_PERCENTAGE = "fixed_percentage"
    KELLY_CRITERION = "kelly"
    VOLATILITY_TARGETING = "volatility_targeting"
    RISK_PARITY = "risk_parity"
    HALF_KELLY = "half_kelly"


class StopLossType(Enum):
    """Stop loss calculation methods."""
    FIXED_PERCENT = "fixed_percent"
    ATR_BASED = "atr_based"
    SUPPORT_RESISTANCE = "support_resistance"
    VOLATILITY_BASED = "volatility_based"
    TRAILING = "trailing"


@dataclass
class RiskParameters:
    """Container for risk management parameters."""
    # Position sizing
    risk_per_trade: float = 1.0  # % of account per trade
    max_position_size: float = 10.0  # Maximum position size in lots
    min_position_size: float = 0.01  # Minimum position size
    
    # Stop loss / Take profit
    stop_loss_type: StopLossType = StopLossType.ATR_BASED
    stop_loss_atr_multiplier: float = 2.0
    stop_loss_fixed_percent: float = 2.0
    take_profit_risk_reward: float = 2.0  # R:R ratio
    
    # Portfolio limits
    max_total_exposure: float = 50.0  # Max % of account at risk
    max_correlated_exposure: float = 30.0  # Max correlated positions
    max_drawdown_limit: float = 10.0  # Max drawdown before stopping
    
    # Risk model
    risk_model: RiskModel = RiskModel.FIXED_PERCENTAGE
    kelly_floor: float = 0.0  # Minimum win rate for Kelly
    kelly_cap: float = 0.25  # Maximum Kelly fraction
    volatility_target: float = 0.15  # Annual vol target


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    lots: float
    risk_amount: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    position_value: float
    account_risk_percent: float


class RiskManager:
    """
    Professional Risk Management System
    
    Implements institutional-grade risk controls:
    - Multiple position sizing models
    - Dynamic stop-loss/take-profit
    - Portfolio-level risk limits
    - Drawdown controls
    - Correlation management
    """
    
    def __init__(self, account_balance: float, params: RiskParameters = None):
        """
        Initialize risk manager.
        
        Parameters
        ----------
        account_balance : float
            Current account balance
        params : RiskParameters
            Risk management parameters
        """
        self.account_balance = account_balance
        self.params = params or RiskParameters()
        
        # Trade history for Kelly calculation
        self.trade_history: List[Dict] = []
        
        # Current positions
        self.open_positions: List[Dict] = []
        
        # Drawdown tracking
        self.peak_balance = account_balance
        self.current_drawdown = 0.0
        
        # Daily limits
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_trades = 20
        
    def update_balance(self, new_balance: float):
        """Update account balance and track drawdown."""
        self.account_balance = new_balance
        
        # Update peak and drawdown
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
            
        self.current_drawdown = (self.peak_balance - new_balance) / self.peak_balance * 100
        
    def check_trading_allowed(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on risk limits.
        
        Returns
        -------
        Tuple[bool, str]
            (allowed, reason)
        """
        # Check drawdown limit
        if self.current_drawdown >= self.params.max_drawdown_limit:
            return False, f"Max drawdown reached: {self.current_drawdown:.2f}%"
            
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return False, f"Daily trade limit reached: {self.daily_trades}"
            
        # Check total exposure
        total_exposure = self.get_total_exposure()
        if total_exposure >= self.params.max_total_exposure:
            return False, f"Max exposure reached: {total_exposure:.2f}%"
            
        return True, "OK"
    
    def calculate_position_size(self, 
                                symbol: str,
                                entry_price: float,
                                volatility: float = None,
                                atr: float = None,
                                win_rate: float = None,
                                avg_win_loss: float = None) -> PositionSizeResult:
        """
        Calculate position size based on risk model.
        
        Parameters
        ----------
        symbol : str
            Trading symbol
        entry_price : float
            Entry price
        volatility : float
            Annualized volatility (for vol targeting)
        atr : float
            Average True Range (for ATR stops)
        win_rate : float
            Historical win rate (for Kelly)
        avg_win_loss : float
            Average win/loss ratio (for Kelly)
            
        Returns
        -------
        PositionSizeResult
            Position sizing result
        """
        # Calculate stop loss
        stop_loss = self.calculate_stop_loss(
            entry_price=entry_price,
            atr=atr,
            is_buy=True  # Will be adjusted for direction
        )
        
        # Calculate risk per share/contract
        risk_per_unit = abs(entry_price - stop_loss)
        
        # Determine position size based on risk model
        if self.params.risk_model == RiskModel.FIXED_PERCENTAGE:
            risk_amount = self.account_balance * (self.params.risk_per_trade / 100)
            
        elif self.params.risk_model == RiskModel.KELLY_CRITERION:
            if win_rate is None or avg_win_loss is None:
                # Fall back to fixed percentage
                risk_amount = self.account_balance * (self.params.risk_per_trade / 100)
            else:
                kelly = self._calculate_kelly(win_rate, avg_win_loss)
                risk_amount = self.account_balance * kelly
                
        elif self.params.risk_model == RiskModel.HALF_KELLY:
            if win_rate is None or avg_win_loss is None:
                risk_amount = self.account_balance * (self.params.risk_per_trade / 100)
            else:
                kelly = self._calculate_kelly(win_rate, avg_win_loss)
                risk_amount = self.account_balance * (kelly / 2)
                
        elif self.params.risk_model == RiskModel.VOLATILITY_TARGETING:
            if volatility is None:
                volatility = 0.20  # Default 20% vol
                
            # Position size inversely proportional to volatility
            vol_ratio = self.params.volatility_target / volatility
            risk_amount = self.account_balance * (self.params.risk_per_trade / 100) * vol_ratio
            
        else:
            risk_amount = self.account_balance * (self.params.risk_per_trade / 100)
            
        # Calculate lot size
        if risk_per_unit > 0:
            units = risk_amount / risk_per_unit
            # Convert to lots (standard lot = 100 units for forex, varies for others)
            if 'XAU' in symbol or 'GOLD' in symbol.upper():
                lots = units / 100  # Gold: 100 oz per lot
            elif 'BTC' in symbol.upper():
                lots = units  # BTC: 1 BTC per lot
            else:
                lots = units / 100000  # Forex: 100k units per lot
        else:
            lots = self.params.min_position_size
            
        # Apply limits
        lots = max(self.params.min_position_size, 
                  min(self.params.max_position_size, lots))
        
        # Calculate take profit
        take_profit = self.calculate_take_profit(
            entry_price=entry_price,
            stop_loss=stop_loss,
            is_buy=True
        )
        
        # Risk-reward ratio
        reward_per_unit = abs(take_profit - entry_price)
        risk_reward = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0
        
        # Position value
        position_value = lots * entry_price * 100  # Approximate
        
        # Actual account risk
        actual_risk = risk_per_unit * lots * 100
        account_risk_percent = actual_risk / self.account_balance * 100
        
        return PositionSizeResult(
            lots=lots,
            risk_amount=actual_risk,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward,
            position_value=position_value,
            account_risk_percent=account_risk_percent
        )
    
    def _calculate_kelly(self, win_rate: float, avg_win_loss: float) -> float:
        """
        Calculate Kelly criterion fraction.
        
        Kelly % = W - [(1-W) / R]
        Where:
        - W = Win probability
        - R = Win/Loss ratio
        
        Parameters
        ----------
        win_rate : float
            Win rate (0-1)
        avg_win_loss : float
            Average win / Average loss ratio
            
        Returns
        -------
        float
            Kelly fraction (capped)
        """
        if win_rate < self.params.kelly_floor:
            return 0.0
            
        kelly = win_rate - ((1 - win_rate) / max(avg_win_loss, 0.01))
        kelly = max(0, min(kelly, self.params.kelly_cap))
        
        return kelly
    
    def calculate_stop_loss(self,
                           entry_price: float,
                           atr: float = None,
                           is_buy: bool = True,
                           high: float = None,
                           low: float = None) -> float:
        """
        Calculate stop loss price.
        
        Parameters
        ----------
        entry_price : float
            Entry price
        atr : float
            Average True Range
        is_buy : bool
            True for long, False for short
        high : float
            Recent high (for support/resistance)
        low : float
            Recent low (for support/resistance)
            
        Returns
        -------
        float
            Stop loss price
        """
        if self.params.stop_loss_type == StopLossType.FIXED_PERCENT:
            if is_buy:
                stop_loss = entry_price * (1 - self.params.stop_loss_fixed_percent / 100)
            else:
                stop_loss = entry_price * (1 + self.params.stop_loss_fixed_percent / 100)
                
        elif self.params.stop_loss_type == StopLossType.ATR_BASED:
            if atr is None:
                atr = entry_price * 0.02  # Default 2% ATR
                
            if is_buy:
                stop_loss = entry_price - (atr * self.params.stop_loss_atr_multiplier)
            else:
                stop_loss = entry_price + (atr * self.params.stop_loss_atr_multiplier)
                
        elif self.params.stop_loss_type == StopLossType.SUPPORT_RESISTANCE:
            if is_buy and low is not None:
                # Place stop below recent low
                stop_loss = low * (1 - 0.01)  # 1% below low
            elif not is_buy and high is not None:
                # Place stop above recent high
                stop_loss = high * (1 + 0.01)
            else:
                # Fall back to fixed percent
                stop_loss = entry_price * (1 - self.params.stop_loss_fixed_percent / 100) if is_buy else entry_price * (1 + self.params.stop_loss_fixed_percent / 100)
                
        elif self.params.stop_loss_type == StopLossType.VOLATILITY_BASED:
            # Use 2 standard deviations
            vol_adjustment = entry_price * 0.02 * 2
            if is_buy:
                stop_loss = entry_price - vol_adjustment
            else:
                stop_loss = entry_price + vol_adjustment
                
        else:
            stop_loss = entry_price * (1 - 0.02) if is_buy else entry_price * (1 + 0.02)
            
        return round(stop_loss, 2)
    
    def calculate_take_profit(self,
                             entry_price: float,
                             stop_loss: float,
                             is_buy: bool = True) -> float:
        """
        Calculate take profit based on risk-reward ratio.
        
        Parameters
        ----------
        entry_price : float
            Entry price
        stop_loss : float
            Stop loss price
        is_buy : bool
            True for long, False for short
            
        Returns
        -------
        float
            Take profit price
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * self.params.take_profit_risk_reward
        
        if is_buy:
            take_profit = entry_price + reward
        else:
            take_profit = entry_price - reward
            
        return round(take_profit, 2)
    
    def calculate_trailing_stop(self,
                               current_price: float,
                               entry_price: float,
                               atr: float,
                               is_buy: bool = True) -> float:
        """
        Calculate trailing stop loss.
        
        Parameters
        ----------
        current_price : float
            Current market price
        entry_price : float
            Original entry price
        atr : float
            Average True Range
        is_buy : bool
            True for long position
            
        Returns
        -------
        float
            Trailing stop price
        """
        if is_buy:
            # For long positions, trail below price
            profit = current_price - entry_price
            
            if profit > 0:
                # Lock in some profit
                trailing_stop = current_price - (atr * self.params.stop_loss_atr_multiplier)
                # Never move stop loss below entry
                trailing_stop = max(trailing_stop, entry_price)
            else:
                # Use original stop
                trailing_stop = entry_price - (atr * self.params.stop_loss_atr_multiplier)
        else:
            # For short positions, trail above price
            profit = entry_price - current_price
            
            if profit > 0:
                trailing_stop = current_price + (atr * self.params.stop_loss_atr_multiplier)
                trailing_stop = min(trailing_stop, entry_price)
            else:
                trailing_stop = entry_price + (atr * self.params.stop_loss_atr_multiplier)
                
        return round(trailing_stop, 2)
    
    def get_total_exposure(self) -> float:
        """
        Calculate total portfolio exposure as % of account.
        
        Returns
        -------
        float
            Total exposure percentage
        """
        total_exposure = 0
        
        for pos in self.open_positions:
            # Approximate exposure
            exposure = pos.get('volume', 0) * pos.get('entry_price', 0) * 100
            total_exposure += exposure / self.account_balance * 100
            
        return total_exposure
    
    def add_position(self, position: Dict):
        """Add position to tracking."""
        self.open_positions.append(position)
        self.daily_trades += 1
        
    def remove_position(self, symbol: str):
        """Remove position from tracking."""
        self.open_positions = [p for p in self.open_positions if p.get('symbol') != symbol]
        
    def record_trade(self, symbol: str, pnl: float, is_win: bool):
        """Record trade for statistics."""
        self.trade_history.append({
            'symbol': symbol,
            'pnl': pnl,
            'is_win': is_win,
            'timestamp': datetime.now()
        })
        
        self.daily_pnl += pnl
        
        # Keep only recent history
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
            
    def get_statistics(self) -> Dict:
        """
        Get risk management statistics.
        
        Returns
        -------
        Dict with statistics
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': self.current_drawdown,
            }
            
        wins = [t['pnl'] for t in self.trade_history if t['is_win'] and t['pnl'] > 0]
        losses = [t['pnl'] for t in self.trade_history if not t['is_win'] or t['pnl'] < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        win_rate = len(wins) / len(self.trade_history) if self.trade_history else 0
        
        return {
            'total_trades': len(self.trade_history),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': avg_win / avg_loss if avg_loss > 0 else 0,
            'profit_factor': profit_factor,
            'max_drawdown': self.current_drawdown,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'total_exposure': self.get_total_exposure(),
        }
    
    def get_correlation_adjustment(self, symbols: List[str]) -> float:
        """
        Get position size adjustment for correlated positions.
        
        Parameters
        ----------
        symbols : List[str]
            List of symbols to check correlation
            
        Returns
        -------
        float
            Adjustment factor (0-1)
        """
        # Define correlated groups
        correlated_groups = {
            'forex_major': ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD'],
            'forex_jpy': ['USDJPY', 'EURJPY', 'GBPJPY'],
            'metals': ['XAUUSD', 'XAGUSD'],
            'crypto': ['BTCUSD', 'ETHUSD'],
        }
        
        # Check how many positions are in same group
        for group_name, group_symbols in correlated_groups.items():
            matching = [s for s in symbols if s.upper() in [g.upper() for g in group_symbols]]
            
            if len(matching) > 1:
                # Reduce size for correlated exposure
                return self.params.max_correlated_exposure / 100 / len(matching)
                
        return 1.0  # No correlation concern


class PortfolioRiskLimits:
    """
    Portfolio-level risk limits and monitoring.
    """
    
    def __init__(self, 
                 max_drawdown: float = 0.10,
                 max_var: float = 0.05,
                 max_sector_exposure: Dict[str, float] = None):
        """
        Initialize portfolio risk limits.
        
        Parameters
        ----------
        max_drawdown : float
            Maximum portfolio drawdown (10%)
        max_var : float
            Maximum Value at Risk (5%)
        max_sector_exposure : Dict[str, float]
            Maximum exposure per sector
        """
        self.max_drawdown = max_drawdown
        self.max_var = max_var
        self.max_sector_exposure = max_sector_exposure or {
            'forex': 0.30,
            'metals': 0.20,
            'crypto': 0.15,
            'indices': 0.20,
        }
        
    def check_var_limit(self, positions: List[Dict], 
                       volatilities: Dict[str, float],
                       correlations: np.ndarray = None) -> Tuple[bool, float]:
        """
        Check if portfolio VaR is within limits.
        
        Parameters
        ----------
        positions : List[Dict]
            Current positions
        volatilities : Dict[str, float]
            Volatility per symbol
        correlations : np.ndarray
            Correlation matrix
            
        Returns
        -------
        Tuple[bool, float]
            (within_limit, var_95)
        """
        if not positions:
            return True, 0.0
            
        # Simple portfolio VaR calculation
        total_var = 0
        
        for pos in positions:
            symbol = pos.get('symbol', '')
            value = pos.get('value', 0)
            vol = volatilities.get(symbol, 0.20)
            
            # Individual position VaR (95%)
            position_var = value * vol * 1.645
            total_var += position_var
            
        # Check against limit
        portfolio_value = sum(p.get('value', 0) for p in positions)
        var_percent = total_var / portfolio_value if portfolio_value > 0 else 0
        
        within_limit = var_percent <= self.max_var
        
        return within_limit, var_percent
