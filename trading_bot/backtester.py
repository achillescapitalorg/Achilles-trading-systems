"""
Backtesting Engine
===================
Professional backtesting framework with realistic simulation.
Includes transaction costs, slippage, and comprehensive performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import warnings


class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Trade:
    """Container for backtest trade record."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    symbol: str
    direction: TradeDirection
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_percent: float
    max_drawdown: float
    max_profit: float
    holding_period: int
    exit_reason: str


@dataclass
class BacktestResult:
    """Container for backtest performance results."""
    # Returns
    total_return: float
    annualized_return: float
    excess_return: float  # Over benchmark
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    avg_holding_period: float
    
    # Risk-adjusted
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float
    
    # Equity curve
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    trades: List[Trade]


class Backtester:
    """
    Professional Backtesting Engine
    
    Features:
    - Vectorized and event-driven backtesting
    - Realistic transaction costs and slippage
    - Multiple timeframes
    - Walk-forward analysis
    - Monte Carlo simulation
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission: float = 0.0001,  # 0.01%
                 slippage: float = 0.0005,  # 0.05%
                 risk_free_rate: float = 0.05):  # 5% annual
        """
        Initialize backtester.
        
        Parameters
        ----------
        initial_capital : float
            Starting capital
        commission : float
            Commission per trade (as decimal)
        slippage : float
            Average slippage (as decimal)
        risk_free_rate : float
            Annual risk-free rate for Sharpe calculation
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        
        # Trading state
        self.capital = initial_capital
        self.position = 0
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.drawdown_curve: List[float] = []
        
    def run(self, 
            data: pd.DataFrame,
            signals: pd.Series,
            symbol: str = 'SYMBOL',
            initial_capital: float = None) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with 'timestamp', 'open', 'high', 'low', 'close', 'volume'
        signals : pd.Series
            Trading signals: 1=BUY, -1=SELL, 0=HOLD
        symbol : str
            Symbol name
        initial_capital : float
            Override initial capital
            
        Returns
        -------
        BacktestResult
            Backtest performance results
        """
        # Reset state
        self.capital = initial_capital or self.initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        
        peak = self.capital
        entry_trade = None
        
        # Iterate through data
        for i in range(len(data)):
            row = data.iloc[i]
            timestamp = row['timestamp'] if 'timestamp' in row else row.name
            current_price = row['close']
            
            signal = signals.iloc[i] if hasattr(signals, 'iloc') else signals[i]
            
            # Execute trades
            if signal == 1 and self.position == 0:
                # Enter long
                entry_price = current_price * (1 + self.slippage)
                self.position = self.capital / entry_price
                entry_trade = {
                    'entry_time': timestamp,
                    'entry_price': entry_price,
                    'direction': TradeDirection.LONG,
                    'size': self.position
                }
                
            elif signal == -1 and self.position == 0:
                # Enter short
                entry_price = current_price * (1 - self.slippage)
                self.position = -self.capital / entry_price
                entry_trade = {
                    'entry_time': timestamp,
                    'entry_price': entry_price,
                    'direction': TradeDirection.SHORT,
                    'size': abs(self.position)
                }
                
            elif signal == 0 and self.position != 0 and entry_trade:
                # Exit position
                if self.position > 0:
                    exit_price = current_price * (1 - self.slippage)
                else:
                    exit_price = current_price * (1 + self.slippage)
                    
                # Calculate PnL
                if entry_trade['direction'] == TradeDirection.LONG:
                    pnl = (exit_price - entry_trade['entry_price']) * self.position
                else:
                    pnl = (entry_trade['entry_price'] - exit_price) * self.position
                    
                # Commission
                commission = (abs(entry_trade['entry_price']) + abs(exit_price)) * abs(self.position) * self.commission
                pnl -= commission
                
                # Record trade
                trade = Trade(
                    entry_time=entry_trade['entry_time'],
                    exit_time=timestamp,
                    symbol=symbol,
                    direction=entry_trade['direction'],
                    entry_price=entry_trade['entry_price'],
                    exit_price=exit_price,
                    size=abs(self.position),
                    pnl=pnl,
                    pnl_percent=pnl / self.capital * 100,
                    max_drawdown=0,  # Will be calculated
                    max_profit=0,
                    holding_period=i - (entry_trade.get('entry_idx', 0)),
                    exit_reason='signal'
                )
                self.trades.append(trade)
                
                # Update capital
                self.capital += pnl
                self.position = 0
                entry_trade = None
                
            # Record equity
            if self.position != 0 and entry_trade:
                # Mark to market
                if self.position > 0:
                    unrealized_pnl = (current_price - entry_trade['entry_price']) * self.position
                else:
                    unrealized_pnl = (entry_trade['entry_price'] - current_price) * abs(self.position)
                equity = self.capital + unrealized_pnl
            else:
                equity = self.capital
                
            self.equity_curve.append(equity)
            
            # Track drawdown
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100 if peak > 0 else 0
            self.drawdown_curve.append(drawdown)
            
        # Close any open position at end
        if self.position != 0 and entry_trade and len(data) > 0:
            final_price = data.iloc[-1]['close']
            if self.position > 0:
                exit_price = final_price * (1 - self.slippage)
            else:
                exit_price = final_price * (1 + self.slippage)
                
            if entry_trade['direction'] == TradeDirection.LONG:
                pnl = (exit_price - entry_trade['entry_price']) * self.position
            else:
                pnl = (entry_trade['entry_price'] - exit_price) * abs(self.position)
                
            commission = (abs(entry_trade['entry_price']) + abs(exit_price)) * abs(self.position) * self.commission
            pnl -= commission
            
            trade = Trade(
                entry_time=entry_trade['entry_time'],
                exit_time=data.iloc[-1]['timestamp'] if 'timestamp' in data else data.index[-1],
                symbol=symbol,
                direction=entry_trade['direction'],
                entry_price=entry_trade['entry_price'],
                exit_price=exit_price,
                size=abs(self.position),
                pnl=pnl,
                pnl_percent=pnl / self.capital * 100,
                max_drawdown=0,
                max_profit=0,
                holding_period=len(data) - entry_trade.get('entry_idx', 0),
                exit_reason='end_of_data'
            )
            self.trades.append(trade)
            self.capital += pnl
            
        return self._calculate_metrics(data)
    
    def _calculate_metrics(self, data: pd.DataFrame) -> BacktestResult:
        """Calculate performance metrics."""
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # Basic returns
        total_return = (equity_series.iloc[-1] - self.initial_capital) / self.initial_capital * 100
        
        # Annualized return
        n_days = len(data)
        years = n_days / 252
        annualized_return = ((equity_series.iloc[-1] / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate / 252
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return / 100) / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        max_drawdown = max(self.drawdown_curve) if self.drawdown_curve else 0
        
        # Drawdown duration
        dd_duration = self._calculate_max_dd_duration(equity_series)
        
        # Calmar ratio
        calmar_ratio = (annualized_return / 100) / (max_drawdown / 100) if max_drawdown > 0 else 0
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        total_trades = len(self.trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        avg_win = total_wins / win_count if win_count > 0 else 0
        avg_loss = total_losses / loss_count if loss_count > 0 else 0
        avg_trade = sum(t.pnl for t in self.trades) / total_trades if total_trades > 0 else 0
        
        avg_holding = sum(t.holding_period for t in self.trades) / total_trades if total_trades > 0 else 0
        
        # VaR and CVaR
        var_95 = returns.quantile(0.05) * 100 if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95 / 100].mean() * 100 if len(returns) > 0 else 0
        
        # Distribution metrics
        skewness = returns.skew() if len(returns) > 0 else 0
        kurtosis = returns.kurtosis() if len(returns) > 0 else 0
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            excess_return=annualized_return - self.risk_free_rate * 100,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=dd_duration,
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            avg_holding_period=avg_holding,
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            equity_curve=equity_series,
            drawdown_curve=pd.Series(self.drawdown_curve),
            trades=self.trades
        )
    
    def _calculate_max_dd_duration(self, equity: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        peak = equity.expanding(min_periods=1).max()
        drawdown = (peak - equity) / peak
        
        in_drawdown = drawdown > 0
        dd_start = None
        max_duration = 0
        current_duration = 0
        
        for i, in_dd in enumerate(in_drawdown):
            if in_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
                
        return max_duration
    
    def run_walk_forward(self,
                        data: pd.DataFrame,
                        signals_generator: Callable,
                        in_sample_periods: int = 252,
                        out_sample_periods: int = 63,
                        symbol: str = 'SYMBOL') -> List[BacktestResult]:
        """
        Run walk-forward analysis.
        
        Parameters
        ----------
        data : pd.DataFrame
            Historical data
        signals_generator : Callable
            Function that takes data and returns signals
        in_sample_periods : int
            Training period length
        out_sample_periods : int
            Testing period length
        symbol : str
            Symbol name
            
        Returns
        -------
        List[BacktestResult]
            Results for each out-of-sample period
        """
        results = []
        
        i = 0
        while i + in_sample_periods + out_sample_periods <= len(data):
            # In-sample period
            train_data = data.iloc[i:i + in_sample_periods]
            
            # Out-of-sample period
            test_data = data.iloc[i + in_sample_periods:i + in_sample_periods + out_sample_periods]
            
            # Generate signals on test data
            signals = signals_generator(train_data, test_data)
            
            # Run backtest
            result = self.run(test_data, signals, symbol)
            results.append(result)
            
            # Move forward
            i += out_sample_periods
            
        return results
    
    def monte_carlo_simulation(self,
                               trades: List[Trade],
                               n_simulations: int = 1000) -> Dict[str, float]:
        """
        Run Monte Carlo simulation on trade sequence.
        
        Parameters
        ----------
        trades : List[Trade]
            Historical trades
        n_simulations : int
            Number of simulations
            
        Returns
        -------
        Dict with simulation statistics
        """
        if not trades:
            return {}
            
        pnls = np.array([t.pnl for t in trades])
        
        simulated_final_equity = []
        simulated_max_dd = []
        
        for _ in range(n_simulations):
            # Random shuffle of trades
            shuffled_pnls = np.random.choice(pnls, size=len(pnls), replace=True)
            
            # Cumulative equity
            equity = self.initial_capital + np.cumsum(shuffled_pnls)
            simulated_final_equity.append(equity[-1])
            
            # Max drawdown
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak * 100
            simulated_max_dd.append(np.max(drawdown))
            
        return {
            'median_final_equity': np.median(simulated_final_equity),
            'mean_final_equity': np.mean(simulated_final_equity),
            'std_final_equity': np.std(simulated_final_equity),
            'prob_profit': np.mean([e > self.initial_capital for e in simulated_final_equity]),
            'median_max_dd': np.median(simulated_max_dd),
            'worst_max_dd': np.max(simulated_max_dd),
            'final_equity_5th_percentile': np.percentile(simulated_final_equity, 5),
            'final_equity_95th_percentile': np.percentile(simulated_final_equity, 95),
        }


def print_backtest_report(result: BacktestResult):
    """Print formatted backtest report."""
    print("\n" + "=" * 70)
    print("BACKTEST PERFORMANCE REPORT")
    print("=" * 70)
    
    print("\n[RETURNS]")
    print(f"  Total Return:        {result.total_return:>10.2f}%")
    print(f"  Annualized Return:   {result.annualized_return:>10.2f}%")
    print(f"  Excess Return:       {result.excess_return:>10.2f}%")
    
    print("\n[RISK METRICS]")
    print(f"  Volatility:          {result.volatility:>10.2f}%")
    print(f"  Sharpe Ratio:        {result.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:       {result.sortino_ratio:>10.2f}")
    print(f"  Calmar Ratio:        {result.calmar_ratio:>10.2f}")
    print(f"  Max Drawdown:        {result.max_drawdown:>10.2f}%")
    print(f"  DD Duration (days):  {result.max_drawdown_duration:>10d}")
    
    print("\n[TRADE STATISTICS]")
    print(f"  Total Trades:        {result.total_trades:>10d}")
    print(f"  Win Rate:            {result.win_rate * 100:>10.1f}%")
    print(f"  Profit Factor:       {result.profit_factor:>10.2f}")
    print(f"  Avg Win:             ${result.avg_win:>10.2f}")
    print(f"  Avg Loss:            ${result.avg_loss:>10.2f}")
    print(f"  Avg Trade:           ${result.avg_trade:>10.2f}")
    print(f"  Avg Holding Period:  {result.avg_holding_period:>10.1f} days")
    
    print("\n[DISTRIBUTION]")
    print(f"  VaR (95%):           {result.var_95:>10.2f}%")
    print(f"  CVaR (95%):          {result.cvar_95:>10.2f}%")
    print(f"  Skewness:            {result.skewness:>10.2f}")
    print(f"  Kurtosis:            {result.kurtosis:>10.2f}")
    
    print("\n" + "=" * 70)
