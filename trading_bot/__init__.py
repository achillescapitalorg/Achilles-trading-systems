"""
Gold/BTC Trading Bot Package
=============================
Professional algorithmic trading system for Gold and Bitcoin.
"""

from .technical_indicators import TechnicalIndicators, SignalStrength, TechnicalSignal
from .quantitative_signals import QuantitativeSignals, SignalType, QuantSignal
from .sentiment_analysis import SentimentAnalyzer, NewsSentimentSignal, SentimentSignal
from .exness_bridge import (
    ExnessMT5Bridge, 
    PaperTradingBridge, 
    TradeOrder, 
    OrderType,
    TradeResult,
    Position
)
from .risk_management import (
    RiskManager, 
    RiskParameters, 
    RiskModel, 
    StopLossType,
    PositionSizeResult
)
from .backtester import Backtester, BacktestResult, print_backtest_report
from .trading_bot import GoldBTCTradingBot, BotConfig, TradingSignal, create_bot_config

__version__ = '1.0.0'
__author__ = 'Quantitative Trading Team'

__all__ = [
    # Main bot
    'GoldBTCTradingBot',
    'BotConfig',
    'TradingSignal',
    'create_bot_config',
    
    # Technical indicators
    'TechnicalIndicators',
    'SignalStrength',
    'TechnicalSignal',
    
    # Quantitative signals
    'QuantitativeSignals',
    'SignalType',
    'QuantSignal',
    
    # Sentiment analysis
    'SentimentAnalyzer',
    'NewsSentimentSignal',
    'SentimentSignal',
    
    # Broker bridge
    'ExnessMT5Bridge',
    'PaperTradingBridge',
    'TradeOrder',
    'OrderType',
    'TradeResult',
    'Position',
    
    # Risk management
    'RiskManager',
    'RiskParameters',
    'RiskModel',
    'StopLossType',
    'PositionSizeResult',
    
    # Backtesting
    'Backtester',
    'BacktestResult',
    'print_backtest_report',
]
