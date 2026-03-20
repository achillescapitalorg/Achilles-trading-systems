"""
Gold/BTC Trading Bot - Main Module
===================================
Professional algorithmic trading bot for Gold (XAUUSD) and Bitcoin (BTCUSD).
Integrates technical analysis, quantitative signals, sentiment analysis,
and risk management for Exness MT5 execution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import json

# Import trading modules
from technical_indicators import TechnicalIndicators, SignalStrength
from quantitative_signals import QuantitativeSignals, SignalType
from sentiment_analysis import SentimentAnalyzer, NewsSentimentSignal
from exness_bridge import ExnessMT5Bridge, PaperTradingBridge, TradeOrder, OrderType
from risk_management import RiskManager, RiskParameters, RiskModel, StopLossType


@dataclass
class TradingSignal:
    """Aggregated trading signal."""
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    technical_signal: str
    quant_signal: str
    sentiment_signal: str
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size: float
    timestamp: datetime
    reasoning: Dict


@dataclass
class BotConfig:
    """Trading bot configuration."""
    # Assets
    assets: List[str] = None
    
    # Trading mode
    paper_trading: bool = True
    mt5_login: int = None
    mt5_password: str = None
    mt5_server: str = None
    
    # Risk
    risk_per_trade: float = 1.0
    max_positions: int = 3
    max_drawdown: float = 10.0
    
    # Signal thresholds
    min_confidence: float = 0.6
    
    # Timeframes
    primary_timeframe: str = 'M15'
    
    def __post_init__(self):
        if self.assets is None:
            self.assets = ['XAUUSD', 'BTCUSD']


class GoldBTCTradingBot:
    """
    Professional Gold/BTC Trading Bot
    
    Architecture:
    1. Data Collection - Fetch OHLCV data from MT5
    2. Technical Analysis - Calculate indicators
    3. Quantitative Signals - Mean reversion, momentum, volatility
    4. Sentiment Analysis - News and social sentiment
    5. Signal Aggregation - Combine all signals with weights
    6. Risk Management - Position sizing, stops
    7. Execution - Place orders via MT5
    
    Decision Matrix:
    - All 3 sources agree (Tech + Quant + Sentiment) → High confidence trade
    - 2 sources agree → Medium confidence trade
    - 1 or 0 sources agree → No trade / Hold
    """
    
    def __init__(self, config: BotConfig = None):
        """
        Initialize trading bot.
        
        Parameters
        ----------
        config : BotConfig
            Bot configuration
        """
        self.config = config or BotConfig()
        
        # Setup logging
        self.logger = logging.getLogger('TradingBot')
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Initialize components
        self._init_broker()
        self._init_risk_manager()
        self._init_sentiment()
        
        # State
        self.positions: Dict[str, Dict] = {}
        self.signal_history: List[TradingSignal] = []
        self.is_running = False
        
        self.logger.info(f"Trading Bot initialized for assets: {self.config.assets}")
        
    def _init_broker(self):
        """Initialize broker connection."""
        if self.config.paper_trading:
            self.broker = PaperTradingBridge(initial_balance=10000)
            self.logger.info("Using Paper Trading mode")
        else:
            self.broker = ExnessMT5Bridge(
                login=self.config.mt5_login,
                password=self.config.mt5_password,
                server=self.config.mt5_server,
                demo=True
            )
            
    def _init_risk_manager(self):
        """Initialize risk manager."""
        params = RiskParameters(
            risk_per_trade=self.config.risk_per_trade,
            max_drawdown_limit=self.config.max_drawdown,
            risk_model=RiskModel.FIXED_PERCENTAGE,
            stop_loss_type=StopLossType.ATR_BASED,
            stop_loss_atr_multiplier=2.0,
            take_profit_risk_reward=2.0,
        )
        
        account_info = self.broker.get_account_info()
        balance = account_info.get('balance', 10000) if account_info else 10000
        
        self.risk_manager = RiskManager(account_balance=balance, params=params)
        
    def _init_sentiment(self):
        """Initialize sentiment analyzer."""
        self.sentiment_analyzer = NewsSentimentSignal()
        
    def connect(self) -> bool:
        """Connect to broker."""
        return self.broker.connect()
    
    def disconnect(self):
        """Disconnect from broker."""
        self.broker.disconnect()
        self.is_running = False
        
    def get_market_data(self, symbol: str, bars: int = 200) -> Optional[pd.DataFrame]:
        """
        Fetch market data.
        
        Parameters
        ----------
        symbol : str
            Trading symbol
        bars : int
            Number of bars to fetch
            
        Returns
        -------
        pd.DataFrame
            OHLCV data
        """
        return self.broker.get_prices(symbol, bars=bars)
    
    def analyze_technicals(self, df: pd.DataFrame) -> Tuple[str, float, Dict]:
        """
        Perform technical analysis.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data
            
        Returns
        -------
        Tuple[str, float, Dict]
            (signal, confidence, details)
        """
        indicators = TechnicalIndicators(df)
        
        # Generate all signals
        signals = indicators.generate_all_signals()
        
        # Get aggregated signal
        signal, confidence = indicators.get_aggregated_signal()
        
        # Detailed analysis
        details = {
            'rsi': indicators.rsi().iloc[-1] if len(df) > 14 else 50,
            'macd_signal': signals[1].signal if len(signals) > 1 else 'NEUTRAL',
            'bollinger_signal': signals[2].signal if len(signals) > 2 else 'NEUTRAL',
            'supertrend': signals[3].signal if len(signals) > 3 else 'NEUTRAL',
            'all_signals': [(s.indicator, s.signal, s.strength.name) for s in signals]
        }
        
        return signal, confidence, details
    
    def analyze_quantitative(self, df: pd.DataFrame) -> Tuple[str, float, Dict]:
        """
        Perform quantitative analysis.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data
            
        Returns
        -------
        Tuple[str, float, Dict]
            (signal, confidence, details)
        """
        quant = QuantitativeSignals(df)
        
        # Generate all signals
        signals = quant.generate_all_signals()
        
        # Get aggregated signal
        signal, confidence, metadata = quant.get_aggregated_signal()
        
        details = {
            'mean_reversion': [s.signal for s in signals if s.signal_type == SignalType.MEAN_REVERSION],
            'momentum': [s.signal for s in signals if s.signal_type == SignalType.MOMENTUM],
            'volatility': [s.signal for s in signals if s.signal_type == SignalType.VOLATILITY_BREAKOUT],
            'market_regime': metadata,
        }
        
        return signal, confidence, details
    
    def analyze_sentiment(self, asset: str) -> Tuple[str, float, Dict]:
        """
        Perform sentiment analysis.
        
        Parameters
        ----------
        asset : str
            Asset symbol
            
        Returns
        -------
        Tuple[str, float, Dict]
            (signal, confidence, details)
        """
        signal, confidence, metadata = self.sentiment_analyzer.get_signal(asset)

        details = {
            'sentiment_score': getattr(metadata.get('sentiment', None), 'sentiment_score', 0) if isinstance(metadata.get('sentiment', None), object) else 0,
            'news_count': metadata.get('news_count', 0),
            'sources': metadata.get('sources', []),
            'economic_events': metadata.get('economic_events', {}),
        }

        return signal, confidence, details
    
    def aggregate_signals(self, 
                         technical: Tuple[str, float, Dict],
                         quantitative: Tuple[str, float, Dict],
                         sentiment: Tuple[str, float, Dict],
                         current_price: float,
                         symbol: str) -> TradingSignal:
        """
        Aggregate all signals into final trading decision.
        
        Parameters
        ----------
        technical : Tuple
            Technical analysis result
        quantitative : Tuple
            Quantitative analysis result
        sentiment : Tuple
            Sentiment analysis result
        current_price : float
            Current market price
        symbol : str
            Trading symbol
            
        Returns
        -------
        TradingSignal
            Final trading signal
        """
        tech_signal, tech_conf, tech_details = technical
        quant_signal, quant_conf, quant_details = quantitative
        sent_signal, sent_conf, sent_details = sentiment
        
        # Weight configuration
        weights = {
            'technical': 0.40,
            'quantitative': 0.35,
            'sentiment': 0.25,
        }
        
        # Convert signals to scores
        signal_map = {'BUY': 1, 'SELL': -1, 'HOLD': 0, 'NEUTRAL': 0}
        
        tech_score = signal_map.get(tech_signal, 0) * tech_conf
        quant_score = signal_map.get(quant_signal, 0) * quant_conf
        sent_score = signal_map.get(sent_signal, 0) * sent_conf
        
        # Weighted average
        total_score = (
            tech_score * weights['technical'] +
            quant_score * weights['quantitative'] +
            sent_score * weights['sentiment']
        )
        
        # Weighted confidence
        total_confidence = (
            tech_conf * weights['technical'] +
            quant_conf * weights['quantitative'] +
            sent_conf * weights['sentiment']
        )
        
        # Determine action
        if total_score > 0.3:
            action = 'BUY'
        elif total_score < -0.3:
            action = 'SELL'
        else:
            action = 'HOLD'
            
        # Check if all sources agree (high conviction)
        sources_agree = (tech_signal == quant_signal == sent_signal) and action != 'HOLD'
        if sources_agree:
            total_confidence = min(total_confidence + 0.2, 1.0)
            
        # Calculate position size and risk levels
        if action != 'HOLD':
            # Get ATR for stops
            indicators = TechnicalIndicators(pd.DataFrame({
                'timestamp': self.broker.get_prices(symbol, bars=100)['time'] if hasattr(self.broker.get_prices(symbol, bars=100), 'time') else [],
            }))
            
            # Simple ATR approximation
            atr = current_price * 0.015

            # Calculate position size
            is_buy = action == 'BUY'
            position_result = self.risk_manager.calculate_position_size(
                symbol=symbol,
                entry_price=current_price,
                atr=atr,
            )
            
            # Adjust stop loss for direction
            if action == 'SELL':
                position_result.stop_loss = self.risk_manager.calculate_stop_loss(
                    entry_price=current_price,
                    atr=atr,
                    is_buy=False
                )
                position_result.take_profit = self.risk_manager.calculate_take_profit(
                    entry_price=current_price,
                    stop_loss=position_result.stop_loss,
                    is_buy=False
                )

            stop_loss = position_result.stop_loss
            take_profit = position_result.take_profit
            position_size = position_result.lots
        else:
            stop_loss = None
            take_profit = None
            position_size = 0
            
        # Reasoning
        reasoning = {
            'technical': {'signal': tech_signal, 'confidence': tech_conf},
            'quantitative': {'signal': quant_signal, 'confidence': quant_conf},
            'sentiment': {'signal': sent_signal, 'confidence': sent_conf},
            'total_score': total_score,
            'sources_agree': sources_agree,
        }
        
        return TradingSignal(
            action=action,
            confidence=total_confidence,
            technical_signal=tech_signal,
            quant_signal=quant_signal,
            sentiment_signal=sent_signal,
            entry_price=current_price if action != 'HOLD' else None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            timestamp=datetime.now(),
            reasoning=reasoning
        )
    
    def execute_trade(self, signal: TradingSignal, symbol: str) -> bool:
        """
        Execute trading signal.
        
        Parameters
        ----------
        signal : TradingSignal
            Trading signal
        symbol : str
            Trading symbol
            
        Returns
        -------
        bool
            Execution success
        """
        if signal.action == 'HOLD':
            return False
            
        # Check risk limits
        allowed, reason = self.risk_manager.check_trading_allowed()
        if not allowed:
            self.logger.warning(f"Trade blocked: {reason}")
            return False
            
        # Check minimum confidence
        if signal.confidence < self.config.min_confidence:
            self.logger.info(f"Signal confidence too low: {signal.confidence:.2f}")
            return False
            
        # Create order
        order = TradeOrder(
            symbol=symbol,
            order_type=OrderType.BUY if signal.action == 'BUY' else OrderType.SELL,
            volume=signal.position_size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            comment=f"Bot: {signal.action} signal",
        )
        
        # Execute
        result = self.broker.place_market_order(order)
        
        if result.success:
            self.positions[symbol] = {
                'symbol': symbol,
                'action': signal.action,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'volume': signal.position_size,
                'timestamp': signal.timestamp,
            }
            
            self.risk_manager.add_position(self.positions[symbol])
            
            self.logger.info(
                f"EXECUTED: {signal.action} {symbol} @ {signal.entry_price:.2f} | "
                f"SL: {signal.stop_loss:.2f} | TP: {signal.take_profit:.2f} | "
                f"Size: {signal.position_size}"
            )
        else:
            self.logger.error(f"Trade execution failed: {result.message}")
            
        return result.success
    
    def manage_positions(self):
        """Manage open positions (trailing stops, take profit)."""
        open_positions = self.broker.get_open_positions()
        
        for pos in open_positions:
            symbol = pos.symbol
            
            # Get current price
            price_info = self.broker.get_current_price(symbol)
            if not price_info:
                continue
                
            current_price = price_info['bid'] if pos.type == 'SELL' else price_info['ask']
            
            # Check if position should be closed
            position_data = self.positions.get(symbol, {})
            
            # Get ATR for trailing stop
            atr = current_price * 0.015
            
            # Calculate trailing stop
            if pos.type == 'BUY':
                trailing_stop = current_price - (atr * 2.0)
                # Lock in profit
                if pos.price_current > pos.price_open * 1.02:
                    trailing_stop = max(trailing_stop, pos.price_open * 1.01)
            else:
                trailing_stop = current_price + (atr * 2.0)
                if pos.price_current < pos.price_open * 0.98:
                    trailing_stop = min(trailing_stop, pos.price_open * 0.99)
                    
            # Update stop loss if better
            if pos.type == 'BUY' and trailing_stop > pos.sl:
                self.broker.modify_stop_loss(symbol, trailing_stop)
            elif pos.type == 'SELL' and trailing_stop < pos.sl:
                self.broker.modify_stop_loss(symbol, trailing_stop)
                
    def run_analysis(self, symbol: str) -> TradingSignal:
        """
        Run complete analysis for a symbol.
        
        Parameters
        ----------
        symbol : str
            Trading symbol
            
        Returns
        -------
        TradingSignal
            Trading signal
        """
        self.logger.info(f"Analyzing {symbol}...")
        
        # Get market data
        df = self.get_market_data(symbol)
        if df is None or len(df) < 50:
            self.logger.warning(f"Insufficient data for {symbol}")
            return TradingSignal(
                action='HOLD',
                confidence=0,
                technical_signal='NEUTRAL',
                quant_signal='NEUTRAL',
                sentiment_signal='NEUTRAL',
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                position_size=0,
                timestamp=datetime.now(),
                reasoning={'error': 'Insufficient data'}
            )
            
        # Get current price
        price_info = self.broker.get_current_price(symbol)
        current_price = price_info['ask'] if price_info else df['close'].iloc[-1]
        
        # Run all analyses
        technical = self.analyze_technicals(df)
        quantitative = self.analyze_quantitative(df)
        
        # Map symbol for sentiment
        asset_map = {'XAUUSD': 'GOLD', 'BTCUSD': 'BTC'}
        asset = asset_map.get(symbol, symbol)
        sentiment = self.analyze_sentiment(asset)
        
        # Aggregate signals
        signal = self.aggregate_signals(
            technical, quantitative, sentiment,
            current_price, symbol
        )
        
        # Log analysis
        self.logger.info(
            f"{symbol}: Tech={technical[0]}({technical[1]:.2f}) | "
            f"Quant={quantitative[0]}({quantitative[1]:.2f}) | "
            f"Sent={sentiment[0]}({sentiment[1]:.2f}) | "
            f"→ {signal.action} ({signal.confidence:.2f})"
        )
        
        self.signal_history.append(signal)
        
        return signal
    
    def run_cycle(self):
        """Run one complete trading cycle."""
        self.logger.info("=" * 50)
        self.logger.info(f"Trading Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 50)
        
        for symbol in self.config.assets:
            # Run analysis
            signal = self.run_analysis(symbol)
            
            # Check if we already have a position
            if symbol in self.positions:
                # Manage existing position
                self.manage_positions()
            else:
                # Execute new signal
                if signal.action in ['BUY', 'SELL']:
                    self.execute_trade(signal, symbol)
                    
        # Update risk manager
        account_info = self.broker.get_account_info()
        if account_info:
            self.risk_manager.update_balance(account_info.get('balance', 0))
            
        # Log summary
        self._log_summary()
        
    def _log_summary(self):
        """Log trading summary."""
        stats = self.risk_manager.get_statistics()
        account = self.broker.get_account_info()
        
        self.logger.info("-" * 50)
        self.logger.info("SUMMARY")
        self.logger.info(f"  Balance: ${account.get('balance', 0):,.2f}")
        self.logger.info(f"  Equity: ${account.get('equity', 0):,.2f}")
        self.logger.info(f"  Open Positions: {len(self.positions)}")
        self.logger.info(f"  Total Trades: {stats.get('total_trades', 0)}")
        self.logger.info(f"  Win Rate: {stats.get('win_rate', 0)*100:.1f}%")
        self.logger.info(f"  Drawdown: {stats.get('max_drawdown', 0):.2f}%")
        self.logger.info("-" * 50)
        
    def start(self, cycle_seconds: int = 300):
        """
        Start the trading bot.
        
        Parameters
        ----------
        cycle_seconds : int
            Seconds between trading cycles
        """
        self.logger.info("Starting Trading Bot...")
        
        if not self.connect():
            self.logger.error("Failed to connect to broker")
            return
            
        self.is_running = True
        
        try:
            while self.is_running:
                self.run_cycle()
                
                # Wait for next cycle
                import time
                for _ in range(cycle_seconds):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
        finally:
            self.disconnect()
            
    def stop(self):
        """Stop the trading bot."""
        self.is_running = False
        self.logger.info("Stopping Trading Bot...")


def create_bot_config(
    paper_trading: bool = True,
    assets: List[str] = None,
    risk_per_trade: float = 1.0,
    max_drawdown: float = 10.0,
    min_confidence: float = 0.6,
) -> BotConfig:
    """
    Factory function to create bot configuration.
    
    Parameters
    ----------
    paper_trading : bool
        Use paper trading mode
    assets : List[str]
        Assets to trade
    risk_per_trade : float
        Risk per trade (%)
    max_drawdown : float
        Maximum drawdown (%)
    min_confidence : float
        Minimum signal confidence
        
    Returns
    -------
    BotConfig
        Bot configuration
    """
    return BotConfig(
        paper_trading=paper_trading,
        assets=assets or ['XAUUSD', 'BTCUSD'],
        risk_per_trade=risk_per_trade,
        max_drawdown=max_drawdown,
        min_confidence=min_confidence,
    )
