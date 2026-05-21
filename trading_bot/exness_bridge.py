"""
Exness MT5 Trading Bridge
==========================
Professional trading execution module for Exness broker via MetaTrader 5.
Supports Gold (XAUUSD), BTCUSD, and other instruments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging


# Try to import MetaTrader5, handle if not available
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None


class OrderType(Enum):
    """Order types for MT5."""
    BUY = "BUY"
    SELL = "SELL"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"


class TimeInForce(Enum):
    """Time in force for orders."""
    GTC = "GTC"  # Good Till Cancel
    DAY = "DAY"  # Day order
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


@dataclass
class TradeOrder:
    """Container for trade order parameters."""
    symbol: str
    order_type: OrderType
    volume: float  # Lot size
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: str = ""
    magic: int = 123456  # EA identifier


@dataclass
class TradeResult:
    """Container for trade execution result."""
    success: bool
    order_id: Optional[int]
    symbol: str
    volume: float
    price: float
    sl: Optional[float]
    tp: Optional[float]
    message: str
    timestamp: datetime


@dataclass
class Position:
    """Container for open position data."""
    symbol: str
    volume: float
    type: str  # 'BUY' or 'SELL'
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float
    swap: float
    commission: float
    time_open: datetime


class ExnessMT5Bridge:
    """
    Exness Broker MT5 Trading Bridge
    
    Provides professional trading execution with:
    - Market and pending orders
    - Position management
    - Risk controls
    - Real-time price feeds
    """
    
    # Supported symbols
    SYMBOLS = {
        'gold': 'XAUUSDm',
        'btc': 'BTCUSD',
        'eth': 'ETHUSD',
        'eurusd': 'EURUSD',
        'gbpusd': 'GBPUSD',
        'usdjpy': 'USDJPY',
    }
    
    def __init__(self, 
                 login: int = None,
                 password: str = None,
                 server: str = None,
                 demo: bool = True):
        """
        Initialize Exness MT5 Bridge.
        
        Parameters
        ----------
        login : int
            MT5 account login
        password : str
            MT5 account password
        server : str
            Exness server name (e.g., 'Exness-MT5Trial')
        demo : bool
            Use demo account
        """
        self.login = login
        self.password = password
        self.server = server
        self.demo = demo
        self.connected = False
        self.positions: Dict[str, Position] = {}
        
        # Setup logging
        self.logger = logging.getLogger('ExnessMT5')
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Trade statistics
        self.trade_history: List[TradeResult] = []
        self.total_trades = 0
        self.winning_trades = 0
        
    def connect(self) -> bool:
        """
        Connect to MT5 terminal.
        
        Returns
        -------
        bool
            Connection success
        """
        if not MT5_AVAILABLE:
            self.logger.error("MetaTrader5 package not installed")
            self.logger.info("Install with: pip install MetaTrader5")
            return False
            
        # Initialize MT5
        if not mt5.initialize():
            self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
            
        # Login if credentials provided
        if self.login and self.password:
            login_result = mt5.login(
                login=self.login,
                password=self.password,
                server=self.server
            )
            
            if not login_result:
                self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False
                
        self.connected = True
        self.logger.info("Connected to MT5")
        
        # Get account info
        account_info = mt5.account_info()
        if account_info:
            self.logger.info(f"Account: {account_info.login}, "
                           f"Balance: ${account_info.balance:.2f}, "
                           f"Server: {account_info.server}")
            
        return True
    
    def disconnect(self):
        """Disconnect from MT5."""
        if self.connected and MT5_AVAILABLE:
            mt5.shutdown()
            self.connected = False
            self.logger.info("Disconnected from MT5")
            
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Get symbol information.
        
        Parameters
        ----------
        symbol : str
            Symbol name (e.g., 'XAUUSD', 'BTCUSD')
            
        Returns
        -------
        Dict with symbol details
        """
        if not self.connected:
            return None
            
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return None
            
        return {
            'name': symbol_info.name,
            'description': symbol_info.description,
            'bid': symbol_info.bid,
            'ask': symbol_info.ask,
            'spread': symbol_info.spread,
            'digits': symbol_info.digits,
            'volume_min': symbol_info.volume_min,
            'volume_max': symbol_info.volume_max,
            'volume_step': symbol_info.volume_step,
            'trade_contract_size': symbol_info.trade_contract_size,
            'trade_tick_value': symbol_info.trade_tick_value,
            'trade_tick_size': symbol_info.trade_tick_size,
        }
    
    def get_prices(self, symbol: str, timeframe: int = None,
                   bars: int = 100) -> Optional[pd.DataFrame]:
        """
        Get historical price data.
        
        Parameters
        ----------
        symbol : str
            Symbol name
        timeframe : int
            MT5 timeframe constant (default: M15)
        bars : int
            Number of bars to retrieve
        """
        if timeframe is None:
            timeframe = 16385 if MT5_AVAILABLE else 15  # M15
            
        if not self.connected:
            return None
            
        # For paper trading or when MT5 not available, generate synthetic data
        if not MT5_AVAILABLE or isinstance(self, PaperTradingBridge):
            return self._generate_synthetic_data(symbol, bars)
            
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        
        if rates is None:
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'
        })

        return df[['time', 'open', 'high', 'low', 'close', 'volume']]

    def _generate_synthetic_data(self, symbol: str, bars: int) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for paper trading/testing.
        
        Parameters
        ----------
        symbol : str
            Symbol name
        bars : int
            Number of bars to generate
            
        Returns
        -------
        pd.DataFrame
            Synthetic OHLCV data
        """
        np = __import__('numpy')
        pd = __import__('pandas')
        
        # Base prices
        base_prices = {
            'XAUUSDm': 2000,
            'BTCUSD': 45000,
            'ETHUSD': 2500,
            'EURUSD': 1.08,
            'GBPUSD': 1.27,
        }
        base_price = base_prices.get(symbol, 100)
        
        # Generate realistic price movements
        np.random.seed(42)
        
        # Daily volatility based on asset
        vol_map = {'XAUUSDm': 0.012, 'BTCUSD': 0.035, 'ETHUSD': 0.045}
        daily_vol = vol_map.get(symbol, 0.01)
        
        # Generate returns with slight upward drift
        returns = np.random.normal(0.0002, daily_vol, bars)
        
        # Generate prices
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC
        data = []
        for i in range(bars):
            close = prices[i]
            daily_range = close * np.random.uniform(0.005, 0.02)
            high = close + np.random.uniform(0, daily_range * 0.7)
            low = close - np.random.uniform(0, daily_range * 0.7)
            open_price = low + np.random.uniform(0.2, 0.8) * (high - low)
            
            data.append({
                'time': pd.Timestamp.now() - pd.Timedelta(days=bars-i),
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=bars-i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.uniform(1000, 10000) * 100,
            })
            
        return pd.DataFrame(data)

    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """
        Get current bid/ask prices.

        Parameters
        ----------
        symbol : str
            Symbol name

        Returns
        -------
        Dict with current prices
        """
        if not self.connected:
            return None

        # For paper trading or when MT5 not available
        if not MT5_AVAILABLE or isinstance(self, PaperTradingBridge):
            base_prices = {
                'XAUUSDm': 2000,
                'BTCUSD': 45000,
                'ETHUSD': 2500,
                'EURUSD': 1.08,
                'GBPUSD': 1.27,
            }
            base = base_prices.get(symbol, 100)
            # Add small random spread
            spread = base * 0.0001
            return {
                'bid': base - spread/2,
                'ask': base + spread/2,
                'last': base,
                'time': datetime.now(),
                'volume': 1000,
            }

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'time': datetime.fromtimestamp(tick.time),
            'volume': tick.volume,
        }
    
    def calculate_lot_size(self, symbol: str, risk_percent: float,
                          stop_loss_pips: float, account_balance: float = None) -> float:
        """
        Calculate position size based on risk.
        
        Parameters
        ----------
        symbol : str
            Symbol name
        risk_percent : float
            Risk as percentage of account (e.g., 1.0 for 1%)
        stop_loss_pips : float
            Stop loss in pips
        account_balance : float
            Account balance (uses MT5 balance if not provided)
            
        Returns
        -------
        float
            Lot size
        """
        if not self.connected:
            return 0.01
            
        # Get account balance
        if account_balance is None:
            account_info = mt5.account_info()
            if account_info:
                account_balance = account_info.balance
            else:
                return 0.01
                
        # Get symbol info
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return 0.01
            
        # Calculate risk amount
        risk_amount = account_balance * (risk_percent / 100)
        
        # Calculate lot size
        tick_value = symbol_info['trade_tick_value']
        tick_size = symbol_info['trade_tick_size']
        
        if tick_size > 0:
            pip_value = tick_value / tick_size * 10  # Approximate
            lot_size = risk_amount / (stop_loss_pips * pip_value)
        else:
            lot_size = 0.01
            
        # Apply symbol constraints
        lot_size = max(symbol_info['volume_min'], 
                      min(symbol_info['volume_max'], lot_size))
        
        # Round to volume step
        step = symbol_info['volume_step']
        lot_size = round(lot_size / step) * step
        
        return lot_size
    
    def place_market_order(self, order: TradeOrder) -> TradeResult:
        """
        Place a market order.
        
        Parameters
        ----------
        order : TradeOrder
            Order parameters
            
        Returns
        -------
        TradeResult
            Execution result
        """
        if not self.connected:
            return TradeResult(
                success=False,
                order_id=None,
                symbol=order.symbol,
                volume=order.volume,
                price=0,
                sl=order.stop_loss,
                tp=order.take_profit,
                message="Not connected to MT5",
                timestamp=datetime.now()
            )
            
        # Get current price
        price_info = self.get_current_price(order.symbol)
        if not price_info:
            return TradeResult(
                success=False,
                order_id=None,
                symbol=order.symbol,
                volume=order.volume,
                price=0,
                sl=order.stop_loss,
                tp=order.take_profit,
                message="Could not get current price",
                timestamp=datetime.now()
            )
            
        # Determine order type and price
        if order.order_type == OrderType.BUY:
            mt5_order_type = mt5.ORDER_TYPE_BUY
            price = price_info['ask']
        elif order.order_type == OrderType.SELL:
            mt5_order_type = mt5.ORDER_TYPE_SELL
            price = price_info['bid']
        else:
            return TradeResult(
                success=False,
                order_id=None,
                symbol=order.symbol,
                volume=order.volume,
                price=0,
                sl=order.stop_loss,
                tp=order.take_profit,
                message="Invalid order type for market order",
                timestamp=datetime.now()
            )
            
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": order.volume,
            "type": mt5_order_type,
            "price": price,
            "sl": order.stop_loss if order.stop_loss else 0,
            "tp": order.take_profit if order.take_profit else 0,
            "deviation": 20,  # Max price deviation in points
            "magic": order.magic,
            "comment": order.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            error_msg = f"Order failed: {mt5.last_error()}"
            success = False
            order_id = None
        elif result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = f"Order failed: {result.comment}"
            success = False
            order_id = None
        else:
            error_msg = "Order placed successfully"
            success = True
            order_id = result.order
            self.total_trades += 1
            if result.profit > 0:
                self.winning_trades += 1
                
        trade_result = TradeResult(
            success=success,
            order_id=order_id,
            symbol=order.symbol,
            volume=order.volume,
            price=price,
            sl=order.stop_loss,
            tp=order.take_profit,
            message=error_msg,
            timestamp=datetime.now()
        )
        
        self.trade_history.append(trade_result)
        self.logger.info(f"Order {order.order_type.value} {order.volume} {order.symbol}: {error_msg}")
        
        return trade_result
    
    def place_pending_order(self, order: TradeOrder) -> TradeResult:
        """
        Place a pending order (limit/stop).
        
        Parameters
        ----------
        order : TradeOrder
            Order parameters with price
            
        Returns
        -------
        TradeResult
            Execution result
        """
        if not self.connected:
            return TradeResult(
                success=False,
                order_id=None,
                symbol=order.symbol,
                volume=order.volume,
                price=0,
                sl=order.stop_loss,
                tp=order.take_profit,
                message="Not connected to MT5",
                timestamp=datetime.now()
            )
            
        # Map order type
        type_map = {
            OrderType.BUY_LIMIT: mt5.ORDER_TYPE_BUY_LIMIT,
            OrderType.SELL_LIMIT: mt5.ORDER_TYPE_SELL_LIMIT,
            OrderType.BUY_STOP: mt5.ORDER_TYPE_BUY_STOP,
            OrderType.SELL_STOP: mt5.ORDER_TYPE_SELL_STOP,
        }
        
        if order.order_type not in type_map:
            return TradeResult(
                success=False,
                order_id=None,
                symbol=order.symbol,
                volume=order.volume,
                price=0,
                sl=order.stop_loss,
                tp=order.take_profit,
                message="Invalid pending order type",
                timestamp=datetime.now()
            )
            
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": order.symbol,
            "volume": order.volume,
            "type": type_map[order.order_type],
            "price": order.price,
            "sl": order.stop_loss if order.stop_loss else 0,
            "tp": order.take_profit if order.take_profit else 0,
            "deviation": 20,
            "magic": order.magic,
            "comment": order.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        success = result.retcode == mt5.TRADE_RETCODE_DONE if result else False
        
        return TradeResult(
            success=success,
            order_id=result.order if result else None,
            symbol=order.symbol,
            volume=order.volume,
            price=order.price,
            sl=order.stop_loss,
            tp=order.take_profit,
            message=result.comment if result else str(mt5.last_error()),
            timestamp=datetime.now()
        )
    
    def close_position(self, symbol: str, volume: float = None) -> TradeResult:
        """
        Close an open position.
        
        Parameters
        ----------
        symbol : str
            Symbol to close
        volume : float
            Volume to close (None for full close)
            
        Returns
        -------
        TradeResult
            Execution result
        """
        if not self.connected:
            return TradeResult(
                success=False,
                order_id=None,
                symbol=symbol,
                volume=0,
                price=0,
                sl=None,
                tp=None,
                message="Not connected to MT5",
                timestamp=datetime.now()
            )
            
        # Get positions
        positions = mt5.positions_get(symbol=symbol)
        
        if not positions:
            return TradeResult(
                success=False,
                order_id=None,
                symbol=symbol,
                volume=0,
                price=0,
                sl=None,
                tp=None,
                message=f"No open position for {symbol}",
                timestamp=datetime.now()
            )
            
        for position in positions:
            # Determine close order type
            if position.type == mt5.POSITION_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:
                close_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
                
            close_volume = volume if volume else position.volume
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": close_volume,
                "type": close_type,
                "position": position.ticket,
                "price": price,
                "deviation": 20,
                "magic": 123456,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                return TradeResult(
                    success=True,
                    order_id=result.order,
                    symbol=symbol,
                    volume=close_volume,
                    price=price,
                    sl=None,
                    tp=None,
                    message=f"Position closed. Profit: {result.profit}",
                    timestamp=datetime.now()
                )
                
        return TradeResult(
            success=False,
            order_id=None,
            symbol=symbol,
            volume=0,
            price=0,
            sl=None,
            tp=None,
            message="Failed to close position",
            timestamp=datetime.now()
        )
    
    def get_open_positions(self) -> List[Position]:
        """
        Get all open positions.
        
        Returns
        -------
        List[Position]
            List of open positions
        """
        if not self.connected:
            return []
            
        positions = mt5.positions_get()
        
        if not positions:
            return []
            
        result = []
        for pos in positions:
            result.append(Position(
                symbol=pos.symbol,
                volume=pos.volume,
                type='BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                price_open=pos.price_open,
                price_current=pos.price_current,
                sl=pos.sl,
                tp=pos.tp,
                profit=pos.profit,
                swap=pos.swap,
                commission=pos.commission,
                time_open=datetime.fromtimestamp(pos.time)
            ))
            
        return result
    
    def get_account_info(self) -> Optional[Dict]:
        """
        Get account information.
        
        Returns
        -------
        Dict with account details
        """
        if not self.connected:
            return None
            
        info = mt5.account_info()
        
        if not info:
            return None
            
        return {
            'login': info.login,
            'server': info.server,
            'balance': info.balance,
            'equity': info.equity,
            'margin': info.margin,
            'margin_free': info.margin_free,
            'margin_level': info.margin_level,
            'profit': info.profit,
            'leverage': info.leverage,
            'currency': info.currency,
        }
    
    def modify_stop_loss(self, symbol: str, new_sl: float) -> bool:
        """
        Modify stop loss for open position.
        
        Parameters
        ----------
        symbol : str
            Symbol to modify
        new_sl : float
            New stop loss price
            
        Returns
        -------
        bool
            Success
        """
        if not self.connected:
            return False
            
        positions = mt5.positions_get(symbol=symbol)
        
        if not positions:
            return False
            
        for pos in positions:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": pos.ticket,
                "sl": new_sl,
                "tp": pos.tp,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                return True
                
        return False
    
    def get_trade_statistics(self) -> Dict:
        """
        Get trading statistics.
        
        Returns
        -------
        Dict with trade statistics
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
            }
            
        winning = sum(1 for t in self.trade_history if t.success)
        total_profit = sum(t.price for t in self.trade_history if t.success)
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'total_profit': total_profit,
            'trade_history': self.trade_history[-10:],  # Last 10 trades
        }


class PaperTradingBridge(ExnessMT5Bridge):
    """
    Paper trading bridge for testing without real money.
    
    Simulates MT5 functionality with virtual trades.
    """
    
    def __init__(self, initial_balance: float = 10000):
        """
        Initialize paper trading.
        
        Parameters
        ----------
        initial_balance : float
            Starting virtual balance
        """
        super().__init__()
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions: Dict[int, Dict] = {}
        self.position_id = 0
        self.connected = True
        self.logger.info(f"Paper trading initialized with ${initial_balance:,.2f}")
        
    def connect(self) -> bool:
        """Connect to paper trading."""
        self.connected = True
        self.logger.info("Paper trading connected")
        return True
        
    def get_account_info(self) -> Dict:
        """Get paper account info."""
        return {
            'login': 999999,
            'server': 'PaperTrading',
            'balance': self.balance,
            'equity': self.equity,
            'margin': 0,
            'margin_free': self.equity,
            'margin_level': 0,
            'profit': sum(p['profit'] for p in self.positions.values()),
            'leverage': 100,
            'currency': 'USD',
        }
        
    def place_market_order(self, order: TradeOrder) -> TradeResult:
        """Place paper market order."""
        # Simulate order execution
        self.position_id += 1
        
        # Get simulated price
        price = 2000 if 'XAU' in order.symbol else 40000 if 'BTC' in order.symbol else 1.0
        
        self.positions[self.position_id] = {
            'symbol': order.symbol,
            'volume': order.volume,
            'type': order.order_type.value,
            'price_open': price,
            'sl': order.stop_loss,
            'tp': order.take_profit,
            'profit': 0,
        }
        
        self.total_trades += 1
        
        return TradeResult(
            success=True,
            order_id=self.position_id,
            symbol=order.symbol,
            volume=order.volume,
            price=price,
            sl=order.stop_loss,
            tp=order.take_profit,
            message="Paper order placed",
            timestamp=datetime.now()
        )
        
    def close_position(self, symbol: str, volume: float = None) -> TradeResult:
        """Close paper position."""
        for pos_id, pos in list(self.positions.items()):
            if pos['symbol'] == symbol:
                # Calculate profit
                current_price = 2000 if 'XAU' in symbol else 40000
                if pos['type'] == 'BUY':
                    profit = (current_price - pos['price_open']) * pos['volume']
                else:
                    profit = (pos['price_open'] - current_price) * pos['volume']
                    
                self.balance += profit
                self.equity = self.balance
                
                del self.positions[pos_id]
                
                return TradeResult(
                    success=True,
                    order_id=pos_id,
                    symbol=symbol,
                    volume=pos['volume'],
                    price=current_price,
                    sl=None,
                    tp=None,
                    message=f"Paper position closed. Profit: ${profit:.2f}",
                    timestamp=datetime.now()
                )
                
        return TradeResult(
            success=False,
            order_id=None,
            symbol=symbol,
            volume=0,
            price=0,
            sl=None,
            tp=None,
            message="No position found",
            timestamp=datetime.now()
        )
        
    def get_open_positions(self) -> List[Position]:
        """Get paper positions."""
        result = []
        for pos_id, pos in self.positions.items():
            current_price = 2000 if 'XAU' in pos['symbol'] else 40000
            if pos['type'] == 'BUY':
                profit = (current_price - pos['price_open']) * pos['volume']
            else:
                profit = (pos['price_open'] - current_price) * pos['volume']
                
            result.append(Position(
                symbol=pos['symbol'],
                volume=pos['volume'],
                type=pos['type'],
                price_open=pos['price_open'],
                price_current=current_price,
                sl=pos['sl'],
                tp=pos['tp'],
                profit=profit,
                swap=0,
                commission=0,
                time_open=datetime.now()
            ))
        return result
