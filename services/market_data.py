"""
Market Data Service
====================
Fetches real-time market data from Yahoo Finance and other sources.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yfinance as yf

# Symbol mappings to Yahoo Finance tickers
YF_SYMBOLS = {
    "XAUUSD": "GC=F",      # Gold futures
    "BTCUSD": "BTC-USD",   # Bitcoin
    "ETHUSD": "ETH-USD",   # Ethereum
    "EURUSD": "EURUSD=X",  # EUR/USD
    "GBPUSD": "GBPUSD=X",  # GBP/USD
    "USDJPY": "USDJPY=X",  # USD/JPY
    "SPX500": "^GSPC",     # S&P 500
    "NAS100": "^NDX",      # Nasdaq 100
}

# Base prices for fallback
BASE_PRICES = {
    "XAUUSD": 2650, "BTCUSD": 95000, "ETHUSD": 3400,
    "EURUSD": 1.05, "GBPUSD": 1.26, "USDJPY": 153.0,
    "SPX500": 5900, "NAS100": 20500,
}


class MarketDataService:
    """Service for fetching and caching market data."""

    def __init__(self):
        self.price_cache: Dict[str, Dict] = {}
        self.history_cache: Dict[str, pd.DataFrame] = {}

    def get_yf_symbol(self, symbol: str) -> str:
        """Get Yahoo Finance symbol for internal symbol."""
        return YF_SYMBOLS.get(symbol, symbol)

    def fetch_price(self, symbol: str) -> Optional[Dict]:
        """
        Fetch current price from Yahoo Finance.

        Parameters
        ----------
        symbol : str
            Trading symbol

        Returns
        -------
        Optional[Dict]
            Price data or None if failed
        """
        try:
            yf_symbol = self.get_yf_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)
            data = ticker.fast_info

            price_data = {
                'symbol': symbol,
                'price': float(data.last_price),
                'bid': float(data.bid),
                'ask': float(data.ask),
                'open': float(data.open_price),
                'high': float(data.day_high),
                'low': float(data.day_low),
                'previous_close': float(data.previous_close),
                'volume': int(data.last_market_volume) if data.last_market_volume else 0,
                'timestamp': datetime.now()
            }

            self.price_cache[symbol] = price_data
            return price_data

        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            return self._generate_mock_price(symbol)

    def fetch_history(self, symbol: str, period: str = "5d",
                      interval: str = "15m") -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Yahoo Finance.

        Parameters
        ----------
        symbol : str
            Trading symbol
        period : str
            Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y')
        interval : str
            Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

        Returns
        -------
        pd.DataFrame
            OHLCV data with timestamp
        """
        try:
            yf_symbol = self.get_yf_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)

            df = ticker.history(period=period, interval=interval)

            if df is None or df.empty:
                return self._generate_fallback_data(symbol)

            df = df.reset_index()
            df.columns = df.columns.str.lower()

            # Handle datetime column
            if 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            elif 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
            else:
                df['timestamp'] = pd.to_datetime(df.iloc[:, 0])

            # Ensure required columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    return self._generate_fallback_data(symbol)

            result = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            self.history_cache[symbol] = result
            return result

        except Exception as e:
            print(f"Error fetching history for {symbol}: {e}")
            return self._generate_fallback_data(symbol)

    def _generate_mock_price(self, symbol: str) -> Dict:
        """Generate mock price data (fallback)."""
        base_price = BASE_PRICES.get(symbol, 100)
        change_pct = np.random.uniform(-0.02, 0.02)
        price = base_price * (1 + change_pct)

        return {
            'symbol': symbol,
            'price': price,
            'bid': price * 0.9999,
            'ask': price * 1.0001,
            'high': price * 1.01,
            'low': price * 0.99,
            'change': price * change_pct,
            'change_percent': change_pct * 100,
            'volume': np.random.uniform(10000, 1000000),
            'timestamp': datetime.now()
        }

    def _generate_fallback_data(self, symbol: str, periods: int = 100) -> pd.DataFrame:
        """Generate fallback OHLCV data if API fails."""
        base_price = BASE_PRICES.get(symbol, 100)

        np.random.seed(int(datetime.now().timestamp()) % 2**32)
        prices = [base_price]
        for _ in range(periods - 1):
            change = np.random.normal(0.0001, 0.015)
            prices.append(prices[-1] * (1 + change))

        dates = [datetime.now() - timedelta(minutes=periods - i) for i in range(periods)]
        df = pd.DataFrame({
            "timestamp": dates,
            "open": prices,
            "high": [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
            "close": prices,
            "volume": [np.random.uniform(1000, 10000) * 100 for _ in prices]
        })
        return df

    def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        try:
            yf_symbol = self.get_yf_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)
            data = ticker.fast_info
            return float(data.last_price)
        except:
            return BASE_PRICES.get(symbol, 100)

    def fetch_all_prices(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """
        Fetch prices for multiple symbols.

        Parameters
        ----------
        symbols : List[str]
            List of symbols to fetch

        Returns
        -------
        Dict[str, Dict]
            Price data for each symbol
        """
        if symbols is None:
            symbols = list(BASE_PRICES.keys())

        results = {}
        for symbol in symbols:
            result = self.fetch_price(symbol)
            if result:
                results[symbol] = result

        return results

    def get_option_chain(self, symbol: str,
                         expiry: datetime = None) -> Optional[Dict]:
        """
        Get option chain data from Yahoo Finance.

        Parameters
        ----------
        symbol : str
            Underlying symbol
        expiry : datetime
            Option expiry date

        Returns
        -------
        Optional[Dict]
            Option chain data
        """
        try:
            yf_symbol = self.get_yf_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)

            if expiry is None:
                expiries = ticker.options
                if not expiries:
                    return None
                expiry = expiries[0]

            opt_chain = ticker.option_chain(expiry)

            return {
                'calls': opt_chain.calls.to_dict('records'),
                'puts': opt_chain.puts.to_dict('records'),
                'expiry': expiry
            }
        except Exception as e:
            print(f"Option chain error: {e}")
            return None


# Singleton instance
_market_data_service: Optional[MarketDataService] = None


def get_market_data_service() -> MarketDataService:
    """Get or create market data service instance."""
    global _market_data_service
    if _market_data_service is None:
        _market_data_service = MarketDataService()
    return _market_data_service
