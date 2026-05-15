"""
Market Data Service
====================
Fetches real-time market data from Yahoo Finance and other sources.
"""

import asyncio
import os
import sys
import numpy as np
import pandas as pd
import time
from contextlib import redirect_stderr
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import yfinance as yf

# Suppress yfinance's misleading "delisted" stderr spam
_YF_NULL = open(os.devnull, "w")


def _suppress_yf_stderr():
    return redirect_stderr(_YF_NULL)

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
        """
        try:
            yf_symbol = self.get_yf_symbol(symbol)
            with _suppress_yf_stderr():
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

    def _period_to_days(self, period: str) -> int:
        """Convert yfinance period string to approximate days."""
        mapping = {
            "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
            "6mo": 180, "1y": 365, "2y": 730, "5y": 1825, "10y": 3650,
            "ytd": 365, "max": 3650,
        }
        # Handle numeric days like "60d"
        if period.endswith("d") and period[:-1].isdigit():
            return int(period[:-1])
        return mapping.get(period.lower(), 30)

    def _fetch_history_chunked_1m(self, symbol: str, days: int) -> pd.DataFrame:
        """
        Yahoo Finance caps 1m data at ~8 days per request.
        Chunk the window into 7-day slices and concatenate.
        """
        yf_symbol = self.get_yf_symbol(symbol)
        end = datetime.now()
        start = end - timedelta(days=days)
        chunks: List[pd.DataFrame] = []
        cursor = start

        while cursor < end:
            chunk_end = min(cursor + timedelta(days=7, hours=23), end)
            try:
                with _suppress_yf_stderr():
                    df_chunk = yf.download(
                        yf_symbol,
                        start=cursor.strftime("%Y-%m-%d"),
                        end=chunk_end.strftime("%Y-%m-%d"),
                        interval="1m",
                        progress=False,
                        auto_adjust=True,
                        threads=False,
                    )
                if df_chunk is not None and not df_chunk.empty:
                    if isinstance(df_chunk.columns, pd.MultiIndex):
                        df_chunk.columns = [
                            c[0] if isinstance(c, tuple) else c
                            for c in df_chunk.columns
                        ]
                    df_chunk.columns = [c.lower() for c in df_chunk.columns]
                    df_chunk = df_chunk[["open", "high", "low", "close", "volume"]].dropna()
                    df_chunk.index = pd.to_datetime(df_chunk.index, utc=True)
                    chunks.append(df_chunk)
            except Exception as e:
                print(f"  [yf 1m] {symbol} chunk {cursor.date()} failed: {e}")
            cursor = chunk_end
            time.sleep(0.3)

        if not chunks:
            return pd.DataFrame()

        out = pd.concat(chunks)
        out = out[~out.index.duplicated(keep="first")].sort_index()
        out.index.name = "timestamp"
        out = out.reset_index()
        return out

    def fetch_history_range(self, symbol: str, start: datetime,
                            end: datetime, interval: str = "15m") -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a specific date range.
        Automatically chunks 1m requests to respect Yahoo's ~8-day limit.
        """
        try:
            yf_symbol = self.get_yf_symbol(symbol)
            days = (end - start).days

            if interval in ("1m", "1min") and days > 7:
                df = self._fetch_history_chunked_1m(symbol, days)
                if not df.empty:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                    start_utc = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
                    end_utc = end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end
                    mask = (df["timestamp"] >= start_utc) & (df["timestamp"] <= end_utc)
                    df = df.loc[mask]
            else:
                with _suppress_yf_stderr():
                    ticker = yf.Ticker(yf_symbol)
                    df = ticker.history(start=start, end=end, interval=interval)
                df = df.reset_index()

            if df is None or df.empty:
                return self._generate_fallback_data(symbol)

            df.columns = df.columns.str.lower()

            if 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            elif 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                df['timestamp'] = pd.to_datetime(df.iloc[:, 0])

            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    return self._generate_fallback_data(symbol)

            result = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            self.history_cache[symbol] = result
            return result

        except Exception as e:
            print(f"Error fetching history range for {symbol}: {e}")
            return self._generate_fallback_data(symbol)

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

            # Yahoo Finance limits 1m data to ~8 days per request.
            # Use chunked download for longer 1m windows.
            if interval in ("1m", "1min"):
                days = self._period_to_days(period)
                if days > 7:
                    df = self._fetch_history_chunked_1m(symbol, days)
                    if df.empty:
                        print(
                            f"[MarketData] 1m chunking failed for {symbol} "
                            f"(period={period}). Falling back to synthetic data."
                        )
                        return self._generate_fallback_data(symbol)
                else:
                    with _suppress_yf_stderr():
                        ticker = yf.Ticker(yf_symbol)
                        df = ticker.history(period=period, interval=interval)
                    df = df.reset_index()
            else:
                with _suppress_yf_stderr():
                    ticker = yf.Ticker(yf_symbol)
                    df = ticker.history(period=period, interval=interval)
                df = df.reset_index()

            if df is None or df.empty:
                return self._generate_fallback_data(symbol)

            df.columns = df.columns.str.lower()

            # Handle datetime column
            if 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            elif 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
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
            err_msg = str(e).lower()
            if "delisted" in err_msg or "no data found" in err_msg:
                print(
                    f"[MarketData] Yahoo Finance data-limit error for {symbol} "
                    f"(period={period}, interval={interval}). "
                    f"Note: 1m data is capped at ~8 days per request by Yahoo. "
                    f"Use interval='15m' or shorter periods for 1m."
                )
            else:
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
            with _suppress_yf_stderr():
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
            with _suppress_yf_stderr():
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
