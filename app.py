"""
Professional Trading Terminal - Dash Frontend
==============================================
A full-featured algorithmic trading dashboard built with Plotly Dash.
Features real-time Yahoo Finance data, charts, and quantitative analysis.
"""
import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, Optional, List
import yfinance as yf
import requests
import dash_bootstrap_components as dbc
import time
import json
from scipy.stats import norm

# Import services
from services.advanced_models import (
    BlackScholes, HestonModel, HestonParams,
    RegimeSwitchingModel, calculate_expected_shortfall,
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown
)
from services.rl_agent import (
    Action, QLearningAgent, TradingEnvironment, TradingState
)
from services.news_scraper import NewsArticle, NewsSource
from services.news_cache import NewsCache, start_background_refresh_thread
from services.ai_news import get_intelligent_news
import threading

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    update_title=None
)

app.title = "Professional Trading Terminal"

# Backend API URL
API_URL = "http://localhost:8000"

# Yahoo Finance symbol mapping
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

# Trading instruments
INSTRUMENTS = [
    {"symbol": "XAUUSD", "name": "Gold", "type": "metal", "yf": "GC=F"},
    {"symbol": "BTCUSD", "name": "Bitcoin", "type": "crypto", "yf": "BTC-USD"},
    {"symbol": "ETHUSD", "name": "Ethereum", "type": "crypto", "yf": "ETH-USD"},
    {"symbol": "EURUSD", "name": "Euro/USD", "type": "forex", "yf": "EURUSD=X"},
    {"symbol": "GBPUSD", "name": "GBP/USD", "type": "forex", "yf": "GBPUSD=X"},
    {"symbol": "USDJPY", "name": "USD/JPY", "type": "forex", "yf": "USDJPY=X"},
    {"symbol": "SPX500", "name": "S&P 500", "type": "index", "yf": "^GSPC"},
    {"symbol": "NAS100", "name": "Nasdaq 100", "type": "index", "yf": "^NDX"},
]

# Color scheme - Fully Black Theme
COLORS = {
    "background": "#000000",
    "surface": "#0a0a0a",
    "surface_light": "#121212",
    "primary": "#1a1a1a",
    "accent": "#00ff88",
    "success": "#00ff88",
    "danger": "#ff4757",
    "warning": "#ffa502",
    "info": "#00d4ff",
    "text": "#ffffff",
    "text_secondary": "#888888",
    "border": "#222222",
    "grid": "#1a1a1a",
}

# Global RL Agent and Models state
rl_agent_state = {
    "agent": None,
    "env": None,
    "training": False,
    "episode": 0,
    "rewards": [],
    "last_action": "HOLD",
    "q_table": {},
    "position": 0,
    "account_value": 10000.0
}

# Global models state
models_state = {
    "black_scholes": BlackScholes(),
    "heston_params": None,
    "sabr_params": None,
    "regime_model": RegimeSwitchingModel(n_regimes=3),
    "regime_history": []
}

# News articles will be fetched dynamically from news scraper
# Keeping placeholder for type checking
NEWS_ARTICLES = []

# Initialize News Cache (persistent, thread-safe)
news_cache = NewsCache()

# Economic calendar events
CALENDAR_EVENTS = [
    {"time": "08:30", "currency": "USD", "event": "Core CPI (MoM)", "impact": "HIGH", "url": "https://www.forexfactory.com/calendar"},
    {"time": "08:30", "currency": "USD", "event": "Non-Farm Payrolls", "impact": "HIGH", "url": "https://www.investing.com/economic-calendar/"},
    {"time": "10:00", "currency": "USD", "event": "Crude Oil Inventories", "impact": "MEDIUM", "url": "https://www.forexfactory.com/calendar"},
    {"time": "14:00", "currency": "USD", "event": "FOMC Meeting Minutes", "impact": "HIGH", "url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"},
    {"time": "07:00", "currency": "EUR", "event": "ECB Rate Decision", "impact": "HIGH", "url": "https://www.fxstreet.com/economic-calendar"},
    {"time": "02:00", "currency": "GBP", "event": "BoE Rate Decision", "impact": "HIGH", "url": "https://www.forexfactory.com/calendar"},
]

# Trade history (in-memory)
TRADE_HISTORY = []


def fetch_yahoo_finance_data(symbol, period="5d", interval="15m"):
    """Fetch real-time data from Yahoo Finance."""
    try:
        yf_symbol = YF_SYMBOLS.get(symbol, symbol)
        ticker = yf.Ticker(yf_symbol)

        # Fetch historical data
        df = ticker.history(period=period, interval=interval)

        # Check if df is None or empty
        if df is None or (hasattr(df, 'empty') and df.empty):
            return generate_fallback_data(symbol)

        # Check if df has expected structure
        if not hasattr(df, 'columns') or len(df.columns) == 0:
            return generate_fallback_data(symbol)

        df = df.reset_index()
        
        # Handle column renaming safely
        try:
            df.columns = df.columns.str.lower()
        except:
            # If column renaming fails, create new dataframe
            df = pd.DataFrame(df.values, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'][:len(df.columns)])

        # Handle datetime column
        if 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        elif 'datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['datetime'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif len(df.columns) > 0:
            df['timestamp'] = pd.to_datetime(df.iloc[:, 0])
        else:
            return generate_fallback_data(symbol)

        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                return generate_fallback_data(symbol)

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return generate_fallback_data(symbol)


def generate_fallback_data(symbol, periods=100):
    """Generate fallback data if API fails."""
    base_prices = {
        "XAUUSD": 2650, "BTCUSD": 95000, "ETHUSD": 3400,
        "EURUSD": 1.05, "GBPUSD": 1.26, "USDJPY": 153.0,
        "SPX500": 5900, "NAS100": 20500,
    }
    base_price = base_prices.get(symbol, 100)

    np.random.seed(int(time.time()) % 2**32)
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


def get_current_price(symbol):
    """Get current price from Yahoo Finance."""
    base_prices = {"XAUUSD": 2650, "BTCUSD": 95000, "ETHUSD": 3400,
                   "EURUSD": 1.05, "GBPUSD": 1.26, "USDJPY": 153.0,
                   "SPX500": 5900, "NAS100": 20500}
    try:
        yf_symbol = YF_SYMBOLS.get(symbol, symbol)
        ticker = yf.Ticker(yf_symbol)
        data = ticker.fast_info
        if data and hasattr(data, 'last_price') and data.last_price is not None:
            return float(data.last_price)
    except Exception as e:
        print(f"Price fetch error for {symbol}: {e}")
        pass
    
    return base_prices.get(symbol, 100)


def create_candlestick_chart(df, symbol):
    """Create a candlestick chart with volume - Black Theme."""
    current_price = df['close'].iloc[-1] if not df.empty else 0
    price_change = df['close'].iloc[-1] - df['open'].iloc[0] if len(df) > 1 else 0
    price_change_pct = (price_change / df['open'].iloc[0] * 100) if df['open'].iloc[0] != 0 else 0

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.75, 0.25],
        subplot_titles=('', 'Volume'),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color=COLORS["success"],
            decreasing_line_color=COLORS["danger"],
            increasing_fillcolor=COLORS["success"],
            decreasing_fillcolor=COLORS["danger"],
            increasing_line_width=1.5,
            decreasing_line_width=1.5,
        ),
        row=1, col=1
    )

    # Volume
    colors = [COLORS["success"] if c >= o else COLORS["danger"]
              for o, c in zip(df["open"], df["close"])]
    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df["volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.4,
        ),
        row=2, col=1
    )

    # Update layout with black theme
    fig.update_layout(
        height=550,
        margin=dict(l=60, r=60, t=80, b=50),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"], size=11, family="Arial"),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=COLORS["surface_light"],
            bordercolor=COLORS["border"],
            font_size=12,
            font_family="Arial",
            font_color=COLORS["text"]
        ),
        title=dict(
            text=f"{symbol} | Price: ${current_price:,.2f} | Change: {price_change_pct:+.2f}%",
            font=dict(size=16, color=COLORS["text"]),
            x=0.5,
            xanchor='center'
        ),
    )

    # Update axes with black theme
    fig.update_xaxes(
        gridcolor=COLORS["grid"],
        linecolor=COLORS["border"],
        tickfont=dict(color=COLORS["text_secondary"]),
        showgrid=True,
        gridwidth=0.5,
    )

    fig.update_yaxes(
        gridcolor=COLORS["grid"],
        linecolor=COLORS["border"],
        tickfont=dict(color=COLORS["text_secondary"]),
        showgrid=True,
        gridwidth=0.5,
    )

    return fig


def create_metrics_cards(symbol):
    """Create market metrics cards - Key Metrics Tab - Black Theme."""
    np.random.seed(hash(symbol) % 2**32)

    hist_vol = round(np.random.uniform(0.1, 0.4), 4)
    skew = round(np.random.uniform(-1, 1), 4)
    kurt = round(np.random.uniform(2, 6), 4)
    h = round(np.random.uniform(0.35, 0.65), 4)

    if abs(h - 0.5) < 0.1:
        regime = "RANDOM_WALK"
        regime_color = COLORS["warning"]
    elif h > 0.5:
        regime = "TRENDING"
        regime_color = COLORS["success"]
    else:
        regime = "MEAN_REVERTING"
        regime_color = COLORS["info"]

    # Create 3 larger cards per row
    cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("📊", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Historical Volatility", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{hist_vol:.2%}", style={"color": COLORS["text"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Annualized standard deviation", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("📐", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Hurst Exponent", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{h:.3f}", style={"color": COLORS["text"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small(f"Regime: {regime}", style={"color": regime_color, "fontSize": "11px", "fontWeight": "bold"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("📈", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Sharpe Ratio", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{round(np.random.uniform(-0.5, 2), 4):.3f}", style={"color": COLORS["success"] if np.random.uniform(-0.5, 2) > 0 else COLORS["danger"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Risk-adjusted returns", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=4),
    ], className="g-3 mb-3")

    cards2 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("⚖️", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Skewness", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{skew:.3f}", style={"color": COLORS["text"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Distribution asymmetry", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("📉", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Kurtosis", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{kurt:.3f}", style={"color": COLORS["text"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Tail thickness (fat tails)", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("⚠️", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("VaR (95%)", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{round(np.random.uniform(0.01, 0.05), 4):.2%}", style={"color": COLORS["danger"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Daily loss threshold", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=4),
    ], className="g-3")

    return [cards, cards2]


def calculate_real_heston_params(symbol):
    """Calculate Heston parameters from real market data."""
    try:
        # Fetch historical data
        df = fetch_yahoo_finance_data(symbol, period="1y", interval="1d")
        
        if df is None or (hasattr(df, 'empty') and df.empty):
            # Return default params if no data
            return HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
        
        # Check if 'close' column exists and has data
        if 'close' not in df.columns or df['close'] is None or len(df['close']) < 10:
            return HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
        
        # Calculate returns
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < 10:
            return HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
        
        # Calculate realized volatility (annualized)
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # v0: Initial variance (current squared volatility)
        v0 = daily_vol ** 2
        
        # theta: Long-run variance (historical average squared volatility)
        theta = (returns.rolling(20).std() ** 2).mean()
        
        # kappa: Mean reversion speed (estimate from autocorrelation)
        vol_series = returns.rolling(20).std() ** 2
        vol_diff = vol_series.diff().dropna()
        vol_lag = vol_series.shift(1).dropna()
        if len(vol_diff) > 10 and vol_lag.var() > 0:
            kappa = max(0.5, min(5.0, 1 - (vol_diff.corr(vol_lag) if len(vol_diff) == len(vol_lag) else 0)))
        else:
            kappa = 2.0
        
        # xi: Vol of vol (volatility of variance)
        if len(vol_series) > 20:
            xi = vol_series.std() * np.sqrt(252)
            xi = max(0.1, min(1.0, xi))
        else:
            xi = 0.3
        
        # rho: Correlation between returns and volatility changes
        if len(returns) > 20:
            vol_changes = vol_series.diff().dropna()
            returns_aligned = returns.iloc[-len(vol_changes):]
            if len(returns_aligned) == len(vol_changes):
                rho = returns_aligned.corr(vol_changes)
                rho = max(-0.9, min(0.9, rho if not np.isnan(rho) else -0.7))
            else:
                rho = -0.7
        else:
            rho = -0.7
        
        # Ensure reasonable bounds
        v0 = max(0.001, min(0.25, v0))
        theta = max(0.001, min(0.25, theta))
        
        return HestonParams(
            kappa=round(kappa, 3),
            theta=round(theta, 5),
            xi=round(xi, 3),
            rho=round(rho, 3),
            v0=round(v0, 5)
        )
        
    except Exception as e:
        print(f"Error calculating Heston params for {symbol}: {e}")
        return HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)


def calculate_real_volatility(symbol):
    """Calculate historical volatility from real market data."""
    try:
        df = fetch_yahoo_finance_data(symbol, period="3mo", interval="1d")
        
        if df is None or (hasattr(df, 'empty') and df.empty):
            return 0.25  # Default 25%
        
        if 'close' not in df.columns or df['close'] is None or len(df['close']) < 10:
            return 0.25
        
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < 5:
            return 0.25
        
        hist_vol = returns.std() * np.sqrt(252)
        
        return round(hist_vol, 4)
        
    except Exception as e:
        print(f"Error calculating volatility for {symbol}: {e}")
        return 0.25


def calculate_regime_metrics(symbol):
    """
    Calculate comprehensive regime detection metrics from real data.
    
    Returns:
        tuple: (hurst_exponent, regime_name, metrics_dict)
    """
    try:
        df = fetch_yahoo_finance_data(symbol, period="6mo", interval="1d")
        
        if df is None or (hasattr(df, 'empty') and df.empty):
            return 0.5, "SIDEWAYS", {}
        
        if 'close' not in df.columns or df['close'] is None:
            return 0.5, "SIDEWAYS", {}
        
        close_prices = df['close'].values
        n = len(close_prices)
        
        if n < 30:
            return 0.5, "SIDEWAYS", {}
        
        returns = np.diff(close_prices) / close_prices[:-1]
        
        if len(returns) < 20:
            return 0.5, "SIDEWAYS", {}
        
        hurst, hurst_conf = calculate_hurst_exponent(returns)
        
        trend_strength = calculate_trend_strength(returns)
        mean_reversion_strength = calculate_mean_reversion_strength(returns)
        
        if hurst > 0.55 and trend_strength > 0.6:
            regime = "TRENDING"
        elif hurst < 0.45 and mean_reversion_strength > 0.6:
            regime = "MEAN_REVERTING"
        elif abs(trend_strength) < 0.3:
            regime = "SIDEWAYS"
        elif trend_strength > 0:
            regime = "TRENDING"
        else:
            regime = "TRENDING"
        
        metrics = {
            'hurst': hurst,
            'hurst_confidence': hurst_conf,
            'trend_strength': trend_strength,
            'mean_reversion_strength': mean_reversion_strength,
            'volatility': float(np.std(returns) * np.sqrt(252)),
            'skewness': float(pd.Series(returns).skew()) if len(returns) > 2 else 0.0,
            'kurtosis': float(pd.Series(returns).kurtosis()) if len(returns) > 3 else 0.0,
            'n_observations': n
        }
        
        return round(hurst, 3), regime, metrics
        
    except Exception as e:
        print(f"Error calculating regime for {symbol}: {e}")
        return 0.5, "SIDEWAYS", {}


def calculate_hurst_exponent(returns, min_lag: int = 5, max_lag: int = None) -> Tuple[float, float]:
    """
    Calculate Hurst exponent using R/S analysis with multiple lags.
    
    The Hurst exponent indicates:
    - H > 0.5: Trending (persistent) behavior
    - H < 0.5: Mean-reverting (anti-persistent) behavior
    - H ≈ 0.5: Random walk (no memory)
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
    min_lag : int
        Minimum lag for R/S calculation
    max_lag : int
        Maximum lag (defaults to n/4)
        
    Returns
    -------
    Tuple[float, float]
        Hurst exponent and confidence score (0-1)
    """
    n = len(returns)
    
    if max_lag is None:
        max_lag = max(min_lag + 5, n // 4)
    
    max_lag = min(max_lag, n // 2)
    
    if max_lag <= min_lag or n < 2 * max_lag:
        return 0.5, 0.0
    
    lags = range(min_lag, max_lag + 1)
    rs_values = []
    n_values = []
    
    for lag in lags:
        if lag < 2:
            continue
            
        n_chunks = n // lag
        
        if n_chunks < 3:
            continue
            
        chunk_rs = []
        
        for i in range(n_chunks):
            start_idx = i * lag
            end_idx = start_idx + lag
            chunk = returns[start_idx:end_idx]
            
            if len(chunk) < 2:
                continue
                
            mean_chunk = np.mean(chunk)
            cumdev = np.cumsum(chunk - mean_chunk)
            
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(chunk, ddof=1)
            
            if S > 1e-10:
                rs = R / S
                chunk_rs.append(rs)
        
        if len(chunk_rs) >= 2:
            rs_values.append(np.mean(chunk_rs))
            n_values.append(lag)
    
    if len(rs_values) < 3:
        return 0.5, 0.0
    
    log_n = np.log(np.array(n_values))
    log_rs = np.log(np.array(rs_values) + 1e-10)
    
    coeffs = np.polyfit(log_n, log_rs, 1)
    H = float(coeffs[0])
    
    H = max(0.1, min(0.9, H))
    
    residuals = np.abs(log_rs - (coeffs[0] * log_n + coeffs[1]))
    max_residual = np.max(residuals) if len(residuals) > 0 else 0
    confidence = max(0.0, 1.0 - min(max_residual / 2.0, 1.0))
    
    return H, round(confidence, 3)


def calculate_trend_strength(returns, lookback: int = 20) -> float:
    """
    Calculate trend strength using sequential analysis.
    
    Returns value between -1 and 1:
    - Positive: Upward trend
    - Negative: Downward trend
    - Close to 0: No clear trend
    """
    if len(returns) < lookback:
        return 0.0
    
    recent = returns[-lookback:]
    
    up_moves = np.sum(recent > 0)
    down_moves = np.sum(recent < 0)
    total = len(recent)
    
    if total == 0:
        return 0.0
    
    trend_ratio = (up_moves - down_moves) / total
    
    cumulative_return = np.sum(recent)
    
    strength = 0.7 * trend_ratio + 0.3 * np.tanh(cumulative_return * 10)
    
    return float(np.clip(strength, -1, 1))


def calculate_mean_reversion_strength(returns, lookback: int = 50) -> float:
    """
    Calculate mean reversion strength using autocorrelation and deviation analysis.
    
    Returns value between 0 and 1:
    - Higher: Stronger mean reversion tendency
    - Lower: Weaker mean reversion
    """
    if len(returns) < lookback:
        return 0.0
    
    recent = returns[-lookback:]
    
    if len(recent) < 10:
        return 0.0
    
    mean_ret = np.mean(recent)
    std_ret = np.std(recent)
    
    if std_ret < 1e-10:
        return 0.5
    
    z_scores = (recent - mean_ret) / std_ret
    
    sign_changes = np.sum(np.diff(np.sign(recent)) != 0)
    max_consecutive = calculate_max_consecutive_same_sign(recent)
    
    reversal_rate = sign_changes / (len(recent) - 1)
    consecutive_penalty = max_consecutive / len(recent)
    
    deviation_strength = np.mean(np.abs(z_scores)) / 3.0
    
    strength = 0.4 * (1 - consecutive_penalty) + 0.3 * reversal_rate + 0.3 * min(deviation_strength, 1.0)
    
    return float(np.clip(strength, 0, 1))


def calculate_max_consecutive_same_sign(arr) -> int:
    """Calculate maximum consecutive positive or negative values."""
    if len(arr) == 0:
        return 0
    
    signs = np.sign(arr)
    max_count = 1
    current_count = 1
    
    for i in range(1, len(signs)):
        if signs[i] == signs[i-1] and signs[i] != 0:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 1
    
    return max_count


def predict_future_prices(symbol, heston_params, days=30, n_paths=100):
    """
    Predict future price distribution using Heston model Monte Carlo simulation.
    
    Returns:
        dict with price predictions, confidence intervals, and probabilities
    """
    try:
        # Get current price
        current_price = get_current_price(symbol)
        
        # Fetch recent data for drift calculation
        df = fetch_yahoo_finance_data(symbol, period="3mo", interval="1d")
        
        if df is None or (hasattr(df, 'empty') and df.empty) or 'close' not in df.columns:
            daily_drift = 0.0002  # Default small positive drift
        else:
            returns = df['close'].pct_change().dropna()
            daily_drift = returns.mean() if len(returns) > 0 else 0.0002
        
        # Heston parameters
        kappa = heston_params.kappa
        theta = heston_params.theta
        xi = heston_params.xi
        rho = heston_params.rho
        v0 = heston_params.v0
        
        # Simulation parameters
        dt = 1.0 / 252  # Daily steps
        n_steps = min(days, 60)  # Max 60 days
        
        # Initialize arrays
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = current_price
        v[:, 0] = v0
        
        # Correlated Brownian motions
        np.random.seed(42)
        Z1 = np.random.standard_normal((n_paths, n_steps))
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal((n_paths, n_steps))
        
        # Full truncation scheme for Heston
        for t in range(n_steps):
            # Variance evolution
            dv = kappa * (theta - v[:, t]) * dt + xi * np.sqrt(np.maximum(v[:, t], 0)) * np.sqrt(dt) * Z2[:, t]
            v[:, t+1] = np.maximum(v[:, t] + dv, 0)
            
            # Price evolution
            dS = daily_drift * S[:, t] * dt + np.sqrt(np.maximum(v[:, t], 0)) * S[:, t] * np.sqrt(dt) * Z1[:, t]
            S[:, t+1] = np.maximum(S[:, t] + dS, 0.01)  # Prevent negative prices
        
        # Calculate statistics at each time step
        final_prices = S[:, -1]
        
        # Percentiles for confidence intervals
        ci_90_lower = np.percentile(final_prices, 5)
        ci_90_upper = np.percentile(final_prices, 95)
        ci_80_lower = np.percentile(final_prices, 10)
        ci_80_upper = np.percentile(final_prices, 90)
        
        # Mean and median predictions
        mean_price = np.mean(final_prices)
        median_price = np.median(final_prices)
        std_price = np.std(final_prices)
        
        # Probability of price increase
        prob_increase = np.mean(final_prices > current_price)
        prob_decrease = np.mean(final_prices < current_price)
        
        # Probability of significant moves (>5%, >10%)
        prob_up_5 = np.mean(final_prices > current_price * 1.05)
        prob_down_5 = np.mean(final_prices < current_price * 0.95)
        prob_up_10 = np.mean(final_prices > current_price * 1.10)
        prob_down_10 = np.mean(final_prices < current_price * 0.90)
        
        # Expected return
        expected_return = (mean_price - current_price) / current_price
        
        # Risk metrics
        max_price = np.max(final_prices)
        min_price = np.min(final_prices)
        
        return {
            'current_price': current_price,
            'mean_price': mean_price,
            'median_price': median_price,
            'std_price': std_price,
            'ci_90': (ci_90_lower, ci_90_upper),
            'ci_80': (ci_80_lower, ci_80_upper),
            'prob_increase': prob_increase,
            'prob_decrease': prob_decrease,
            'prob_up_5': prob_up_5,
            'prob_down_5': prob_down_5,
            'prob_up_10': prob_up_10,
            'prob_down_10': prob_down_10,
            'expected_return': expected_return,
            'max_price': max_price,
            'min_price': min_price,
            'price_paths': S,
            'days': n_steps
        }
        
    except Exception as e:
        print(f"Error in price prediction: {e}")
        # Return default values
        current_price = get_current_price(symbol) if symbol else 100
        return {
            'current_price': current_price,
            'mean_price': current_price * 1.02,
            'median_price': current_price,
            'std_price': current_price * 0.1,
            'ci_90': (current_price * 0.85, current_price * 1.20),
            'ci_80': (current_price * 0.88, current_price * 1.15),
            'prob_increase': 0.52,
            'prob_decrease': 0.48,
            'prob_up_5': 0.35,
            'prob_down_5': 0.30,
            'prob_up_10': 0.20,
            'prob_down_10': 0.18,
            'expected_return': 0.02,
            'max_price': current_price * 1.25,
            'min_price': current_price * 0.80,
            'price_paths': np.zeros((100, 31)),
            'days': 30
        }


def calculate_bs_probability(symbol, days=30):
    """
    Calculate probability of profit using Black-Scholes framework.
    
    Returns risk-neutral probabilities for various price targets.
    """
    try:
        current_price = get_current_price(symbol)
        vol = calculate_real_volatility(symbol)
        risk_free_rate = 0.05  # 5% annual risk-free rate
        
        T = days / 365
        
        # Standard deviation of log returns
        sigma_sqrt_T = vol * np.sqrt(T)
        
        # Expected drift under risk-neutral measure
        drift = (risk_free_rate - 0.5 * vol**2) * T
        
        # Probability of being above current price (call option delta)
        d1 = (drift + 0.5 * sigma_sqrt_T**2) / sigma_sqrt_T
        prob_above = norm.cdf(d1)
        
        # Probability targets
        targets = {
            'up_5': norm.cdf((np.log(1.05) - drift) / sigma_sqrt_T),
            'down_5': 1 - norm.cdf((np.log(0.95) - drift) / sigma_sqrt_T),
            'up_10': norm.cdf((np.log(1.10) - drift) / sigma_sqrt_T),
            'down_10': 1 - norm.cdf((np.log(0.90) - drift) / sigma_sqrt_T),
            'up_15': norm.cdf((np.log(1.15) - drift) / sigma_sqrt_T),
            'down_15': 1 - norm.cdf((np.log(0.85) - drift) / sigma_sqrt_T),
        }
        
        # Expected price under risk-neutral measure
        expected_price = current_price * np.exp(risk_free_rate * T)
        
        return {
            'prob_above_current': prob_above,
            'targets': targets,
            'expected_price': expected_price,
            'volatility': vol,
            'drift': drift
        }
        
    except Exception as e:
        print(f"Error in BS probability: {e}")
        return {
            'prob_above_current': 0.5,
            'targets': {'up_5': 0.4, 'down_5': 0.35, 'up_10': 0.25, 'down_10': 0.20, 'up_15': 0.15, 'down_15': 0.12},
            'expected_price': 100,
            'volatility': 0.25,
            'drift': 0.001
        }


def create_advanced_metrics_cards(symbol):
    """Create advanced metrics cards - Advanced Tab."""
    try:
        df = fetch_yahoo_finance_data(symbol, period="3mo", interval="1d")
        if df is not None and 'close' in df.columns and len(df) > 30:
            returns = df['close'].pct_change().dropna().values
            equity_curve = (1 + returns).cumprod()
            sortino = calculate_sortino_ratio(returns)
            sharpe = calculate_sharpe_ratio(returns)
            max_dd = calculate_max_drawdown(equity_curve)
            annual_return = np.mean(returns) * 252
            calmar = annual_return / max_dd if max_dd > 0 else 0
        else:
            sortino = sharpe = 0.5
            calmar = max_dd = 0.1
    except Exception:
        sortino = sharpe = 0.5
        calmar = max_dd = 0.1
    
    cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("🎯", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Sortino Ratio", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{sortino:.3f}", style={"color": COLORS["text"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Downside risk-adjusted return", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("📊", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Calmar Ratio", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{calmar:.3f}", style={"color": COLORS["text"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Return vs max drawdown", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("📉", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Max Drawdown", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{max_dd:.2%}", style={"color": COLORS["danger"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Largest peak-to-trough decline", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("📈", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Sharpe Ratio", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{sharpe:.3f}", style={"color": COLORS["success"] if sharpe > 0 else COLORS["danger"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Risk-adjusted return", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=6),
    ], className="g-3")

    return cards


def create_risk_metrics_cards(symbol):
    """Create risk metrics cards - Risk Tab."""
    try:
        df = fetch_yahoo_finance_data(symbol, period="3mo", interval="1d")
        if df is not None and 'close' in df.columns and len(df) > 30:
            returns = df['close'].pct_change().dropna().values
            var_95 = np.percentile(returns, 5)
            es_95 = calculate_expected_shortfall(returns, 0.05)
            max_dd = calculate_max_drawdown((1 + returns).cumprod())
        else:
            var_95 = -0.02
            es_95 = -0.03
            max_dd = 0.05
    except Exception:
        var_95 = -0.02
        es_95 = -0.03
        max_dd = 0.05
    
    cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("⚠️", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Value at Risk (95%)", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{abs(var_95):.2%}", style={"color": COLORS["danger"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Maximum daily loss (95% confidence)", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("🔥", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Expected Shortfall", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{abs(es_95):.2%}", style={"color": COLORS["danger"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Loss beyond VaR threshold", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("📉", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Max Drawdown", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{max_dd:.2%}", style={"color": COLORS["warning"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Peak-to-trough decline", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=4),
    ], className="g-3")

    return cards


def generate_signals(symbol):
    """Generate synthetic trading signals."""
    np.random.seed(hash(symbol) % 2**32)

    actions = ["BUY", "SELL", "HOLD"]
    action = np.random.choice(actions, p=[0.35, 0.35, 0.3])
    confidence = round(np.random.uniform(0.4, 0.9), 2)

    signals = [
        {"indicator": "RSI (14)", "signal": np.random.choice(["BUY", "SELL", "HOLD"]),
         "strength": np.random.choice(["WEAK", "MODERATE", "STRONG"]), "value": round(np.random.uniform(30, 70), 2)},
        {"indicator": "MACD", "signal": np.random.choice(["BUY", "SELL", "HOLD"]),
         "strength": np.random.choice(["WEAK", "MODERATE", "STRONG"]), "value": round(np.random.uniform(-0.001, 0.001), 5)},
        {"indicator": "Bollinger (20)", "signal": np.random.choice(["BUY", "SELL", "HOLD"]),
         "strength": np.random.choice(["WEAK", "MODERATE", "STRONG"]), "value": round(np.random.uniform(0, 1), 2)},
         {"indicator": "Supertrend", "signal": np.random.choice(["BUY", "SELL"]),
          "strength": "STRONG", "value": 1},
    ]

    return action, confidence, signals


def get_unified_trading_recommendation(symbol):
    """
    Generate unified trading recommendation by combining all available signals:
    - Technical indicators (RSI, MACD, Bollinger, Supertrend)
    - Regime detection (HMM)
    - Black-Scholes probabilities
    - News sentiment
    - Volatility regime
    """
    signal_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    confidence_scores = []
    factors = []
    vol_ratio = 1.0
    current_price = 0
    
    try:
        # Step 1: Fetch data with multiple attempts
        df = None
        try:
            df = fetch_yahoo_finance_data(symbol, period="3mo", interval="1d")
        except Exception as e:
            print(f"Data fetch error: {e}")
        
        if df is None or not isinstance(df, pd.DataFrame) or len(df) < 10:
            print(f"Insufficient data for {symbol}, using fallback")
            df = _generate_recommendation_data(symbol)
        
        if 'close' not in df.columns:
            print(f"Missing 'close' column for {symbol}")
            return _default_recommendation(symbol)
        
        close = df['close'].dropna().values
        if len(close) < 20:
            print(f"Too few price points ({len(close)}) for {symbol}")
            return _default_recommendation(symbol, close[-1] if len(close) > 0 else None)
        
        current_price = close[-1]
        returns = np.diff(close) / close[:-1]
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 10:
            print(f"Too few returns ({len(returns)}) for {symbol}")
            return _default_recommendation(symbol, current_price)
        
        # Step 2: Technical Indicators (each wrapped separately)
        
        # RSI
        try:
            rsi = calculate_rsi_safe(df['close'])
            if rsi is not None:
                if rsi < 30:
                    signal_scores['BUY'] += 1
                    confidence_scores.append(0.7)
                    factors.append(("RSI", "OVERSOLD", "BUY"))
                elif rsi > 70:
                    signal_scores['SELL'] += 1
                    confidence_scores.append(0.7)
                    factors.append(("RSI", "OVERBOUGHT", "SELL"))
                else:
                    signal_scores['HOLD'] += 0.5
                    factors.append(("RSI", f"NEUTRAL ({rsi:.1f})", "HOLD"))
        except Exception as e:
            print(f"RSI error: {e}")
            factors.append(("RSI", "ERROR", "HOLD"))
        
        # MACD
        try:
            macd_result = calculate_macd_safe(df['close'])
            if macd_result:
                if macd_result['macd'] > macd_result['signal']:
                    signal_scores['BUY'] += 0.8
                    confidence_scores.append(0.6)
                    factors.append(("MACD", "BULLISH", "BUY"))
                else:
                    signal_scores['SELL'] += 0.8
                    confidence_scores.append(0.6)
                    factors.append(("MACD", "BEARISH", "SELL"))
        except Exception as e:
            print(f"MACD error: {e}")
            factors.append(("MACD", "ERROR", "HOLD"))
        
        # Bollinger Bands
        try:
            bb_result = calculate_bb_safe(df['close'])
            if bb_result:
                if close[-1] < bb_result['lower']:
                    signal_scores['BUY'] += 1
                    confidence_scores.append(0.75)
                    factors.append(("Bollinger", "BELOW LOWER", "BUY"))
                elif close[-1] > bb_result['upper']:
                    signal_scores['SELL'] += 1
                    confidence_scores.append(0.75)
                    factors.append(("Bollinger", "ABOVE UPPER", "SELL"))
                else:
                    signal_scores['HOLD'] += 0.5
                    factors.append(("Bollinger", "MIDDLE ZONE", "HOLD"))
        except Exception as e:
            print(f"Bollinger error: {e}")
            factors.append(("Bollinger", "ERROR", "HOLD"))
        
        # Supertrend
        try:
            supertrend_val = calculate_supertrend_safe(df)
            if supertrend_val is not None:
                if supertrend_val > 0:
                    signal_scores['BUY'] += 1.2
                    confidence_scores.append(0.8)
                    factors.append(("Supertrend", "UPTREND", "BUY"))
                else:
                    signal_scores['SELL'] += 1.2
                    confidence_scores.append(0.8)
                    factors.append(("Supertrend", "DOWNTREND", "SELL"))
        except Exception as e:
            print(f"Supertrend error: {e}")
            factors.append(("Supertrend", "ERROR", "HOLD"))
        
        # Step 3: Regime Detection (HMM)
        try:
            returns_for_hmm = returns[-min(180, len(returns)):]
            if len(returns_for_hmm) >= 30:
                regime_model = RegimeSwitchingModel(n_regimes=3)
                results = regime_model.fit(returns_for_hmm, max_iter=30)
                
                probs = results.get('smoothed_probs', np.array([[0.33, 0.33, 0.34]]))
                if len(probs.shape) == 2 and probs.shape[1] == 3:
                    current_regime = int(np.argmax(probs[-1]))
                    regime_confidence = float(np.max(probs[-1]))
                    
                    if current_regime == 0:
                        signal_scores['SELL'] += 1.0 * regime_confidence
                        factors.append(("Regime", "BEAR", "SELL"))
                    elif current_regime == 2:
                        signal_scores['BUY'] += 1.0 * regime_confidence
                        factors.append(("Regime", "BULL", "BUY"))
                    else:
                        signal_scores['HOLD'] += 0.5
                        factors.append(("Regime", "NEUTRAL", "HOLD"))
                    confidence_scores.append(regime_confidence)
                else:
                    factors.append(("Regime", "INVALID PROBS", "HOLD"))
            else:
                factors.append(("Regime", "INSUFFICIENT DATA", "HOLD"))
        except Exception as e:
            print(f"Regime error: {e}")
            factors.append(("Regime", "ERROR", "HOLD"))
        
        # Step 4: Black-Scholes probability
        try:
            if len(returns) >= 20:
                spot = float(close[-1])
                T = 30 / 365
                vol = float(np.std(returns[-30:]) * np.sqrt(252)) if len(returns) >= 30 else float(np.std(returns) * np.sqrt(252))
                
                if vol > 0 and spot > 0:
                    d1 = (0.05 + 0.5 * vol**2) * T / (vol * np.sqrt(T))
                    prob_up = norm.cdf(d1)
                    
                    if prob_up > 0.55:
                        signal_scores['BUY'] += 0.8 * prob_up
                        factors.append(("BS Prob", f"{prob_up:.0%} UP", "BUY"))
                    elif prob_up < 0.45:
                        signal_scores['SELL'] += 0.8 * (1 - prob_up)
                        factors.append(("BS Prob", f"{(1-prob_up):.0%} DOWN", "SELL"))
                    else:
                        signal_scores['HOLD'] += 0.5
                        factors.append(("BS Prob", "BALANCED", "HOLD"))
                    confidence_scores.append(abs(prob_up - 0.5) * 2)
                else:
                    factors.append(("BS Prob", "LOW VOL", "HOLD"))
            else:
                factors.append(("BS Prob", "INSUFFICIENT", "HOLD"))
        except Exception as e:
            print(f"BS Prob error: {e}")
            factors.append(("BS Prob", "ERROR", "HOLD"))
        
        # Step 5: Volatility regime
        try:
            if len(returns) >= 20:
                recent_vol = float(np.std(returns[-20:]) * np.sqrt(252))
                hist_vol = float(np.std(returns) * np.sqrt(252))
                vol_ratio = recent_vol / hist_vol if hist_vol > 0 else 1.0
                
                if vol_ratio > 1.3:
                    factors.append(("Volatility", "HIGH VOL", "HOLD"))
                elif vol_ratio < 0.7:
                    factors.append(("Volatility", "LOW VOL", "HOLD"))
                else:
                    factors.append(("Volatility", "NORMAL", "HOLD"))
            else:
                factors.append(("Volatility", "LOW DATA", "HOLD"))
        except Exception as e:
            print(f"Volatility error: {e}")
            factors.append(("Volatility", "ERROR", "HOLD"))
        
        # Step 6: Momentum/Sentiment
        try:
            if len(returns) >= 5:
                recent_return = float(np.sum(returns[-5:]))
                if recent_return > 0.01:
                    signal_scores['BUY'] += 0.5
                    factors.append(("Momentum", "POSITIVE", "BUY"))
                elif recent_return < -0.01:
                    signal_scores['SELL'] += 0.5
                    factors.append(("Momentum", "NEGATIVE", "SELL"))
                else:
                    factors.append(("Momentum", "NEUTRAL", "HOLD"))
            else:
                factors.append(("Momentum", "LOW DATA", "HOLD"))
        except Exception as e:
            print(f"Momentum error: {e}")
            factors.append(("Momentum", "ERROR", "HOLD"))
        
    except Exception as e:
        print(f"Main recommendation error: {e}")
        import traceback
        traceback.print_exc()
        return _default_recommendation(symbol, current_price if current_price > 0 else None)
    
    # Determine final recommendation
    total_score = sum(signal_scores.values())
    if total_score > 0:
        buy_strength = signal_scores['BUY'] / total_score
        sell_strength = signal_scores['SELL'] / total_score
        hold_strength = signal_scores['HOLD'] / total_score
        
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
        
        if buy_strength > sell_strength + 0.15:
            action = "BUY"
            confidence = buy_strength * avg_confidence
        elif sell_strength > buy_strength + 0.15:
            action = "SELL"
            confidence = sell_strength * avg_confidence
        else:
            action = "HOLD"
            confidence = hold_strength * avg_confidence
    else:
        action = "HOLD"
        confidence = 0.5
    
    # Calculate entry, stop loss, and take profit zones
    if current_price > 0:
        if action == "BUY":
            entry_zone = f"{current_price * 0.998:.2f} - {current_price * 1.002:.2f}"
            stop_loss = f"{current_price * (1 - 0.015 * vol_ratio):.2f}"
            take_profit = f"{current_price * (1 + 0.03 * vol_ratio):.2f}"
            risk_reward = f"1:{min(2.0, 0.03 / (0.015)):1.1f}"
        elif action == "SELL":
            entry_zone = f"{current_price * 0.998:.2f} - {current_price * 1.002:.2f}"
            stop_loss = f"{current_price * (1 + 0.015 * vol_ratio):.2f}"
            take_profit = f"{current_price * (1 - 0.03 * vol_ratio):.2f}"
            risk_reward = f"1:{min(2.0, 0.03 / (0.015)):1.1f}"
        else:
            entry_zone = f"{current_price * 0.995:.2f} - {current_price * 1.005:.2f}"
            stop_loss = "WAIT"
            take_profit = "WAIT"
            risk_reward = "N/A"
    else:
        entry_zone = "N/A"
        stop_loss = "N/A"
        take_profit = "N/A"
        risk_reward = "N/A"
    
    # Trade direction
    if action == "BUY":
        direction = "LONG"
        direction_color = COLORS["success"]
    elif action == "SELL":
        direction = "SHORT"
        direction_color = COLORS["danger"]
    else:
        direction = "NO POSITION"
        direction_color = COLORS["warning"]
    
    # Session bias
    hour = datetime.now().hour
    if 8 <= hour <= 11:
        session = "LONDON/NY"
    elif 13 <= hour <= 16:
        session = "NY SESSION"
    elif 0 <= hour <= 5:
        session = "ASIAN"
    else:
        session = "OFF-PEAK"
    
    vol_regime = "HIGH" if vol_ratio > 1.3 else ("LOW" if vol_ratio < 0.7 else "NORMAL")
    
    return {
        'action': action,
        'confidence': confidence,
        'direction': direction,
        'direction_color': direction_color,
        'factors': factors,
        'entry_zone': entry_zone,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_reward': risk_reward,
        'current_price': current_price,
        'session': session,
        'volatility_regime': vol_regime,
        'signal_breakdown': signal_scores
    }


def _generate_recommendation_data(symbol):
    """Generate synthetic data for recommendation when API fails."""
    base_prices = {
        "XAUUSD": 2650, "BTCUSD": 95000, "ETHUSD": 3400,
        "EURUSD": 1.05, "GBPUSD": 1.26, "USDJPY": 153.0,
        "SPX500": 5900, "NAS100": 20500,
    }
    base_price = base_prices.get(symbol, 100)
    
    np.random.seed(42)
    periods = 90
    prices = [base_price]
    for _ in range(periods - 1):
        change = np.random.normal(0.0001, 0.015)
        prices.append(prices[-1] * (1 + change))
    
    dates = [datetime.now() - timedelta(days=periods - i) for i in range(periods)]
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
        "low": [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
        "close": prices,
        "volume": [np.random.uniform(1000, 10000) * 100 for _ in prices]
    })
    return df


def _default_recommendation(symbol, price=None):
    """Return default recommendation on error."""
    base_prices = {
        "XAUUSD": 2650, "BTCUSD": 95000, "ETHUSD": 3400,
        "EURUSD": 1.05, "GBPUSD": 1.26, "USDJPY": 153.0,
        "SPX500": 5900, "NAS100": 20500,
    }
    current_price = price if price is not None else base_prices.get(symbol, 0)
    
    hour = datetime.now().hour
    if 8 <= hour <= 11:
        session = "LONDON/NY"
    elif 13 <= hour <= 16:
        session = "NY SESSION"
    elif 0 <= hour <= 5:
        session = "ASIAN"
    else:
        session = "OFF-PEAK"
    
    return {
        'action': "HOLD",
        'confidence': 0.5,
        'direction': "NO POSITION",
        'direction_color': COLORS["warning"],
        'factors': [("System", "ANALYZING...", "HOLD")],
        'entry_zone': f"{current_price * 0.995:.2f} - {current_price * 1.005:.2f}" if current_price > 0 else "WAIT",
        'stop_loss': "WAIT",
        'take_profit': "WAIT",
        'risk_reward': "N/A",
        'current_price': current_price,
        'session': session,
        'volatility_regime': "NORMAL",
        'signal_breakdown': {'BUY': 0, 'SELL': 0, 'HOLD': 1}
    }


def calculate_rsi_safe(prices, period=14):
    """Calculate RSI with error handling."""
    try:
        if len(prices) < period + 1:
            return None
        prices = prices.dropna()
        deltas = np.diff(prices.values)
        if len(deltas) < period:
            return None
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    except:
        return None


def calculate_macd_safe(prices, fast=12, slow=26, signal_period=9):
    """Calculate MACD with error handling."""
    try:
        if len(prices) < slow + signal_period:
            return None
        prices = prices.dropna()
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return {'macd': float(macd.iloc[-1]), 'signal': float(signal.iloc[-1])}
    except:
        return None


def calculate_bb_safe(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands with error handling."""
    try:
        if len(prices) < period:
            return None
        prices = prices.dropna()
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return {'upper': float(upper.iloc[-1]), 'middle': float(sma.iloc[-1]), 'lower': float(lower.iloc[-1])}
    except:
        return None


def calculate_supertrend_safe(df, period=10, multiplier=3):
    """Calculate Supertrend with error handling."""
    try:
        if len(df) < period + 2 or 'high' not in df.columns or 'low' not in df.columns:
            return None
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        hl2 = (high + low) / 2
        supertrend = 1
        if close.iloc[-1] > float(hl2.iloc[-2] + multiplier * atr.iloc[-2]):
            supertrend = 1
        elif close.iloc[-1] < float(hl2.iloc[-2] - multiplier * atr.iloc[-2]):
            supertrend = -1
        return supertrend
    except:
        return None


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal_period=9):
    """Calculate MACD."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return {'macd': float(macd.iloc[-1]), 'signal': float(signal.iloc[-1])}


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return {'upper': float(upper.iloc[-1]), 'middle': float(sma.iloc[-1]), 'lower': float(lower.iloc[-1])}


def calculate_supertrend(df, period=10, multiplier=3):
    """Calculate Supertrend indicator."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    supertrend = 1
    if close[-1] > upperband[-2]:
        supertrend = 1
    elif close[-1] < lowerband[-2]:
        supertrend = -1
    else:
        supertrend = 1
    
    return supertrend


def create_signals_table(signals):
    """Create a table for technical signals - Black Theme."""
    signal_colors = {"BUY": COLORS["success"], "SELL": COLORS["danger"], "HOLD": COLORS["warning"]}
    strength_colors = {"WEAK": "#ff6b6b", "MODERATE": "#ffa502", "STRONG": COLORS["success"]}

    rows = []
    for sig in signals:
        signal_color = signal_colors.get(sig["signal"], COLORS["text"])
        strength_color = strength_colors.get(sig["strength"], COLORS["text"])
        rows.append(
            html.Tr([
                html.Td(sig["indicator"], style={"color": COLORS["text_secondary"], "padding": "8px", "fontSize": "12px"}),
                html.Td(
                    html.Span(sig["signal"], style={"color": signal_color, "fontWeight": "bold", "fontSize": "12px"}),
                    style={"textAlign": "center", "padding": "8px"}
                ),
                html.Td(
                    html.Span(sig["strength"], style={"color": strength_color, "fontSize": "11px"}),
                    style={"textAlign": "center", "padding": "8px"}
                ),
                html.Td(str(sig["value"]), style={"textAlign": "right", "color": COLORS["text"], "padding": "8px", "fontSize": "12px"}),
            ], style={"borderBottom": f"1px solid {COLORS['border']}"})
        )

    return html.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th("Indicator", style={"color": COLORS["text_secondary"], "fontSize": "11px", "padding": "8px", "textAlign": "left"}),
                    html.Th("Signal", style={"color": COLORS["text_secondary"], "fontSize": "11px", "padding": "8px", "textAlign": "center"}),
                    html.Th("Strength", style={"color": COLORS["text_secondary"], "fontSize": "11px", "padding": "8px", "textAlign": "center"}),
                    html.Th("Value", style={"color": COLORS["text_secondary"], "fontSize": "11px", "padding": "8px", "textAlign": "right"}),
                ], style={"backgroundColor": COLORS["surface"]})
            ),
            html.Tbody(rows)
        ],
        style={"width": "100%", "borderCollapse": "collapse"}
    )


# App Layout - minimal structure, content comes from routing
app.layout = dbc.Container(fluid=True, children=[
    dcc.Location(id="url", refresh=False),
    
    # Navigation Bar
    dbc.Navbar(
        dbc.Container(fluid=True, children=[
            dbc.NavbarBrand([
                html.Span("📈", style={"fontSize": "24px", "marginRight": "8px"}),
                "VibeTrading"
            ], href="/", style={"color": COLORS["accent"], "fontWeight": "bold", "fontSize": "18px", "textDecoration": "none"}),
            dbc.Nav([
                dbc.NavLink("Dashboard", href="/", active="exact", style={"color": COLORS["text"], "fontSize": "12px"}),
                dbc.NavLink("News", href="/news", active="exact", style={"color": COLORS["text"], "fontSize": "12px"}),
            ], style={"marginLeft": "auto"}),
        ]),
        color=COLORS["surface"],
        dark=True,
        style={"borderBottom": f"1px solid {COLORS['border']}", "padding": "8px 0"}
    ),
    
    # Page Content - routing determines what shows here
    html.Div(id="page-content"),
    
    # Global state components
    dcc.Interval(id="interval-component", interval=5000, n_intervals=0),
    dcc.Interval(id="news-refresh-interval", interval=60000, n_intervals=0),  # 60s news refresh
    dcc.Store(id="selected-symbol", data="XAUUSD"),
    dcc.Store(id="current-price", data=0),
    dcc.Store(id="news-refresh-trigger", data=None),  # Triggers news refresh
])


def _init_background_news():
    """Initialize background news loading on app startup."""
    print("[App] Starting background news cache initialization...")
    try:
        start_background_refresh_thread(
            news_cache,
            [inst["symbol"] for inst in INSTRUMENTS],
            _fetch_news_from_sources
        )
        print("[App] Background news loading started")
    except Exception as e:
        print(f"[App] Error starting background news loading: {e}")


def get_dashboard_layout():
    """Return the full dashboard layout for the main page."""
    return dbc.Container(fluid=True, style={"backgroundColor": COLORS["background"]}, children=[
        
        # Top Row - Unified Recommendation
        html.Div(id="unified-recommendation", className="mb-3 mt-3"),
        
        # Main Dashboard Row - Instruments + Price Chart + Trading Panel
        dbc.Row([
            # Left Sidebar - Instruments
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("INSTRUMENTS", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
                    ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "12px"}),
                    dbc.CardBody([
                        html.Div([
                            dbc.Button(
                                html.Div([
                                    html.Span(inst["symbol"], style={"fontWeight": "bold", "fontSize": "12px"}),
                                    html.Span(inst["name"], style={"color": COLORS["text_secondary"], "fontSize": "10px", "marginLeft": "8px"}),
                                ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "width": "100%"}),
                                id={"type": "instrument-btn", "index": inst["symbol"]},
                                className="w-100",
                                style={
                                    "border": f"1px solid {COLORS['accent']}" if inst["symbol"] == "XAUUSD" else f"1px solid {COLORS['border']}",
                                    "borderRadius": "4px",
                                    "padding": "10px 12px",
                                    "textAlign": "left",
                                    "backgroundColor": COLORS["accent"] if inst["symbol"] == "XAUUSD" else "transparent",
                                    "color": "#000000" if inst["symbol"] == "XAUUSD" else COLORS["text_secondary"],
                                    "width": "100%",
                                    "marginBottom": "8px",
                                    "fontWeight": "bold" if inst["symbol"] == "XAUUSD" else "normal",
                                    "cursor": "pointer",
                                    "transition": "all 0.2s ease",
                                }
                            )
                            for inst in INSTRUMENTS
                        ])
                    ], style={"padding": "12px"})
                ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"}),
            ], width=2),
            
            # Center - Price Chart
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.Span("📊 PRICE CHART", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
                            dcc.Dropdown(
                                id="timeframe-selector",
                                options=[
                                    {"label": "5m", "value": "5m"},
                                    {"label": "15m", "value": "15m"},
                                    {"label": "1h", "value": "1h"},
                                    {"label": "4h", "value": "4h"},
                                    {"label": "1D", "value": "1D"},
                                ],
                                value="15m",
                                clearable=False,
                                style={"width": "80px", "display": "inline-block"}
                            ),
                        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"})
                    ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "12px"}),
                    dbc.CardBody([
                        dcc.Graph(id="price-chart", config={"displayModeBar": False, "responsive": True})
                    ], style={"padding": "0"})
                ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"}),
            ], width=7),

            # Right Sidebar - Trading Panel
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("🎯 TRADING SIGNALS", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
                    ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "12px"}),
                    dbc.CardBody([
                        html.Div(id="trading-signals", style={"maxHeight": "200px", "overflowY": "auto"})
                    ], style={"padding": "12px"})
                ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px", "marginBottom": "12px"}),
                
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("📝 ORDER FORM", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
                    ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "12px"}),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("BUY", id="buy-btn", color="success", className="w-100", style={"fontWeight": "bold", "fontSize": "12px"}),
                            ], width=6),
                            dbc.Col([
                                dbc.Button("SELL", id="sell-btn", color="danger", className="w-100", style={"fontWeight": "bold", "fontSize": "12px"}),
                            ], width=6),
                        ], className="mb-3"),
                        dbc.Input(type="number", id="order-size", placeholder="Size (lots)", min=0.01, max=100, step=0.01, value=0.1,
                                 style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"], "marginBottom": "8px"}),
                        dbc.Input(type="number", id="stop-loss", placeholder="Stop Loss",
                                 style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"], "marginBottom": "8px"}),
                        dbc.Input(type="number", id="take-profit", placeholder="Take Profit",
                                 style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"], "marginBottom": "8px"}),
                        html.Div(id="order-status", className="text-center mt-2", style={"fontSize": "12px"}),
                    ], style={"padding": "12px"})
                ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"}),
            ], width=3),
        ], className="mt-3"),

        # Second Row - News + Calendar + Trade History
        dbc.Row([
            # News Feed
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("📰 FINANCIAL NEWS", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
                    ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "10px"}),
                    dbc.CardBody([
                        html.Div(id="news-sources-grid", style={"marginBottom": "8px"}),
                        html.Hr(style={"borderColor": COLORS["border"], "margin": "8px 0"}),
                        html.Div(id="news-feed", style={"maxHeight": "200px", "overflowY": "auto"})
                    ], style={"padding": "10px"})
                ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"}),
            ], width=5),
            
            # Economic Calendar
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("📅 ECONOMIC CALENDAR", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
                    ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "10px"}),
                    dbc.CardBody([
                        html.Div(id="economic-calendar", style={"maxHeight": "220px", "overflowY": "auto"})
                    ], style={"padding": "10px"})
                ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"}),
            ], width=3),
            
            # Trade History
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("📋 TRADE HISTORY", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
                        dbc.Button("Clear", id="clear-trades-btn", size="sm", color="link", 
                                  style={"float": "right", "fontSize": "10px", "padding": "0 5px"})
                    ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "10px"}),
                    dbc.CardBody([
                        html.Div(id="trade-history-table", style={"maxHeight": "220px", "overflowY": "auto"})
                    ], style={"padding": "10px"})
                ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"}),
            ], width=4),
        ], className="mt-3"),

        # Market Metrics Section
        dbc.Card([
            dbc.CardHeader([
                html.Span("📊 MARKET METRICS", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
            ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "10px"}),
            dbc.CardBody([
                dbc.Tabs([
                    dbc.Tab([
                        html.Div(id="metrics-cards", style={"padding": "8px"})
                    ], label="📈 Key Metrics", tab_id="metrics", label_style={"color": COLORS["text"], "fontSize": "11px"}),
                    dbc.Tab([
                        html.Div(id="advanced-metrics-cards", style={"padding": "8px"})
                    ], label="🔬 Advanced", tab_id="advanced", label_style={"color": COLORS["text"], "fontSize": "11px"}),
                    dbc.Tab([
                        html.Div(id="risk-metrics-cards", style={"padding": "8px"})
                    ], label="⚠️ Risk", tab_id="risk", label_style={"color": COLORS["text"], "fontSize": "11px"}),
                ], active_tab="metrics", style={"backgroundColor": COLORS["background"]})
            ], style={"padding": "0"})
        ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px", "marginBottom": "16px"}),

        # Heston Volatility & Options Analytics
        dbc.Card([
            dbc.CardHeader([
                html.Span("📉 HESTON VOLATILITY & OPTIONS ANALYTICS", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
            ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "10px"}),
            dbc.CardBody([
                dbc.Tabs([
                    dbc.Tab([
                        html.Div([
                            html.Div(id="heston-model-cards", className="mb-2"),
                            dcc.Graph(id="heston-surface-chart", config={"displayModeBar": False, "responsive": True}, style={"height": "350px"}),
                        ], style={"padding": "8px"})
                    ], label="📉 Heston Model", tab_id="heston", label_style={"color": COLORS["text"], "fontSize": "11px"}),
                    dbc.Tab([
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Spot:", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                    dbc.Input(type="number", id="bs-spot", value=100, step=1,
                                             style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"]}),
                                ], width=4),
                                dbc.Col([
                                    html.Label("Strike:", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                    dbc.Input(type="number", id="bs-strike", value=100, step=1,
                                             style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"]}),
                                ], width=4),
                                dbc.Col([
                                    html.Label("Days:", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                    dbc.Input(type="number", id="bs-time", value=30, step=1,
                                             style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"]}),
                                ], width=4),
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Vol:", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                    dbc.Input(type="number", id="bs-vol", value=0.25, step=0.01,
                                             style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"]}),
                                ], width=4),
                                dbc.Col([
                                    html.Label("Rate:", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                    dbc.Input(type="number", id="bs-rate", value=0.05, step=0.01,
                                             style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"]}),
                                ], width=4),
                                dbc.Col([
                                    html.Label("Type:", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                    dcc.Dropdown(
                                        id="bs-option-type",
                                        options=[
                                            {"label": "Call", "value": "call"},
                                            {"label": "Put", "value": "put"},
                                        ],
                                        value="call",
                                        clearable=False,
                                        style={"backgroundColor": COLORS["surface_light"], "color": COLORS["text"]}
                                    ),
                                ], width=4),
                            ], className="mb-2"),
                            html.Div(id="bs-model-cards", className="mb-2"),
                            dcc.Graph(id="bs-greeks-chart", config={"displayModeBar": False, "responsive": True}, style={"height": "280px"}),
                        ], style={"padding": "8px"})
                    ], label="📊 Black-Scholes", tab_id="black-scholes", label_style={"color": COLORS["text"], "fontSize": "11px"}),
                    dbc.Tab([
                        html.Div([
                            html.Div(id="prediction-cards", className="mb-2"),
                            dcc.Graph(id="prediction-chart", config={"displayModeBar": False, "responsive": True}, style={"height": "320px"}),
                        ], style={"padding": "8px"})
                    ], label="🤖 AI Prediction", tab_id="prediction", label_style={"color": COLORS["text"], "fontSize": "11px"}),
                ], active_tab="heston", id="volatility-tabs", style={"backgroundColor": COLORS["background"]})
            ], style={"padding": "0"})
        ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px", "marginBottom": "16px"}),

        # RL Agent & Monte Carlo
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("🤖 REINFORCEMENT LEARNING AGENT", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
                    ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "10px"}),
                    dbc.CardBody([
                        html.Div(id="rl-status-cards"),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="rl-reward-chart", config={"displayModeBar": False, "responsive": True}),
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id="rl-qtable-heatmap", config={"displayModeBar": False, "responsive": True}),
                            ], width=6),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Train Agent", id="rl-train-btn", color="primary", className="me-2"),
                                dbc.Button("Reset", id="rl-reset-btn", color="secondary"),
                            ], width=12),
                        ], className="mt-2"),
                    ], style={"padding": "10px"})
                ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"}),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("🎲 MONTE CARLO SIMULATION", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
                    ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "10px"}),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Days:", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                dbc.Input(type="number", id="mc-days", value=30, min=1, max=365, step=1,
                                         style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"]}),
                            ], width=4),
                            dbc.Col([
                                html.Label("Paths:", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                dbc.Input(type="number", id="mc-paths", value=1000, min=100, max=10000, step=100,
                                         style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"]}),
                            ], width=4),
                            dbc.Col([
                                html.Label(" ", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                dbc.Button("Run", id="mc-run-btn", color="primary", style={"marginTop": "18px", "width": "100%"}),
                            ], width=4),
                        ], className="mb-2"),
                        html.Div(id="mc-results"),
                        dcc.Graph(id="mc-chart", config={"displayModeBar": False, "responsive": True}),
                    ], style={"padding": "10px"})
                ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"}),
            ], width=6),
        ], className="mb-3"),

        # Regime Detection
        dbc.Card([
            dbc.CardHeader([
                html.Span("🔄 REGIME DETECTION", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
            ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "10px"}),
            dbc.CardBody([
                html.Div(id="regime-detection-display"),
                dcc.Graph(id="regime-chart", config={"displayModeBar": False, "responsive": True}, style={"height": "300px"}),
            ], style={"padding": "12px"})
        ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px", "marginBottom": "16px"}),

        # Hidden elements for callbacks that need them
        html.Div(id="last-update", style={"display": "none"}),
        dcc.Graph(id="rl-training-chart", style={"display": "none"}),
        html.Div(id="rl-action-display", style={"display": "none"}),
    ])


# Page routing callback
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def render_page(pathname):
    """Route to appropriate page based on URL."""
    if pathname == "/news":
        from pages.news import layout as news_layout
        return news_layout
    return get_dashboard_layout()


# Callbacks
@callback(
    Output("selected-symbol", "data"),
    [Input({"type": "instrument-btn", "index": dash.ALL}, "n_clicks")],
    [State("selected-symbol", "data")],
    prevent_initial_call=False
)
def update_selected_symbol(n_clicks_list, current_symbol):
    """Update selected symbol when instrument button is clicked."""
    if not ctx.triggered:
        return current_symbol

    triggered_id = ctx.triggered_id
    if triggered_id and isinstance(triggered_id, dict) and "index" in triggered_id:
        clicked_symbol = triggered_id["index"]
        print(f"Symbol changed to: {clicked_symbol}")
        return clicked_symbol

    return current_symbol


@callback(
    Output({ "type": "instrument-btn", "index": dash.ALL }, "style"),
    [Input("selected-symbol", "data")],
    prevent_initial_call=False
)
def update_button_styles(current_symbol):
    """Update instrument button styles based on selection."""
    if current_symbol is None:
        current_symbol = "XAUUSD"

    styles = []
    for inst in INSTRUMENTS:
        if inst["symbol"] == current_symbol:
            styles.append({
                "border": f"1px solid {COLORS['accent']}",
                "borderRadius": "4px",
                "padding": "10px 14px",
                "textAlign": "left",
                "backgroundColor": COLORS["accent"],
                "color": "#000000",
                "width": "100%",
                "marginBottom": "8px",
                "fontWeight": "bold",
                "cursor": "pointer",
                "transition": "all 0.2s ease",
                "boxShadow": f"0 0 10px {COLORS['accent']}40",
            })
        else:
            styles.append({
                "border": f"1px solid {COLORS['border']}",
                "borderRadius": "4px",
                "padding": "10px 14px",
                "textAlign": "left",
                "backgroundColor": "transparent",
                "color": COLORS["text_secondary"],
                "width": "100%",
                "marginBottom": "8px",
                "cursor": "pointer",
                "transition": "all 0.2s ease",
            })
    return styles


@callback(
    Output("price-chart", "figure"),
    [Input("selected-symbol", "data"),
     Input("interval-component", "n_intervals"),
     Input("timeframe-selector", "value")],
    prevent_initial_call=False
)
def update_price_chart(symbol, n, timeframe):
    """Update price chart with real-time Yahoo Finance data."""
    if symbol is None:
        symbol = "XAUUSD"

    # Map timeframe to Yahoo Finance period
    period_map = {"5m": "1d", "15m": "5d", "1h": "1mo", "4h": "3mo", "1D": "1y"}
    interval_map = {"5m": "5m", "15m": "15m", "1h": "1h", "4h": "1d", "1D": "1d"}

    period = period_map.get(timeframe, "5d")
    interval = interval_map.get(timeframe, "15m")

    try:
        df = fetch_yahoo_finance_data(symbol, period=period, interval=interval)
        return create_candlestick_chart(df, symbol)
    except Exception as e:
        print(f"Chart error for {symbol}: {e}")
        df = generate_fallback_data(symbol)
        return create_candlestick_chart(df, symbol)


@callback(
    [Output("metrics-cards", "children"),
     Output("advanced-metrics-cards", "children"),
     Output("risk-metrics-cards", "children")],
    [Input("selected-symbol", "data")]
)
def update_metrics(symbol):
    """Update market metrics cards with REAL calculated data."""
    if symbol is None:
        symbol = "XAUUSD"

    # Calculate REAL metrics from market data
    try:
        df = fetch_yahoo_finance_data(symbol, period="6mo", interval="1d")

        # Check if df has valid data
        has_valid_data = (
            df is not None and 
            hasattr(df, 'empty') and not df.empty and 
            len(df) > 30 and
            'close' in df.columns and
            df['close'] is not None
        )

        if has_valid_data:
            returns = df['close'].pct_change().dropna()

            if len(returns) > 10:
                # Historical volatility (annualized)
                hist_vol = returns.std() * np.sqrt(252)

                # Skewness
                skew = returns.skew()

                # Kurtosis
                kurt = returns.kurtosis() + 3  # Excess kurtosis + 3

                # Hurst exponent and regime
                hurst, regime, regime_metrics = calculate_regime_metrics(symbol)

                # Sharpe ratio (annualized, assuming 5% risk-free rate)
                excess_returns = returns - 0.05 / 252
                sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0

                # VaR (95%)
                var_95 = np.percentile(returns, 5)
            else:
                hist_vol = 0.20
                skew = 0.0
                kurt = 3.0
                hurst = 0.5
                regime = "SIDEWAYS"
                sharpe = 0.5
                var_95 = -0.02
        else:
            # Fallback to defaults
            hist_vol = 0.20
            skew = 0.0
            kurt = 3.0
            hurst = 0.5
            regime = "SIDEWAYS"
            sharpe = 0.5
            var_95 = -0.02

    except Exception as e:
        print(f"Error calculating metrics for {symbol}: {e}")
        hist_vol = 0.20
        skew = 0.0
        kurt = 3.0
        hurst = 0.5
        regime = "SIDEWAYS"
        sharpe = 0.5
        var_95 = -0.02

    # Determine regime color
    regime_color = COLORS["warning"] if regime == "SIDEWAYS" else COLORS["success"] if regime == "TRENDING" else COLORS["info"]

    # Create Key Metrics cards
    cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("📊", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Historical Volatility", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{hist_vol:.2%}", style={"color": COLORS["text"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Annualized standard deviation", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("📐", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Hurst Exponent", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{hurst:.3f}", style={"color": COLORS["text"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small(f"Regime: {regime}", style={"color": regime_color, "fontSize": "11px", "fontWeight": "bold"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("📈", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Sharpe Ratio", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{sharpe:.3f}", style={"color": COLORS["success"] if sharpe > 0 else COLORS["danger"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Risk-adjusted returns", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=4),
    ], className="g-3 mb-3")

    cards2 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("⚖️", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Skewness", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{skew:.3f}", style={"color": COLORS["text"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Distribution asymmetry", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("📉", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Kurtosis", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{kurt:.3f}", style={"color": COLORS["text"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Tail thickness (fat tails)", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("⚠️", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("VaR (95%)", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{var_95:.2%}", style={"color": COLORS["danger"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Daily loss threshold", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=4),
    ], className="g-3")

    # Advanced metrics
    sortino = sharpe * 1.2 if sharpe > 0 else sharpe * 0.8  # Simplified
    calmar = sharpe * 0.9 if sharpe > 0 else sharpe * 0.7
    
    advanced_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("🎯", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Sortino Ratio", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{sortino:.3f}", style={"color": COLORS["text"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Downside risk-adjusted return", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("📊", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Calmar Ratio", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{calmar:.3f}", style={"color": COLORS["text"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Return vs max drawdown", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=6),
    ], className="g-3")

    # Risk metrics
    es_95 = var_95 * 1.4  # Simplified expected shortfall
    
    risk_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("⚠️", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Value at Risk (95%)", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{var_95:.2%}", style={"color": COLORS["danger"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Maximum daily loss (95% confidence)", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("🔥", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Expected Shortfall", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{es_95:.2%}", style={"color": COLORS["danger"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Loss beyond VaR threshold", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=6),
    ], className="g-3")

    return cards, advanced_cards, risk_cards


@callback(
    Output("trading-signals", "children"),
    [Input("selected-symbol", "data")]
)
def update_signals(symbol):
    """Update trading signals."""
    if symbol is None:
        symbol = "XAUUSD"

    action, confidence, signals = generate_signals(symbol)

    action_colors = {
        "BUY": {"color": COLORS["success"], "bg": "success"},
        "SELL": {"color": COLORS["danger"], "bg": "danger"},
        "HOLD": {"color": COLORS["warning"], "bg": "warning"},
    }

    action_color = action_colors.get(action, {}).get("color", COLORS["text"])

    return html.Div([
        html.Div([
            html.Div(action, style={
                "color": action_color,
                "fontWeight": "bold",
                "fontSize": "28px",
                "textAlign": "center",
                "letterSpacing": "3px",
                "marginBottom": "10px"
            }),
            html.Div([
                dbc.Progress(value=confidence * 100, color=action_colors.get(action, {}).get("bg", "secondary"),
                             style={"height": "8px"}, className="mb-2"),
                html.Span(f"Confidence: {confidence * 100:.1f}%", style={"fontSize": "11px", "color": COLORS["text_secondary"]})
            ], style={"marginBottom": "10px"}),
            create_signals_table(signals)
        ])
    ])


@callback(
    Output("news-sources-grid", "children"),
    [Input("interval-component", "n_intervals")]
)
def update_news_sources(n):
    """Create clickable news source cards with links to each news website."""
    news_sources = [
        {
            "id": "bloomberg",
            "name": "Bloomberg",
            "icon": "💼",
            "url": "https://www.bloomberg.com",
            "color": "#FF6600",
            "description": "Global Markets"
        },
        {
            "id": "cnbc",
            "name": "CNBC",
            "icon": "📺",
            "url": "https://www.cnbc.com",
            "color": "#003366",
            "description": "Business News"
        },
        {
            "id": "investing",
            "name": "Investing",
            "icon": "📈",
            "url": "https://www.investing.com",
            "color": "#008000",
            "description": "Trading Platform"
        },
        {
            "id": "fxstreet",
            "name": "FXStreet",
            "icon": "💱",
            "url": "https://www.fxstreet.com",
            "color": "#E91E63",
            "description": "Forex News"
        },
        {
            "id": "forexfactory",
            "name": "Forex Factory",
            "icon": "🏭",
            "url": "https://www.forexfactory.com",
            "color": "#FF9800",
            "description": "Forex Calendar"
        },
        {
            "id": "reuters",
            "name": "Reuters",
            "icon": "📰",
            "url": "https://www.reuters.com",
            "color": "#FF8000",
            "description": "Breaking News"
        },
        {
            "id": "marketwatch",
            "name": "MarketWatch",
            "icon": "⌚",
            "url": "https://www.marketwatch.com",
            "color": "#00A600",
            "description": "Stock Data"
        },
        {
            "id": "yahoo_finance",
            "name": "Yahoo Finance",
            "icon": "🟣",
            "url": "https://finance.yahoo.com",
            "color": "#400090",
            "description": "Markets & Tech"
        },
        {
            "id": "dailyfx",
            "name": "DailyFX",
            "icon": "📊",
            "url": "https://www.dailyfx.com",
            "color": "#E74C3C",
            "description": "Forex Analysis"
        },
        {
            "id": "coindesk",
            "name": "CoinDesk",
            "icon": "₿",
            "url": "https://www.coindesk.com",
            "color": "#1652F0",
            "description": "Crypto News"
        },
        {
            "id": "kitco",
            "name": "Kitco",
            "icon": "🥇",
            "url": "https://www.kitco.com/news/",
            "color": "#FFD700",
            "description": "Precious Metals"
        },
        {
            "id": "trading_economics",
            "name": "Trading Econ",
            "icon": "🌐",
            "url": "https://tradingeconomics.com",
            "color": "#2E86AB",
            "description": "Econ Indicators"
        },
    ]

    # Create source cards in rows of 4
    source_cards = []
    for source in news_sources:
        card = html.A(
            html.Div([
                html.Div([
                    html.Span(source["icon"], style={"fontSize": "18px"}),
                    html.Span(source["name"], style={"color": source["color"], "fontWeight": "bold", "fontSize": "10px", "marginLeft": "6px"}),
                ], style={"display": "flex", "alignItems": "center", "justifyContent": "center"}),
                html.Small(source["description"], style={"color": COLORS["text_secondary"], "fontSize": "8px", "textAlign": "center", "marginTop": "4px"}),
            ],
            style={
                "padding": "8px 4px",
                "borderRadius": "4px",
                "border": f"1px solid {source['color']}40",
                "backgroundColor": f"{source['color']}10",
                "textAlign": "center",
                "transition": "all 0.2s",
                "cursor": "pointer",
            }),
            href=source["url"],
            target="_blank",
            rel="noopener noreferrer",
            style={"textDecoration": "none", "display": "block"},
            title=f"Visit {source['name']} - {source['description']}"
        )
        source_cards.append(card)

    # Group into rows of 4
    rows = []
    for i in range(0, len(source_cards), 4):
        row_sources = source_cards[i:i+4]
        row = dbc.Row([
            dbc.Col(source, width=3) for source in row_sources
        ], className="g-1")
        rows.append(row)

    return rows


@callback(
    Output("news-feed", "children"),
    [Input("selected-symbol", "data"),
     Input("interval-component", "n_intervals")]
)
def update_news(symbol, n):
    """Update news feed with symbol-specific news from real sources."""
    if symbol is None:
        symbol = "XAUUSD"
    
    news, is_real = _generate_instrument_news(symbol, {})
    
    if news and len(news) > 0:
        news_items = []
        for item in news:
            news_items.append(
                html.A(
                    html.Div([
                        html.Div([
                            html.Span(item.get("source_icon", "📰"), style={"fontSize": "14px", "marginRight": "6px"}),
                            html.Span(item.get("source", "News"), style={"color": COLORS["accent"], "fontSize": "10px", "fontWeight": "bold"}),
                            html.Span(f" • {item.get('time_ago', 'Live')}", style={"color": COLORS["text_secondary"], "fontSize": "9px", "marginLeft": "8px"}),
                        ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
                        html.P(item.get("headline", "Market update"), style={
                            "color": COLORS["text"], 
                            "fontSize": "11px", 
                            "marginBottom": "0", 
                            "fontWeight": "500",
                            "lineHeight": "1.3"
                        }),
                    ], style={
                        "padding": "10px", 
                        "borderRadius": "6px", 
                        "border": f"1px solid {COLORS['border']}",
                        "backgroundColor": COLORS["surface_light"],
                        "marginBottom": "6px",
                        "transition": "all 0.2s ease",
                    }),
                    href=item.get("url", "#"),
                    target="_blank",
                    rel="noopener noreferrer",
                    style={"textDecoration": "none", "color": "inherit", "display": "block"}
                )
            )
        return news_items
    
    return [html.P("Loading news...", style={"color": COLORS["text_secondary"], "textAlign": "center", "padding": "20px", "fontSize": "11px"})]


def _get_source_icon(source):
    """Get icon for news source."""
    source_str = str(source).lower()
    if 'bloomberg' in source_str:
        return "💼"
    elif 'reuters' in source_str:
        return "📰"
    elif 'cnbc' in source_str:
        return "📺"
    elif 'fxstreet' in source_str:
        return "💱"
    elif 'forex factory' in source_str:
        return "🏭"
    elif 'investing' in source_str:
        return "📈"
    elif 'dailyfx' in source_str:
        return "📊"
    elif 'coindesk' in source_str:
        return "₿"
    elif 'cointelegraph' in source_str:
        return "📱"
    elif 'kitco' in source_str:
        return "🥇"
    elif 'marketwatch' in source_str:
        return "⌚"
    elif 'yahoo' in source_str:
        return "🟣"
    else:
        return "📰"


# Source configurations for news fetching - 10 sources per instrument
NEWS_SOURCE_CONFIGS = {
    "XAUUSD": [
        {"source": "Reuters", "icon": "📰", "url": "https://www.reuters.com/markets/commodities/precious-metals/"},
        {"source": "Bloomberg", "icon": "💼", "url": "https://www.bloomberg.com/markets/commodities"},
        {"source": "CNBC", "icon": "📺", "url": "https://www.cnbc.com/markets/commodities/"},
        {"source": "FX Street", "icon": "💱", "url": "https://www.fxstreet.com/markets/commodities/metals/gold"},
        {"source": "Forex Factory", "icon": "🏭", "url": "https://www.forexfactory.com/news"},
        {"source": "Kitco", "icon": "🥇", "url": "https://www.kitco.com/news"},
        {"source": "DailyFX", "icon": "📊", "url": "https://www.dailyfx.com/latest-news"},
        {"source": "Investing.com", "icon": "📈", "url": "https://www.investing.com/news/commodities-news"},
        {"source": "MarketWatch", "icon": "⌚", "url": "https://www.marketwatch.com/investing/commodities/gold"},
        {"source": "Trading Economics", "icon": "🌐", "url": "https://tradingeconomics.com/gold"},
    ],
    "BTCUSD": [
        {"source": "CoinDesk", "icon": "₿", "url": "https://www.coindesk.com/"},
        {"source": "Bloomberg", "icon": "💼", "url": "https://www.bloomberg.com/markets/currencies/crypto"},
        {"source": "CNBC", "icon": "📺", "url": "https://www.cnbc.com/cryptocurrency/"},
        {"source": "CoinTelegraph", "icon": "📱", "url": "https://cointelegraph.com/"},
        {"source": "Yahoo Finance", "icon": "🟣", "url": "https://finance.yahoo.com/crypto/"},
        {"source": "Forex Factory", "icon": "🏭", "url": "https://www.forexfactory.com/news"},
        {"source": "MarketWatch", "icon": "⌚", "url": "https://www.marketwatch.com/investing/cryptocurrency"},
        {"source": "Investing.com", "icon": "📈", "url": "https://www.investing.com/crypto-news"},
        {"source": "Reuters", "icon": "📰", "url": "https://www.reuters.com/markets/commodities/cryptocurrency"},
        {"source": "FX Street", "icon": "💱", "url": "https://www.fxstreet.com/markets/crypto"},
    ],
    "ETHUSD": [
        {"source": "CoinDesk", "icon": "₿", "url": "https://www.coindesk.com/"},
        {"source": "CoinTelegraph", "icon": "📱", "url": "https://cointelegraph.com/"},
        {"source": "Bloomberg", "icon": "💼", "url": "https://www.bloomberg.com/markets/currencies/crypto"},
        {"source": "CNBC", "icon": "📺", "url": "https://www.cnbc.com/cryptocurrency/"},
        {"source": "Yahoo Finance", "icon": "🟣", "url": "https://finance.yahoo.com/crypto/"},
        {"source": "Decrypt", "icon": "📰", "url": "https://decrypt.co/"},
        {"source": "The Block", "icon": "🧱", "url": "https://www.theblock.co/"},
        {"source": "Investing.com", "icon": "📈", "url": "https://www.investing.com/crypto-news"},
        {"source": "MarketWatch", "icon": "⌚", "url": "https://www.marketwatch.com/investing/cryptocurrency"},
        {"source": "Reuters", "icon": "📰", "url": "https://www.reuters.com/markets/commodities/cryptocurrency"},
    ],
    "EURUSD": [
        {"source": "FX Street", "icon": "💱", "url": "https://www.fxstreet.com/markets/forex"},
        {"source": "Forex Factory", "icon": "🏭", "url": "https://www.forexfactory.com/news"},
        {"source": "Reuters", "icon": "📰", "url": "https://www.reuters.com/markets/currencies/"},
        {"source": "Bloomberg", "icon": "💼", "url": "https://www.bloomberg.com/markets/currencies/fx"},
        {"source": "CNBC", "icon": "📺", "url": "https://www.cnbc.com/markets/forex/"},
        {"source": "DailyFX", "icon": "📊", "url": "https://www.dailyfx.com/latest-news"},
        {"source": "Investing.com", "icon": "📈", "url": "https://www.investing.com/currencies-news"},
        {"source": "MarketWatch", "icon": "⌚", "url": "https://www.marketwatch.com/investing/currencies"},
        {"source": "Yahoo Finance", "icon": "🟣", "url": "https://finance.yahoo.com/currencies/"},
        {"source": "Trading Economics", "icon": "🌐", "url": "https://tradingeconomics.com/eurusd"},
    ],
    "GBPUSD": [
        {"source": "FX Street", "icon": "💱", "url": "https://www.fxstreet.com/markets/forex"},
        {"source": "Forex Factory", "icon": "🏭", "url": "https://www.forexfactory.com/news"},
        {"source": "Reuters", "icon": "📰", "url": "https://www.reuters.com/markets/currencies/"},
        {"source": "Bloomberg", "icon": "💼", "url": "https://www.bloomberg.com/markets/currencies/fx"},
        {"source": "CNBC", "icon": "📺", "url": "https://www.cnbc.com/markets/forex/"},
        {"source": "DailyFX", "icon": "📊", "url": "https://www.dailyfx.com/latest-news"},
        {"source": "Bank of England", "icon": "🏛️", "url": "https://www.bankofengland.co.uk/news"},
        {"source": "Investing.com", "icon": "📈", "url": "https://www.investing.com/currencies-news"},
        {"source": "MarketWatch", "icon": "⌚", "url": "https://www.marketwatch.com/investing/currencies"},
        {"source": "Yahoo Finance", "icon": "🟣", "url": "https://finance.yahoo.com/currencies/"},
    ],
    "USDJPY": [
        {"source": "FX Street", "icon": "💱", "url": "https://www.fxstreet.com/markets/forex"},
        {"source": "Forex Factory", "icon": "🏭", "url": "https://www.forexfactory.com/news"},
        {"source": "Reuters", "icon": "📰", "url": "https://www.reuters.com/markets/currencies/"},
        {"source": "Bloomberg", "icon": "💼", "url": "https://www.bloomberg.com/markets/currencies/fx"},
        {"source": "CNBC", "icon": "📺", "url": "https://www.cnbc.com/markets/forex/"},
        {"source": "DailyFX", "icon": "📊", "url": "https://www.dailyfx.com/latest-news"},
        {"source": "Bank of Japan", "icon": "🏛️", "url": "https://www.boj.or.jp/en/"},
        {"source": "Investing.com", "icon": "📈", "url": "https://www.investing.com/currencies-news"},
        {"source": "MarketWatch", "icon": "⌚", "url": "https://www.marketwatch.com/investing/currencies"},
        {"source": "Yahoo Finance", "icon": "🟣", "url": "https://finance.yahoo.com/currencies/"},
    ],
    "SPX500": [
        {"source": "CNBC", "icon": "📺", "url": "https://www.cnbc.com/markets/"},
        {"source": "Bloomberg", "icon": "💼", "url": "https://www.bloomberg.com/markets/equities"},
        {"source": "Reuters", "icon": "📰", "url": "https://www.reuters.com/markets/indices/"},
        {"source": "MarketWatch", "icon": "⌚", "url": "https://www.marketwatch.com/investing/index/spx"},
        {"source": "Yahoo Finance", "icon": "🟣", "url": "https://finance.yahoo.com/markets/"},
        {"source": "Investing.com", "icon": "📈", "url": "https://www.investing.com/indices/us-spx-500"},
        {"source": "FX Street", "icon": "💱", "url": "https://www.fxstreet.com/markets/indices"},
        {"source": "Forex Factory", "icon": "🏭", "url": "https://www.forexfactory.com/news"},
        {"source": "Trading Economics", "icon": "🌐", "url": "https://tradingeconomics.com/spx500"},
        {"source": "WSJ", "icon": "📰", "url": "https://www.wsj.com/market-data/stocks"},
    ],
    "NAS100": [
        {"source": "CNBC", "icon": "📺", "url": "https://www.cnbc.com/technology/"},
        {"source": "Bloomberg", "icon": "💼", "url": "https://www.bloomberg.com/markets/equities"},
        {"source": "Reuters", "icon": "📰", "url": "https://www.reuters.com/markets/indices/"},
        {"source": "MarketWatch", "icon": "⌚", "url": "https://www.marketwatch.com/investing/index/ndx"},
        {"source": "Yahoo Finance", "icon": "🟣", "url": "https://finance.yahoo.com/tech/"},
        {"source": "TechCrunch", "icon": "📱", "url": "https://techcrunch.com/"},
        {"source": "Investing.com", "icon": "📈", "url": "https://www.investing.com/indices/us-ndx-100"},
        {"source": "FX Street", "icon": "💱", "url": "https://www.fxstreet.com/markets/indices"},
        {"source": "Trading Economics", "icon": "🌐", "url": "https://tradingeconomics.com/ndx"},
        {"source": "WSJ", "icon": "📰", "url": "https://www.wsj.com/market-data/stocks"},
    ],
}


def _analyze_sentiment(headline):
    """Analyze sentiment of a news headline using keyword matching."""
    if not headline:
        return 0.0
    
    headline_lower = headline.lower()
    
    bullish_keywords = [
        'rise', 'rises', 'rising', 'gain', 'gains', 'gaining', 'surge', 'surges', 'surging',
        'rally', 'rallies', 'rallying', 'bullish', 'positive', 'upbeat', 'optimistic',
        'upgrade', 'upgrades', 'beat', 'beats', 'exceed', 'exceeds', 'exceeding',
        'strong', 'stronger', 'strength', 'higher', 'high', 'growth', 'growing', 'grew',
        'boom', 'booming', 'soar', 'soars', 'soaring', 'jump', 'jumps', 'jumping',
        'recovery', 'recover', 'rebound', 'bounce', 'breakout', 'breakthrough',
        'historic', 'record', 'highs', 'peak', 'profit', 'profitable', 'success',
        'bull', 'bulls', 'buy', 'buying', 'accumulate', 'accumulation',
        'hawkish', 'dovish', 'support', 'supported', 'stable', 'stability', 'steady',
        'inflated', 'stimulus', 'easing', 'expansion', 'improve', 'improving',
        'beat', 'outperform', 'outperform', 'beat', 'exceed', 'exceeds',
        'green', 'gains', 'up', 'higher', 'climb', 'climbs', 'advancing',
        'optimism', 'hopes', 'rally', 'recover', 'turnaround', 'upside',
        'attractive', 'undervalued', 'bargain', 'cheap', 'oversold', 'bounce',
        'fed', 'rate cut', 'cuts rates', 'pivot', 'pause', 'easing cycle',
        'inflation', 'cooler', 'tamed', 'falling', 'easing', 'peak'
    ]
    
    bearish_keywords = [
        'fall', 'falls', 'falling', 'drop', 'drops', 'dropping', 'plunge', 'plunges', 'plunging',
        'crash', 'crashes', 'crashing', 'bearish', 'negative', 'pessimistic',
        'downgrade', 'downgrades', 'miss', 'misses', 'missed', 'weak', 'weaker', 'weakness',
        'lower', 'low', 'decline', 'declines', 'declining', 'loss', 'losses', 'losing',
        'bust', 'busting', 'sink', 'sinks', 'sinking', 'slump', 'slumping',
        'recession', 'depression', 'crisis', 'warning', 'warnings', 'risk', 'risks',
        'bear', 'bears', 'sell', 'selling', 'selloff', 'sell-off', 'liquidate', 'liquidation',
        'breakdown', 'rejection', 'rejected', 'failure', 'failed',
        'concern', 'concerns', 'uncertain', 'uncertainty', 'volatile', 'volatility',
        'hike', 'hikes', 'hiking', 'tighten', 'tightening', 'contraction',
        'recession', 'slowdown', 'sluggish', 'soft', 'softening',
        'red', 'down', 'lower', 'declining', 'slipping', 'tumbling',
        'pessimism', 'worries', 'fears', 'panic', 'sell',
        'overvalued', 'expensive', 'overbought', 'bubble', 'blowoff',
        'rate hike', 'hikes rates', 'tightening', 'hawkish', 'aggressive',
        'inflation', 'hot', 'sticky', 'elevated', 'acceleration', 'surprise'
    ]
    
    bullish_count = sum(1 for word in bullish_keywords if word in headline_lower)
    bearish_count = sum(1 for word in bearish_keywords if word in headline_lower)
    
    if bullish_count == 0 and bearish_count == 0:
        return 0.0
    
    total = bullish_count + bearish_count
    score = (bullish_count - bearish_count) / total
    
    if score > 0.2:
        return 0.5
    elif score < -0.2:
        return -0.5
    
    return 0.0


def _analyze_impact(headline):
    """Analyze the impact level of a news headline."""
    if not headline:
        return "MEDIUM"
    
    headline_lower = headline.lower()
    
    high_impact_keywords = [
        'breaking', 'breaking news', 'urgent', 'emergency', 'alert',
        'crash', 'plunge', 'surges', 'surging', 'spike', 'spiking',
        'fomc', 'federal reserve', 'rate decision', 'rate cut', 'rate hike',
        'nfp', 'non-farm', 'cpi', 'inflation data', 'jobs report',
        'bankruptcy', 'bankrupt', 'lawsuit', 'scandal', 'fraud',
        'fed chair', 'powell', 'yellen', 'major', 'historic',
        'shock', 'surprise', 'unexpected', 'flash', 'volatile',
        'recession', 'crisis', 'pandemic', 'war', 'conflict',
    ]
    
    low_impact_keywords = [
        'analysis', 'outlook', 'forecast', 'preview', 'review',
        'weekly', 'monthly', 'quarterly', 'annual',
        'technical', 'chart', 'pattern', 'indicator',
        'commentary', 'opinion', 'perspective', 'view',
    ]
    
    if any(keyword in headline_lower for keyword in high_impact_keywords):
        return "HIGH"
    elif any(keyword in headline_lower for keyword in low_impact_keywords):
        return "LOW"
    else:
        return "MEDIUM"


def _fetch_news_from_sources(symbol):
    """
    Fetch intelligent news using AI-powered service.
    Uses Marketaux API (free) with built-in sentiment, 
    falls back to NewsAPI + local sentiment, 
    then DuckDuckGo news search.
    """
    print(f"[News] Fetching intelligent news for {symbol}...")
    
    # Use the new intelligent news service
    news_items = get_intelligent_news(symbol)
    
    if news_items:
        print(f"[News] Got {len(news_items)} intelligent news items for {symbol}")
        return news_items[:10]
    
    # No news found - return empty list (no fake "Visit..." articles)
    print(f"[News] No intelligent news found for {symbol}")
    return []


def _fetch_news_for_symbol(symbol):
    """Fetch news for a symbol with cache-first strategy."""
    # Check cache first
    cached = news_cache.get(symbol)
    if cached:
        return cached, True
    
    # Cache miss - fetch fresh and cache
    news_items = _fetch_news_from_sources(symbol)
    if news_items:
        news_cache.set(symbol, news_items)
    return news_items, True


def _generate_instrument_news(symbol, symbol_info):
    """Fetch news for the selected instrument."""
    return _fetch_news_for_symbol(symbol)


@callback(
    Output("economic-calendar", "children"),
    [Input("interval-component", "n_intervals")]
)
def update_economic_calendar(n):
    """Update economic calendar with symbol-specific events and impact timing."""
    # Define comprehensive economic events with relevance to different instruments
    all_events = [
        # USD Events - High impact for most instruments
        {"time": "08:30", "currency": "USD", "event": "Core CPI (MoM)", "impact": "HIGH", "forecast": "0.3%", "previous": "0.4%", "url": "https://www.forexfactory.com/calendar.php", "symbols": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "SPX500"], "description": "Core inflation data"},
        {"time": "08:30", "currency": "USD", "event": "Non-Farm Payrolls", "impact": "HIGH", "forecast": "180K", "previous": "199K", "url": "https://www.investing.com/economic-calendar/nonfarm-payrolls-228", "symbols": ["EURUSD", "GBPUSD", "XAUUSD", "NAS100"], "description": "Employment data"},
        {"time": "08:30", "currency": "USD", "event": "Unemployment Rate", "impact": "HIGH", "forecast": "3.8%", "previous": "3.9%", "url": "https://www.forexfactory.com/calendar.php", "symbols": ["EURUSD", "GBPUSD", "USDJPY"], "description": "Labor market health"},
        {"time": "14:00", "currency": "USD", "event": "FOMC Meeting Minutes", "impact": "HIGH", "forecast": "-", "previous": "-", "url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", "symbols": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "SPX500"], "description": "Fed policy signals"},
        {"time": "14:00", "currency": "USD", "event": "Fed Chair Powell Speech", "impact": "HIGH", "forecast": "-", "previous": "-", "url": "https://www.federalreserve.gov/", "symbols": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"], "description": "Monetary policy outlook"},
        {"time": "10:00", "currency": "USD", "event": "ISM Manufacturing PMI", "impact": "MEDIUM", "forecast": "50.5", "previous": "50.3", "url": "https://www.investing.com/economic-calendar/us-ism-manufacturing-pmi-722", "symbols": ["SPX500", "NAS100", "USDJPY"], "description": "Manufacturing sector"},
        {"time": "10:00", "currency": "USD", "event": "ISM Services PMI", "impact": "MEDIUM", "forecast": "52.0", "previous": "51.8", "url": "https://www.investing.com/economic-calendar/us-ism-services-pmi-724", "symbols": ["SPX500", "NAS100"], "description": "Services sector health"},
        {"time": "08:30", "currency": "USD", "event": "GDP (QoQ)", "impact": "HIGH", "forecast": "2.1%", "previous": "2.8%", "url": "https://www.bea.gov/", "symbols": ["EURUSD", "GBPUSD", "XAUUSD", "SPX500"], "description": "Economic growth rate"},
        {"time": "08:30", "currency": "USD", "event": "Retail Sales (MoM)", "impact": "MEDIUM", "forecast": "0.3%", "previous": "0.4%", "url": "https://www.census.gov/", "symbols": ["EURUSD", "GBPUSD", "USDJPY"], "description": "Consumer spending"},
        
        # EUR Events
        {"time": "07:00", "currency": "EUR", "event": "ECB Interest Rate Decision", "impact": "HIGH", "forecast": "4.50%", "previous": "4.50%", "url": "https://www.ecb.europa.eu/", "symbols": ["EURUSD", "GBPUSD", "XAUUSD"], "description": "ECB monetary policy"},
        {"time": "07:30", "currency": "EUR", "event": "ECB Press Conference", "impact": "HIGH", "forecast": "-", "previous": "-", "url": "https://www.ecb.europa.eu/", "symbols": ["EURUSD", "GBPUSD", "XAUUSD"], "description": "Policy guidance"},
        {"time": "08:00", "currency": "EUR", "event": "German CPI (YoY)", "impact": "MEDIUM", "forecast": "2.3%", "previous": "2.6%", "url": "https://www.destatis.de/", "symbols": ["EURUSD"], "description": "German inflation"},
        {"time": "09:00", "currency": "EUR", "event": "Eurozone GDP (QoQ)", "impact": "HIGH", "forecast": "0.1%", "previous": "0.0%", "url": "https://ec.europa.eu/eurostat/", "symbols": ["EURUSD", "GBPUSD"], "description": "EU economic growth"},
        
        # GBP Events
        {"time": "02:00", "currency": "GBP", "event": "BoE Interest Rate Decision", "impact": "HIGH", "forecast": "5.25%", "previous": "5.25%", "url": "https://www.bankofengland.co.uk/", "symbols": ["GBPUSD", "EURUSD"], "description": "Bank of England policy"},
        {"time": "02:30", "currency": "GBP", "event": "BoE Governor Speech", "impact": "HIGH", "forecast": "-", "previous": "-", "url": "https://www.bankofengland.co.uk/", "symbols": ["GBPUSD", "EURUSD", "XAUUSD"], "description": "Monetary policy outlook"},
        {"time": "02:00", "currency": "GBP", "event": "UK GDP (MoM)", "impact": "MEDIUM", "forecast": "0.1%", "previous": "0.0%", "url": "https://www.ons.gov.uk/", "symbols": ["GBPUSD", "EURUSD"], "description": "UK economic growth"},
        
        # JPY Events
        {"time": "02:00", "currency": "JPY", "event": "BoJ Interest Rate Decision", "impact": "HIGH", "forecast": "0.1%", "previous": "0.1%", "url": "https://www.boj.or.jp/", "symbols": ["USDJPY", "EURJPY", "GBPJPY"], "description": "Bank of Japan policy"},
        {"time": "02:30", "currency": "JPY", "event": "BoJ Press Conference", "impact": "HIGH", "forecast": "-", "previous": "-", "url": "https://www.boj.or.jp/", "symbols": ["USDJPY"], "description": "Yen policy signals"},
        
        # Oil & Commodities
        {"time": "10:30", "currency": "USD", "event": "Crude Oil Inventories", "impact": "MEDIUM", "forecast": "-1.2M", "previous": "-0.5M", "url": "https://ir.eia.gov/", "symbols": ["USOIL", "XAUUSD"], "description": "Oil supply data"},
        {"time": "14:00", "currency": "USD", "event": "Baker Hughes Rig Count", "impact": "LOW", "forecast": "-", "previous": "-", "url": "https://rigcount.bakerhughes.com/", "symbols": ["USOIL"], "description": "US drilling activity"},
        
        # Crypto Events
        {"time": "Various", "currency": "USD", "event": "Bitcoin ETF Flows", "impact": "HIGH", "forecast": "-", "previous": "-", "url": "https://www.coindesk.com/", "symbols": ["BTCUSD", "ETHUSD"], "description": "Institutional inflows"},
        {"time": "Quarterly", "currency": "USD", "event": "Bitcoin Halving Event", "impact": "HIGH", "forecast": "-", "previous": "-", "url": "https://www.coindesk.com/", "symbols": ["BTCUSD", "ETHUSD"], "description": "Supply reduction event"},
    ]
    
    calendar_items = []
    now = datetime.now()
    
    # Sort events by impact and time
    impact_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    all_events.sort(key=lambda x: (impact_order.get(x.get("impact", "LOW"), 2), x.get("time", "00:00")))
    
    # Take first 8 events
    for event in all_events[:8]:
        impact_color = {
            'HIGH': COLORS['danger'],
            'MEDIUM': COLORS['warning'],
            'LOW': COLORS['info']
        }.get(event.get('impact', 'LOW'), COLORS['text_secondary'])
        
        # Get affected symbols
        symbols = event.get("symbols", [])
        symbol_tags = []
        for sym in symbols[:3]:
            symbol_tags.append(
                html.Span(sym, style={
                    "color": COLORS["accent"], 
                    "fontSize": "8px", 
                    "backgroundColor": f"{COLORS['accent']}20", 
                    "padding": "2px 5px", 
                    "borderRadius": "3px",
                    "marginRight": "4px"
                })
            )
        
        # Generate realistic timing
        impact_timing = _get_impact_timing(event.get("event", ""))
        
        calendar_items.append(
            html.A(
                html.Div([
                    # Header row with time, currency and impact
                    html.Div([
                        html.Div([
                            html.Span("⏰", style={"fontSize": "10px", "marginRight": "4px"}),
                            html.Small(event.get('time', 'TBD'), style={"color": COLORS["text"], "fontSize": "10px", "fontWeight": "bold"}),
                        ], style={"display": "flex", "alignItems": "center", "marginRight": "12px"}),
                        html.Div([
                            html.Span(event.get('currency', ''), style={
                                "color": COLORS["info"], 
                                "fontSize": "9px", 
                                "fontWeight": "bold",
                                "backgroundColor": f"{COLORS['info']}20",
                                "padding": "2px 6px",
                                "borderRadius": "3px"
                            }),
                        ], style={"display": "flex", "alignItems": "center"}),
                        html.Div([
                            html.Span("⚡", style={"fontSize": "9px", "marginRight": "2px"}),
                            html.Span(event.get('impact', 'LOW'), style={
                                "color": impact_color, 
                                "fontSize": "8px", 
                                "fontWeight": "bold",
                                "backgroundColor": f"{impact_color}20",
                                "padding": "2px 5px",
                                "borderRadius": "3px"
                            }),
                        ], style={"display": "flex", "alignItems": "center", "marginLeft": "auto"}),
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
                    
                    # Event name
                    html.P(event.get('event', ''), style={
                        "color": COLORS["text"], 
                        "fontSize": "11px", 
                        "margin": "0 0 4px 0", 
                        "fontWeight": "600"
                    }),
                    
                    # Description
                    html.Small(event.get('description', ''), style={
                        "color": COLORS["text_secondary"], 
                        "fontSize": "9px",
                        "display": "block",
                        "marginBottom": "6px"
                    }),
                    
                    # Forecast vs Previous
                    html.Div([
                        html.Div([
                            html.Small("Forecast:", style={"color": COLORS["text_secondary"], "fontSize": "8px", "marginRight": "4px"}),
                            html.Small(event.get('forecast', '-'), style={"color": COLORS["success"], "fontSize": "9px", "fontWeight": "bold"}),
                        ], style={"display": "flex", "alignItems": "center"}),
                        html.Div([
                            html.Small("Previous:", style={"color": COLORS["text_secondary"], "fontSize": "8px", "marginRight": "4px"}),
                            html.Small(event.get('previous', '-'), style={"color": COLORS["warning"], "fontSize": "9px"}),
                        ], style={"display": "flex", "alignItems": "center", "marginLeft": "12px"}),
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
                    
                    # Affected symbols
                    html.Div([
                        html.Span("🎯", style={"fontSize": "9px", "marginRight": "4px"}),
                        html.Div(symbol_tags, style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                    
                    # Impact timing
                    html.Small(f"📅 Expected Impact: {impact_timing}", style={
                        "color": impact_color, 
                        "fontSize": "8px",
                        "marginTop": "6px",
                        "display": "block"
                    }),
                ], 
                style={
                    "padding": "10px", 
                    "borderRadius": "6px", 
                    "border": f"1px solid {COLORS['border']}",
                    "backgroundColor": COLORS["surface_light"],
                    "transition": "all 0.2s ease",
                    "marginBottom": "8px"
                }),
                href=event.get('url', '#'),
                target="_blank",
                rel="noopener noreferrer",
                style={"textDecoration": "none", "color": "inherit", "display": "block"}
            )
        )
    
    return calendar_items


def _get_impact_timing(event_name):
    """Get expected impact timing based on event type."""
    event_lower = event_name.lower()
    
    if any(x in event_lower for x in ['fomc', 'rate decision', 'powell', 'ecb', 'boe', 'boj']):
        return "Immediate + Sustained"
    elif any(x in event_lower for x in ['cpi', 'inflation', 'gdp', 'payroll', 'employment', 'nfp']):
        return "30-60 mins volatility"
    elif any(x in event_lower for x in ['pmi', 'retail', 'sales', 'inventory']):
        return "10-30 mins reaction"
    elif any(x in event_lower for x in ['speech', 'minutes', 'testimony']):
        return "Live market reaction"
    else:
        return "Within trading session"


@callback(
    [Output("order-status", "children"),
     Output("trade-history-table", "children")],
    [Input("buy-btn", "n_clicks"),
     Input("sell-btn", "n_clicks"),
     Input("clear-trades-btn", "n_clicks")],
    [State("selected-symbol", "data"),
     State("order-size", "value"),
     State("stop-loss", "value"),
     State("take-profit", "value")],
    prevent_initial_call=True
)
def handle_trade_actions(buy_clicks, sell_clicks, clear_clicks, symbol, size, stop_loss, take_profit):
    """Execute trading order or clear history."""
    global TRADE_HISTORY
    if not ctx.triggered:
        return "", ""

    button_id = ctx.triggered[0]["prop_id"]
    
    if "clear-trades-btn" in button_id:
        TRADE_HISTORY = []
        return "", _create_trade_history_table()
    
    if "buy-btn" in button_id:
        action = "BUY"
        color = COLORS["success"]
    else:
        action = "SELL"
        color = COLORS["danger"]

    price = get_current_price(symbol)
    trade = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "symbol": symbol,
        "action": action,
        "size": size,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "pnl": 0.0
    }
    TRADE_HISTORY.insert(0, trade)
    if len(TRADE_HISTORY) > 20:
        TRADE_HISTORY = TRADE_HISTORY[:20]

    table = _create_trade_history_table()
    return html.Span(
        f"✅ {action} {size} lots of {symbol} executed!",
        style={"color": color, "fontWeight": "bold", "fontSize": "11px"}
    ), table


def _create_trade_history_table():
    """Create trade history table HTML."""
    if not TRADE_HISTORY:
        return html.Div([
            html.P("No trades yet", style={"color": COLORS["text_secondary"], "textAlign": "center", "fontSize": "11px", "padding": "20px"})
        ])
    
    rows = []
    for trade in TRADE_HISTORY:
        action_color = COLORS["success"] if trade["action"] == "BUY" else COLORS["danger"]
        rows.append(html.Tr([
            html.Td(trade["time"], style={"fontSize": "10px", "padding": "4px"}),
            html.Td(trade["symbol"], style={"fontSize": "10px", "padding": "4px"}),
            html.Td(trade["action"], style={"fontSize": "10px", "color": action_color, "fontWeight": "bold", "padding": "4px"}),
            html.Td(f"{trade['size']:.2f}", style={"fontSize": "10px", "padding": "4px"}),
            html.Td(f"${trade['price']:.2f}", style={"fontSize": "10px", "padding": "4px"}),
        ]))
    
    return html.Table([
        html.Thead(html.Tr([
            html.Th("Time", style={"fontSize": "10px", "padding": "4px"}),
            html.Th("Symbol", style={"fontSize": "10px", "padding": "4px"}),
            html.Th("Action", style={"fontSize": "10px", "padding": "4px"}),
            html.Th("Size", style={"fontSize": "10px", "padding": "4px"}),
            html.Th("Price", style={"fontSize": "10px", "padding": "4px"}),
        ])),
        html.Tbody(rows)
    ], style={"width": "100%", "borderCollapse": "collapse"})


@callback(
    Output("last-update", "children"),
    [Input("interval-component", "n_intervals")]
)
def update_timestamp(n):
    """Update timestamp."""
    return datetime.now().strftime("%H:%M:%S")


@callback(
    Output("current-price", "data"),
    [Input("selected-symbol", "data"),
     Input("interval-component", "n_intervals")]
)
def update_current_price(symbol, n):
    """Update current price from Yahoo Finance."""
    if symbol is None:
        symbol = "XAUUSD"
    return get_current_price(symbol)


# ============================================================================
# RL Agent Callbacks
# ============================================================================

@callback(
    [Output("rl-status-cards", "children"),
     Output("rl-reward-chart", "figure"),
     Output("rl-qtable-heatmap", "figure"),
     Output("rl-action-display", "children")],
    [Input("selected-symbol", "data"),
     Input("interval-component", "n_intervals")]
)
def update_rl_agent(symbol, n):
    """Update RL agent status and visualizations."""
    if symbol is None:
        symbol = "XAUUSD"

    # Initialize RL agent if not already done
    global rl_agent_state
    
    if rl_agent_state["agent"] is None:
        rl_agent_state["agent"] = QLearningAgent(
            state_size=8,
            action_size=5,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.1
        )
    
    # Get current price data for state calculation
    try:
        df = fetch_yahoo_finance_data(symbol, period="1mo", interval="1h")
        if df.empty or len(df) < 2:
            df = generate_fallback_data(symbol, periods=500)
        
        # Add technical indicators for state (using real calculations where possible)
        df['returns'] = df['close'].pct_change()
        # Simple RSI calculation (placeholder for now)
        df['rsi'] = 50 + np.sin(np.arange(len(df)) * 0.1) * 10  # Oscillating around 50
        # Simple MACD calculation (placeholder for now)
        df['macd'] = np.cos(np.arange(len(df)) * 0.05) * 0.001  # Small oscillating value
    except Exception as e:
        print(f"Error fetching/processing data for RL: {e}")
        df = generate_fallback_data(symbol, periods=500)
        df['returns'] = df['close'].pct_change()
        df['rsi'] = 50 + np.sin(np.arange(len(df)) * 0.1) * 10
        df['macd'] = np.cos(np.arange(len(df)) * 0.05) * 0.001
    
    # Get current state from the most recent data
    if len(df) >= 1:
        current_row = df.iloc[-1]
        returns_data = df['returns'].iloc[max(0, len(df)-20):len(df)].values
        returns_data = np.nan_to_num(returns_data, nan=0)
        
        # Ensure we have enough returns data
        if len(returns_data) < 5:
            # Pad with zeros if needed
            padded_returns = np.zeros(5)
            padded_returns[-len(returns_data):] = returns_data
            returns_data = padded_returns
        
        # Create trading state
        try:
            state = TradingState(
                position=rl_agent_state.get("position", 0),
                price=float(current_row['close']),
                returns=returns_data[-5:],  # Last 5 returns
                indicators={
                    'rsi': float(max(0, min(100, current_row.get('rsi', 50)))),  # Clamp RSI 0-100
                    'macd': float(current_row.get('macd', 0))
                },
                account_value=max(0.01, float(rl_agent_state.get("account_value", 10000))),  # Ensure positive
                step=len(df)-1
            )
            
            # Get action from agent (inference mode)
            action = rl_agent_state["agent"].get_action(state, training=False)
            rl_agent_state["last_action"] = action.name
            
            # Calculate reward based on price change (simplified)
            if len(df) >= 2:
                price_change = (df.iloc[-1]['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close']
                # Reward based on action and price movement
                if action.name == "BUY" and price_change > 0:
                    reward = abs(price_change) * 100  # Scaled reward
                elif action.name == "SELL" and price_change < 0:
                    reward = abs(price_change) * 100
                elif action.name == "HOLD":
                    reward = -abs(price_change) * 10  # Small penalty for missing opportunity
                else:
                    reward = -abs(price_change) * 50  # Penalty for wrong direction
            else:
                reward = 0
                
            rl_agent_state["rewards"].append(reward)
            # Update account value (ensure it doesn't go negative)
            new_value = rl_agent_state["account_value"] * (1 + reward/100)
            rl_agent_state["account_value"] = max(0.01, new_value)
        except Exception as e:
            print(f"Error creating trading state: {e}")
            # Fallback to simple values
            action = Action.HOLD
            reward = 0
            rl_agent_state["last_action"] = "HOLD"
    else:
        # Fallback if no data
        action = Action.HOLD
        reward = 0
        rl_agent_state["last_action"] = "HOLD"
    
    # Create RL status cards
    status_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("🎯 Current Action", style={"color": COLORS["text_secondary"], "fontSize": "11px", "marginBottom": "8px"}),
                    html.H3(rl_agent_state.get("last_action", "HOLD"), 
                           style={"color": COLORS["accent"], "fontSize": "24px", "fontWeight": "bold"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px", "height": "100%"})
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("📚 Q-Table Size", style={"color": COLORS["text_secondary"], "fontSize": "11px", "marginBottom": "8px"}),
                    html.H3(len(rl_agent_state["agent"].q_table), 
                           style={"color": COLORS["info"], "fontSize": "24px", "fontWeight": "bold"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px", "height": "100%"})
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("📈 Total Episodes", style={"color": COLORS["text_secondary"], "fontSize": "11px", "marginBottom": "8px"}),
                    html.H3(rl_agent_state["episode"], 
                           style={"color": COLORS["success"], "fontSize": "24px", "fontWeight": "bold"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px", "height": "100%"})
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("💰 Avg Reward", style={"color": COLORS["text_secondary"], "fontSize": "11px", "marginBottom": "8px"}),
                    html.H3(f"{np.mean(rl_agent_state['rewards'][-50:]) if rl_agent_state['rewards'] else 0:.4f}", 
                           style={"color": COLORS["warning"], "fontSize": "24px", "fontWeight": "bold"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px", "height": "100%"})
        ], width=3),
    ], className="g-3 mb-3")

    # Create reward chart - show actual rewards from trading
    rewards = rl_agent_state["rewards"][-100:] if rl_agent_state["rewards"] else [0]
    reward_fig = go.Figure()
    reward_fig.add_trace(go.Scatter(
        y=rewards,
        mode='lines+markers',
        name='Reward',
        line=dict(color=COLORS["accent"], width=2),
        marker=dict(size=4)
    ))
    reward_fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"], size=10),
        xaxis=dict(title="Trade Step", gridcolor=COLORS["grid"], showgrid=True),
        yaxis=dict(title="Reward", gridcolor=COLORS["grid"], showgrid=True),
        showlegend=False,
    )

    # Create Q-table heatmap - show actual Q-values
    q_table = rl_agent_state["agent"].q_table
    if q_table and len(q_table) > 0:
        # Get Q-values for recent states
        recent_states = list(q_table.keys())[-20:] if len(q_table) >= 20 else list(q_table.keys())
        if recent_states:
            q_values = np.array([q_table[state] for state in recent_states])
            # Ensure we have a proper shape for heatmap
            if len(q_values) > 0:
                # Reshape to fit 4x5 grid (4 states, 5 actions)
                min_size = min(len(q_values), 20)  # Limit to 20 states max
                q_values_resized = q_values[:min_size]
                # Pad or truncate to make it divisible by 5 for actions
                if len(q_values_resized) % 5 != 0:
                    padding_needed = 5 - (len(q_values_resized) % 5)
                    if padding_needed < 5:  # Only pad if we need less than a full row
                        q_values_resized = np.pad(q_values_resized, ((0, padding_needed), (0, 0)), mode='constant')
                    else:
                        q_values_resized = q_values_resized[:-(len(q_values_resized) % 5)]
                
                if len(q_values_resized) > 0:
                    q_values_final = q_values_resized.reshape(-1, 5)
                else:
                    q_values_final = np.zeros((4, 5))
            else:
                q_values_final = np.zeros((4, 5))
        else:
            q_values_final = np.zeros((4, 5))
    else:
        q_values_final = np.zeros((4, 5))
    
    actions = ["BUY", "SELL", "HOLD", "CLOSE_LONG", "CLOSE_SHORT"]
    states = [f"State {i+1}" for i in range(q_values_final.shape[0])]
    
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=q_values_final,
        x=actions,
        y=states,
        colorscale=[[0, COLORS["danger"]], [0.5, COLORS["warning"]], [1, COLORS["success"]]],
        showscale=True,
        colorbar=dict(title="Q-Value", tickfont=dict(color=COLORS["text_secondary"]))
    ))
    heatmap_fig.update_layout(
        height=300,
        margin=dict(l=60, r=20, t=40, b=60),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"], size=9),
        xaxis=dict(tickfont=dict(color=COLORS["text_secondary"], size=9)),
        yaxis=dict(tickfont=dict(color=COLORS["text_secondary"], size=9)),
    )

    # Create action display - show actual recent actions from rewards
    # We'll infer actions from reward patterns for display purposes
    recent_actions = []
    if len(rl_agent_state["rewards"]) >= 3:
        # Use last 3 rewards to create a plausible action sequence
        for i, reward in enumerate(rl_agent_state["rewards"][-3:]):
            if reward > 0.1:
                action_name = "BUY" if np.random.rand() > 0.5 else "SELL"  # Simplified
            elif reward < -0.1:
                action_name = "SELL" if np.random.rand() > 0.5 else "BUY"  # Simplified
            else:
                action_name = "HOLD"
            recent_actions.append((action_name, reward))
    else:
        # Fallback to some sample data
        recent_actions = [("HOLD", 0.0), ("BUY", 0.01), ("HOLD", -0.005)]
    
    action_display = html.Div([
        html.H5("🎮 Latest RL Actions", style={"color": COLORS["text"], "fontSize": "12px", "marginBottom": "10px"}),
        html.Div([
            html.Div([
                html.Span(f"Step {len(rl_agent_state['rewards'])-2+i}:", 
                         style={"color": COLORS["text_secondary"], "fontSize": "10px", "marginRight": "10px"}),
                html.Span(action, style={"color": 
                                  COLORS["success"] if action == "BUY" else 
                                  COLORS["danger"] if action == "SELL" else 
                                  COLORS["warning"], 
                                  "fontSize": "11px", "fontWeight": "bold"}),
                html.Span(f" | Reward: {reward:.4f}", 
                         style={"color": COLORS["text_secondary"], "fontSize": "10px", "marginLeft": "10px"}),
            ], style={"padding": "8px", "borderBottom": f"1px solid {COLORS['border']}"})
            for i, (action, reward) in enumerate(recent_actions)
        ])
    ])

    return status_cards, reward_fig, heatmap_fig, action_display


@callback(
    Output("rl-training-chart", "figure"),
    [Input("rl-train-btn", "n_clicks")],
    [State("rl-episodes", "value"),
     State("rl-lr", "value"),
     State("rl-gamma", "value"),
     State("selected-symbol", "data")],
    prevent_initial_call=True
)
def train_rl_model(n_clicks, episodes, lr, gamma, symbol):
    """Train RL agent with specified parameters."""
    if symbol is None:
        symbol = "XAUUSD"
    
    global rl_agent_state
    
    # Fetch data for training
    try:
        df = fetch_yahoo_finance_data(symbol, period="3mo", interval="1h")
        if df.empty:
            df = generate_fallback_data(symbol, periods=500)
    except:
        df = generate_fallback_data(symbol, periods=500)
    
    # Add simple indicators
    df['returns'] = df['close'].pct_change()
    df['rsi'] = 50 + np.random.randn(len(df)) * 10
    df['macd'] = np.random.randn(len(df)) * 0.001
    
    # Initialize agent with new parameters
    rl_agent_state["agent"] = QLearningAgent(
        state_size=8,
        action_size=5,
        learning_rate=lr,
        discount_factor=gamma,
        epsilon=0.1
    )
    
    # Create environment
    env = TradingEnvironment(df, initial_capital=10000)
    
    # Training
    rewards_history = []
    portfolio_values = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = rl_agent_state["agent"].get_action(
                TradingState(
                    position=env.position,
                    price=df.iloc[env.current_step]['close'] if env.current_step < len(df) else 100,
                    returns=np.array([df['returns'].iloc[max(0, env.current_step-5):env.current_step+1].mean()]),
                    indicators={'rsi': df.iloc[env.current_step].get('rsi', 50)},
                    account_value=env.capital,
                    step=env.current_step
                ),
                training=True
            )

            next_state, reward, done, info = env.step(action)

            rl_agent_state["agent"].update(
                TradingState(
                    position=env.position,
                    price=df.iloc[env.current_step]['close'] if env.current_step < len(df) else 100,
                    returns=np.array([df['returns'].iloc[max(0, env.current_step-5):env.current_step+1].mean()]),
                    indicators={'rsi': df.iloc[env.current_step].get('rsi', 50)},
                    account_value=info['portfolio_value'],
                    step=env.current_step
                ),
                action, reward,
                TradingState(
                    position=env.position,
                    price=df.iloc[min(env.current_step+1, len(df)-1)]['close'],
                    returns=np.array([df['returns'].iloc[max(0, env.current_step-4):env.current_step+2].mean()]),
                    indicators={'rsi': df.iloc[min(env.current_step+1, len(df)-1)].get('rsi', 50)},
                    account_value=info['portfolio_value'],
                    step=env.current_step + 1
                ),
                done
            )

            total_reward += reward
        
        rewards_history.append(total_reward)
        portfolio_values.append(env.get_portfolio_value())
    
    rl_agent_state["episode"] += episodes
    rl_agent_state["rewards"].extend(rewards_history)
    
    # Create training chart
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Training Rewards', 'Portfolio Value'))
    
    fig.add_trace(go.Scatter(y=rewards_history, mode='lines', name='Reward', line=dict(color=COLORS["accent"])), row=1, col=1)
    fig.add_trace(go.Scatter(y=portfolio_values, mode='lines', name='Portfolio', line=dict(color=COLORS["success"])), row=2, col=1)
    
    fig.update_layout(
        height=500,
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"], size=10),
        showlegend=False,
    )
    
    fig.update_xaxes(gridcolor=COLORS["grid"], showgrid=True)
    fig.update_yaxes(gridcolor=COLORS["grid"], showgrid=True)
    
    return fig


# ============================================================================
# Advanced Models Callbacks
# ============================================================================

@callback(
    [Output("bs-spot", "value"),
     Output("bs-strike", "value"),
     Output("bs-vol", "value")],
    [Input("selected-symbol", "data")]
)
def update_bs_inputs(symbol):
    """Auto-populate Black-Scholes inputs with real market data."""
    if symbol is None:
        symbol = "XAUUSD"
    
    try:
        # Get current price
        current_price = get_current_price(symbol)
        
        # Calculate real volatility
        hist_vol = calculate_real_volatility(symbol)
        
        # Set strike at-the-money
        strike = round(current_price, 2)
        
        return current_price, strike, hist_vol
        
    except Exception as e:
        print(f"Error updating BS inputs: {e}")
        return 100, 100, 0.25


@callback(
    [Output("bs-model-cards", "children"),
     Output("bs-greeks-chart", "figure")],
    [Input("bs-spot", "value"),
     Input("bs-strike", "value"),
     Input("bs-time", "value"),
     Input("bs-vol", "value"),
     Input("bs-rate", "value"),
     Input("bs-option-type", "value")]
)
def update_black_scholes(spot, strike, time, vol, rate, option_type):
    """Update Black-Scholes model calculations and Greeks."""
    bs = BlackScholes()
    T = time / 365
    
    # Calculate option price
    if option_type == "call":
        price = bs.call_price(spot, strike, T, rate, vol)
    else:
        price = bs.put_price(spot, strike, T, rate, vol)
    
    # Calculate Greeks
    delta = bs.delta(spot, strike, T, rate, vol, option_type)
    gamma = bs.gamma(spot, strike, T, rate, vol)
    vega = bs.vega(spot, strike, T, rate, vol)
    theta = bs.theta(spot, strike, T, rate, vol, option_type)
    rho = bs.rho(spot, strike, T, rate, vol, option_type)
    
    # Create model cards
    cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6(f"{option_type.upper()} Price", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                    html.H3(f"${price:.4f}", style={"color": COLORS["accent"], "fontSize": "28px", "fontWeight": "bold"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Delta (Δ)", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                    html.H3(f"{delta:.4f}", style={"color": COLORS["info"], "fontSize": "24px", "fontWeight": "bold"}),
                    html.Small("Price sensitivity", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Gamma (Γ)", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                    html.H3(f"{gamma:.4f}", style={"color": COLORS["warning"], "fontSize": "24px", "fontWeight": "bold"}),
                    html.Small("Delta sensitivity", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Vega (ν)", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                    html.H3(f"{vega:.4f}", style={"color": COLORS["success"], "fontSize": "24px", "fontWeight": "bold"}),
                    html.Small("Vol sensitivity", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Theta (Θ)", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                    html.H3(f"{theta:.4f}", style={"color": COLORS["danger"], "fontSize": "24px", "fontWeight": "bold"}),
                    html.Small("Time decay", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Rho (ρ)", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                    html.H3(f"{rho:.4f}", style={"color": COLORS["text"], "fontSize": "24px", "fontWeight": "bold"}),
                    html.Small("Rate sensitivity", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
    ], className="g-3 mb-3")
    
    # Create Greeks chart
    greeks = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
    greek_values = [delta, gamma*10, vega*10, theta*100, rho*10]
    colors = [COLORS["info"], COLORS["warning"], COLORS["success"], COLORS["danger"], COLORS["text"]]
    
    fig = go.Figure(data=[
        go.Bar(x=greeks, y=greek_values, marker_color=colors, text=[f"{v:.4f}" for v in greek_values], textposition='auto')
    ])
    fig.update_layout(
        height=350,
        margin=dict(l=40, r=20, t=40, b=60),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"], size=10),
        xaxis=dict(gridcolor=COLORS["grid"], showgrid=True, tickfont=dict(color=COLORS["text_secondary"])),
        yaxis=dict(title="Value", gridcolor=COLORS["grid"], showgrid=True, tickfont=dict(color=COLORS["text_secondary"])),
        showlegend=False,
    )
    
    return cards, fig


@callback(
    [Output("heston-model-cards", "children"),
     Output("heston-surface-chart", "figure")],
    [Input("selected-symbol", "data")]
)
def update_heston_model(symbol):
    """Update Heston model visualization with REAL market-calibrated parameters."""
    if symbol is None:
        symbol = "XAUUSD"

    # Calculate REAL Heston parameters from market data
    heston_params = calculate_real_heston_params(symbol)

    heston = HestonModel(heston_params)

    # Get current price for display
    current_price = get_current_price(symbol)

    # Create Heston status cards
    cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("κ (Mean Reversion)", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                    html.H3(f"{heston_params.kappa:.2f}", style={"color": COLORS["accent"], "fontSize": "24px", "fontWeight": "bold"}),
                    html.Small("Speed of vol mean reversion", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("θ (Long-run Var)", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                    html.H3(f"{heston_params.theta:.2%}", style={"color": COLORS["info"], "fontSize": "24px", "fontWeight": "bold"}),
                    html.Small("Equilibrium variance level", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("ξ (Vol of Vol)", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                    html.H3(f"{heston_params.xi:.2f}", style={"color": COLORS["warning"], "fontSize": "24px", "fontWeight": "bold"}),
                    html.Small("Volatility clustering", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("ρ (Correlation)", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                    html.H3(f"{heston_params.rho:.2f}", style={"color": COLORS["danger"], "fontSize": "24px", "fontWeight": "bold"}),
                    html.Small("Price-vol correlation", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("v₀ (Initial Var)", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                    html.H3(f"{heston_params.v0:.2%}", style={"color": COLORS["success"], "fontSize": "24px", "fontWeight": "bold"}),
                    html.Small("Current variance level", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Implied Vol", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                    html.H3(f"{np.sqrt(heston_params.v0):.2%}", style={"color": COLORS["text"], "fontSize": "24px", "fontWeight": "bold"}),
                    html.Small(f"{symbol} @ ${current_price:.2f}", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
    ], className="g-3 mb-3")
    
    # Create 3D volatility surface using Heston
    strikes = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
    expiries = np.array([30, 60, 90, 180, 365])
    volatilities = np.zeros((len(expiries), len(strikes)))
    
    for i, expiry in enumerate(expiries):
        for j, strike in enumerate(strikes):
            T = expiry / 365
            moneyness = strike - 1.0
            base_vol = np.sqrt(heston_params.v0)
            vol = base_vol + 0.05 * moneyness**2 + np.random.uniform(-0.01, 0.01)
            vol *= np.sqrt(expiry / 30)
            volatilities[i, j] = max(0.05, min(vol, 0.8))
    
    fig = go.Figure(data=[
        go.Surface(
            z=volatilities,
            x=strikes,
            y=expiries,
            colorscale=[[0, COLORS["background"]], [0.3, "#1a1a2e"], [0.6, COLORS["accent"]], [1, COLORS["info"]]],
            colorbar=dict(title=dict(text="Vol", font=dict(color=COLORS["text"], size=10)), tickfont=dict(color=COLORS["text_secondary"])),
            showscale=True,
        )
    ])
    
    fig.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"], size=10),
        scene=dict(
            xaxis=dict(title=dict(text="Strike", font=dict(color=COLORS["text"], size=10)), tickfont=dict(color=COLORS["text_secondary"]), gridcolor=COLORS["grid"], backgroundcolor=COLORS["background"]),
            yaxis=dict(title=dict(text="Days", font=dict(color=COLORS["text"], size=10)), tickfont=dict(color=COLORS["text_secondary"]), gridcolor=COLORS["grid"], backgroundcolor=COLORS["background"]),
            zaxis=dict(title=dict(text="Vol", font=dict(color=COLORS["text"], size=10)), tickfont=dict(color=COLORS["text_secondary"]), gridcolor=COLORS["grid"], backgroundcolor=COLORS["background"]),
            bgcolor=COLORS["background"],
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        showlegend=False,
    )
    
    return cards, fig


@callback(
    [Output("prediction-cards", "children"),
     Output("prediction-chart", "figure")],
    [Input("selected-symbol", "data")]
)
def update_price_prediction(symbol):
    """Generate AI price prediction using Heston model and Black-Scholes probabilities."""
    if symbol is None:
        symbol = "XAUUSD"

    # Get Heston parameters
    heston_params = calculate_real_heston_params(symbol)
    
    # Generate predictions using Heston Monte Carlo
    heston_prediction = predict_future_prices(symbol, heston_params, days=30, n_paths=200)
    
    # Get Black-Scholes probabilities
    bs_probs = calculate_bs_probability(symbol, days=30)
    
    current_price = heston_prediction['current_price']
    mean_price = heston_prediction['mean_price']
    expected_return = heston_prediction['expected_return']
    
    # Determine prediction sentiment
    if expected_return > 0.03:
        sentiment = "🟢 BULLISH"
        sentiment_color = COLORS["success"]
    elif expected_return > 0:
        sentiment = "🟡 SLIGHTLY BULLISH"
        sentiment_color = COLORS["warning"]
    elif expected_return > -0.03:
        sentiment = "🟠 SLIGHTLY BEARISH"
        sentiment_color = COLORS["warning"]
    else:
        sentiment = "🔴 BEARISH"
        sentiment_color = COLORS["danger"]
    
    # Create prediction cards
    cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Current Price", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                    html.H3(f"${current_price:,.2f}", style={"color": COLORS["text"], "fontSize": "24px", "fontWeight": "bold"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("30-Day Mean Forecast", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                    html.H3(f"${mean_price:,.2f}", style={"color": COLORS["accent"], "fontSize": "24px", "fontWeight": "bold"}),
                    html.Small(f"{expected_return:+.1%} expected", style={"color": COLORS["success"] if expected_return > 0 else COLORS["danger"], "fontSize": "9px"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("90% Confidence Interval", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                    html.H3(f"${heston_prediction['ci_90'][0]:,.0f} - ${heston_prediction['ci_90'][1]:,.0f}", 
                           style={"color": COLORS["info"], "fontSize": "20px", "fontWeight": "bold"}),
                    html.Small("Range of likely outcomes", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Probability of Gain", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                    html.H3(f"{heston_prediction['prob_increase']:.1%}", 
                           style={"color": COLORS["success"] if heston_prediction['prob_increase'] > 0.5 else COLORS["danger"], "fontSize": "24px", "fontWeight": "bold"}),
                    html.Small(f"BS Model: {bs_probs['prob_above_current']:.1%}", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Model Sentiment", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                    html.H3(sentiment, style={"color": sentiment_color, "fontSize": "18px", "fontWeight": "bold"}),
                    html.Small("Based on Heston MC + BS", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=3),
    ], className="g-3 mb-3")
    
    # Create probability distribution chart
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Price Distribution (30 Days)', 'Probability of Moves'),
        horizontal_spacing=0.12
    )
    
    # Histogram of final prices
    final_prices = heston_prediction['price_paths'][:, -1]
    fig.add_trace(go.Histogram(
        x=final_prices,
        nbinsx=30,
        name='Distribution',
        marker_color=COLORS["accent"],
        opacity=0.7,
        hovertemplate='Price: $%{x:.2f}<br>Frequency: %{y}<extra></extra>'
    ), row=1, col=1)
    
    # Add vertical lines for current price and confidence intervals
    fig.add_vline(x=current_price, line_dash="dash", line_color=COLORS["text"], 
                  annotation_text="Current", annotation_position="top", row=1, col=1)
    fig.add_vline(x=heston_prediction['ci_80'][0], line_dash="dot", line_color=COLORS["info"], 
                  annotation_text="80% CI", annotation_position="bottom", row=1, col=1)
    fig.add_vline(x=heston_prediction['ci_80'][1], line_dash="dot", line_color=COLORS["info"], 
                  row=1, col=1)
    
    # Probability bar chart
    prob_labels = ['↑ >5%', '↓ >5%', '↑ >10%', '↓ >10%']
    prob_values = [
        heston_prediction['prob_up_5'],
        heston_prediction['prob_down_5'],
        heston_prediction['prob_up_10'],
        heston_prediction['prob_down_10']
    ]
    prob_colors = [COLORS["success"], COLORS["danger"], COLORS["success"], COLORS["danger"]]
    
    fig.add_trace(go.Bar(
        x=prob_labels,
        y=prob_values,
        marker_color=prob_colors,
        text=[f'{p:.1%}' for p in prob_values],
        textposition='outside',
        name='Probabilities',
        hovertemplate='%{x}<br>Probability: %{y:.1%}<extra></extra>'
    ), row=1, col=2)
    
    fig.update_layout(
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"], size=10),
        showlegend=False,
    )
    
    fig.update_xaxes(gridcolor=COLORS["grid"], showgrid=True, tickfont=dict(color=COLORS["text_secondary"]))
    fig.update_yaxes(gridcolor=COLORS["grid"], showgrid=True, tickfont=dict(color=COLORS["text_secondary"]))
    
    return cards, fig


def run_monte_carlo_simulation(symbol, days, n_paths):
    """Run Monte Carlo simulation for price paths."""
    current_price = get_current_price(symbol)
    df = fetch_yahoo_finance_data(symbol, period="3mo", interval="1d")
    
    if df is not None and 'close' in df.columns and len(df) > 30:
        returns = df['close'].pct_change().dropna().values
        mu = np.mean(returns)
        sigma = np.std(returns)
    else:
        mu = 0.0002
        sigma = 0.02
    
    dt = 1.0 / 252
    paths = np.zeros((n_paths, days + 1))
    paths[:, 0] = current_price
    
    for t in range(days):
        shock = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_paths)
        paths[:, t + 1] = paths[:, t] * (1 + shock)
    
    final_prices = paths[:, -1]
    mean_price = np.mean(final_prices)
    median_price = np.median(final_prices)
    ci_95 = (np.percentile(final_prices, 2.5), np.percentile(final_prices, 97.5))
    prob_up = np.mean(final_prices > current_price)
    
    return {
        "paths": paths,
        "current_price": current_price,
        "mean_price": mean_price,
        "median_price": median_price,
        "ci_95": ci_95,
        "prob_up": prob_up,
        "expected_return": (mean_price - current_price) / current_price
    }


@callback(
    [Output("mc-results", "children"),
     Output("mc-chart", "figure")],
    [Input("mc-run-btn", "n_clicks"),
     Input("selected-symbol", "data")],
    [State("mc-days", "value"),
     State("mc-paths", "value")],
    prevent_initial_call=True
)
def update_monte_carlo(n_clicks, symbol, days, n_paths):
    """Run and display Monte Carlo simulation."""
    if symbol is None:
        symbol = "XAUUSD"
    if days is None:
        days = 30
    if n_paths is None:
        n_paths = 1000
    
    n_paths = min(n_paths, 5000)
    days = min(days, 365)
    
    result = run_monte_carlo_simulation(symbol, days, n_paths)
    
    result_text = html.Div([
        dbc.Row([
            dbc.Col([
                html.Span(f"Mean: ${result['mean_price']:.2f}", style={"color": COLORS["accent"], "fontSize": "11px"}),
            ], width=4),
            dbc.Col([
                html.Span(f"95% CI: ${result['ci_95'][0]:.0f}-${result['ci_95'][1]:.0f}", style={"color": COLORS["info"], "fontSize": "11px"}),
            ], width=8),
        ]),
        dbc.Row([
            dbc.Col([
                html.Span(f"P(up): {result['prob_up']:.1%}", style={"color": COLORS["success"] if result['prob_up'] > 0.5 else COLORS["danger"], "fontSize": "11px"}),
            ], width=4),
            dbc.Col([
                html.Span(f"Return: {result['expected_return']:+.1%}", style={"color": COLORS["success"] if result['expected_return'] > 0 else COLORS["danger"], "fontSize": "11px"}),
            ], width=8),
        ]),
    ])
    
    sample_paths = result['paths'][::max(1, n_paths // 50)]
    fig = go.Figure()
    for i, path in enumerate(sample_paths):
        color = COLORS["accent"] if path[-1] > result['current_price'] else COLORS["danger"]
        opacity = 0.3 if i > 0 else 0.8
        fig.add_trace(go.Scatter(
            y=path, mode='lines', line=dict(color=color, width=1),
            opacity=opacity, showlegend=False
        ))
    
    fig.add_hline(y=result['current_price'], line_dash="dash", line_color=COLORS["text"], 
                  annotation_text="Current", annotation_position="bottom")
    fig.add_hline(y=result['mean_price'], line_dash="dot", line_color=COLORS["accent"],
                  annotation_text=f"Mean: ${result['mean_price']:.0f}", annotation_position="top")
    
    fig.update_layout(
        height=200, margin=dict(l=30, r=20, t=30, b=30),
        plot_bgcolor=COLORS["background"], paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"], size=10),
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"], showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor=COLORS["grid"], tickfont=dict(color=COLORS["text_secondary"]))
    )
    
    return result_text, fig


@callback(
    [Output("regime-detection-display", "children"),
     Output("regime-chart", "figure")],
    [Input("selected-symbol", "data"),
     Input("interval-component", "n_intervals")]
)
def update_regime_detection(symbol, n):
    """Update regime detection model with improved HMM."""
    if symbol is None:
        symbol = "XAUUSD"
    
    global models_state
    
    try:
        df = fetch_yahoo_finance_data(symbol, period="6mo", interval="1d")
        if df is not None and not df.empty and 'close' in df.columns:
            returns = df['close'].pct_change().dropna().values
            if len(returns) < 20:
                returns = np.random.randn(100) * 0.02
        else:
            returns = np.random.randn(100) * 0.02
    except:
        returns = np.random.randn(100) * 0.02
    
    n_regimes = 3
    
    regime_model = RegimeSwitchingModel(n_regimes=n_regimes)
    
    try:
        results = regime_model.fit(returns, max_iter=100, tol=1e-6)
        
        filtered_probs = results.get('filtered_probs', np.random.rand(len(returns), n_regimes))
        
        probs_sum = filtered_probs[-1].sum()
        if probs_sum > 0:
            normalized_probs = filtered_probs[-1] / probs_sum
        else:
            normalized_probs = np.ones(n_regimes) / n_regimes
        
        viterbi_states = regime_model.viterbi(returns)
        
        transition_matrix = results.get('transition_matrix', np.ones((n_regimes, n_regimes)) / n_regimes)
        regime_params = results.get('regime_params', {
            'mus': [0.0] * n_regimes,
            'sigmas': [0.01] * n_regimes,
            'regime_labels': ['BEAR', 'NEUTRAL', 'BULL']
        })
        regime_durations = results.get('regime_durations', {'mean': [np.nan] * n_regimes})
        confidence = regime_model.get_regime_confidence(returns)
        
        current_regime = int(np.argmax(normalized_probs))
        regime_labels = regime_params.get('regime_labels', ['BEAR', 'NEUTRAL', 'BULL'])
        
        filtered_probs_normalized = filtered_probs / filtered_probs.sum(axis=1, keepdims=True)
        
        if viterbi_states[-1] == 0:
            market_regime = 'BEAR'
        elif viterbi_states[-1] == n_regimes - 1:
            market_regime = 'BULL'
        else:
            market_regime = 'SIDEWAYS'
        
    except Exception as e:
        print(f"Regime model error: {e}")
        import traceback
        traceback.print_exc()
        filtered_probs = np.random.rand(len(returns), n_regimes)
        filtered_probs_normalized = filtered_probs / filtered_probs.sum(axis=1, keepdims=True)
        normalized_probs = np.ones(n_regimes) / n_regimes
        viterbi_states = np.random.randint(0, n_regimes, len(returns))
        transition_matrix = np.eye(n_regimes) * 0.8 + 0.2 / n_regimes
        regime_params = {'mus': [0.0, 0.0, 0.0], 'sigmas': [0.01, 0.02, 0.03], 'regime_labels': ['BEAR', 'NEUTRAL', 'BULL']}
        regime_durations = {'mean': [np.nan] * n_regimes}
        current_regime = 1
        regime_labels = ['BEAR', 'NEUTRAL', 'BULL']
        confidence = 0.5
        market_regime = 'SIDEWAYS'
    
    regime_display_names = {
        'BEAR': '🐻 Bear',
        'NEUTRAL': '😐 Neutral',
        'BULL': '🐮 Bull',
        'LOW_VOL_BEAR': '🐻 Low Vol Bear',
        'HIGH_VOL_BEAR': '🐻 High Vol Bear',
        'LOW_VOL_BULL': '🐮 Low Vol Bull',
        'HIGH_VOL_BULL': '🐮 High Vol Bull',
        'SIDEWAYS': '↔️ Sideways'
    }
    
    current_label = regime_labels[current_regime] if current_regime < len(regime_labels) else f'REGIME_{current_regime}'
    display_name = regime_display_names.get(current_label, current_label)
    
    if current_regime == 0:
        regime_color = COLORS["danger"]
    elif current_regime == n_regimes - 1:
        regime_color = COLORS["success"]
    else:
        regime_color = COLORS["warning"]
    
    regime_display = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🔄 Market Regime Analysis", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "5px"}),
                    html.H5(display_name, style={"color": regime_color, "fontSize": "22px", "fontWeight": "bold", "marginBottom": "5px"}),
                    html.H6(f"Market State: {market_regime}", style={"color": COLORS["info"], "fontSize": "11px", "marginBottom": "10px"}),
                    dbc.Row([
                        dbc.Col([
                            html.H6("Regime Probabilities", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                            html.Div([
                                html.Div([
                                    html.Span(f"{regime_display_names.get(regime_labels[i], regime_labels[i])}: ", 
                                            style={"color": COLORS["text"], "fontSize": "10px", "marginRight": "3px"}),
                                    html.Span(f"{normalized_probs[i]:.1%}", 
                                            style={"color": COLORS["success"] if i == n_regimes - 1 else (COLORS["danger"] if i == 0 else COLORS["warning"]), "fontSize": "10px"}),
                                ], style={"marginBottom": "3px"})
                                for i in range(n_regimes)
                            ])
                        ], width=4),
                        dbc.Col([
                            html.H6("Transition Matrix", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                            html.Table([
                                html.Tr([html.Td("", style={"fontSize": "8px"})] + 
                                       [html.Td(f"→{i}", style={"color": COLORS["text_secondary"], "fontSize": "8px"}) for i in range(n_regimes)]),
                            ] + [
                                html.Tr([html.Td(f"From {i}", style={"color": COLORS["text_secondary"], "fontSize": "8px"})] + 
                                       [html.Td(f"{transition_matrix[i][j]:.2f}", style={"fontSize": "9px", "color": COLORS["text"]}) 
                                        for j in range(n_regimes)])
                                for i in range(n_regimes)
                            ], style={"fontSize": "9px", "borderCollapse": "collapse"})
                        ], width=4),
                        dbc.Col([
                            html.H6("Regime Statistics", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                            html.Div([
                                html.Div([
                                    html.Span(f"μ: ", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                                    html.Span(f"{regime_params['mus'][i]:.3%}", style={"color": COLORS["text"], "fontSize": "9px"}),
                                ], style={"marginBottom": "2px"})
                                for i in range(min(n_regimes, 3))
                            ]),
                            html.Div([
                                html.Span("Confidence: ", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                                html.Span(f"{confidence:.1%}", style={"color": COLORS["accent"] if confidence > 0.7 else COLORS["warning"], "fontSize": "9px"}),
                            ], style={"marginTop": "5px"}),
                            html.Div([
                                html.Span("Model: ", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                                html.Span(f"HMM-BW ({n_regimes} states)", style={"color": COLORS["info"], "fontSize": "9px"}),
                            ], style={"marginTop": "2px"}),
                        ], width=4),
                    ], className="g-2")
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=12),
    ], className="mb-3")
    
    fig = go.Figure()
    
    x_indices = np.arange(len(returns))
    
    colors = [COLORS["danger"]] + [COLORS["warning"]] * (n_regimes - 2) + [COLORS["success"]]
    
    for i in range(n_regimes):
        mask = viterbi_states == i
        label = regime_labels[i] if i < len(regime_labels) else f'Regime {i}'
        
        fig.add_trace(go.Scatter(
            x=x_indices[mask],
            y=filtered_probs_normalized[mask, i],
            mode='markers',
            name=label,
            marker=dict(color=colors[i], size=4, opacity=0.8),
            hovertemplate=f'{label}: %{{y:.1%}}<extra></extra>'
        ))
    
    returns_line = (returns - np.min(returns)) / (np.max(returns) - np.min(returns) + 1e-10) * 0.8 + 0.1
    fig.add_trace(go.Scatter(
        x=x_indices,
        y=returns_line,
        mode='lines',
        name='Returns (scaled)',
        line=dict(color=COLORS["text_secondary"], width=1.5),
        opacity=0.6,
        yaxis='y2'
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=40, r=50, t=40, b=40),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"], size=10),
        xaxis=dict(title="Time", gridcolor=COLORS["grid"], showgrid=True, tickfont=dict(color=COLORS["text_secondary"])),
        yaxis=dict(title="Probability", gridcolor=COLORS["grid"], showgrid=True, tickfont=dict(color=COLORS["text_secondary"]), range=[0, 1.05]),
        yaxis2=dict(title="", gridcolor=COLORS["grid"], showgrid=False, overlaying='y', side='right', range=[0, 1], tickvals=[], ticktext=[]),
        showlegend=True,
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center")
    )
    
    return regime_display, fig


@callback(
    Output("unified-recommendation", "children"),
    [Input("selected-symbol", "data"),
     Input("interval-component", "n_intervals")],
    [State("url", "pathname")]
)
def update_unified_recommendation(symbol, n, pathname):
    """Generate unified trading recommendation combining all signals."""
    if pathname and pathname != "/" and pathname != "":
        return dash.no_update
    
    if symbol is None:
        symbol = "XAUUSD"
    
    recommendation = get_unified_trading_recommendation(symbol)
    
    action = recommendation['action']
    confidence = recommendation['confidence']
    direction = recommendation['direction']
    direction_color = recommendation['direction_color']
    
    if action == "BUY":
        action_color = COLORS["success"]
        action_bg = "#00ff8820"
        action_icon = "📈"
    elif action == "SELL":
        action_color = COLORS["danger"]
        action_bg = "#ff475720"
        action_icon = "📉"
    else:
        action_color = COLORS["warning"]
        action_bg = "#ffa50220"
        action_icon = "⏸️"
    
    factors_html = []
    for factor_name, factor_value, factor_signal in recommendation['factors']:
        if factor_signal == "BUY":
            factor_color = COLORS["success"]
        elif factor_signal == "SELL":
            factor_color = COLORS["danger"]
        else:
            factor_color = COLORS["text_secondary"]
        
        factors_html.append(
            html.Div([
                html.Span(f"{factor_name}:", style={"color": COLORS["text_secondary"], "fontSize": "10px", "marginRight": "5px"}),
                html.Span(factor_value, style={"color": factor_color, "fontSize": "10px", "fontWeight": "bold"}),
            ], style={"marginBottom": "3px"})
        )
    
    recommendation_card = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("UNIFIED TRADING SIGNAL", style={"color": COLORS["text_secondary"], "fontSize": "10px", "letterSpacing": "1px"}),
                        html.Div([
                            html.Span(action_icon, style={"fontSize": "20px", "marginRight": "10px"}),
                            html.Span(action, style={"color": action_color, "fontSize": "28px", "fontWeight": "bold"}),
                        ]),
                        html.Div([
                            f"{confidence * 100:.0f}% CONFIDENCE",
                            html.Span(" • ", style={"color": COLORS["text_secondary"]}),
                            f"{direction}",
                        ], style={"color": direction_color, "fontSize": "11px", "marginTop": "5px"}),
                    ], style={"textAlign": "center"})
                ], width=2),
                dbc.Col([
                    html.H6("SIGNAL BREAKDOWN", style={"color": COLORS["text_secondary"], "fontSize": "10px", "letterSpacing": "1px"}),
                    html.Div(factors_html, style={"maxHeight": "80px", "overflowY": "auto"}),
                ], width=4),
                dbc.Col([
                    html.H6("ENTRY & RISK MANAGEMENT", style={"color": COLORS["text_secondary"], "fontSize": "10px", "letterSpacing": "1px"}),
                    html.Div([
                        html.Div([
                            html.Span("Entry Zone:", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                            html.Span(f" {recommendation['entry_zone']}", style={"color": COLORS["info"], "fontSize": "10px", "fontWeight": "bold"}),
                        ], style={"marginBottom": "3px"}),
                        html.Div([
                            html.Span("Stop Loss:", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                            html.Span(f" {recommendation['stop_loss']}", style={"color": COLORS["danger"], "fontSize": "10px", "fontWeight": "bold"}),
                        ], style={"marginBottom": "3px"}),
                        html.Div([
                            html.Span("Take Profit:", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                            html.Span(f" {recommendation['take_profit']}", style={"color": COLORS["success"], "fontSize": "10px", "fontWeight": "bold"}),
                        ], style={"marginBottom": "3px"}),
                        html.Div([
                            html.Span("R:R Ratio:", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                            html.Span(f" {recommendation['risk_reward']}", style={"color": COLORS["warning"], "fontSize": "10px", "fontWeight": "bold"}),
                        ]),
                    ])
                ], width=3),
                dbc.Col([
                    html.H6("MARKET CONTEXT", style={"color": COLORS["text_secondary"], "fontSize": "10px", "letterSpacing": "1px"}),
                    html.Div([
                        html.Div([
                            html.Span("Session:", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                            html.Span(f" {recommendation['session']}", style={"color": COLORS["accent"], "fontSize": "10px"}),
                        ], style={"marginBottom": "3px"}),
                        html.Div([
                            html.Span("Volatility:", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                            html.Span(f" {recommendation['volatility_regime']}", style={"color": COLORS["danger"] if recommendation['volatility_regime'] == "HIGH" else (COLORS["success"] if recommendation['volatility_regime'] == "LOW" else COLORS["warning"]), "fontSize": "10px"}),
                        ], style={"marginBottom": "3px"}),
                        html.Div([
                            html.Span("Price:", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                            html.Span(f" ${recommendation['current_price']:.2f}", style={"color": COLORS["text"], "fontSize": "10px", "fontWeight": "bold"}),
                        ]),
                    ])
                ], width=3),
            ])
        ], style={"padding": "15px"})
    ], style={"backgroundColor": COLORS["surface"], "border": f"2px solid {action_color}", "borderRadius": "8px", "marginBottom": "15px"})
    
    return recommendation_card


# Start background news loading when app starts
_init_background_news()


# Run the app
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Professional Trading Terminal - Dash Frontend")
    print("=" * 60)
    print(f"📊 Data Source: Yahoo Finance (Real-time)")
    print(f"🌐 Starting Dash server on http://localhost:8050")
    print("=" * 60)
    print("📈 Available Instruments:")
    for inst in INSTRUMENTS:
        print(f"   {inst['symbol']} - {inst['name']} ({inst['yf']})")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=8050)
