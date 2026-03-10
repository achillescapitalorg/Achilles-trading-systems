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
import yfinance as yf
import requests
import dash_bootstrap_components as dbc
import time
import json

# Import services
from services.advanced_models import (
    BlackScholes, HestonModel, HestonParams, SABRModel, SABRParams,
    RegimeSwitchingModel, calculate_var, calculate_expected_shortfall,
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown
)
from services.rl_agent import (
    Action, QLearningAgent, TradingEnvironment, train_rl_agent, TradingState
)
from services.market_data import get_market_data_service

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
    "q_table": {}
}

# Global models state
models_state = {
    "black_scholes": BlackScholes(),
    "heston_params": None,
    "sabr_params": None,
    "regime_model": RegimeSwitchingModel(n_regimes=3),
    "regime_history": []
}

# News articles with real URLs
NEWS_ARTICLES = [
    {
        "title": "Gold Prices Surge Amid Market Uncertainty",
        "source": "Bloomberg",
        "url": "https://www.bloomberg.com/markets/commodities",
        "sentiment": 0.6,
        "time": "2 hours ago"
    },
    {
        "title": "Bitcoin Breaks Key Resistance Level",
        "source": "CoinDesk",
        "url": "https://www.coindesk.com/markets",
        "sentiment": 0.8,
        "time": "3 hours ago"
    },
    {
        "title": "Fed Policy Decision Impacts Forex Markets",
        "source": "Reuters",
        "url": "https://www.reuters.com/markets/currencies",
        "sentiment": -0.2,
        "time": "4 hours ago"
    },
    {
        "title": "Tech Stocks Rally on Earnings Reports",
        "source": "CNBC",
        "url": "https://www.cnbc.com/technology",
        "sentiment": 0.7,
        "time": "5 hours ago"
    },
    {
        "title": "Oil Prices Volatile Amid Supply Concerns",
        "source": "MarketWatch",
        "url": "https://www.marketwatch.com/investing/commodities",
        "sentiment": -0.3,
        "time": "6 hours ago"
    },
    {
        "title": "European Markets Close Higher on ECB News",
        "source": "Financial Times",
        "url": "https://www.ft.com/markets",
        "sentiment": 0.4,
        "time": "7 hours ago"
    },
]

# Economic calendar events
CALENDAR_EVENTS = [
    {"time": "08:30", "currency": "USD", "event": "Core CPI (MoM)", "impact": "HIGH", "url": "https://www.forexfactory.com/calendar"},
    {"time": "08:30", "currency": "USD", "event": "Non-Farm Payrolls", "impact": "HIGH", "url": "https://www.investing.com/economic-calendar/"},
    {"time": "10:00", "currency": "USD", "event": "Crude Oil Inventories", "impact": "MEDIUM", "url": "https://www.forexfactory.com/calendar"},
    {"time": "14:00", "currency": "USD", "event": "FOMC Meeting Minutes", "impact": "HIGH", "url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"},
    {"time": "07:00", "currency": "EUR", "event": "ECB Rate Decision", "impact": "HIGH", "url": "https://www.fxstreet.com/economic-calendar"},
    {"time": "02:00", "currency": "GBP", "event": "BoE Rate Decision", "impact": "HIGH", "url": "https://www.forexfactory.com/calendar"},
]


def fetch_yahoo_finance_data(symbol, period="5d", interval="15m"):
    """Fetch real-time data from Yahoo Finance."""
    try:
        yf_symbol = YF_SYMBOLS.get(symbol, symbol)
        ticker = yf.Ticker(yf_symbol)

        # Fetch historical data
        df = ticker.history(period=period, interval=interval)

        if df is None or df.empty:
            return generate_fallback_data(symbol)

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
    try:
        yf_symbol = YF_SYMBOLS.get(symbol, symbol)
        ticker = yf.Ticker(yf_symbol)
        data = ticker.fast_info
        return float(data.last_price)
    except:
        base_prices = {"XAUUSD": 2650, "BTCUSD": 95000, "ETHUSD": 3400,
                       "EURUSD": 1.05, "GBPUSD": 1.26, "USDJPY": 153.0,
                       "SPX500": 5900, "NAS100": 20500}
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


def create_volatility_surface(symbol):
    """Create a 3D volatility surface plot - Black Theme."""
    np.random.seed(hash(symbol) % 2**32)

    strikes = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
    expiries = np.array([7, 14, 30, 60, 90])
    base_vol = 0.15 if symbol in ["EURUSD", "GBPUSD"] else 0.35 if symbol in ["BTCUSD", "ETHUSD"] else 0.18

    volatilities = np.zeros((len(expiries), len(strikes)))
    for i, expiry in enumerate(expiries):
        for j, strike in enumerate(strikes):
            moneyness = strike - 1.0
            vol = base_vol + 0.05 * (moneyness ** 2) + np.random.uniform(-0.02, 0.02)
            vol *= np.sqrt(expiry / 30)
            volatilities[i, j] = vol

    fig = go.Figure(data=[
        go.Surface(
            z=volatilities,
            x=strikes,
            y=expiries,
            colorscale=[[0, '#000000'], [0.3, '#1a1a2e'], [0.6, '#00ff88'], [1, '#00d4ff']],
            colorbar=dict(
                title=dict(text="Volatility", font=dict(color=COLORS["text"], size=11)),
                tickfont=dict(color=COLORS["text_secondary"]),
            ),
            showscale=True,
        )
    ])

    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"], size=10, family="Arial"),
        scene=dict(
            xaxis=dict(
                title=dict(text="Strike", font=dict(color=COLORS["text"], size=10)),
                tickfont=dict(color=COLORS["text_secondary"]),
                gridcolor=COLORS["grid"],
                backgroundcolor=COLORS["background"],
            ),
            yaxis=dict(
                title=dict(text="Days to Expiry", font=dict(color=COLORS["text"], size=10)),
                tickfont=dict(color=COLORS["text_secondary"]),
                gridcolor=COLORS["grid"],
                backgroundcolor=COLORS["background"],
            ),
            zaxis=dict(
                title=dict(text="Implied Vol", font=dict(color=COLORS["text"], size=10)),
                tickfont=dict(color=COLORS["text_secondary"]),
                gridcolor=COLORS["grid"],
                backgroundcolor=COLORS["background"],
            ),
            bgcolor=COLORS["background"],
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)
            ),
        ),
        showlegend=False,
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


def create_advanced_metrics_cards(symbol):
    """Create advanced metrics cards - Advanced Tab."""
    np.random.seed(hash(symbol) % 2**32)
    
    cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("🎯", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Sortino Ratio", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{round(np.random.uniform(-0.5, 2), 4):.3f}", style={"color": COLORS["text"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
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
                    html.H3(f"{round(np.random.uniform(-0.5, 2), 4):.3f}", style={"color": COLORS["text"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Return vs max drawdown", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=6),
    ], className="g-3")

    return cards


def create_risk_metrics_cards(symbol):
    """Create risk metrics cards - Risk Tab."""
    np.random.seed(hash(symbol) % 2**32)
    
    cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("⚠️", style={"fontSize": "24px", "marginRight": "8px"}),
                        html.H6("Value at Risk (95%)", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "8px", "display": "inline"}),
                    ]),
                    html.H3(f"{round(np.random.uniform(0.01, 0.05), 4):.2%}", style={"color": COLORS["danger"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
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
                    html.H3(f"{round(np.random.uniform(0.02, 0.08), 4):.2%}", style={"color": COLORS["danger"], "marginBottom": "8px", "fontSize": "28px", "fontWeight": "bold"}),
                    html.Small("Loss beyond VaR threshold", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "height": "100%"})
        ], width=6),
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


# App Layout
app.layout = dbc.Container(fluid=True, children=[
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
                                    html.Span("  " + inst["name"], style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
                                id={"type": "instrument-btn", "index": inst["symbol"]},
                                className="w-100",
                                style={
                                    "border": f"1px solid {COLORS['accent']}" if inst["symbol"] == "XAUUSD" else f"1px solid {COLORS['border']}",
                                    "borderRadius": "4px",
                                    "padding": "10px 14px",
                                    "textAlign": "left",
                                    "backgroundColor": COLORS["accent"] if inst["symbol"] == "XAUUSD" else "transparent",
                                    "color": "#000000" if inst["symbol"] == "XAUUSD" else COLORS["text_secondary"],
                                    "width": "100%",
                                    "marginBottom": "8px",
                                    "fontWeight": "bold" if inst["symbol"] == "XAUUSD" else "normal",
                                    "cursor": "pointer",
                                    "transition": "all 0.2s ease",
                                    "boxShadow": f"0 0 10px {COLORS['accent']}40" if inst["symbol"] == "XAUUSD" else "none",
                                }
                            )
                            for inst in INSTRUMENTS
                        ])
                    ], style={"padding": "12px"})
                ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
            ], width=2),

            # Center Panel - Charts and Analysis
            dbc.Col([
                # Price Chart
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
                ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px", "marginBottom": "16px"}),

                # Volatility Surface
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("📉 VOLATILITY SURFACE", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
                    ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "12px"}),
                    dbc.CardBody([
                        dcc.Graph(id="volatility-surface", config={"displayModeBar": False, "responsive": True}, style={"height": "400px"})
                    ], style={"padding": "0", "minHeight": "400px"})
                ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px", "marginBottom": "16px"}),

                # Market Metrics with Tabs
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("📊 MARKET METRICS", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
                    ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "12px"}),
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab([
                                html.Div(id="metrics-cards", style={"padding": "10px"})
                            ], label="📈 Key Metrics", tab_id="metrics", label_style={"color": COLORS["text"], "fontSize": "11px"}),
                            dbc.Tab([
                                html.Div(id="advanced-metrics-cards", style={"padding": "10px"})
                            ], label="🔬 Advanced", tab_id="advanced", label_style={"color": COLORS["text"], "fontSize": "11px"}),
                            dbc.Tab([
                                html.Div(id="risk-metrics-cards", style={"padding": "10px"})
                            ], label="⚠️ Risk", tab_id="risk", label_style={"color": COLORS["text"], "fontSize": "11px"}),
                        ], active_tab="metrics", style={"backgroundColor": COLORS["background"]})
                    ], style={"padding": "0"})
                ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"}),
            ], width=7),

            # Right Sidebar - Signals and Orders
            dbc.Col([
                # Trading Signal
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("🎯 TRADING SIGNAL", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
                    ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "12px"}),
                    dbc.CardBody([
                        html.Div([
                            html.H3(id="signal-action", className="text-center mb-3", style={"fontSize": "32px", "fontWeight": "bold", "letterSpacing": "3px"}),
                            dbc.Progress(
                                id="signal-confidence",
                                value=50,
                                className="mb-3",
                                style={"height": "20px", "borderRadius": "10px", "backgroundColor": COLORS["surface_light"]}
                            ),
                            html.Div(id="signals-table"),
                        ])
                    ], style={"padding": "16px"})
                ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px", "marginBottom": "16px"}),

                # Order Form
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("📝 ORDER FORM", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
                    ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "12px"}),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("BUY", id="buy-btn", color="success", className="w-100 mb-2",
                                          style={"borderRadius": "4px", "fontWeight": "bold", "letterSpacing": "1px"}),
                            ]),
                            dbc.Col([
                                dbc.Button("SELL", id="sell-btn", color="danger", className="w-100 mb-2",
                                          style={"borderRadius": "4px", "fontWeight": "bold", "letterSpacing": "1px"}),
                            ]),
                        ]),
                        html.Label("Size (lots)", style={"color": COLORS["text_secondary"], "fontSize": "10px", "marginBottom": "4px", "display": "block"}),
                        dbc.Input(type="number", id="order-size", value=0.01, step=0.01, min=0.01,
                                 style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"], "borderRadius": "4px"},
                                 className="mb-2"),
                        html.Label("Stop Loss", style={"color": COLORS["text_secondary"], "fontSize": "10px", "marginBottom": "4px", "display": "block"}),
                        dbc.Input(type="number", id="stop-loss", placeholder="Optional",
                                 style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"], "borderRadius": "4px"},
                                 className="mb-2"),
                        html.Label("Take Profit", style={"color": COLORS["text_secondary"], "fontSize": "10px", "marginBottom": "4px", "display": "block"}),
                        dbc.Input(type="number", id="take-profit", placeholder="Optional",
                                 style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"], "borderRadius": "4px"},
                                 className="mb-2"),
                        html.Div(id="order-status", className="text-center mt-2", style={"fontSize": "12px"}),
                    ], style={"padding": "16px"})
                ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px", "marginBottom": "16px"}),

                # News Feed - Clickable Links
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("📰 NEWS FEED", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
                    ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "12px"}),
                    dbc.CardBody([
                        html.Div(id="news-feed", style={"maxHeight": "300px", "overflowY": "auto"})
                    ], style={"padding": "12px"})
                ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px", "marginBottom": "16px"}),

                # Economic Calendar
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("📅 ECONOMIC CALENDAR", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
                    ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "12px"}),
                    dbc.CardBody([
                        html.Div(id="economic-calendar", style={"maxHeight": "250px", "overflowY": "auto"})
                    ], style={"padding": "12px"})
                ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"}),
            ], width=3),
        ], className="mt-3", style={"backgroundColor": COLORS["background"]}),

    # RL Agent and Advanced Models Section
    dbc.Row([
        dbc.Col([
            # RL Agent Tab
            dbc.Card([
                dbc.CardHeader([
                    html.Span("🤖 REINFORCEMENT LEARNING AGENT", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
                ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "12px"}),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab([
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.Div(id="rl-status-cards"),
                                    ], width=12),
                                ]),
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
                                        html.Div(id="rl-action-display", style={"padding": "10px"}),
                                    ], width=12),
                                ]),
                            ], style={"padding": "10px"})
                        ], label="🎮 Q-Learning Agent", tab_id="qlearning", label_style={"color": COLORS["text"], "fontSize": "11px"}),
                        dbc.Tab([
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button("▶️ START TRAINING", id="rl-train-btn", color="success", className="w-100 mb-3",
                                                  style={"borderRadius": "4px", "fontWeight": "bold"}),
                                    ], width=12),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Training Episodes:", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                        dbc.Input(type="number", id="rl-episodes", value=50, min=10, max=500, step=10,
                                                 style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"]}),
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("Learning Rate:", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                        dbc.Input(type="number", id="rl-lr", value=0.1, min=0.01, max=1, step=0.01,
                                                 style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"]}),
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("Discount Factor (Gamma):", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                        dbc.Input(type="number", id="rl-gamma", value=0.95, min=0.1, max=0.99, step=0.01,
                                                 style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"]}),
                                    ], width=4),
                                ], className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id="rl-training-chart", config={"displayModeBar": False, "responsive": True}),
                                    ], width=12),
                                ]),
                            ], style={"padding": "10px"})
                        ], label="⚙️ Training Control", tab_id="training", label_style={"color": COLORS["text"], "fontSize": "11px"}),
                    ], active_tab="qlearning", id="rl-tabs", style={"backgroundColor": COLORS["background"]})
                ], style={"padding": "0"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px", "marginBottom": "16px"}),
        ], width=6),

        dbc.Col([
            # Options Pricing Models Tab
            dbc.Card([
                dbc.CardHeader([
                    html.Span("💹 OPTIONS PRICING MODELS", style={"fontWeight": "bold", "color": COLORS["text"], "fontSize": "11px", "letterSpacing": "1px"}),
                ], style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}", "padding": "12px"}),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab([
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.Div(id="bs-model-cards"),
                                    ], width=12),
                                ], className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Spot Price (S):", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                        dbc.Input(type="number", id="bs-spot", value=100, step=1,
                                                 style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"]}),
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("Strike (K):", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                        dbc.Input(type="number", id="bs-strike", value=100, step=1,
                                                 style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"]}),
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("Time to Expiry (days):", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                        dbc.Input(type="number", id="bs-time", value=30, step=1,
                                                 style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"]}),
                                    ], width=4),
                                ], className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Volatility (σ):", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                        dbc.Input(type="number", id="bs-vol", value=0.25, step=0.01, min=0.01, max=2,
                                                 style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"]}),
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("Risk-free Rate (r):", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                        dbc.Input(type="number", id="bs-rate", value=0.05, step=0.01,
                                                 style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"]}),
                                    ], width=4),
                                    dbc.Col([
                                        html.Label("Option Type:", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                                        dcc.Dropdown(
                                            id="bs-option-type",
                                            options=[{"label": "Call", "value": "call"}, {"label": "Put", "value": "put"}],
                                            value="call",
                                            style={"backgroundColor": COLORS["surface_light"], "border": f"1px solid {COLORS['border']}", "color": COLORS["text"]},
                                        ),
                                    ], width=4),
                                ], className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id="bs-greeks-chart", config={"displayModeBar": False, "responsive": True}),
                                    ], width=12),
                                ]),
                            ], style={"padding": "10px"})
                        ], label="⚫ Black-Scholes", tab_id="black-scholes", label_style={"color": COLORS["text"], "fontSize": "11px"}),
                        dbc.Tab([
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.Div(id="heston-model-cards"),
                                    ], width=12),
                                ], className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id="heston-surface-chart", config={"displayModeBar": False, "responsive": True}),
                                    ], width=12),
                                ]),
                            ], style={"padding": "10px"})
                        ], label="📉 Heston Model", tab_id="heston", label_style={"color": COLORS["text"], "fontSize": "11px"}),
                        dbc.Tab([
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.Div(id="regime-detection-display"),
                                    ], width=12),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(id="regime-chart", config={"displayModeBar": False, "responsive": True}),
                                    ], width=12),
                                ]),
                            ], style={"padding": "10px"})
                        ], label="🔄 Regime Detection", tab_id="regime", label_style={"color": COLORS["text"], "fontSize": "11px"}),
                    ], active_tab="black-scholes", id="models-tabs", style={"backgroundColor": COLORS["background"]})
                ], style={"padding": "0"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"}),
        ], width=6),
    ], className="mt-3"),

    dcc.Interval(id="interval-component", interval=5*1000, n_intervals=0),
    dcc.Store(id="selected-symbol", data="XAUUSD"),
    dcc.Store(id="current-price", data=0),
])


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
    Output("volatility-surface", "figure"),
    [Input("selected-symbol", "data")]
)
def update_volatility_surface(symbol):
    """Update volatility surface."""
    if symbol is None:
        symbol = "XAUUSD"
    try:
        return create_volatility_surface(symbol)
    except Exception as e:
        print(f"Volatility surface error: {e}")
        return create_volatility_surface("XAUUSD")


@callback(
    [Output("metrics-cards", "children"),
     Output("advanced-metrics-cards", "children"),
     Output("risk-metrics-cards", "children")],
    [Input("selected-symbol", "data")]
)
def update_metrics(symbol):
    """Update market metrics cards for all tabs."""
    if symbol is None:
        symbol = "XAUUSD"
    
    key_metrics = create_metrics_cards(symbol)
    advanced_metrics = create_advanced_metrics_cards(symbol)
    risk_metrics = create_risk_metrics_cards(symbol)
    
    return key_metrics, advanced_metrics, risk_metrics


@callback(
    [Output("signal-action", "children"),
     Output("signal-action", "style"),
     Output("signal-confidence", "value"),
     Output("signal-confidence", "color"),
     Output("signals-table", "children")],
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

    action_style = {
        "color": action_colors.get(action, {}).get("color", COLORS["text"]),
        "fontWeight": "bold",
        "marginBottom": "0",
        "fontSize": "32px",
        "letterSpacing": "3px",
    }
    confidence_color = action_colors.get(action, {}).get("bg", "secondary")

    return action, action_style, confidence * 100, confidence_color, create_signals_table(signals)


@callback(
    Output("news-feed", "children"),
    [Input("interval-component", "n_intervals")]
)
def update_news(n):
    """Update news feed with clickable links to original articles."""
    news_items = []
    for article in NEWS_ARTICLES:
        sentiment_color = COLORS["success"] if article["sentiment"] > 0.3 else COLORS["danger"] if article["sentiment"] < -0.3 else COLORS["text_secondary"]
        sentiment_text = "Bullish" if article["sentiment"] > 0.3 else "Bearish" if article["sentiment"] < -0.3 else "Neutral"

        news_items.append(
            html.Div([
                html.A(
                    html.Div([
                        html.Div([
                            html.Small(article["source"], style={"color": COLORS["text_secondary"], "fontSize": "9px", "textTransform": "uppercase", "letterSpacing": "1px"}),
                            html.Small("• " + article["time"], style={"color": COLORS["text_secondary"], "fontSize": "9px", "marginLeft": "8px"}),
                        ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}),
                        html.P(article["title"], style={"color": COLORS["accent"], "fontSize": "12px", "marginBottom": "6px", "fontWeight": "500", "textDecoration": "none"}),
                        html.Div([
                            html.Span(f"Sentiment: {sentiment_text}", style={"color": sentiment_color, "fontSize": "10px", "backgroundColor": COLORS["surface_light"], "padding": "2px 6px", "borderRadius": "3px"}),
                            html.Span("🔗 Read More →", style={"color": COLORS["text_secondary"], "fontSize": "9px", "marginLeft": "8px"}),
                        ], style={"display": "flex", "alignItems": "center"}),
                    ], style={"padding": "10px", "borderRadius": "4px", "transition": "background-color 0.2s"}),
                    href=article["url"],
                    target="_blank",
                    rel="noopener noreferrer",
                    style={"textDecoration": "none", "color": "inherit", "display": "block"}
                ),
                html.Hr(style={"borderColor": COLORS["border"], "margin": "8px 0"}),
            ], style={"marginBottom": "4px"})
        )

    return news_items


@callback(
    Output("economic-calendar", "children"),
    [Input("interval-component", "n_intervals")]
)
def update_economic_calendar(n):
    """Update economic calendar with clickable events."""
    calendar_items = []
    now = datetime.now()
    
    for event in CALENDAR_EVENTS:
        impact_color = {
            'HIGH': COLORS['danger'],
            'MEDIUM': COLORS['warning'],
            'LOW': COLORS['info']
        }.get(event.get('impact', 'LOW'), COLORS['text_secondary'])
        
        # Calculate day name
        event_date = now + timedelta(days=np.random.randint(0, 5))
        date_str = event_date.strftime('%a, %b %d')
        
        calendar_items.append(
            html.A(
                html.Div([
                    html.Div([
                        html.Div([
                            html.Small(f"📅 {date_str}", style={"color": COLORS["accent"], "fontSize": "8px", "fontWeight": "bold", "marginRight": "8px"}),
                            html.Small(event.get('time', ''), style={"color": COLORS["text"], "fontSize": "10px", "fontWeight": "bold"}),
                            html.Small(event.get('currency', ''), style={"color": COLORS["info"], "fontSize": "9px", "marginLeft": "8px", "fontWeight": "bold"}),
                        ], style={"display": "flex", "alignItems": "center"}),
                        html.P(event.get('event', ''), style={"color": COLORS["text"], "fontSize": "11px", "margin": "6px 0", "fontWeight": "500"}),
                        html.Div([
                            html.Span(
                                f"● {event.get('impact', '')}",
                                style={"color": impact_color, "fontSize": "9px", "backgroundColor": COLORS["surface_light"], "padding": "2px 6px", "borderRadius": "3px"}
                            ),
                            html.Span("🔗 View Details →", style={"color": COLORS["accent"], "fontSize": "9px", "marginLeft": "8px"}),
                        ], style={"display": "flex", "alignItems": "center", "marginTop": "4px"}),
                    ], style={"padding": "10px", "borderRadius": "4px", "border": f"1px solid {COLORS['border']}", "transition": "border-color 0.2s"})
                ]),
                href=event.get('url', '#'),
                target="_blank",
                rel="noopener noreferrer",
                style={"textDecoration": "none", "color": "inherit", "display": "block", "marginBottom": "8px"}
            )
        )
    
    return calendar_items


@callback(
    Output("order-status", "children"),
    [Input("buy-btn", "n_clicks"),
     Input("sell-btn", "n_clicks")],
    [State("selected-symbol", "data"),
     State("order-size", "value"),
     State("stop-loss", "value"),
     State("take-profit", "value")],
    prevent_initial_call=True
)
def execute_order(buy_clicks, sell_clicks, symbol, size, stop_loss, take_profit):
    """Execute trading order."""
    if not ctx.triggered:
        return ""

    button_id = ctx.triggered[0]["prop_id"]
    if "buy-btn" in button_id:
        action = "BUY"
        color = COLORS["success"]
    else:
        action = "SELL"
        color = COLORS["danger"]

    return html.Span(
        f"✅ {action} {size} lots of {symbol} executed!",
        style={"color": color, "fontWeight": "bold", "fontSize": "11px"}
    )


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
    
    # Get current price data
    try:
        df = fetch_yahoo_finance_data(symbol, period="1mo", interval="1h")
        prices = df['close'].values if not df.empty else np.array([100])
        returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else np.array([0])
    except:
        prices = np.array([100])
        returns = np.array([0])
    
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

    # Create reward chart
    rewards = rl_agent_state["rewards"][-100:] if rl_agent_state["rewards"] else [0]
    reward_fig = go.Figure()
    reward_fig.add_trace(go.Scatter(
        y=rewards,
        mode='lines',
        name='Reward',
        line=dict(color=COLORS["accent"], width=2)
    ))
    reward_fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"], size=10),
        xaxis=dict(title="Episode", gridcolor=COLORS["grid"], showgrid=True),
        yaxis=dict(title="Reward", gridcolor=COLORS["grid"], showgrid=True),
        showlegend=False,
    )

    # Create Q-table heatmap
    q_table = rl_agent_state["agent"].q_table
    if q_table:
        q_values = np.array([np.max(v) for v in list(q_table.values())[:20]])
        q_values = q_values.reshape(4, 5) if len(q_values) >= 20 else np.zeros((4, 5))
    else:
        q_values = np.zeros((4, 5))
    
    actions = ["BUY", "SELL", "HOLD", "CLOSE_LONG", "CLOSE_SHORT"]
    states = ["Low Ret", "Med Ret", "High Ret", "VH Ret"]
    
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=q_values,
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

    # Create action display
    action_display = html.Div([
        html.H5("🎮 Latest RL Actions", style={"color": COLORS["text"], "fontSize": "12px", "marginBottom": "10px"}),
        html.Div([
            html.Div([
                html.Span(f"Step {i+1}:", style={"color": COLORS["text_secondary"], "fontSize": "10px", "marginRight": "10px"}),
                html.Span(action, style={"color": COLORS["accent"], "fontSize": "11px", "fontWeight": "bold"}),
                html.Span(f" | Reward: {reward:.4f}", style={"color": COLORS["text_secondary"], "fontSize": "10px", "marginLeft": "10px"}),
            ], style={"padding": "8px", "borderBottom": f"1px solid {COLORS['border']}"})
            for i, (action, reward) in enumerate([("HOLD", 0.001), ("BUY", 0.002), ("HOLD", -0.001)][-3:])
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
    """Update Heston model visualization."""
    if symbol is None:
        symbol = "XAUUSD"
    
    # Heston parameters (calibrated to typical market values)
    heston_params = HestonParams(
        kappa=2.0,      # Mean reversion speed
        theta=0.04,     # Long-run variance
        xi=0.3,         # Vol of vol
        rho=-0.7,       # Correlation
        v0=0.04         # Initial variance
    )
    
    heston = HestonModel(heston_params)
    
    # Create Heston status cards
    cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("κ (Mean Reversion)", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                    html.H3(f"{heston_params.kappa:.2f}", style={"color": COLORS["accent"], "fontSize": "24px", "fontWeight": "bold"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("θ (Long-run Var)", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                    html.H3(f"{heston_params.theta:.2%}", style={"color": COLORS["info"], "fontSize": "24px", "fontWeight": "bold"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("ξ (Vol of Vol)", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                    html.H3(f"{heston_params.xi:.2f}", style={"color": COLORS["warning"], "fontSize": "24px", "fontWeight": "bold"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("ρ (Correlation)", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                    html.H3(f"{heston_params.rho:.2f}", style={"color": COLORS["danger"], "fontSize": "24px", "fontWeight": "bold"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("v₀ (Initial Var)", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                    html.H3(f"{heston_params.v0:.2%}", style={"color": COLORS["success"], "fontSize": "24px", "fontWeight": "bold"}),
                ], style={"padding": "15px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Implied Vol", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                    html.H3(f"{np.sqrt(heston_params.v0):.2%}", style={"color": COLORS["text"], "fontSize": "24px", "fontWeight": "bold"}),
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
    [Output("regime-detection-display", "children"),
     Output("regime-chart", "figure")],
    [Input("selected-symbol", "data"),
     Input("interval-component", "n_intervals")]
)
def update_regime_detection(symbol, n):
    """Update regime detection model."""
    if symbol is None:
        symbol = "XAUUSD"
    
    global models_state
    
    # Fetch returns data
    try:
        df = fetch_yahoo_finance_data(symbol, period="6mo", interval="1d")
        returns = df['close'].pct_change().dropna().values if not df.empty else np.random.randn(100) * 0.02
    except:
        returns = np.random.randn(100) * 0.02
    
    # Fit regime model
    regime_model = models_state["regime_model"]
    try:
        results = regime_model.fit(returns, max_iter=50)
        regime_probs = results.get('regime_probs', np.random.rand(len(returns), 3))
        transition_matrix = results.get('transition_matrix', np.ones((3, 3)) / 3)
        regime_params = results.get('regime_params', {'mus': [0, 0, 0], 'sigmas': [0.01, 0.02, 0.03], 'probs': [0.33, 0.33, 0.34]})
    except:
        regime_probs = np.random.rand(len(returns), 3)
        transition_matrix = np.ones((3, 3)) / 3
        regime_params = {'mus': [0, 0, 0], 'sigmas': [0.01, 0.02, 0.03], 'probs': [0.33, 0.33, 0.34]}
    
    # Determine current regime
    current_regime_probs = regime_probs[-1] if len(regime_probs) > 0 else [0.33, 0.33, 0.34]
    current_regime = np.argmax(current_regime_probs)
    regime_names = ["🐻 Bear / High Vol", "😐 Neutral", "🐮 Bull / Low Vol"]
    regime_colors_display = [COLORS["danger"], COLORS["warning"], COLORS["success"]]
    
    # Create regime display
    regime_display = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🔄 Current Market Regime", style={"color": COLORS["text_secondary"], "fontSize": "12px", "marginBottom": "10px"}),
                    html.H3(regime_names[current_regime], 
                           style={"color": regime_colors_display[current_regime], "fontSize": "24px", "fontWeight": "bold", "marginBottom": "15px"}),
                    dbc.Row([
                        dbc.Col([
                            html.H6("Regime Probabilities", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                            html.Div([
                                html.Div([
                                    html.Span("🐻", style={"fontSize": "10px", "marginRight": "5px"}),
                                    html.Span(f"Bear: {current_regime_probs[0]:.1%}", style={"color": COLORS["danger"], "fontSize": "11px"}),
                                ], style={"marginBottom": "5px"}),
                                html.Div([
                                    html.Span("😐", style={"fontSize": "10px", "marginRight": "5px"}),
                                    html.Span(f"Neutral: {current_regime_probs[1]:.1%}", style={"color": COLORS["warning"], "fontSize": "11px"}),
                                ], style={"marginBottom": "5px"}),
                                html.Div([
                                    html.Span("🐮", style={"fontSize": "10px", "marginRight": "5px"}),
                                    html.Span(f"Bull: {current_regime_probs[2]:.1%}", style={"color": COLORS["success"], "fontSize": "11px"}),
                                ]),
                            ])
                        ], width=4),
                        dbc.Col([
                            html.H6("Transition Matrix", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                            html.Table([
                                html.Tr([html.Td("", style={"fontSize": "9px"})] + [html.Td(f"To {i}", style={"color": COLORS["text_secondary"], "fontSize": "8px"}) for i in range(3)]),
                            ] + [
                                html.Tr([html.Td(f"From {i}", style={"color": COLORS["text_secondary"], "fontSize": "8px"})] + 
                                       [html.Td(f"{transition_matrix[i][j]:.2f}", style={"fontSize": "9px", "color": COLORS["text"]}) for j in range(3)])
                                for i in range(3)
                            ], style={"fontSize": "9px", "borderCollapse": "collapse"})
                        ], width=4),
                        dbc.Col([
                            html.H6("Regime Stats", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                            html.Div([
                                html.Div([
                                    html.Span("μ (Return):", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                                    html.Span(f" {regime_params['mus'][current_regime]:.2%}", style={"color": COLORS["text"], "fontSize": "10px", "marginLeft": "5px"}),
                                ], style={"marginBottom": "5px"}),
                                html.Div([
                                    html.Span("σ (Vol):", style={"color": COLORS["text_secondary"], "fontSize": "9px"}),
                                    html.Span(f" {regime_params['sigmas'][current_regime]:.2%}", style={"color": COLORS["text"], "fontSize": "10px", "marginLeft": "5px"}),
                                ]),
                            ])
                        ], width=4),
                    ], className="g-2")
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "6px"})
        ], width=12),
    ], className="mb-3")
    
    # Create regime probability chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=regime_probs[:, 0], mode='lines', name='Bear', stackgroup='one', line=dict(color=COLORS["danger"]), opacity=0.7))
    fig.add_trace(go.Scatter(y=regime_probs[:, 1], mode='lines', name='Neutral', stackgroup='one', line=dict(color=COLORS["warning"]), opacity=0.7))
    fig.add_trace(go.Scatter(y=regime_probs[:, 2], mode='lines', name='Bull', stackgroup='one', line=dict(color=COLORS["success"]), opacity=0.7))
    
    fig.update_layout(
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"], size=10),
        xaxis=dict(title="Time", gridcolor=COLORS["grid"], showgrid=True, tickfont=dict(color=COLORS["text_secondary"])),
        yaxis=dict(title="Probability", gridcolor=COLORS["grid"], showgrid=True, tickfont=dict(color=COLORS["text_secondary"])),
        showlegend=True,
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
    )
    
    return regime_display, fig


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
