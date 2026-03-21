# Trading Terminal - Project Context

## Overview

A professional algorithmic trading dashboard built with **Plotly Dash** featuring real-time market data, quantitative finance models, and a standalone trading bot for Gold (XAUUSD), Bitcoin (BTCUSD), and other instruments.

**Note:** This project uses a Dash-based frontend. A React/TypeScript frontend and Flask backend were originally planned but not implemented.

## Current Status (Updated March 2026)

### Recently Completed
- Fixed Keltner Channel bug (`middle == lower` → `middle == middle`)
- Fixed sentiment score return value in trading_bot.py
- Fixed invalid hex color `#CHOCO` → `#E67E22` in news_scraper.py
- Added unified trading recommendation card (RSI, MACD, Bollinger, Supertrend, Regime HMM, Black-Scholes, Volatility, Momentum)
- Integrated `get_financial_news()` for real news with synthetic fallback
- Cleaned unused imports (SABRModel, train_rl_agent, get_market_data_service, calculate_var)
- Updated advanced metrics to use real calculations (Sharpe, Sortino, Max Drawdown, Calmar)
- Added Monte Carlo simulation UI with configurable days/paths
- Added trade history table with clear functionality
- Improved regime detection with proper Baum-Welch EM, Viterbi, forward-backward smoothing

## Architecture

### Main Application (`/frontend`)
- **Framework**: Plotly Dash (standalone web app)
- **Backend**: None (Dash handles both frontend and API)
- **Styling**: Dash Bootstrap Components (Cyborg theme) + Custom CSS
- **Charts**: Plotly (Candlestick, 3D Surface)
- **State**: Python in-memory state + Dash callbacks

### Trading Bot (`/trading_bot/`)
- **Standalone**: Can run independently of the Dash UI
- **Broker Integration**: Exness MT5 (paper trading by default)
- **Data**: Yahoo Finance for market data

### Services (`/services/`)
- **Data Layer**: Yahoo Finance API integration
- **ML/Quant**: Black-Scholes, Heston, Regime Switching, Q-Learning agent
- **News**: Multi-source news scraping

## Project Structure

```
frontend/
├── app.py                      # Main Dash application (~4200 lines)
├── volatility_models.py         # GARCH, EGARCH, Heston models (1051 lines)
├── requirements.txt            # Python dependencies
├── assets/
│   ├── chart_enhancements.js   # Custom chart JS
│   └── style.css               # Pure black theme CSS
├── services/
│   ├── __init__.py
│   ├── market_data.py          # Yahoo Finance data fetching
│   ├── news_scraper.py         # Multi-source news (1452 lines)
│   ├── advanced_models.py      # Black-Scholes, Heston, VaR, Sharpe/Sortino/MaxDD (1299 lines)
│   └── rl_agent.py             # Q-Learning trading agent (592 lines)
├── trading_bot/
│   ├── __init__.py
│   ├── trading_bot.py          # Main multi-signal bot (686 lines)
│   ├── exness_bridge.py        # MT5/Exness execution (1009 lines)
│   ├── technical_indicators.py # 50+ indicators (553 lines)
│   ├── quantitative_signals.py # Mean reversion, momentum (599 lines)
│   ├── risk_management.py      # Kelly, position sizing (627 lines)
│   ├── sentiment_analysis.py   # News sentiment (573 lines)
│   ├── backtester.py          # Backtesting engine (523 lines)
│   ├── run_bot.py             # Bot runner script
│   └── config.json             # Bot configuration
├── layouts/                    # (Empty - reserved for future layouts)
├── config/                    # (Empty - reserved for configuration)
└── plan/                      # (Empty - project planning files)
```

## Key Dependencies

### Core
- `dash>=2.14.0` - Web framework
- `dash-bootstrap-components>=1.5.0` - UI components
- `dash-ag-grid>=2.21.0` - Advanced data grid
- `plotly>=5.18.0` - Visualization

### Data Analysis
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `scikit-learn>=1.3.0` - Machine learning

### Market Data
- `yfinance>=0.2.31` - Yahoo Finance API
- `ccxt>=4.0.0` - Crypto exchange API

### HTTP & Scraping
- `requests>=2.31.0` - HTTP requests
- `httpx>=0.25.0` - Async HTTP
- `aiohttp>=3.9.0` - Async networking
- `beautifulsoup4>=4.12.0` - HTML parsing
- `lxml>=4.9.0` - XML/HTML parser

### Machine Learning
- `torch>=2.1.0` - PyTorch (RL agent)
- `gymnasium>=0.29.0` - RL environments

## Core Modules

### `app.py` (~4200 lines)
Main Dash application handling:
- Real-time price charts with candlesticks
- Volatility surface (3D implied volatility)
- Market metrics (Hurst exponent, skewness, kurtosis, Sharpe, Sortino, Calmar, Max Drawdown)
- Technical indicator signals (RSI, MACD, Bollinger, Supertrend)
- **Unified Trading Recommendation** combining all indicators
- Order form with stop loss/take profit
- Trade history table (in-memory, persists during session)
- News feed with sentiment analysis (real + synthetic fallback)
- Monte Carlo simulation UI (configurable days/paths)
- **Regime detection** with improved HMM (Baum-Welch EM, Viterbi, forward-backward)
- Auto-refresh (5-second intervals)

### `volatility_models.py` (1051 lines)
Professional volatility models:
- **GARCH(p, q)** - Generalized ARCH
- **EGARCH** - Exponential GARCH
- **GJR-GARCH** - Asymmetric GARCH
- **Heston** - Stochastic Volatility Model
- **Realized Volatility** - High-frequency estimators
- **Parkinson** - Range-based estimator
- **Garman-Klass** - OHLC estimator
- **Yang-Zhang** - Drift-independent estimator

### Services

#### `market_data.py`
- Yahoo Finance symbol mapping
- Real-time price fetching
- Historical OHLCV data
- Fallback mock data generation

#### `news_scraper.py` (1452 lines)
Multi-source financial news:
- Bloomberg, CNBC, Reuters
- FXStreet, Forex Factory, DailyFX
- Investing.com, Yahoo Finance
- CoinDesk, Crypto Panic
- MarketWatch, Kitco

#### `advanced_models.py` (1299 lines)
Quantitative finance models:
- **Black-Scholes** - Option pricing + Greeks
- **Heston Model** - Stochastic volatility
- **Regime Switching** - Market regime detection (improved with Baum-Welch, Viterbi)
- **VaR/CVaR** - Risk metrics
- **Sharpe/Sortino/Max Drawdown/Calmar** - Performance metrics
- `detect_regime()` - Unified regime detection function

#### `rl_agent.py` (592 lines)
Reinforcement learning trading:
- **QLearningAgent** - Q-table based learning
- **TradingEnvironment** - Gym-like environment
- **Deep RL** - PyTorch neural network agent
- Bellman equation for value iteration

### Trading Bot (`/trading_bot/`)

#### `trading_bot.py` (686 lines)
Multi-signal trading bot architecture:
1. Data Collection - Fetch OHLCV from MT5
2. Technical Analysis - Calculate indicators
3. Quantitative Signals - Mean reversion, momentum
4. Sentiment Analysis - News sentiment
5. Signal Aggregation - Weighted combination
6. Risk Management - Position sizing, stops
7. Execution - Place orders via MT5

#### `exness_bridge.py` (1009 lines)
MT5/Exness trading bridge:
- **ExnessMT5Bridge** - Real MT5 connection
- **PaperTradingBridge** - Simulated trading
- Market and pending orders
- Position management
- Real-time price feeds

#### `technical_indicators.py` (553 lines)
50+ technical indicators:
- **Trend**: EMA, SMA, WMA, Hull MA, VWAP, Ichimoku
- **Momentum**: RSI, MACD, Stochastic, CCI, Williams %R
- **Volatility**: Bollinger Bands, ATR, Keltner Channel (fixed)
- **Volume**: OBV, Volume Profile, Money Flow
- **Custom**: Supertrend, Pivot Points, Fibonacci

#### `quantitative_signals.py` (599 lines)
Institutional-grade signals:
- **Mean Reversion** - Bollinger, RSI, Statistical
- **Momentum** - Time-series, Cross-sectional
- **Volatility Breakout** - ATR-based
- **Statistical Arbitrage** - Pairs trading
- **Market Regime** - Regime detection

#### `risk_management.py` (627 lines)
Professional risk controls:
- **Position Sizing**: Fixed %, Kelly Criterion, Volatility Targeting
- **Stop Loss**: Fixed, ATR-based, Support/Resistance
- **Portfolio Limits**: Max exposure, correlation
- **Drawdown Controls**: Trailing stops, circuit breakers

#### `sentiment_analysis.py` (573 lines)
Multi-source sentiment:
- Social media (Twitter, Reddit)
- Financial news APIs
- Crypto-specific sources
- Gold/precious metals reports
- Economic calendar events

#### `backtester.py` (523 lines)
Backtesting framework:
- Event-driven backtesting
- Transaction costs and slippage
- Walk-forward analysis
- Monte Carlo simulation
- Performance metrics (Sharpe, Sortino, Calmar)

## Trading Instruments

| Symbol | Name | Type | Yahoo Finance |
|--------|------|------|---------------|
| XAUUSD | Gold | Metal | GC=F |
| BTCUSD | Bitcoin | Crypto | BTC-USD |
| ETHUSD | Ethereum | Crypto | ETH-USD |
| EURUSD | Euro/USD | Forex | EURUSD=X |
| GBPUSD | GBP/USD | Forex | GBPUSD=X |
| USDJPY | USD/JPY | Forex | USDJPY=X |
| SPX500 | S&P 500 | Index | ^GSPC |
| NAS100 | Nasdaq 100 | Index | ^NDX |

## UI Theme

**Pure Black Theme:**
- Background: `#000000` (Pure Black)
- Surface: `#0a0a0a` (Near Black)
- Surface Light: `#121212`
- Primary: `#1a1a1a`
- Accent/Success: `#00ff88` (Neon Green)
- Danger: `#ff4757` (Red)
- Warning: `#ffa502` (Orange)
- Info: `#00d4ff` (Cyan)
- Text: `#ffffff`
- Text Secondary: `#888888`

## Configuration

### Environment Variables
- `API_URL` - Backend API URL (default: http://localhost:8000)

### Chart Timeframes
- M5, M15, H1, H4, D1

### Bot Configuration (`config.json`)
- Assets to trade
- Risk parameters
- Signal thresholds
- MT5 credentials (for live trading)

## Data Sources

### Primary
- **Yahoo Finance** - Real-time prices and historical data

### Fallback
- **Synthetic data generation** - If Yahoo Finance unavailable

## Development

### Running the Dashboard
```bash
cd frontend
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
python app.py
```
**Dashboard:** http://localhost:8050

### Running the Trading Bot
```bash
cd frontend/trading_bot
pip install -r requirements.txt
python run_bot.py
```

## Important Notes

1. **Standalone Architecture**: Dash app runs independently (no separate backend required)
2. **Paper Trading Default**: Trading bot uses paper trading by default
3. **MT5 Optional**: Requires MetaTrader 5 terminal for live trading
4. **Yahoo Finance Limits**: Free tier has rate limits; app gracefully degrades to synthetic data
5. **Risk Warning**: Trading involves substantial risk; always test with paper trading first

## Known Issues & Architecture Gaps

1. **Trading Bot Integration**: Bot runs standalone, not integrated with dashboard
2. **SABR Model**: Excluded from main app (not actively used)
3. **RL Training**: Agent can be trained but results not prominently displayed
4. **Backtest Panel**: Not implemented
5. **Bot Status Panel**: Not implemented
6. **Alert System**: Not implemented

## Future Enhancements (Planned)

- [ ] React/TypeScript frontend (per original spec)
- [ ] Flask backend with WebSocket support
- [ ] SQLite database for trade persistence
- [ ] Real-time MT5/Exness trading execution
- [ ] User authentication and portfolios
- [ ] Mobile-responsive design
- [ ] Integrate trading bot with dashboard UI
- [ ] Add persistent trade history (database)
- [ ] Implement backtest panel
- [ ] Implement alert system
