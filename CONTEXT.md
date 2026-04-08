# Trading Terminal - Project Context

## Overview

A professional algorithmic trading dashboard built with **Plotly Dash** featuring real-time market data, quantitative finance models, and a standalone trading bot for Gold (XAUUSD), Bitcoin (BTCUSD), and other instruments.

**Note:** This project uses a Dash-based frontend. A React/TypeScript frontend and Flask backend were originally planned but not implemented.

## Current Status (Updated April 2026)

### Recently Completed in This Session

1. **Replaced DeepSeek with Local AI (FinBERT + Ollama)**
   - Created `services/local_ai_service.py` - New unified AI service
   - Uses FinBERT (ProsusAI/finbert) for financial sentiment analysis
   - Uses Ollama for local LLM-powered market analysis
   - Added keyword-based fallback when Ollama not available

2. **Fixed AI Prediction Tab**
   - Fixed Heston parameter calculation (theta was incorrectly calculated using rolling mean)
   - Changed probability thresholds from 5%/10% to 1%/2% for more realistic 30-day predictions
   - Now recalculates probabilities directly from price paths
   - Added asset-specific default probabilities

3. **Added AI Analysis Tab**
   - New tab in "HESTON VOLATILITY & OPTIONS ANALYTICS" section
   - Input field for asking market questions
   - Shows FinBERT and Ollama status
   - Keyword-based fallback provides trading tips when Ollama not installed

4. **Added Markov Model Predictions**
   - Added NEXT REGIME prediction with probability
   - Added PREDICTION direction (Bullish/Bearish/Sideways)
   - Added RISK LEVEL (HIGH/MEDIUM/LOW)
   - Added prediction explanations and trading tips for each regime

5. **Updated Requirements**
   - Added `transformers>=4.30.0` for FinBERT
   - Added `tokenizers>=0.13.0`

6. **Updated News Service**
   - Replaced DeepSeek sentiment with FinBERT in `services/ai_news.py`
   - Updated `pages/news.py` to use local AI service

## Architecture

### Main Application (`/frontend`)
- **Framework**: Plotly Dash (standalone web app)
- **Backend**: None (Dash handles both frontend and API)
- **Styling**: Dash Bootstrap Components (Cyborg theme) + Custom CSS
- **Charts**: Plotly (Candlestick, 3D Surface)
- **State**: Python in-memory state + Dash callbacks

### AI Services
- **FinBERT**: Financial sentiment analysis (HuggingFace ProsusAI/finbert)
- **Ollama**: Local LLM for market analysis (requires separate installation)
- **Keyword Fallback**: Provides trading tips when Ollama not available

### Trading Bot (`/trading_bot/`)
- **Standalone**: Can run independently of the Dash UI
- **Broker Integration**: Exness MT5 (paper trading by default)
- **Data**: Yahoo Finance for market data

## Project Structure

```
frontend/
├── app.py                      # Main Dash application (~5500 lines)
├── volatility_models.py       # GARCH, EGARCH, Heston models (1051 lines)
├── requirements.txt           # Python dependencies
├── assets/
│   ├── chart_enhancements.js  # Custom chart JS
│   └── style.css               # Pure black theme CSS
├── services/
│   ├── __init__.py
│   ├── market_data.py          # Yahoo Finance data fetching
│   ├── news_scraper.py         # Multi-source news (1452 lines)
│   ├── advanced_models.py      # Black-Scholes, Heston, VaR, Sharpe/Sortino/MaxDD
│   ├── markov_model.py         # Hidden Markov Model regime detection
│   ├── ai_news.py              # AI-powered news service (now uses FinBERT)
│   ├── local_ai_service.py     # NEW: FinBERT + Ollama wrapper
│   ├── deepseek_sentiment.py   # OLD: DeepSeek (still present, not used)
│   ├── news_cache.py           # Persistent news cache
│   └── rl_agent.py             # Q-Learning trading agent
├── trading_bot/
│   ├── __init__.py
│   ├── trading_bot.py          # Main multi-signal bot
│   ├── exness_bridge.py        # MT5/Exness execution
│   ├── technical_indicators.py # 50+ indicators
│   ├── quantitative_signals.py # Mean reversion, momentum
│   ├── risk_management.py      # Kelly, position sizing
│   ├── sentiment_analysis.py   # News sentiment
│   ├── backtester.py           # Backtesting engine
│   ├── run_bot.py              # Bot runner script
│   └── config.json             # Bot configuration
├── pages/
│   ├── __init__.py             # Page registry
│   ├── dashboard.py            # Dashboard page layout
│   └── news.py                 # News page with tabs
├── layouts/
├── config/
└── plan/
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

### Machine Learning (UPDATED)
- `torch>=2.1.0` - PyTorch (RL agent)
- `gymnasium>=0.29.0` - RL environments
- `transformers>=4.30.0` - FinBERT for sentiment analysis (NEW)
- `tokenizers>=0.13.0` - Tokenizers for transformers (NEW)

## Core Modules

### `app.py` (~5500 lines)
Main Dash application handling:
- **Multi-page routing** via `dcc.Location` and URL pathname callbacks
- Dashboard page (/) with real-time charts and signals
- News page (/news) with comprehensive news aggregation
- Analysis page (/analysis) placeholder
- Real-time price charts with candlesticks
- Volatility surface (3D implied volatility)
- Market metrics (Hurst exponent, skewness, kurtosis, Sharpe, Sortino, Calmar, Max Drawdown)
- Technical indicator signals (RSI, MACD, Bollinger, Supertrend, Stochastic, CCI, Williams %R, ADX)
- **Unified Trading Recommendation** combining all indicators
- Order form with stop loss/take profit
- Trade history table (in-memory, persists during session)
- News feed with sentiment analysis (FinBERT + synthetic fallback)
- Monte Carlo simulation UI (configurable days/paths)
- **Regime detection** with improved HMM
- **Markov Model** with predictions (NEXT REGIME, PREDICTION, RISK LEVEL)
- **AI Analysis Tab** - Chat with local AI for market insights
- Auto-refresh (5-second intervals)

### `services/local_ai_service.py` (NEW - 370 lines)
Unified local AI service:
- **FinBERTSentiment**: Financial sentiment analysis using ProsusAI/finbert
- **OllamaClient**: Local LLM wrapper for market analysis
- **LocalAIService**: Unified interface for both
- Provides:
  - `analyze_sentiment()` - Single text sentiment
  - `analyze_sentiment_batch()` - Batch sentiment analysis
  - `analyze_market()` - Market analysis with LLM
  - `chat()` - General Q&A with LLM

### `services/ai_news.py` (UPDATED)
- Now uses FinBERT instead of DeepSeek for sentiment
- Falls back to keyword-based sentiment when FinBERT fails

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
- **Regime Switching** - Market regime detection
- **VaR/CVaR** - Risk metrics
- **Sharpe/Sortino/Max Drawdown/Calmar** - Performance metrics

#### `markov_model.py`
- Hidden Markov Model for regime detection
- Creates 3 regimes: LOW_VOL, NORMAL, HIGH_VOL

#### `rl_agent.py` (592 lines)
Reinforcement learning trading:
- **QLearningAgent** - Q-table based learning
- **TradingEnvironment** - Gym-like environment
- **Deep RL** - PyTorch neural network agent

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
- `NEWSAPI_KEY` - NewsAPI.org key (working)
- `MARKETAUX_API_KEY` - Marketaux API (not working - returns 401)
- `DEEPSEEK_API_KEY` - DeepSeek API (not working - returns 402)

### Chart Timeframes
- 5m, 15m, 1h, 4h, 1D

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

### News
- **NewsAPI** - Working (has valid key)
- **Multiple scrapers** - Bloomberg, CNBC, FXStreet, etc.

### Sentiment (UPDATED)
- **FinBERT** - Local transformer model (working)
- **Keyword fallback** - When FinBERT fails

## Development

### Running the Dashboard
```bash
cd frontend
source venv/bin/activate
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

### Installing Ollama (Optional - for better AI)
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3.2:1b
```

## Important Notes

1. **Standalone Architecture**: Dash app runs independently (no separate backend required)
2. **Paper Trading Default**: Trading bot uses paper trading by default
3. **MT5 Optional**: Requires MetaTrader 5 terminal for live trading
4. **Yahoo Finance Limits**: Free tier has rate limits; app gracefully degrades to synthetic data
5. **Risk Warning**: Trading involves substantial risk; always test with paper trading first
6. **Ollama Not Installed**: AI Analysis uses keyword fallback until Ollama is installed
7. **FinBERT Loading**: First request may be slow as model loads (cached after)

## Known Issues & Architecture Gaps

1. **Trading Bot Integration**: Bot runs standalone, not integrated with dashboard
2. **Ollama Not Installed**: AI Analysis tab works but with limited functionality
3. **RL Training**: Agent can be trained but results not prominently displayed
4. **Backtest Panel**: Not implemented
5. **Bot Status Panel**: Not implemented
6. **Alert System**: Not implemented
7. **Server Connection Issues**: Sometimes app doesn't respond to curl but works in browser

## Future Enhancements (Planned)

- [ ] Install and integrate Ollama for AI Analysis
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
