# Trading Terminal - Project Context

## Overview

A professional algorithmic trading dashboard built with **Plotly Dash** featuring real-time market data, quantitative finance models, and a standalone trading bot for Gold (XAUUSD), Bitcoin (BTCUSD), and other instruments.

**Note:** This project uses a Dash-based frontend. A React/TypeScript frontend and Flask backend were originally planned but not implemented.

---

## Current Status (Updated April 2026)

### Recently Completed in This Session

1. **Replaced DeepSeek with Local AI (FinBERT + Ollama)**
   - Created `services/local_ai_service.py` - New unified AI service
   - Uses FinBERT (ProsusAI/finbert) for financial sentiment analysis
   - Uses Ollama (llama3.2:1b) for local LLM-powered market analysis
   - Added keyword-based fallback when Ollama not available
   - Ollama is now installed and working

2. **Fixed AI Prediction Tab**
   - Fixed Heston parameter calculation (theta was incorrectly calculated)
   - Changed probability thresholds from 5%/10% to 1%/2% for realistic 30-day predictions
   - Recalculates probabilities directly from price paths
   - Added asset-specific default probabilities

3. **Added AI Analysis Tab**
   - New tab in "HESTON VOLATILITY & OPTIONS ANALYTICS" section
   - Input field for asking market questions
   - Shows FinBERT and Ollama status indicators
   - Keyword-based fallback provides trading tips when Ollama not installed
   - Now working with Ollama (llama3.2:1b model installed)

4. **Added Markov Model Predictions**
   - Added NEXT REGIME prediction with probability
   - Added PREDICTION direction (Bullish/Bearish/Sideways)
   - Added RISK LEVEL (HIGH/MEDIUM/LOW)
   - Added prediction explanations and trading tips for each regime

5. **Enhanced News Page**
   - Added Economic Calendar tab with upcoming events
   - Added AI Analysis tab for market summary using Ollama
   - Changed sentiment meter label from "DeepSeek" to "FinBERT"

6. **Updated Requirements**
   - Added `transformers>=4.30.0` for FinBERT
   - Added `tokenizers>=0.13.0`

7. **Updated News Service**
   - Replaced DeepSeek sentiment with FinBERT in `services/ai_news.py`
   - Updated `pages/news.py` to use local AI service
   - Fixed Ollama streaming issue (added `"stream": False` to API call)

---

## Architecture

### Main Application (`/frontend`)
- **Framework**: Plotly Dash (standalone web app)
- **Backend**: None (Dash handles both frontend and API)
- **Styling**: Dash Bootstrap Components (Cyborg theme) + Custom CSS
- **Charts**: Plotly (Candlestick, 3D Surface)
- **State**: Python in-memory state + Dash callbacks

### AI Services
- **FinBERT**: Financial sentiment analysis (HuggingFace ProsusAI/finbert) - WORKING
- **Ollama**: Local LLM for market analysis (llama3.2:1b installed and running)
- **Keyword Fallback**: Provides trading tips when Ollama not available

### Trading Bot (`/trading_bot/`)
- **Standalone**: Can run independently of the Dash UI
- **Broker Integration**: Exness MT5 (paper trading by default)
- **Data**: Yahoo Finance for market data

---

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
│   ├── ai_news.py              # AI-powered news service (uses FinBERT)
│   ├── local_ai_service.py     # FinBERT + Ollama wrapper
│   ├── deepseek_sentiment.py   # OLD: DeepSeek (not used)
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

---

## Key Functions in `app.py`

### Data Fetching
- `fetch_yahoo_finance_data(symbol, period, interval)` - Fetch OHLCV data from Yahoo Finance
- `get_current_price(symbol)` - Get current price for a symbol
- `generate_fallback_data(symbol, periods)` - Generate synthetic data when Yahoo fails

### Technical Indicators (Safe Versions)
- `calculate_rsi_safe(prices, period=14)` - RSI with error handling
- `calculate_macd_safe(prices, fast, slow, signal_period)` - MACD with error handling
- `calculate_bb_safe(prices, period, std_dev)` - Bollinger Bands with error handling
- `calculate_supertrend_safe(df, period, multiplier)` - Supertrend with error handling
- `calculate_stochastic_safe(df, k_period, d_period)` - Stochastic Oscillator
- `calculate_cci_safe(df, period)` - Commodity Channel Index
- `calculate_williams_r_safe(df, period)` - Williams %R
- `calculate_adx_safe(df, period)` - Average Directional Index

### Financial Models
- `calculate_real_heston_params(symbol)` - Calculate Heston parameters from real market data
- `calculate_real_volatility(symbol)` - Calculate realized volatility
- `calculate_regime_metrics(symbol)` - Calculate regime detection metrics
- `calculate_hurst_exponent(returns, min_lag, max_lag)` - Hurst exponent for trend/mean-reversion
- `calculate_trend_strength(returns, lookback)` - Trend strength indicator
- `calculate_mean_reversion_strength(returns, lookback)` - Mean reversion strength
- `predict_future_prices(symbol, heston_params, days, n_paths)` - Monte Carlo price simulation
- `calculate_bs_probability(symbol, days)` - Black-Scholes probability calculation

### Trading Signals
- `generate_signals(symbol)` - Generate all technical indicator signals
- `get_unified_trading_recommendation(symbol)` - Combined BUY/SELL/HOLD recommendation

### Callbacks (Dash UI Updates)
- `update_price_chart(symbol, n, timeframe)` - Update main candlestick chart
- `update_metrics(symbol)` - Update market metrics cards
- `update_signals(symbol)` - Update trading signals panel
- `update_news(symbol, n)` - Update news feed
- `update_economic_calendar(n)` - Update economic calendar events
- `handle_trade_actions(...)` - Handle buy/sell/clear trade actions
- `update_heston_model(symbol)` - Update Heston volatility surface
- `update_price_prediction(symbol)` - Update AI price prediction
- `update_sabr_model(symbol)` - Update SABR volatility smile
- `update_markov_model(symbol)` - Update Markov regime detection
- `update_ai_chat(n_clicks, symbol, user_message)` - Handle AI chat interactions
- `update_monte_carlo(n_clicks, symbol, days, n_paths)` - Run Monte Carlo simulation
- `update_regime_detection(symbol, n)` - Update regime detection display
- `update_unified_recommendation(symbol, n, pathname)` - Update unified recommendation

---

## Key Classes and Functions in `services/`

### `services/ai_news.py` - UnifiedNewsService
- `get_news(symbol, max_items)` - Main method to fetch news from all sources
- `_fetch_all_sources_parallel(symbol, keywords)` - Parallel fetching from multiple sources
- `_fetch_newsapi(keywords)` - Fetch from NewsAPI (your key works)
- `_fetch_google_news(keywords, symbol)` - Fetch from Google News RSS
- `_fetch_forexcom_news(keywords)` - Fetch from Forex.com
- `_fetch_marketaux(symbol)` - Fetch from Marketaux API (broken - returns 401)
- `_deduplicate_and_rank(items)` - Deduplicate and rank by source priority

**News Sources Currently Used:**
- NewsAPI (working)
- Google News RSS (5-day filter)
- Forex.com
- Marketaux (broken)

### `services/local_ai_service.py` - LocalAIService
- `get_sentiment(text)` - Analyze single text sentiment
- `get_sentiment_batch(texts)` - Batch sentiment analysis
- `analyze_market(symbol, news, price_data)` - Market analysis with LLM
- `chat(message, context)` - Chat with Ollama LLM

### `services/news_scraper.py` - NewsAggregator
- `fetch_symbol_news(symbol, limit)` - Fetch news for specific symbol
- `fetch_all_news(limit_per_source)` - Fetch general news
- `get_news_sources()` - Get list of available sources

### `services/markov_model.py` - MarkovRegimeModel
- `fit(prices, returns)` - Fit HMM to price data
- `predict_next_regime()` - Predict next regime
- `get_regime_description(regime)` - Get regime explanation

### `services/market_data.py`
- `get_market_data(symbol)` - Get comprehensive market data
- `get_instrument_info(symbol)` - Get instrument details

---

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

### HTTP & Scraping
- `requests>=2.31.0` - HTTP requests
- `httpx>=0.25.0` - Async HTTP
- `aiohttp>=3.9.0` - Async networking
- `beautifulsoup4>=4.12.0` - HTML parsing
- `lxml>=4.9.0` - XML/HTML parser

### Machine Learning
- `torch>=2.1.0` - PyTorch (RL agent)
- `gymnasium>=0.29.0` - RL environments
- `transformers>=4.30.0` - FinBERT for sentiment analysis
- `tokenizers>=0.13.0` - Tokenizers for transformers

---

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

---

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

---

## Configuration

### Environment Variables
- `NEWSAPI_KEY` - NewsAPI.org key (working)
- `MARKETAUX_API_KEY` - Marketaux API (not working - returns 401)
- `DEEPSEEK_API_KEY` - DeepSeek API (not working - returns 402)

### Chart Timeframes
- 5m, 15m, 1h, 4h, 1D

---

## Data Sources

### Primary
- **Yahoo Finance** - Real-time prices and historical data

### Fallback
- **Synthetic data generation** - If Yahoo Finance unavailable

### News (Current)
- **NewsAPI** - Working (has valid key)
- **Google News RSS** - Filtered by 5 days
- **Forex.com** - Web scraping

### Sentiment (Updated)
- **FinBERT** - Local transformer model (working)
- **Ollama** - Local LLM (llama3.2:1b installed and running)
- **Keyword fallback** - When Ollama not available

---

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

### Ollama Status
Ollama is now installed with llama3.2:1b model. To ensure it's running:
```bash
ollama serve  # Start Ollama server in background
ollama list   # Check installed models
```

---

## Important Notes

1. **Standalone Architecture**: Dash app runs independently (no separate backend required)
2. **Paper Trading Default**: Trading bot uses paper trading by default
3. **MT5 Optional**: Requires MetaTrader 5 terminal for live trading
4. **Yahoo Finance Limits**: Free tier has rate limits; app gracefully degrades to synthetic data
5. **Risk Warning**: Trading involves substantial risk; always test with paper trading first
6. **Ollama Now Working**: AI Analysis tab uses Ollama for intelligent responses
7. **FinBERT Loaded**: First request may be slow as model loads (cached after)

---

## Known Issues & Architecture Gaps

1. **Trading Bot Integration**: Bot runs standalone, not integrated with dashboard
2. **News Database**: No SQLite database for historical news storage (planned)
3. **No Time Filter**: News doesn't have time filtering (1min, 1hour, Today, etc.)
4. **Refresh Button**: Doesn't fetch fresh news, just rebuilds from cache
5. **RL Training**: Agent can be trained but results not prominently displayed
6. **Backtest Panel**: Not implemented in UI
7. **Alert System**: Not implemented

---

## Future Enhancements (Planned)

- [ ] Add News database (SQLite) with 90-day retention
- [ ] Fix refresh button to actually fetch fresh news
- [ ] Add time filter to news (1min, 1hour, Today, Week, Month, All)
- [ ] Add more web scraping sources (no APIs)
- [ ] View old news via database with calendar picker
- [ ] Enhanced AI context with all platform data for better advice
- [ ] Add sentiment trends chart over time
- [ ] React/TypeScript frontend
- [ ] Flask backend with WebSocket support
- [ ] SQLite database for trade persistence
- [ ] Real-time MT5/Exness trading execution
- [ ] Integrate trading bot with dashboard UI
- [ ] Implement backtest panel
- [ ] Implement alert system