# Trading Terminal - Project Context

## Overview

A professional algorithmic trading dashboard built with **Plotly Dash** featuring real-time market data, quantitative finance models, and a standalone trading bot for Gold (XAUUSD), Bitcoin (BTCUSD), and other instruments.

**Note:** This project uses a Dash-based frontend. A React/TypeScript frontend and Flask backend were originally planned but not implemented.

---

## Current Status (Updated April 2026)

### Recently Completed Features

1. **Replaced DeepSeek with Local AI (FinBERT + Ollama)**
   - Created `services/local_ai_service.py` - New unified AI service
   - Uses FinBERT (ProsusAI/finbert) for financial sentiment analysis
   - Uses Ollama (llama3.2:1b) for local LLM-powered market analysis
   - Added keyword-based fallback when Ollama not available

2. **News Database**
   - Created `services/news_db.py` - SQLite database with 90-day retention
   - Auto-cleanup of news older than 90 days
   - Full-text search on headlines
   - Query by date range, instrument, time filter

3. **Market Context Builder**
   - Created `services/market_context_builder.py` - Assembles ALL platform data
   - Parallel fetch: price, signals, Heston, Markov, GARCH, risk, MC, news
   - 8-second deadline with fallbacks
   - Formats context for both Claude and Ollama prompts

4. **Enhanced News Page**
   - Time filter dropdown: All / Last Hour / Today / Week / Month
   - Calendar picker for specific date lookup
   - Falls back to cache if DB empty
   - Added Economic Calendar tab
   - Added AI Analysis tab

5. **More RSS Sources**
   - Added: Bing News, Seeking Alpha, Economist
   - Auto-saves fetched news to DB

6. **Fixed AI Prediction Tab**
   - Fixed Heston parameter calculation (theta was incorrectly calculated)
   - Changed probability thresholds from 5%/10% to 1%/2% for realistic 30-day predictions
   - Added asset-specific default probabilities

7. **Added Markov Model Predictions**
   - NEXT REGIME prediction with probability
   - PREDICTION direction (Bullish/Bearish/Sideways)
   - RISK LEVEL (HIGH/MEDIUM/LOW)

---

## Architecture

### Main Application (`/frontend`)
- **Framework**: Plotly Dash (standalone web app)
- **Backend**: None (Dash handles both frontend and API)
- **Styling**: Dash Bootstrap Components (Cyborg theme) + Custom CSS
- **Charts**: Plotly (Candlestick, 3D Surface)
- **State**: Python in-memory state + Dash callbacks

### AI Services
- **FinBERT**: Financial sentiment analysis (HuggingFace ProsusAI/finbert)
- **Ollama**: Local LLM for market analysis (llama3.2:1b model)
- **Keyword Fallback**: Provides trading tips when Ollama not available

### Trading Bot (`/trading_bot/`)
- **Standalone**: Can run independently of the Dash UI
- **Broker Integration**: Exness MT5 (paper trading by default)
- **Data**: Yahoo Finance for market data

---

## Project Structure

```
frontend/
├── app.py                      # Main Dash application (~6086 lines)
├── volatility_models.py        # GARCH, EGARCH, Heston models (1051 lines)
├── requirements.txt            # Python dependencies
├── assets/
│   ├── chart_enhancements.js  # Custom chart JS
│   └── style.css              # Pure black theme CSS
├── services/
│   ├── __init__.py
│   ├── market_data.py         # Yahoo Finance data fetching (260 lines)
│   ├── news_scraper.py       # Multi-source news (1452 lines)
│   ├── advanced_models.py    # Black-Scholes, Heston, VaR (1648 lines)
│   ├── markov_model.py       # Hidden Markov Model (270 lines)
│   ├── ai_news.py           # AI-powered news service (1000 lines)
│   ├── local_ai_service.py   # FinBERT + Ollama wrapper (420 lines)
│   ├── news_db.py          # SQLite news database (391 lines)
│   ├── news_cache.py        # Persistent news cache (366 lines)
│   ├── market_context_builder.py # AI context builder (623 lines)
│   ├── markov_model.py     # Markov regime detection
│   ├── claude_service.py   # Claude API integration (199 lines)
│   ├── rl_agent.py        # Q-Learning trading agent (630 lines)
│   ├── gold_rl_trainer.py  # Deep RL trainer (991 lines)
│   ├── gold_rl_env.py    # RL environment (419 lines)
│   ├── gold_rl_dueling.py # Dueling DQN (279 lines)
│   └── deepseek_sentiment.py # OLD: DeepSeek (not used)
├── trading_bot/
│   ├── __init__.py
│   ├── trading_bot.py          # Main multi-signal bot
│   ├── exness_bridge.py      # MT5/Exness execution
│   ├── technical_indicators.py # 50+ indicators
│   ├── quantitative_signals.py # Mean reversion, momentum
│   ├── risk_management.py   # Kelly, position sizing
│   ├── sentiment_analysis.py  # News sentiment
│   ├── backtester.py        # Backtesting engine
│   ├── run_bot.py           # Bot runner script
│   └── config.json          # Bot configuration
├── pages/
│   ├── __init__.py
│   ├── dashboard.py        # Dashboard page layout
│   ├── news.py             # News page with tabs (487 lines)
│   └── smma_strategy.py    # SMMA/RL strategy (1074 lines)
├── data/
│   └── news.db            # SQLite news database
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

### `services/news_db.py` - NewsDatabase
- `save_news(news_items, instrument)` - Save news to SQLite database
- `get_news_by_date_range(start, end, instrument, limit)` - Query by date range
- `get_news_by_time_filter(instrument, time_filter, limit)` - Query by time filter
- `get_news_by_date(date_str, instrument, limit)` - Query by specific date
- `search_news(query, limit)` - Full-text search on headlines
- `get_sentiment_trend(instrument, days)` - Daily sentiment summary
- `cleanup_old_news(days)` - Auto-cleanup old news

### `services/ai_news.py` - UnifiedNewsService
- `get_news(symbol, max_items)` - Main method to fetch news from all sources
- `_fetch_all_sources_parallel(symbol, keywords)` - Parallel fetching
- `_fetch_newsapi(keywords)` - Fetch from NewsAPI
- `_fetch_google_news(keywords, symbol)` - Fetch from Google News RSS
- `_fetch_forexcom_news(keywords)` - Fetch from Forex.com
- `_fetch_marketaux(symbol)` - Fetch from Marketaux API
- `_deduplicate_and_rank(items)` - Deduplicate and rank by source priority

**News Sources:**
- NewsAPI (working)
- Google News RSS
- Forex.com
- RSS Feeds: ForexLive, FXStreet, Kitco, DailyFX, CNBC, etc.

### `services/local_ai_service.py` - LocalAIService
- `get_sentiment(text)` - Analyze single text sentiment
- `get_sentiment_batch(texts)` - Batch sentiment analysis
- `analyze_market(symbol, news, price_data)` - Market analysis with LLM
- `chat(message, context)` - Chat with Ollama LLM

### `services/market_context_builder.py` - MarketContext
- `build_market_context(symbol)` - Assemble all platform data
- `format_for_prompt(ctx)` - Format for Claude/gpt4all prompts
- `format_for_ollama_prompt(ctx)` - Compact format for small LLMs

**MarketContext dataclass contains:**
- Price, trend direction, current regime
- Technical signals (RSI, MACD, BB, etc.)
- Unified recommendation (action, confidence, entry, SL, TP)
- Heston parameters (kappa, theta, xi, rho, v0)
- Markov regime (current, next, transition probability)
- GARCH forecast (1-day, 5-day, 22-day volatility)
- Risk metrics (VaR, Expected Shortfall, Max Drawdown, Sharpe, Sortino)
- Monte Carlo (30-day mean price, probability up, 95% CI)
- FinBERT aggregation (dominant sentiment, weighted score, breakdown)
- Top news headlines

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
- `MARKETAUX_API_KEY` - Marketaux API (may return 401)
- `DEEPSEEK_API_KEY` - DeepSeek API (not used)
- `ANTHAVANTAGE_KEY` - Alpha Vantage API (optional)

### Chart Timeframes
- 5m, 15m, 1h, 4h, 1D

---

## Data Sources

### Primary
- **Yahoo Finance** - Real-time prices and historical data

### Fallback
- **Synthetic data generation** - If Yahoo Finance unavailable

### News (Current)
- **NewsAPI** - Working
- **Google News RSS** - Filtered by 5 days
- **RSS Feeds** - ForexLive, FXStreet, Kitco, DailyFX, CNBC, etc.

### Sentiment (Updated)
- **FinBERT** - Local transformer model
- **Ollama** - Local LLM (llama3.2:1b)
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
To use local LLM for market analysis:
```bash
ollama serve  # Start Ollama server in background
ollama list   # Check installed models
ollama run llama3.2:1b  # Install model if needed
```

---

## Important Notes

1. **Standalone Architecture**: Dash app runs independently (no separate backend required)
2. **Paper Trading Default**: Trading bot uses paper trading by default
3. **MT5 Optional**: Requires MetaTrader 5 terminal for live trading
4. **Yahoo Finance Limits**: Free tier has rate limits; app gracefully degrades to synthetic data
5. **Risk Warning**: Trading involves substantial risk; always test with paper trading first
6. **FinBERT Slow on First Request**: First sentiment analysis may be slow as model loads
7. **Ollama Optional**: Falls back to keyword engine if Ollama not available

---

## Known Issues

1. **Network Errors**: When offline, API calls fail - app uses cached/fallback data
2. **Duplicate Callbacks**: Fixed with prevent_initial_call=True
3. **NewsAPI Rate Limits**: May be throttled on free tier
4. **Marketaux API**: May return 401 (API key issues)

---

## Future Enhancements

- [ ] React/TypeScript frontend
- [ ] Flask backend with WebSocket support
- [ ] SQLite database for trade persistence
- [ ] Real-time MT5/Exness trading execution
- [ ] Integrate trading bot with dashboard UI
- [ ] Implement backtest panel
- [ ] Implement alert system
- [ ] Add sentiment trends chart over time

---

## File Statistics

| File | Lines | Purpose |
|------|-------|---------|
| app.py | 6086 | Main Dash application |
| services/advanced_models.py | 1648 | Financial models |
| services/news_scraper.py | 1452 | News scraping |
| pages/smma_strategy.py | 1074 | SMMA/RL trading |
| volatility_models.py | 1051 | Volatility models |
| services/ai_news.py | 1000 | AI news service |
| services/gold_rl_trainer.py | 991 | Deep RL trainer |
| services/market_context_builder.py | 623 | AI context |
| services/rl_agent.py | 630 | RL agent |
| services/local_ai_service.py | 420 | FinBERT + Ollama |
| services/gold_rl_env.py | 419 | RL environment |
| services/news_db.py | 391 | SQLite DB |
| services/gold_rl_seq_agent.py | 388 | Seq RL agent |
| services/news_cache.py | 366 | News cache |
| pages/news.py | 487 | News page |
| services/claude_service.py | 199 | Claude API |
| services/markov_model.py | 270 | Markov model |
| services/market_data.py | 260 | Market data |
| services/gold_rl_backtest.py | 429 | Backtesting |
| services/gold_rl_dueling.py | 279 | Dueling DQN |
| services/deepseek_sentiment.py | 298 | DeepSeek (unused) |