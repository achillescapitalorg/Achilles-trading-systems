# Trading Terminal - Project Context

## Overview
A comprehensive trading terminal application with a Flask backend, React/TypeScript frontend, and MT5/Exness trading integration.

## Architecture

### Backend (`/backend`)
- **Framework**: Flask with async support
- **Database**: SQLite (`trading_data.db`)
- **Key Patterns**: Singleton, Factory, Strategy, Pipeline patterns

### Frontend (`/frontend`)
- **Framework**: React with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS + shadcn/ui components
- **Charts**: Recharts
- **State**: React Query for server state

## Key Dependencies

### Backend
- `flask`, `flask-cors`, `flask-socketio`
- `sqlalchemy` (ORM)
- `python-dotenv`
- `yfinance` (stock data)
- `MetaTrader5` (MT5 trading)
- `requests`, `beautifulsoup4` (web scraping)
- `apscheduler` (scheduling)
- `ta`, `pandas`, `numpy` (technical analysis)

### Frontend
- `react`, `react-dom`, `react-router-dom`
- `@tanstack/react-query`
- `recharts` (visualization)
- `framer-motion` (animations)
- `lucide-react` (icons)
- `clsx`, `tailwind-merge` (utilities)
- `zustand` (client state)

## Core Components

### Backend Modules

#### `app.py` (3068 lines)
Main Flask application with WebSocket support (`flask-socketio`). Handles:
- Trading signals and portfolio management
- Real-time market data streaming
- WebSocket events for live updates
- API endpoints for all trading operations

#### `ExnessBroker` (`/backend/services/exness_service.py`)
- Implements `BrokerInterface`
- Manages MT5 connection via Exness
- Methods: `connect()`, `get_account_info()`, `get_positions()`, `get_pending_orders()`, `open_position()`, `close_position()`, `modify_position()`, `get_symbols()`, `get_candles()`

#### `DataManager` (`/backend/services/data_manager.py`)
- Singleton pattern for market data
- Methods: `get_candles()`, `get_ticker()`, `get_forex_symbols()`
- Data sources: Yahoo Finance, Exness MT5

#### `VolatilityModels` (`/backend/services/volatility_models.py`)
Volatility calculations and regime detection:
- `VolatilityCalculator`: ATR, Bollinger Bands, Standard Deviation
- `VolatilityRegimeDetector`: Identifies low/medium/high volatility regimes
- `VolatilitySignalGenerator`: Generates volatility-based trading signals

#### `NewsService` (`/backend/services/news_service.py`)
- Async web scraping for news
- Uses `httpx` for async HTTP requests
- Scrapes forexfactory.com and news sources

#### `TradingBot` (`/backend/services/trading_bot.py`)
- Multi-timeframe analysis
- Implements entry/exit strategies
- Risk management rules

### Frontend Components

#### Dashboard (`/src/components/dashboard/`)
- `Dashboard.tsx` - Main trading dashboard
- `MarketOverview.tsx` - Market summary and indices
- `PortfolioSummary.tsx` - Account and position overview
- `SignalTable.tsx` - Trading signals display

#### Charts (`/src/components/charts/`)
- `MainChart.tsx` - Primary price chart with indicators
- `IndicatorPanel.tsx` - Technical indicator configuration
- `ChartToolbar.tsx` - Timeframe and chart type controls

#### Trading (`/src/components/trading/`)
- `TradePanel.tsx` - Trade execution interface
- `OrderBook.tsx` - Market depth display
- `PositionsPanel.tsx` - Open positions management
- `HistoryPanel.tsx` - Trade history
- `SignalGenerator.tsx` - Trading signal generation UI

#### Broker Integration (`/src/components/broker/`)
- `BrokerConnection.tsx` - MT5/Exness connection status
- `BrokerSettings.tsx` - Broker configuration

## API Endpoints

### Market Data
- `GET /api/market/candles/<symbol>` - OHLCV data
- `GET /api/market/ticker/<symbol>` - Current price
- `GET /api/market/forex-symbols` - Available forex pairs

### Trading
- `GET /api/trading/positions` - Open positions
- `POST /api/trading/open-position` - Open new position
- `POST /api/trading/close-position` - Close position
- `GET /api/trading/history` - Trade history

### Signals
- `GET /api/signals` - All trading signals
- `POST /api/signals/generate` - Generate new signal
- `POST /api/signals/execute/<id>` - Execute signal

### Broker
- `GET /api/broker/info` - Account information
- `GET /api/broker/connection-status` - MT5 connection status

### WebSocket Events
- `market_update` - Real-time price updates
- `signal_update` - New signals generated
- `position_update` - Position changes
- `volatility_update` - Volatility regime changes

## Configuration

### Environment Variables (`.env`)
- `MT5_PATH` - MetaTrader 5 terminal path
- `EXNESS_ACCOUNT` - Exness account ID
- `FLASK_SECRET_KEY` - Flask secret key
- `DATABASE_URL` - SQLite database path

### Chart Configuration
- Default indicators: SMA, EMA, RSI, MACD, Bollinger Bands
- Timeframes: M1, M5, M15, H1, H4, D1

## Technical Analysis Features

### Indicators
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- VWAP
- Stochastic Oscillator
- ADX (Average Directional Index)
- Fibonacci Retracements

### Volatility Analysis
- Real-time ATR calculation
- Bollinger Band width analysis
- Standard deviation monitoring
- Regime detection (low/medium/high volatility)

## Development

### Running the Application
```bash
# Backend
cd backend
python app.py

# Frontend
npm run dev
```

### Database Schema
- `signals` table - Trading signals
- `portfolio` table - Portfolio holdings
- `trades` table - Trade history
- `positions` table - Current positions
- `market_data` table - Cached market data

## Important Notes

1. **MT5 Connection**: Requires MetaTrader 5 terminal installed and running
2. **Data Sources**: Yahoo Finance for stocks, MT5/Exness for forex/metal
3. **Async Operations**: News service uses async/await for non-blocking scraping
4. **WebSocket Updates**: Real-time updates via Socket.IO for live trading data
5. **Risk Management**: Built-in position sizing and stop-loss calculations
