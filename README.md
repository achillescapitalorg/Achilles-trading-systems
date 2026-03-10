# Trading Terminal - Dash Frontend

A professional algorithmic trading dashboard built with **Plotly Dash** featuring real-time market data.

![Dashboard](https://img.shields.io/badge/Status-Running-success)
![Data](https://img.shields.io/badge/Data-Yahoo%20Finance-blue)
![Theme](https://img.shields.io/badge/Theme-Pure%20Black-black)

## ✨ Features

- 📊 **Real-time Price Charts** - Live candlestick charts with volume from Yahoo Finance
- 📉 **Volatility Surface** - 3D visualization of implied volatility
- 📈 **Market Metrics** - Hurst exponent, skewness, kurtosis, historical volatility
- 🎯 **Trading Signals** - Technical indicators (RSI, MACD, Bollinger, Supertrend)
- 📝 **Order Form** - Buy/Sell orders with stop loss and take profit
- 📰 **News Feed** - Clickable links to original articles
- 🔄 **Auto-Refresh** - Updates every 5 seconds

## 📊 Trading Instruments (Real-time Yahoo Finance)

| Symbol | Name | Yahoo Ticker |
|--------|------|--------------|
| XAUUSD | Gold | GC=F |
| BTCUSD | Bitcoin | BTC-USD |
| ETHUSD | Ethereum | ETH-USD |
| EURUSD | Euro/USD | EURUSD=X |
| GBPUSD | GBP/USD | GBPUSD=X |
| USDJPY | USD/JPY | USDJPY=X |
| SPX500 | S&P 500 | ^GSPC |
| NAS100 | Nasdaq 100 | ^NDX |

## 🚀 Quick Start

```bash
# Navigate to frontend directory
cd trading_terminal/frontend

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start the Dash app
python app.py
```

**Frontend:** http://localhost:8050

## 🔌 Backend Integration

The frontend connects to the FastAPI backend at **http://localhost:8000**.

To start the backend:
```bash
cd ../backend
python main_advanced.py
```

## 📁 Project Structure

```
frontend/
├── app.py              # Main Dash application
├── requirements.txt    # Python dependencies
├── assets/
│   └── style.css      # Pure black theme CSS
├── components/        # Reusable components
└── layouts/           # Page layouts
```

## 🎨 UI Theme

**Pure Black Theme** - A sleek, professional dark interface:
- Background: `#000000` (Pure Black)
- Surface: `#0a0a0a` (Near Black)
- Accent: `#00ff88` (Neon Green)
- Success: `#00ff88` | Danger: `#ff4757` | Warning: `#ffa502`

## 📈 Data Sources

- **Primary**: Yahoo Finance API (Real-time)
- **Fallback**: Synthetic data generation (if API unavailable)

### Timeframes Available
- 5m, 15m, 1h, 4h, 1D

## 🔗 News Feed

All news items are clickable and redirect to original sources:
- Bloomberg
- CoinDesk
- Reuters
- CNBC
- MarketWatch
- Financial Times

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| B | Buy order |
| S | Sell order |
| 1-8 | Select instrument |

## 🛠️ Technology Stack

- **Frontend Framework**: Plotly Dash 4.0
- **UI Components**: Dash Bootstrap Components (Cyborg theme)
- **Charts**: Plotly (Candlestick, 3D Surface)
- **Data**: yfinance (Yahoo Finance API)
- **Styling**: Custom Pure Black CSS

## 📊 Chart Features

- ✅ Real-time price updates
- ✅ Volume bars with color coding
- ✅ Price change percentage display
- ✅ Interactive hover tooltips
- ✅ Black theme optimized

## ⚠️ Disclaimer

**This software is for EDUCATIONAL purposes only.**

- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- Always test in paper trading first
- Never risk more than you can afford to lose
- Data delays may occur with free API tiers

## 📝 API Rate Limits

Yahoo Finance free tier has rate limits. If charts show fallback data:
- Wait a few minutes between refreshes
- Consider using a different timeframe
- The app gracefully degrades to synthetic data

## 🎯 Signal Indicators

| Indicator | Description |
|-----------|-------------|
| RSI (14) | Relative Strength Index |
| MACD | Moving Average Convergence Divergence |
| Bollinger (20) | Bollinger Bands |
| Supertrend | Trend following indicator |

## 📰 News Sentiment

- 🟢 **Bullish** (> 0.3) - Positive market sentiment
- 🟡 **Neutral** (-0.3 to 0.3) - Mixed sentiment
- 🔴 **Bearish** (< -0.3) - Negative market sentiment

---

**Built with ❤️ for traders**
