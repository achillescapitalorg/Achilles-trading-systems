# Gold/BTC Algorithmic Trading Bot

A professional-grade algorithmic trading system for **Gold (XAUUSD)** and **Bitcoin (BTCUSD)** with multi-source signal confirmation, quantitative analysis, and Exness MT5 integration.

## Features

### 🎯 Multi-Source Signal Confirmation
The bot uses a **3-layer confirmation system** before executing trades:

| Layer | Description | Weight |
|-------|-------------|--------|
| **Technical Analysis** | RSI, MACD, Bollinger Bands, Supertrend, ADX, ATR | 40% |
| **Quantitative Signals** | Mean Reversion, Momentum, Volatility Breakout, Market Regime | 35% |
| **Sentiment Analysis** | News, Social Media, Economic Calendar | 25% |

### 📊 Technical Indicators
- **Trend**: EMA, SMA, HMA, VWAP, ADX, Supertrend
- **Momentum**: RSI, Stochastic, Williams %R, CCI, ROC
- **Volatility**: Bollinger Bands, Keltner Channel, Donchian Channel, ATR
- **Volume**: OBV, MFI, CMF, Accumulation/Distribution
- **Advanced**: Fisher Transform, Hurst Exponent, Squeeze Indicator

### 📈 Quantitative Strategies
- **Mean Reversion**: Statistical z-score, Bollinger mean reversion, RSI extremes
- **Momentum**: Time-series momentum, RSI divergence detection
- **Volatility Breakout**: ATR breakout, Volatility squeeze detection
- **Market Regime**: Regime detection (bull/bear, high/low vol)

### 📰 Sentiment Analysis
- Twitter sentiment analysis
- Reddit sentiment (r/wallstreetbets, r/cryptocurrencies)
- News API integration
- CryptoPanic crypto news
- Gold-specific reports (World Gold Council, LBMA)
- Economic calendar impact analysis

### 💼 Risk Management
- **Position Sizing**: Fixed %, Kelly Criterion, Volatility Targeting
- **Stop Loss**: ATR-based, Fixed %, Support/Resistance, Trailing
- **Portfolio Limits**: Max exposure, Max drawdown, Daily limits
- **Take Profit**: Risk-reward based (default 2:1)

### 🔧 Trading Modes
- **Paper Trading**: Test strategies with virtual money
- **Live Trading**: Execute real trades via Exness MT5
- **Backtesting**: Historical performance analysis

## Installation

```bash
cd Algorithmic_Trading_Machine_Learning/trading_bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Paper Trading (Recommended for Testing)
```bash
python run_bot.py --paper
```

### Demo Analysis
```bash
python run_bot.py --demo
```

### Backtesting
```bash
python run_bot.py --backtest --symbol XAUUSD
python run_bot.py --backtest --symbol BTCUSD
```

### Live Trading (Requires Exness Account)
```bash
# First, configure your Exness credentials in config.json
python run_bot.py --live
```

## Configuration

Edit `config.json` to customize:

```json
{
  "trading": {
    "paper_trading": true,
    "assets": ["XAUUSD", "BTCUSD"],
    "cycle_seconds": 300
  },
  "risk": {
    "risk_per_trade": 1.0,
    "max_drawdown": 10.0,
    "min_confidence": 0.6
  },
  "signals": {
    "weights": {
      "technical": 0.40,
      "quantitative": 0.35,
      "sentiment": 0.25
    }
  }
}
```

## Architecture

```
trading_bot/
├── __init__.py              # Package exports
├── trading_bot.py           # Main bot orchestration
├── technical_indicators.py  # Technical analysis
├── quantitative_signals.py  # Quant strategies
├── sentiment_analysis.py    # News/sentiment
├── exness_bridge.py         # MT5 execution
├── risk_management.py       # Risk controls
├── backtester.py            # Backtesting engine
├── run_bot.py               # Entry point
├── config.json              # Configuration
└── requirements.txt         # Dependencies
```

## Signal Generation Process

```
1. Fetch OHLCV data from MT5
         ↓
2. Calculate Technical Indicators → Technical Signal (BUY/SELL/HOLD)
         ↓
3. Run Quantitative Analysis → Quant Signal (BUY/SELL/HOLD)
         ↓
4. Fetch News/Sentiment → Sentiment Signal (BUY/SELL/HOLD)
         ↓
5. Aggregate Signals (weighted average)
         ↓
6. Check Risk Limits (drawdown, exposure)
         ↓
7. Calculate Position Size & Stop Levels
         ↓
8. Execute Trade via MT5 (if confidence > threshold)
```

## Decision Matrix

| Technical | Quantitative | Sentiment | Action |
|-----------|--------------|-----------|--------|
| BUY | BUY | BUY | **Strong BUY** (High confidence) |
| BUY | BUY | HOLD | **BUY** (Medium confidence) |
| BUY | SELL | HOLD | **HOLD** (Conflicting) |
| HOLD | HOLD | HOLD | **HOLD** (No signal) |
| SELL | SELL | SELL | **Strong SELL** (High confidence) |

## Risk Management

### Position Sizing Formula
```
Risk Amount = Account Balance × (Risk % / 100)
Position Size = Risk Amount / (Entry - Stop Loss)
```

### Stop Loss Calculation
```
ATR-Based: Stop = Entry ± (ATR × Multiplier)
Trailing: Stop follows price, never moves against position
```

### Drawdown Controls
- Trading stops if max drawdown reached
- Daily loss limits
- Maximum correlated exposure limits

## Backtest Performance Metrics

- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Calmar Ratio**: Return vs max drawdown
- **Profit Factor**: Gross profit / Gross loss
- **Win Rate**: Percentage of winning trades
- **Max Drawdown**: Largest peak-to-trough decline
- **VaR/CVaR**: Value at Risk metrics

## Exness MT5 Setup

1. Open Exness account at [exness.com](https://www.exness.com)
2. Download and install MetaTrader 5
3. Create trading account (Demo or Live)
4. Get login credentials
5. Update `config.json`:
```json
{
  "exness": {
    "login": 12345678,
    "password": "your_password",
    "server": "Exness-MT5Trial",
    "demo": true
  }
}
```

## API Keys (Optional)

For enhanced sentiment analysis, add API keys to `config.json`:

```json
{
  "api_keys": {
    "twitter": "your_twitter_api_key",
    "newsapi": "your_newsapi_key",
    "cryptopanic": "your_cryptopanic_key"
  }
}
```

## Disclaimer

⚠️ **Trading involves substantial risk of loss. This software is for educational purposes only.**

- Past performance does not guarantee future results
- Always test strategies in paper trading first
- Never risk more than you can afford to lose
- The authors are not responsible for any trading losses

## License

MIT License - See LICENSE file for details.

## Support

For issues and feature requests, please open an issue on GitHub.
