# AGENTS.md — Trading Terminal (Dash Frontend)

> This file is for AI coding agents. It describes the project's architecture, conventions, and development workflows. Read this before modifying code.

---

## 1. Project Overview

This is a **professional algorithmic trading dashboard** built with **Plotly Dash**. It is a single Python process that serves a web UI at `http://localhost:8050`. There is no separate backend server — Dash callbacks handle both the frontend rendering and the business logic.

The project also contains a **standalone trading bot** (`trading_bot/`) that can run independently of the dashboard for paper or live trading via MetaTrader 5 / Exness.

**Primary purpose:** Real-time market data visualization, quantitative finance modelling, AI-driven news sentiment, and signal generation for XAUUSD, BTCUSD, EURUSD, GBPUSD, and major indices.

**Important disclaimer:** The code contains extensive trading logic, but the project is explicitly marked as **for educational purposes only**. Never assume live trading code is production-ready without review.

---

## 2. Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Web framework | Plotly Dash | >=2.14.0 |
| UI components | Dash Bootstrap Components (Cyborg theme) | >=1.5.0 |
| Charts | Plotly | >=5.18.0 |
| Data | pandas, numpy, scipy, scikit-learn | >=2.0, >=1.24, >=1.7, >=1.3 |
| Market data | yfinance | >=0.2.31 |
| Deep learning | PyTorch | >=2.1.0 |
| RL environments | gymnasium | >=0.29.0 |
| Sentiment | transformers (FinBERT), tokenizers | >=4.30.0 |
| Scraping | beautifulsoup4, lxml, feedparser | >=4.12, >=4.9, >=6.0 |
| HTTP | requests, httpx, aiohttp | >=2.31, >=0.25, >=3.9 |
| ML classifiers | xgboost, pykalman, hmmlearn | >=2.0, >=0.9.5, >=0.3.0 |
| Local LLM | Ollama (optional, external) | llama3.2:1b |
| Cloud AI | Anthropic Claude API (optional) | >=0.40.0 |

**No package manager files** such as `pyproject.toml`, `setup.py`, `setup.cfg`, or `tox.ini` exist. Dependencies are declared only in `requirements.txt`.

---

## 3. Project Structure

```
frontend/                         # Project root (working directory)
├── app.py                        # Main Dash application (~6 000 lines)
├── requirements.txt              # Python dependencies
├── .env                          # Secrets (API keys, not committed)
├── CONTEXT.md                    # Human-maintained project context
├── README.md                     # Human-facing quick-start guide
│
├── assets/
│   ├── style.css                 # Pure-black theme overrides
│   └── chart_enhancements.js     # Custom Plotly chart JS
│
├── pages/                        # Dash multi-page routing
│   ├── __init__.py
│   ├── dashboard.py              # Tiny redirect to app.py root
│   ├── news.py                   # News page with async loading
│   ├── precision_strategy.py     # 6-layer precision trading UI
│   └── smma_strategy.py          # SMMA / synthetic order-book UI
│
├── services/                     # Core business logic
│   ├── __init__.py
│   ├── advanced_models.py        # Black-Scholes, Heston, SABR, VaR
│   ├── ai_news.py                # Unified news aggregator
│   ├── claude_service.py         # Anthropic Claude wrapper
│   ├── deepseek_sentiment.py     # Deprecated / unused
│   ├── local_ai_service.py       # FinBERT + Ollama wrapper
│   ├── market_context_builder.py # Assembles all data for AI prompts
│   ├── market_data.py            # Yahoo Finance helpers
│   ├── markov_model.py           # HMM regime detection
│   ├── news_cache.py             # Persistent JSON news cache
│   ├── news_db.py                # SQLite news database (90-day retention)
│   ├── news_scraper.py           # Multi-source scrapers
│   ├── precision_trading_system.py # 1-min precision engine (~144k lines)
│   ├── rl_agent.py               # Tabular Q-learning agent
│   ├── gold_rl_trainer.py        # Dueling Double DQN trainer
│   ├── gold_rl_env.py            # Gold 1-min RL environment
│   ├── gold_rl_dueling.py        # Dueling DQN network
│   ├── gold_rl_seq_agent.py      # Sequence-based RL agent
│   ├── gold_rl_backtest.py       # RL backtest engine
│   ├── training_pipeline.py      # Precision model training pipeline
│   ├── trading_memory.py         # SQLite trading memory (disabled by default)
│   └── temp1.1.py                # Scratch / temp file (ignore)
│
├── trading_bot/                  # Standalone bot (independent of Dash)
│   ├── run_bot.py                # CLI entry point
│   ├── trading_bot.py            # Core bot logic
│   ├── backtester.py             # Backtesting engine
│   ├── exness_bridge.py          # MT5/Exness execution
│   ├── technical_indicators.py   # 50+ indicators
│   ├── quantitative_signals.py   # Mean-reversion, momentum
│   ├── risk_management.py        # Kelly, position sizing
│   ├── sentiment_analysis.py     # News sentiment
│   ├── config.json               # Bot JSON config
│   └── requirements.txt          # Bot-specific deps (subset)
│
├── data/                         # Runtime data & model artifacts
│   ├── historical/               # CSV OHLCV data (1m, 1h, daily)
│   ├── models/                   # Saved model pickles per asset
│   ├── experiments/              # Experiment outputs
│   ├── feature_store/            # Feature caches
│   ├── news.db                   # SQLite news DB
│   ├── news_cache.json           # JSON news cache
│   ├── gold_rl_model.pt          # PyTorch DQN weights
│   ├── gold_rl_seq_model.pt      # PyTorch sequence-model weights
│   ├── gold_rl_scaler.pkl        # sklearn StandardScaler
│   ├── gold_rl_history.json      # Training history
│   ├── rl_state.pkl              # Tabular Q-table checkpoint
│   └── precision_backtest.db     # Backtest results DB
│
├── notebooks/                    # Jupyter training notebooks
│   ├── precision_full_training.ipynb
│   └── precision_gpu_transformer_training.ipynb
│
├── scripts/
│   └── build_historical_data.py  # Build canonical CSVs from yfinance
│
├── docs/
│   └── precision_v5_research.md  # Research notes & SOTA targets
│
└── plan/                         # Empty design-plan directory
```

### Key architectural notes

- **`app.py`** is monolithic (~6 000 lines). It defines the Dash app singleton, global state, all callback functions, and the main layout. Dash multi-page support is used (`pages/` folder), but the bulk of the UI logic still lives in `app.py`.
- **`services/precision_trading_system.py`** is the largest single file (~144 000 lines). It contains the full 6-layer precision trading engine (microstructure cleaning, flow/toxicity features, ML signal generation, execution logic). Treat it as a library — do not try to read it top-to-bottom.
- **The `trading_bot/` package is fully standalone.** It has its own `requirements.txt` and `config.json` and can be run without the Dash app.
- **There is no `components/` or `layouts/` directory in active use.** The `plan/` and `layouts/` directories are essentially empty.

---

## 4. Build, Run, and Test Commands

### Virtual environment

The project uses a `venv/` directory inside the project root. **Always activate it before working.**

```bash
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

If you only need the trading bot:

```bash
pip install -r trading_bot/requirements.txt
```

### Run the dashboard

```bash
python app.py
```

- URL: `http://localhost:8050`
- The app auto-reloads when Python files change (Dash dev mode is implicit via `__name__ == '__main__'`).

### Run the trading bot

```bash
cd trading_bot
python run_bot.py --paper    # Paper trading mode (default)
python run_bot.py --live     # Live Exness / MT5 (requires MT5 terminal)
python run_bot.py --backtest # Run backtest
```

### Run the historical-data builder

```bash
python scripts/build_historical_data.py        # All pairs
python scripts/build_historical_data.py XAUUSD # Single pair
```

### Note on testing

**There is no formal test suite** (no `pytest`, no `unittest` files, no CI pipeline). Validation is performed through:

1. **Jupyter notebooks** in `notebooks/` — full training pipelines with walk-forward CV.
2. **Backtesting engines** — `services/gold_rl_backtest.py` and `trading_bot/backtester.py`.
3. **Dash UI smoke tests** — start the app and verify callbacks load without `Duplicate callback outputs` errors.
4. **Research docs** — `docs/precision_v5_research.md` tracks known bugs, SOTA benchmarks, and realistic metric targets.

When you modify callback logic, always restart `app.py` and check the browser console / terminal for callback exceptions.

---

## 5. Code Style and Conventions

### Language

All code, comments, and docstrings are written in **English**.

### Imports

- **Lazy imports are common and encouraged** for heavy optional dependencies (torch, transformers, anthropic). Use `try/except ImportError` and set a boolean flag (e.g., `TORCH_AVAILABLE`).
- **Pre-import PIL** in the main thread before any background threads start, to avoid a race condition with `transformers` / `torchvision`:
  ```python
  try:
      import PIL.Image
      import PIL.ImageFile
      import PIL.JpegImagePlugin
  except ImportError:
      pass
  ```
- **Load `.env` before service imports** in `app.py` so that `os.getenv()` works for modules that read env vars at import time.

### Global state

- Dash callbacks rely on **module-level global variables** for state that must survive across callbacks (e.g., `rl_agent_state`, `models_state`, `news_cache`).
- **Always use `threading.Lock`** when mutable global state is accessed from callbacks, because Dash callbacks run in a `ThreadPoolExecutor` by default.
- Example pattern seen throughout:
  ```python
  _systems: Dict[str, PrecisionTradingSystem] = {}
  _systems_lock = threading.Lock()
  ```

### Graceful degradation

- Every external API or optional service must fail gracefully:
  - Yahoo Finance rate-limited? → `generate_fallback_data()` returns synthetic OHLCV.
  - Ollama not running? → Keyword-based sentiment fallback.
  - FinBERT not installed? → Skip sentiment refinement.
  - Claude API key missing? → Fallback to Ollama → keyword engine.
- This is not just a preference; it is an architectural requirement because the app is often run on laptops with intermittent internet.

### Colour constants

The UI uses a **pure-black theme**. The canonical palette is defined in `app.py` as `COLORS` and copied into page modules as a local `C` dict. When adding new UI code, reuse these exact values:

```python
COLORS = {
    "background": "#000000",
    "surface": "#0a0a0a",
    "surface_light": "#121212",
    "primary": "#1a1a1a",
    "accent": "#00ff88",      # Neon green (success)
    "success": "#00ff88",
    "danger": "#ff4757",      # Red
    "warning": "#ffa502",     # Orange
    "info": "#00d4ff",        # Cyan
    "text": "#ffffff",
    "text_secondary": "#888888",
    "border": "#222222",
    "grid": "#1a1a1a",
}
```

### File size

- `app.py` and `services/precision_trading_system.py` are intentionally monolithic. Do **not** refactor them into dozens of small files unless explicitly asked — the authors prefer locality over modularity for these core files.
- For new features, prefer adding a new `services/*.py` module and importing it into `app.py`.

### Type hints

- Type hints are used sporadically. New code should use them where practical (`typing.Dict`, `typing.Optional`, `typing.List`).
- Dataclasses (`@dataclass`) are preferred for configuration objects.

---

## 6. Configuration and Secrets

### Environment variables (`.env`)

The `.env` file is **not committed** (see `.gitignore`). Expected variables:

| Variable | Purpose |
|----------|---------|
| `NEWSAPI_KEY` | NewsAPI.org access |
| `MARKETAUX_API_KEY` | Marketaux API (often returns 401 on free tier) |
| `ANTHAVANTAGE_KEY` | Alpha Vantage (optional) |
| `ANTHROPIC_API_KEY` | Claude AI service |
| `DEEPSEEK_API_KEY` | Deprecated; unused |
| `ENABLE_TRADING_MEMORY` | Set to `true` to enable GraphMem SQLite logging |

### JSON config (`trading_bot/config.json`)

Controls the standalone bot: paper/live mode, risk per trade, indicator parameters, Exness credentials, and API keys.

---

## 7. Data and Persistence

### SQLite databases

| File | Schema | Retention |
|------|--------|-----------|
| `data/news.db` | News articles, full-text search, sentiment | 90 days (auto-cleanup) |
| `data/precision_backtest.db` | Backtest trades, metrics | Manual |
| `trading_memory.db` | Trading decisions, signals | Manual (disabled by default) |

### Pickled / serialized model artifacts

| File | Format | Description |
|------|--------|-------------|
| `data/gold_rl_model.pt` | PyTorch state_dict | Dueling DQN weights |
| `data/gold_rl_seq_model.pt` | PyTorch state_dict | Sequence-model weights |
| `data/gold_rl_scaler.pkl` | pickle (sklearn) | StandardScaler |
| `data/rl_state.pkl` | pickle (custom) | Q-table + epsilon |
| `data/models/*/*.pkl` | pickle | Precision ensemble per asset |

**Warning:** Loading pickle files on startup is done in **daemon threads** so that a corrupt or OOM-prone model never blocks the Dash app from booting.

---

## 8. Security Considerations

- **Do not commit `.env`.** It contains live API keys.
- **Pickle files are trusted.** The code `pickle.load()`s model artifacts from `data/`. Do not load pickles from untrusted sources.
- **No input sanitization on SQLite.** The news DB uses parameterized queries, but if you extend raw SQL paths, use placeholders.
- **No authentication.** The Dash app runs on localhost without login. Do not expose port 8050 to the public internet without a reverse proxy and auth.
- **Trading bot defaults to paper mode.** The `config.json` has `"paper_trading": true`. Changing this to live requires an active MT5 terminal and valid Exness credentials.
- **Yahoo Finance data is free-tier.** Rate limits apply; the app falls back to synthetic data, but repeated rapid polling may get your IP throttled.

---

## 9. Common Pitfalls for Agents

1. **Duplicate callback outputs** — If you see this at boot, it usually means `app.py` was imported twice. The file contains a `sys.modules.setdefault("app", ...)` guard; respect it.
2. **FinBERT OOM** — Running FinBERT inference inside a Dash callback that fires every 15 seconds will exhaust RAM. The existing code caches sentiment aggregates for 60 seconds and runs refinement in daemon threads. Follow that pattern.
3. **yfinance MultiIndex columns** — Newer versions return `MultiIndex` columns. The code already flattens them, but if you write new data-fetching code, handle both shapes.
4. **Precision system is huge** — `precision_trading_system.py` is ~144 000 lines. If you need to change a feature, grep for the specific class or function rather than reading the whole file.
5. **No tests to run** — Do not expect `pytest` to pass. Verify changes by running the app and checking the UI manually.
6. **Thread safety** — Dash callbacks are concurrent. Any mutable global state must be protected with `threading.Lock`.

---

## 10. Useful References

- `CONTEXT.md` — Human-maintained changelog and architecture notes (more detailed than this file for specific features).
- `docs/precision_v5_research.md` — Research-backed feature additions, bug fixes, and realistic SOTA targets.
- `README.md` — Quick-start for human users.
- `.github/copilot-instructions.md` — Brief note about `graphify-out/GRAPH_REPORT.md` for architecture questions.
