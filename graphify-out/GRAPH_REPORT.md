# Graph Report - frontend  (2026-05-01)

## Corpus Check
- 42 files · ~118,640 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1834 nodes · 5458 edges · 79 communities detected
- Extraction: 46% EXTRACTED · 54% INFERRED · 0% AMBIGUOUS · INFERRED: 2932 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]

## God Nodes (most connected - your core abstractions)
1. `MarkovRegimeModel` - 189 edges
2. `GoldRLTrainer` - 183 edges
3. `RegimeSwitchingModel` - 173 edges
4. `NewsCache` - 169 edges
5. `QLearningAgent` - 166 edges
6. `HestonModel` - 166 edges
7. `TradingEnvironment` - 164 edges
8. `SABRModel` - 161 edges
9. `NewsArticle` - 161 edges
10. `Action` - 158 edges

## Surprising Connections (you probably didn't know these)
- `fetch_yahoo_finance_data()` --calls--> `update_smma_strategy()`  [INFERRED]
  app.py → pages/smma_strategy.py
- `fetch_yahoo_finance_data()` --calls--> `update_rl_signal()`  [INFERRED]
  app.py → pages/smma_strategy.py
- `fetch_yahoo_finance_data()` --calls--> `execute_auto_trade()`  [INFERRED]
  app.py → pages/smma_strategy.py
- `fetch_yahoo_finance_data()` --calls--> `_fetch_recent_data()`  [INFERRED]
  app.py → pages/precision_strategy.py
- `fetch_yahoo_finance_data()` --calls--> `_fetch_training_data()`  [INFERRED]
  app.py → pages/precision_strategy.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.02
Nodes (195): Enum, reset(), Backtester, BacktestResult, print_backtest_report(), Backtesting Engine =================== Professional backtesting framework with r, Run backtest on historical data.                  Parameters         ----------, Container for backtest trade record. (+187 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (266): _analyze_impact(), _analyze_sentiment(), calculate_adx_safe(), calculate_atr_wilder(), calculate_bb_safe(), calculate_bollinger_bands(), calculate_bs_probability(), calculate_cci_safe() (+258 more)

### Community 2 - "Community 2"
Cohesion: 0.02
Nodes (90): End-to-end Gold RL training + backtest demonstration. Runs the new sequence-base, create_volatility_model(), ModelParameters, Goldman Sachs-Level Volatility Modeling Framework ==============================, Compute GARCH(p, q) log-likelihood.                  Parameters         --------, Fit GARCH(p, q) model using Maximum Likelihood Estimation.                  Para, Compute the full conditional variance series., Compute AIC, BIC, and other fit statistics. (+82 more)

### Community 3 - "Community 3"
Cohesion: 0.02
Nodes (76): _align_tf(), ARIMABaseline, Asset, AssetConfig, _compute_atr(), compute_order_book_imbalance(), compute_realized_volatility(), compute_signed_volume() (+68 more)

### Community 4 - "Community 4"
Cohesion: 0.03
Nodes (68): _build_fold_table(), _empty_wf_fig(), _fetch_recent_data(), _fetch_training_data(), _format_pf(), _get_system(), Precision Strategy Page — 1-minute multi-asset trading system ==================, Fetch recent 1-min OHLCV data via app.py's existing helper. (+60 more)

### Community 5 - "Community 5"
Cohesion: 0.03
Nodes (78): _build_news_tabs(), _calculate_instrument_sentiment(), _detect_topic(), _fetch_and_cache(), _get_ai_market_summary(), _get_all_news(), _get_economic_calendar(), _get_news_with_aggregate() (+70 more)

### Community 6 - "Community 6"
Cohesion: 0.03
Nodes (69): Asset, AssetConfig, BacktestEngine, _calculate_max_drawdown(), _calculate_sharpe(), _compute_atr(), _compute_macd(), _compute_rsi() (+61 more)

### Community 7 - "Community 7"
Cohesion: 0.03
Nodes (54): Trading Terminal Services ========================== Core services for market da, get_market_data_service(), MarketDataService, Market Data Service ==================== Fetches real-time market data from Yaho, Generate mock price data (fallback)., Generate fallback OHLCV data if API fails., Get current price for symbol., Fetch prices for multiple symbols.          Parameters         ---------- (+46 more)

### Community 8 - "Community 8"
Cohesion: 0.04
Nodes (58): create_advanced_metrics_cards(), Create advanced metrics cards - Advanced Tab., calculate_calmar_ratio(), calculate_max_drawdown(), calculate_sharpe_ratio(), calculate_sortino_ratio(), calculate_var(), calibrate() (+50 more)

### Community 9 - "Community 9"
Cohesion: 0.07
Nodes (31): analyze(), analyze_batch(), analyze_market(), analyze_sentiment(), analyze_sentiment_batch(), chat_with_ai(), get_ai_status(), get_local_ai_service() (+23 more)

### Community 10 - "Community 10"
Cohesion: 0.07
Nodes (28): cleanup_old_news(), get_news_by_instrument(), get_news_db(), get_recent_news(), get_sentiment_trend(), NewsDatabase, News Database Module - SQLite Storage with 90-day Retention ====================, Get news within a date range.                  Args:             start_date: Sta (+20 more)

### Community 11 - "Community 11"
Cohesion: 0.08
Nodes (32): ClaudeService, get_claude_service(), Claude AI Service ================= Primary AI backend using the Anthropic Claud, Lazy-init the Anthropic client (thread-safe)., Send a user question to Claude with the full market context prepended., Return a dict suitable for the UI status bar., Return the process-wide ClaudeService singleton., Thin wrapper around the Anthropic Messages API.      Usage:         svc = get_cl (+24 more)

### Community 12 - "Community 12"
Cohesion: 0.08
Nodes (17): DeepQNetwork, DeepRLAgent, Reinforcement Learning Trading Agent ===================================== Imple, Load agent state from disk. Returns True on success., Deep Q-Network for trading., Deep Reinforcement Learning Agent using DQN with Experience Replay., Initialize Deep RL agent.              Parameters             ----------, Get action using epsilon-greedy policy. (+9 more)

### Community 13 - "Community 13"
Cohesion: 0.1
Nodes (19): analyze_batch_sentiment(), analyze_sentiment(), DeepSeekSentiment, get_aggregate_sentiment(), KeywordSentiment, DeepSeek LLM Sentiment Analysis Service Uses DeepSeek API for intelligent financ, Call DeepSeek API for batch headline analysis., Normalize DeepSeek response to standard format. (+11 more)

### Community 14 - "Community 14"
Cohesion: 0.09
Nodes (14): _init_db(), Trading Memory - Auto-captures trading decisions using GraphMem ================, Log pattern discovery., Save to GraphMem or SQLite fallback., Save to SQLite fallback., Query memory - GraphMem or SQLite., Query SQLite fallback., Get memories for specific asset. (+6 more)

### Community 15 - "Community 15"
Cohesion: 0.16
Nodes (18): build_pair(), cross_check(), fetch_binance(), fetch_ecb_pair(), fetch_stooq(), fetch_yf(), fetch_yf_1m_chunked(), _load_ecb() (+10 more)

### Community 16 - "Community 16"
Cohesion: 0.09
Nodes (6): Calculate Heston characteristic function for option pricing.         Uses the He, Calculate implied volatility using bisection method., Generate implied volatility surface from Heston model.                  Paramete, Get volatility term structure (vol vs time).                  Shows how implied, Simulate asset price and volatility paths using QE (Quadratic Exponential) schem, Get human-readable model interpretation.

### Community 17 - "Community 17"
Cohesion: 0.18
Nodes (7): Markov Regime-Switching Model for Trading Uses Hidden Markov Models (HMM) to det, Fallback regime detection without HMM library., Calculate empirical transition matrix from state sequence., Label regimes based on return/volatility characteristics., Run complete Markov regime analysis.          Args:         symbol: Trading symb, Fit HMM model to price data.                  Args:             prices: Price se, run_markov_analysis()

### Community 18 - "Community 18"
Cohesion: 0.24
Nodes (9): BacktestResult, _deflated_sharpe_ratio(), Rigorous Gold RL Backtester ============================ Realistic, bias-free ba, Bailey & López de Prado (2014) Deflated Sharpe Ratio.     Adjusts the observed S, Bar-by-bar trade simulation with realistic costs and intra-bar SL/TP fills., Walk-forward expanding-window backtest. For each fold, generate signals     on t, run_backtest(), Trade (+1 more)

### Community 19 - "Community 19"
Cohesion: 0.27
Nodes (8): addDrawingButton(), hidePriceTooltip(), initChartEnhancements(), setupChartKeyboardShortcuts(), setupCrosshairEnhancements(), setupPlotlyDrawingTools(), showPriceTooltip(), zoomChart()

### Community 20 - "Community 20"
Cohesion: 0.4
Nodes (3): Calculate implied volatility using Hagan's (2002) approximation., Calculate ATM volatility., Generate complete volatility smile from calibrated SABR.                  Return

### Community 21 - "Community 21"
Cohesion: 0.33
Nodes (4): PortfolioRiskLimits, Portfolio-level risk limits and monitoring., Initialize portfolio risk limits.                  Parameters         ----------, Check if portfolio VaR is within limits.                  Parameters         ---

### Community 22 - "Community 22"
Cohesion: 0.33
Nodes (2): Get color for regime., Get description for regime.

### Community 23 - "Community 23"
Cohesion: 0.5
Nodes (3): get_dashboard_layout(), Dashboard Page - Minimal placeholder The full dashboard is in app.py layout, Return empty layout - dashboard is in main app.py

### Community 24 - "Community 24"
Cohesion: 0.5
Nodes (2): Discretize continuous state for Q-table lookup., Update Q-value using Bellman equation.          Q(s,a) = Q(s,a) + α * [r + γ * m

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Initialize Q-Learning agent.          Parameters         ----------         stat

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Page Registry for Multi-Page Dash Application

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Generate volatility smile data for a given expiry.                  Returns ATM

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Get human-readable model interpretation.

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Predict next regime and its probability.                  Returns         ------

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Align higher timeframe data to lower timeframe index.

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Compute Average True Range.

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Check if FinBERT can be loaded.

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Load the financial-sentiment model singleton.          Tries each entry in MODEL

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Analyze sentiment of financial text.                  Returns:             Dict

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Fallback keyword-based sentiment analysis.

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Analyze multiple texts.

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Absorption: large volume with minimal price movement (institutional accumulation

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Reindex a higher-TF close series onto the 1m index via forward-fill.

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Add mtf_score and mtf_confluence to a 1m DataFrame.

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Expected Calibration Error (lower is better; <0.05 = well calibrated).

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Expected Calibration Error (lower is better; <0.05 = well calibrated).

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Calculate d1 parameter.

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Calculate d2 parameter.

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Calculate European call option price.

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Calculate European put option price.

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Calculate option delta.

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Calculate option gamma.

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Calculate option vega.

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Calculate option theta.

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Calculate option rho.

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Calculate implied volatility using Newton-Raphson method.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Calibrate SABR parameters to market volatilities using L-BFGS-B.

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Calibrate SABR from typical market data.                  Can work with limited

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Gradient-boosted tree classifier on microstructure features     + isotonic proba

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Backward-compat for old pickles missing v3 attributes.

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Fixed fractional sizing: risk max_risk_per_trade_pct of equity.

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Quarter-Kelly sizing.          Kelly fraction f* = (W*B − (1−W)) / B   where B =

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Return (win_rate, avg_win, avg_loss) from history, or None.

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Run all feature layers in order: clean → flow → structure → MTF.

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Continuous direction prediction for the UI panel.         Returns: action / conf

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): Simulate intra-bar fill realism using OHLC ordering by direction.          For L

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): Train on `train_df`, evaluate on `test_df`. Returns stats + closed         trade

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): Single train/test backtest. Returns metrics + equity curve.         Now uses int

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Anchored walk-forward backtest with intra-bar fill simulation.          Splits `

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Capture generated signal to trading memory.

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): Capture backtest results to trading memory.

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): Fixed fractional sizing: risk max_risk_per_trade_pct of equity.

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (1): Quarter-Kelly sizing.          Kelly fraction f* = (W*B − (1−W)) / B   where B =

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (1): Return (win_rate, avg_win, avg_loss) from history, or None.

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (1): Run all feature layers in order: clean → flow → structure → MTF.

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (1): Continuous direction prediction for the UI panel.         Returns: action / conf

### Community 72 - "Community 72"
Cohesion: 1.0
Nodes (1): Simulate intra-bar fill realism using OHLC ordering by direction.          For L

### Community 73 - "Community 73"
Cohesion: 1.0
Nodes (1): Expected Calibration Error (lower is better; <0.05 = well calibrated).

### Community 74 - "Community 74"
Cohesion: 1.0
Nodes (1): Train on `train_df`, evaluate on `test_df`. Returns stats + closed         trade

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (1): Single train/test backtest. Returns metrics + equity curve.         Now uses int

### Community 76 - "Community 76"
Cohesion: 1.0
Nodes (1): Anchored walk-forward backtest with intra-bar fill simulation.          Splits `

### Community 77 - "Community 77"
Cohesion: 1.0
Nodes (1): Capture generated signal to trading memory.

### Community 78 - "Community 78"
Cohesion: 1.0
Nodes (1): Capture backtest results to trading memory.

## Knowledge Gaps
- **581 isolated node(s):** `================================================================================`, `Per-asset configuration with research-backed parameters.`, `Layer 0: Fundamental and macro context integration.      Components:     - Econo`, `Identify trading session regime.`, `Compute news impact score (0-1, higher = more dangerous).          Args:` (+576 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 22`** (4 nodes): `.get_regime_color()`, `.get_regime_description()`, `Get color for regime.`, `Get description for regime.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (4 nodes): `._discretize_state()`, `.update()`, `Discretize continuous state for Q-table lookup.`, `Update Q-value using Bellman equation.          Q(s,a) = Q(s,a) + α * [r + γ * m`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (2 nodes): `.__init__()`, `Initialize Q-Learning agent.          Parameters         ----------         stat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (2 nodes): `__init__.py`, `Page Registry for Multi-Page Dash Application`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (2 nodes): `.generate_volatility_smile()`, `Generate volatility smile data for a given expiry.                  Returns ATM`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (2 nodes): `Get human-readable model interpretation.`, `.get_model_info()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (2 nodes): `Predict next regime and its probability.                  Returns         ------`, `.predict_next_regime()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Align higher timeframe data to lower timeframe index.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Compute Average True Range.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Check if FinBERT can be loaded.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Load the financial-sentiment model singleton.          Tries each entry in MODEL`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Analyze sentiment of financial text.                  Returns:             Dict`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Fallback keyword-based sentiment analysis.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Analyze multiple texts.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Absorption: large volume with minimal price movement (institutional accumulation`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Reindex a higher-TF close series onto the 1m index via forward-fill.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Add mtf_score and mtf_confluence to a 1m DataFrame.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Expected Calibration Error (lower is better; <0.05 = well calibrated).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Expected Calibration Error (lower is better; <0.05 = well calibrated).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Calculate d1 parameter.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Calculate d2 parameter.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Calculate European call option price.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Calculate European put option price.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Calculate option delta.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Calculate option gamma.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Calculate option vega.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Calculate option theta.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Calculate option rho.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Calculate implied volatility using Newton-Raphson method.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Calibrate SABR parameters to market volatilities using L-BFGS-B.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Calibrate SABR from typical market data.                  Can work with limited`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Gradient-boosted tree classifier on microstructure features     + isotonic proba`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Backward-compat for old pickles missing v3 attributes.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Fixed fractional sizing: risk max_risk_per_trade_pct of equity.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Quarter-Kelly sizing.          Kelly fraction f* = (W*B − (1−W)) / B   where B =`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Return (win_rate, avg_win, avg_loss) from history, or None.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Run all feature layers in order: clean → flow → structure → MTF.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Continuous direction prediction for the UI panel.         Returns: action / conf`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `Simulate intra-bar fill realism using OHLC ordering by direction.          For L`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `Train on `train_df`, evaluate on `test_df`. Returns stats + closed         trade`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `Single train/test backtest. Returns metrics + equity curve.         Now uses int`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Anchored walk-forward backtest with intra-bar fill simulation.          Splits ``
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Capture generated signal to trading memory.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `Capture backtest results to trading memory.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `Fixed fractional sizing: risk max_risk_per_trade_pct of equity.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `Quarter-Kelly sizing.          Kelly fraction f* = (W*B − (1−W)) / B   where B =`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `Return (win_rate, avg_win, avg_loss) from history, or None.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (1 nodes): `Run all feature layers in order: clean → flow → structure → MTF.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (1 nodes): `Continuous direction prediction for the UI panel.         Returns: action / conf`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 72`** (1 nodes): `Simulate intra-bar fill realism using OHLC ordering by direction.          For L`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 73`** (1 nodes): `Expected Calibration Error (lower is better; <0.05 = well calibrated).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 74`** (1 nodes): `Train on `train_df`, evaluate on `test_df`. Returns stats + closed         trade`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 75`** (1 nodes): `Single train/test backtest. Returns metrics + equity curve.         Now uses int`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 76`** (1 nodes): `Anchored walk-forward backtest with intra-bar fill simulation.          Splits ``
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 77`** (1 nodes): `Capture generated signal to trading memory.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 78`** (1 nodes): `Capture backtest results to trading memory.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `NewsSource` connect `Community 1` to `Community 0`, `Community 8`, `Community 7`?**
  _High betweenness centrality (0.122) - this node is a cross-community bridge._
- **Why does `Action` connect `Community 1` to `Community 0`, `Community 8`, `Community 12`, `Community 7`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Why does `VolatilityModels` connect `Community 2` to `Community 1`, `Community 11`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Are the 179 inferred relationships involving `MarkovRegimeModel` (e.g. with `Professional Trading Terminal - Dash Frontend ==================================` and `Try to restore a previously trained RL agent from disk.`) actually correct?**
  _`MarkovRegimeModel` has 179 INFERRED edges - model-reasoned connections that need verification._
- **Are the 159 inferred relationships involving `GoldRLTrainer` (e.g. with `Professional Trading Terminal - Dash Frontend ==================================` and `Try to restore a previously trained RL agent from disk.`) actually correct?**
  _`GoldRLTrainer` has 159 INFERRED edges - model-reasoned connections that need verification._
- **Are the 157 inferred relationships involving `RegimeSwitchingModel` (e.g. with `Professional Trading Terminal - Dash Frontend ==================================` and `Try to restore a previously trained RL agent from disk.`) actually correct?**
  _`RegimeSwitchingModel` has 157 INFERRED edges - model-reasoned connections that need verification._
- **Are the 153 inferred relationships involving `NewsCache` (e.g. with `Professional Trading Terminal - Dash Frontend ==================================` and `Try to restore a previously trained RL agent from disk.`) actually correct?**
  _`NewsCache` has 153 INFERRED edges - model-reasoned connections that need verification._