# Graph Report - frontend  (2026-05-01)

## Corpus Check
- 42 files · ~118,379 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1744 nodes · 4329 edges · 63 communities detected
- Extraction: 58% EXTRACTED · 42% INFERRED · 0% AMBIGUOUS · INFERRED: 1807 edges (avg confidence: 0.52)
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

## God Nodes (most connected - your core abstractions)
1. `MarkovRegimeModel` - 114 edges
2. `GoldRLTrainer` - 108 edges
3. `RegimeSwitchingModel` - 98 edges
4. `NewsCache` - 94 edges
5. `QLearningAgent` - 91 edges
6. `HestonModel` - 91 edges
7. `TradingEnvironment` - 89 edges
8. `SABRModel` - 86 edges
9. `NewsArticle` - 86 edges
10. `Action` - 83 edges

## Surprising Connections (you probably didn't know these)
- `fetch_yahoo_finance_data()` --calls--> `update_rl_signal()`  [INFERRED]
  app.py → pages/smma_strategy.py
- `fetch_yahoo_finance_data()` --calls--> `execute_auto_trade()`  [INFERRED]
  app.py → pages/smma_strategy.py
- `fetch_yahoo_finance_data()` --calls--> `update_smma_strategy()`  [INFERRED]
  app.py → pages/smma_strategy.py
- `fetch_yahoo_finance_data()` --calls--> `_fetch_recent_data()`  [INFERRED]
  app.py → pages/precision_strategy.py
- `fetch_yahoo_finance_data()` --calls--> `_fetch_training_data()`  [INFERRED]
  app.py → pages/precision_strategy.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.02
Nodes (178): reset(), Backtester, Run Monte Carlo simulation on trade sequence.                  Parameters, Professional Backtesting Engine          Features:     - Vectorized and event-dr, Initialize backtester.                  Parameters         ----------         in, ExnessMT5Bridge, OrderType, PaperTradingBridge (+170 more)

### Community 1 - "Community 1"
Cohesion: 0.02
Nodes (97): Enum, _align_tf(), ARIMABaseline, Asset, AssetConfig, _compute_atr(), compute_order_book_imbalance(), compute_realized_volatility() (+89 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (138): generate_fallback_data(), _lazy_load_models(), _load_rl_state_from_disk(), Professional Trading Terminal - Dash Frontend ==================================, Calculate probability of profit using Black-Scholes framework.          Returns, Create advanced metrics cards - Advanced Tab., Create risk metrics cards - Risk Tab., Generate trading signals from real technical indicators. (+130 more)

### Community 3 - "Community 3"
Cohesion: 0.03
Nodes (65): _build_fold_table(), _empty_wf_fig(), _fetch_recent_data(), _fetch_training_data(), _format_pf(), _get_system(), Precision Strategy Page — 1-minute multi-asset trading system ==================, Fetch recent 1-min OHLCV data via app.py's existing helper. (+57 more)

### Community 4 - "Community 4"
Cohesion: 0.03
Nodes (66): _init_background_news(), CachedNewsItem, NewsCacheData, News Cache Module - Persistent cache with background refresh Provides thread-saf, Save cache to disk. Thread-safe., Get cached news for a symbol.                  Returns:             List of news, Get all cached news for all symbols., A single cached news item. (+58 more)

### Community 5 - "Community 5"
Cohesion: 0.03
Nodes (68): create_volatility_model(), ModelParameters, Goldman Sachs-Level Volatility Modeling Framework ==============================, Compute GARCH(p, q) log-likelihood.                  Parameters         --------, Fit GARCH(p, q) model using Maximum Likelihood Estimation.                  Para, Compute the full conditional variance series., Compute AIC, BIC, and other fit statistics., Compute EGARCH(1,1) log-likelihood.                  EGARCH captures leverage ef (+60 more)

### Community 6 - "Community 6"
Cohesion: 0.03
Nodes (57): BacktestEngine, _compute_atr(), _compute_macd(), _compute_rsi(), FlowAnalyzer, generate_synthetic_data(), HMMRegimeDetector, LorentzianClassifier (+49 more)

### Community 7 - "Community 7"
Cohesion: 0.04
Nodes (30): DuelingDQN, GoldDuelingAgent, Dueling Double DQN Agent for Gold 1-Minute Trading =============================, Returns int action (0/1/2) — NOT an Action enum., Return raw Q-values for confidence scoring., Perform one DQN update step using Double DQN target., Pre-train the Q-network as a 3-class classifier on direction labels., Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s))) (+22 more)

### Community 8 - "Community 8"
Cohesion: 0.04
Nodes (60): create_advanced_metrics_cards(), create_risk_metrics_cards(), calculate_calmar_ratio(), calculate_expected_shortfall(), calculate_max_drawdown(), calculate_sharpe_ratio(), calculate_sortino_ratio(), calculate_var() (+52 more)

### Community 9 - "Community 9"
Cohesion: 0.05
Nodes (56): _fetch_news_from_sources(), _build_news_tabs(), _calculate_instrument_sentiment(), _detect_topic(), _fetch_and_cache(), _get_ai_market_summary(), _get_all_news(), _get_economic_calendar() (+48 more)

### Community 10 - "Community 10"
Cohesion: 0.05
Nodes (69): _analyze_impact(), _analyze_sentiment(), calculate_adx_safe(), calculate_atr_wilder(), calculate_bb_safe(), calculate_bollinger_bands(), calculate_bs_probability(), calculate_cci_safe() (+61 more)

### Community 11 - "Community 11"
Cohesion: 0.07
Nodes (31): analyze(), analyze_batch(), analyze_market(), analyze_sentiment(), analyze_sentiment_batch(), chat_with_ai(), get_ai_status(), get_local_ai_service() (+23 more)

### Community 12 - "Community 12"
Cohesion: 0.07
Nodes (28): cleanup_old_news(), get_news_by_instrument(), get_news_db(), get_recent_news(), get_sentiment_trend(), NewsDatabase, News Database Module - SQLite Storage with 90-day Retention ====================, Get news within a date range.                  Args:             start_date: Sta (+20 more)

### Community 13 - "Community 13"
Cohesion: 0.1
Nodes (19): analyze_batch_sentiment(), analyze_sentiment(), DeepSeekSentiment, get_aggregate_sentiment(), KeywordSentiment, DeepSeek LLM Sentiment Analysis Service Uses DeepSeek API for intelligent financ, Call DeepSeek API for batch headline analysis., Normalize DeepSeek response to standard format. (+11 more)

### Community 14 - "Community 14"
Cohesion: 0.14
Nodes (24): _atr_wilder(), _categorical_x(), _delta(), execute_auto_trade(), _fig_delta(), _fig_orderbook(), _fig_price(), _hurst() (+16 more)

### Community 15 - "Community 15"
Cohesion: 0.09
Nodes (14): _init_db(), Trading Memory - Auto-captures trading decisions using GraphMem ================, Log pattern discovery., Save to GraphMem or SQLite fallback., Save to SQLite fallback., Query memory - GraphMem or SQLite., Query SQLite fallback., Get memories for specific asset. (+6 more)

### Community 16 - "Community 16"
Cohesion: 0.11
Nodes (13): get_market_data_service(), MarketDataService, Market Data Service ==================== Fetches real-time market data from Yaho, Generate mock price data (fallback)., Generate fallback OHLCV data if API fails., Get current price for symbol., Fetch prices for multiple symbols.          Parameters         ----------, Get option chain data from Yahoo Finance.          Parameters         ---------- (+5 more)

### Community 17 - "Community 17"
Cohesion: 0.1
Nodes (10): DeepQNetwork, DeepRLAgent, Load agent state from disk. Returns True on success., Deep Q-Network for trading., Deep Reinforcement Learning Agent using DQN with Experience Replay., Initialize Deep RL agent.              Parameters             ----------, Get action using epsilon-greedy policy., Store transition in replay buffer. (+2 more)

### Community 18 - "Community 18"
Cohesion: 0.16
Nodes (18): build_pair(), cross_check(), fetch_binance(), fetch_ecb_pair(), fetch_stooq(), fetch_yf(), fetch_yf_1m_chunked(), _load_ecb() (+10 more)

### Community 19 - "Community 19"
Cohesion: 0.2
Nodes (11): BacktestResult, _capture_backtest_to_memory(), _deflated_sharpe_ratio(), Rigorous Gold RL Backtester ============================ Realistic, bias-free ba, Bailey & López de Prado (2014) Deflated Sharpe Ratio.     Adjusts the observed S, Bar-by-bar trade simulation with realistic costs and intra-bar SL/TP fills., Walk-forward expanding-window backtest. For each fold, generate signals     on t, Capture backtest results to trading memory. (+3 more)

### Community 20 - "Community 20"
Cohesion: 0.18
Nodes (7): Markov Regime-Switching Model for Trading Uses Hidden Markov Models (HMM) to det, Fallback regime detection without HMM library., Calculate empirical transition matrix from state sequence., Label regimes based on return/volatility characteristics., Run complete Markov regime analysis.          Args:         symbol: Trading symb, Fit HMM model to price data.                  Args:             prices: Price se, run_markov_analysis()

### Community 21 - "Community 21"
Cohesion: 0.27
Nodes (8): addDrawingButton(), hidePriceTooltip(), initChartEnhancements(), setupChartKeyboardShortcuts(), setupCrosshairEnhancements(), setupPlotlyDrawingTools(), showPriceTooltip(), zoomChart()

### Community 22 - "Community 22"
Cohesion: 0.17
Nodes (8): BacktestResult, Run backtest on historical data.                  Parameters         ----------, Container for backtest trade record., Calculate performance metrics., Calculate maximum drawdown duration in days., Run walk-forward analysis.                  Parameters         ----------, Container for backtest performance results., Trade

### Community 23 - "Community 23"
Cohesion: 0.4
Nodes (3): Calculate implied volatility using Hagan's (2002) approximation., Calculate ATM volatility., Generate complete volatility smile from calibrated SABR.                  Return

### Community 24 - "Community 24"
Cohesion: 0.33
Nodes (4): PortfolioRiskLimits, Portfolio-level risk limits and monitoring., Initialize portfolio risk limits.                  Parameters         ----------, Check if portfolio VaR is within limits.                  Parameters         ---

### Community 25 - "Community 25"
Cohesion: 0.5
Nodes (3): get_dashboard_layout(), Dashboard Page - Minimal placeholder The full dashboard is in app.py layout, Return empty layout - dashboard is in main app.py

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Page Registry for Multi-Page Dash Application

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Align higher timeframe data to lower timeframe index.

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Compute Average True Range.

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Check if FinBERT can be loaded.

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Load the financial-sentiment model singleton.          Tries each entry in MODEL

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Analyze sentiment of financial text.                  Returns:             Dict

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Fallback keyword-based sentiment analysis.

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Analyze multiple texts.

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Absorption: large volume with minimal price movement (institutional accumulation

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Reindex a higher-TF close series onto the 1m index via forward-fill.

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Add mtf_score and mtf_confluence to a 1m DataFrame.

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Simulate intra-bar fill realism using OHLC ordering by direction.          For L

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Expected Calibration Error (lower is better; <0.05 = well calibrated).

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Calculate d1 parameter.

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Calculate d2 parameter.

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Calculate European call option price.

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Calculate European put option price.

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Calculate option delta.

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Calculate option gamma.

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Calculate option vega.

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Calculate option theta.

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Calculate option rho.

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Calculate implied volatility using Newton-Raphson method.

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Calibrate SABR parameters to market volatilities using L-BFGS-B.

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Calibrate SABR from typical market data.                  Can work with limited

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Fixed fractional sizing: risk max_risk_per_trade_pct of equity.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Quarter-Kelly sizing.          Kelly fraction f* = (W*B − (1−W)) / B   where B =

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Return (win_rate, avg_win, avg_loss) from history, or None.

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Run all feature layers in order: clean → flow → structure → MTF.

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Continuous direction prediction for the UI panel.         Returns: action / conf

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Simulate intra-bar fill realism using OHLC ordering by direction.          For L

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Expected Calibration Error (lower is better; <0.05 = well calibrated).

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Train on `train_df`, evaluate on `test_df`. Returns stats + closed         trade

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Single train/test backtest. Returns metrics + equity curve.         Now uses int

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Anchored walk-forward backtest with intra-bar fill simulation.          Splits `

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): Capture generated signal to trading memory.

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): Capture backtest results to trading memory.

## Knowledge Gaps
- **567 isolated node(s):** `================================================================================`, `Per-asset configuration with research-backed parameters.`, `Layer 0: Fundamental and macro context integration.      Components:     - Econo`, `Identify trading session regime.`, `Compute news impact score (0-1, higher = more dangerous).          Args:` (+562 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 26`** (2 nodes): `__init__.py`, `Page Registry for Multi-Page Dash Application`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Align higher timeframe data to lower timeframe index.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Compute Average True Range.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Check if FinBERT can be loaded.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Load the financial-sentiment model singleton.          Tries each entry in MODEL`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Analyze sentiment of financial text.                  Returns:             Dict`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Fallback keyword-based sentiment analysis.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Analyze multiple texts.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Absorption: large volume with minimal price movement (institutional accumulation`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Reindex a higher-TF close series onto the 1m index via forward-fill.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Add mtf_score and mtf_confluence to a 1m DataFrame.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Simulate intra-bar fill realism using OHLC ordering by direction.          For L`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Expected Calibration Error (lower is better; <0.05 = well calibrated).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Calculate d1 parameter.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Calculate d2 parameter.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Calculate European call option price.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Calculate European put option price.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Calculate option delta.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Calculate option gamma.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Calculate option vega.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Calculate option theta.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Calculate option rho.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Calculate implied volatility using Newton-Raphson method.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Calibrate SABR parameters to market volatilities using L-BFGS-B.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Calibrate SABR from typical market data.                  Can work with limited`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Fixed fractional sizing: risk max_risk_per_trade_pct of equity.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Quarter-Kelly sizing.          Kelly fraction f* = (W*B − (1−W)) / B   where B =`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Return (win_rate, avg_win, avg_loss) from history, or None.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Run all feature layers in order: clean → flow → structure → MTF.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Continuous direction prediction for the UI panel.         Returns: action / conf`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Simulate intra-bar fill realism using OHLC ordering by direction.          For L`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Expected Calibration Error (lower is better; <0.05 = well calibrated).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Train on `train_df`, evaluate on `test_df`. Returns stats + closed         trade`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Single train/test backtest. Returns metrics + equity curve.         Now uses int`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Anchored walk-forward backtest with intra-bar fill simulation.          Splits ``
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `Capture generated signal to trading memory.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `Capture backtest results to trading memory.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Action` connect `Community 2` to `Community 1`, `Community 17`?**
  _High betweenness centrality (0.113) - this node is a cross-community bridge._
- **Why does `VolatilityModels` connect `Community 5` to `Community 2`, `Community 14`, `Community 7`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Why does `NewsSource` connect `Community 2` to `Community 1`, `Community 4`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Are the 104 inferred relationships involving `MarkovRegimeModel` (e.g. with `Professional Trading Terminal - Dash Frontend ==================================` and `Try to restore a previously trained RL agent from disk.`) actually correct?**
  _`MarkovRegimeModel` has 104 INFERRED edges - model-reasoned connections that need verification._
- **Are the 84 inferred relationships involving `GoldRLTrainer` (e.g. with `Professional Trading Terminal - Dash Frontend ==================================` and `Try to restore a previously trained RL agent from disk.`) actually correct?**
  _`GoldRLTrainer` has 84 INFERRED edges - model-reasoned connections that need verification._
- **Are the 82 inferred relationships involving `RegimeSwitchingModel` (e.g. with `Professional Trading Terminal - Dash Frontend ==================================` and `Try to restore a previously trained RL agent from disk.`) actually correct?**
  _`RegimeSwitchingModel` has 82 INFERRED edges - model-reasoned connections that need verification._
- **Are the 78 inferred relationships involving `NewsCache` (e.g. with `Professional Trading Terminal - Dash Frontend ==================================` and `Try to restore a previously trained RL agent from disk.`) actually correct?**
  _`NewsCache` has 78 INFERRED edges - model-reasoned connections that need verification._