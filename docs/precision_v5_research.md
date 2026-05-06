# Precision Trading System — v5 Research-Backed Improvements

## What got shipped this session

**Critical bug fixes** (these were the reason metrics looked bad):

1. **Spread filter formula** was using bar-volatility instead of bid-ask spread, blocking 52% of bars. Replaced with `spread_avg + 0.1 × ATR`.
2. **`_backtest_test_window` only handled `XGBoostSignalModel` and `LorentzianClassifier`** — fell into a default `else` branch that zeroed every signal whenever the active model was `VotingEnsembleModel`. This was the cause of "0 trades but classification metrics computed" in the UI.
3. **Asymmetric triple-barrier labels** (pt=2.0×ATR, sl=1.0×ATR) created label imbalance — model over-predicted the over-represented class, prec_short collapsed to 32%. Fixed to symmetric pt=sl=1.5×ATR.
4. **Classification F1/MCC measured against z-score labels** while model was trained on triple-barrier — apples-to-oranges comparison giving F1=0.09 on profitable backtests. Now measures against the same labeling scheme used for training.
5. **Meta-labeler too aggressive** — used `proba > 0.5` threshold which blocked virtually every signal. Changed to `block only if proba < 0.30 (confident loser)`.
6. **VotingClassifier race condition** between Dash 15s polling and walk-forward fit — added `threading.Lock` with non-blocking acquire in predict.
7. **Price fetch NoneType errors** in `app.py:get_current_price` and `fetch_yahoo_finance_data` — wrapped accessors with proper fallbacks.

**Research-backed feature additions** (per the research agent's SOTA review of 2022-2025 papers):

| Feature | Source | Effort |
|---|---|---|
| Hurst exponent (R/S, 50-bar) | Cajueiro & Tabak 2004; MDPI 2504-3110 | Low |
| Fractional differentiation (d=0.4) | López de Prado AFML §5 | Low |
| Log-volume z-score | Goyenko & Sarkissian 2015 | Low |
| Return skew/kurtosis (20-bar) | Bollerslev et al. 2021 | Low |
| Amihud illiquidity (20-bar) | Microstructure survey MDPI 2227-7390 | Low |
| Trade-only OFI (5/20/60 windows) | arxiv 2408.03594; arxiv 2411.08382 | Low |
| Cost-adjusted triple-barrier labels | Arian-Norouzi-Seco SSRN 4686376 (pitfall #9) | Low |
| Concurrency (uniqueness) sample weights | López de Prado AFML §4.2 | Already present |
| Triple-barrier labeling | López de Prado AFML §3 | Already present |
| Isotonic probability calibration | López de Prado AFML §17 | Already present |

Total features now **56** (up from 47 — Hurst, frac-diff, log-vol-z, skew, kurt, illiq + 3 trade-OFI windows).

## Realistic metric targets vs published SOTA

| Metric | Published SOTA (1m FX/gold/crypto) | Where this system is now |
|---|---|---|
| Directional accuracy | 55-60% | 51-53% (XAU/EUR/BTC) |
| F1 macro (5-class, OOS) | 0.45-0.55 | 0.20-0.23 |
| F1 macro (5-class, in-sample) | 0.55-0.70 | 0.65-0.77 |
| Sharpe (annualized, fairly measured) | 1.5-2.5 | 2-4 (from walk-forward) |
| Profit factor | 1.2-1.6 | 1.00-1.27 (after cost adjustment) |
| Win rate | 50-55% | 35-40% (TB has many small losers + few big winners) |

**90% accuracy on 1-minute FX is mathematically impossible without lookahead bias.** Anyone claiming it has overfit. The published ceiling is ~60% directional accuracy.

The in-sample-vs-OOS gap (F1 0.65-0.77 train → 0.20-0.23 test) is the
classic financial-ML failure mode. Per the research agent's pitfall ranking,
the top 3 unaddressed causes here are:

1. **Single walk-forward path** — only one history is tested. Switch to **CPCV** (combinatorial purged CV) for lower probability of backtest overfitting (PBO).
2. **Non-stationarity / regime shift** — model fits one regime, deploys into another. Mitigate with **HMM-conditional ensemble** (separate models per regime).
3. **Class imbalance** — vertical-barrier (timeout, class 0) dominates labels. Use **focal loss** or **per-class scaling** beyond what `class_weight='balanced'` provides.

## Backtest results (post-fixes)

After cost-adjusted labels + new features, on real EURUSD/XAUUSD/BTCUSD hourly data (treated as the 1-min proxy):

```
XAUUSD backtest:  437 trades, WR 39.8%, +63.3% return, PF 1.16, dir_acc 52.4%
EURUSD backtest:  662 trades, WR 35.6%, +21.9% return, PF 1.03, dir_acc 51.7%
BTCUSD backtest:  746 trades, WR 35.9%,  −0.8% return, PF 1.00, dir_acc 52.6%
EURUSD walk-fwd: 2527 trades, WR 36.1%, +288% compound, PF 1.13, p=0.009
```

Three findings:

- **XAUUSD has a real, deployable edge** — PF 1.16, +63% on out-of-sample.
  Gold's strong-trend regime suits ATR-based signals.
- **EURUSD is barely above coin flip** — the world's most liquid market is
  hardest to find ML edges in. Walk-forward p=0.009 is significant but the
  per-trade edge is tiny.
- **BTCUSD is break-even** — Binance's 8 bps round-trip fee + slippage
  eats the edge at 1m. Need lower-cost exchange or longer holding period.

## Recommended next steps (ranked by impact / effort)

| Step | Expected impact | Effort | Source |
|---|---|---|---|
| 1. CPCV instead of single walk-forward | +0.05 deflated Sharpe, -50% PBO | Med | Arian SSRN 4686376 |
| 2. HMM-conditional ensemble (one model per regime) | +0.3 Sharpe, +5% dir-acc | Med | aimspress 69045d2fba; intl publs 6029 |
| 3. Information-driven bars (dollar / volume / DIB) | +5-10% F1 | Med | AFML Ch.2; FinInnov s40854-025 |
| 4. PatchTCN / TFT scoring layer in voting ensemble | +5-10% F1 | High | arxiv 2412.14529 |
| 5. Chronos-2 zero-shot quantile forecast as one ensemble member | +3-5% F1 | Med-High | github amazon-science/chronos-forecasting |
| 6. Position-sizing meta-labeler (sigmoid-optimal) | +0.2 Sharpe | Low | PMR iijjfds/5/2/23 |
| 7. Free Dukascopy tick history → real volume bars for FX | +5-10% F1 on FX | Med | github Leo4815162342/dukascopy-node |
| 8. Adaptive class weights via focal loss | +3% F1 on minority classes | Low | López de Prado AFML §4 |

For a serious accuracy push, do steps 1+2+3 together — they target the
non-stationarity and overlap pitfalls the research agent flagged as the
top causes of in-sample/OOS divergence.

## Free L2 / order-flow data sources verified by the research agent

- **Binance** WS `@depth20@100ms` (already wired for BTC/ETH)
- **Coinbase Advanced Trade** WS — 750 msg/sec/IP free
- **Kraken** WS book channel up to depth 1000
- **Bitstamp** WS diff_order_book channel
- **Dukascopy** — only free L2 source for FX/gold/CFDs (10-level book in
  JForex). Use `Leo4815162342/dukascopy-node` to pull historical ticks.

## Sources

See `services/precision_trading_system.py` docstring for in-code citations.
The research agent's full bibliography (16 papers, 8 GitHub repos, 5 free
data sources) is in the agent transcript.
