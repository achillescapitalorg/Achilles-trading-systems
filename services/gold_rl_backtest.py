"""
Rigorous Gold RL Backtester
============================
Realistic, bias-free backtesting with industrial-grade metrics.

Features:
  - Realistic gold trading costs: $0.30 spread + $0.10 slippage per leg
  - Trade-by-trade execution with intra-bar SL/TP fills
  - Walk-forward validation across K folds
  - Comprehensive metrics:
      • Direction accuracy (5-bar fwd labels)
      • Win rate, profit factor
      • Sharpe ratio AND Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
      • Max drawdown + duration
      • Calmar, Sortino ratios
      • Average trade R-multiple
  - Returns DataFrame of all trades for forensic analysis
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
from datetime import datetime
from scipy import stats


# ── Realistic gold futures (GC=F) trading costs ──────────────────────────────
# Source: CME Group GC contract spec + typical retail broker conditions
# NOTE: Legacy fixed costs kept for backward compatibility.
# Use use_realistic_costs=True to enable session/volatility-dependent costs.
GOLD_SPREAD_USD     = 0.30      # bid-ask half-spread per leg
GOLD_SLIPPAGE_USD   = 0.10      # market impact slippage per leg
GOLD_TICK_SIZE      = 0.10
GOLD_VALUE_PER_DOL  = 100.0     # 1 contract = 100 oz, $1 move = $100

# Import new realistic cost model
from services.realistic_costs import RealisticCostModel, detect_volatility_regime

# Default realistic cost model instance
_DEFAULT_COST_MODEL = RealisticCostModel(commission_per_lot=7.0)


@dataclass
class Trade:
    entry_time:     int
    exit_time:      int
    direction:      int          # 1 = long, -1 = short
    entry_price:    float
    exit_price:     float
    size:           float        # in oz
    pnl_dollar:     float
    pnl_pct:        float
    exit_reason:    str          # 'tp', 'sl', 'signal_flip', 'eod'
    bars_held:      int


@dataclass
class BacktestResult:
    trades:               List[Trade]   = field(default_factory=list)
    equity_curve:         List[float]   = field(default_factory=list)
    timestamps:           List          = field(default_factory=list)
    total_return:         float = 0.0
    sharpe:               float = 0.0
    deflated_sharpe:      float = 0.0
    sortino:              float = 0.0
    calmar:               float = 0.0
    max_drawdown:         float = 0.0
    max_dd_duration:      int   = 0
    win_rate:             float = 0.0
    profit_factor:        float = 0.0
    avg_win:              float = 0.0
    avg_loss:             float = 0.0
    avg_trade_pnl:        float = 0.0
    n_trades:             int   = 0
    n_winning:            int   = 0
    n_losing:             int   = 0
    direction_accuracy:   float = 0.0
    initial_capital:      float = 10_000.0
    final_capital:        float = 10_000.0

    def summary(self) -> str:
        return (
            "═══════════════════════════════════════════════════════\n"
            "          GOLD RL BACKTEST RESULTS\n"
            "═══════════════════════════════════════════════════════\n"
            f"  Initial Capital   : ${self.initial_capital:>14,.2f}\n"
            f"  Final Capital     : ${self.final_capital:>14,.2f}\n"
            f"  Total Return      : {self.total_return:>14.2%}\n"
            f"  Number of Trades  : {self.n_trades:>14d}\n"
            f"  Winning / Losing  : {self.n_winning:>6d} / {self.n_losing:<6d}\n"
            "───────────────────────────────────────────────────────\n"
            f"  Win Rate          : {self.win_rate:>14.2%}\n"
            f"  Direction Accuracy: {self.direction_accuracy:>14.2%}\n"
            f"  Profit Factor     : {self.profit_factor:>14.3f}\n"
            f"  Avg Winner        : ${self.avg_win:>14,.2f}\n"
            f"  Avg Loser         : ${self.avg_loss:>14,.2f}\n"
            f"  Avg Trade PnL     : ${self.avg_trade_pnl:>14,.2f}\n"
            "───────────────────────────────────────────────────────\n"
            f"  Sharpe Ratio      : {self.sharpe:>14.3f}\n"
            f"  Deflated Sharpe   : {self.deflated_sharpe:>14.3f}\n"
            f"  Sortino Ratio     : {self.sortino:>14.3f}\n"
            f"  Calmar Ratio      : {self.calmar:>14.3f}\n"
            f"  Max Drawdown      : {self.max_drawdown:>14.2%}\n"
            f"  Max DD Duration   : {self.max_dd_duration:>14d}\n"
            "═══════════════════════════════════════════════════════\n"
            f"  Verdict: " + self._verdict() + "\n"
            "═══════════════════════════════════════════════════════\n"
        )

    def _verdict(self) -> str:
        if self.deflated_sharpe > 1.0 and self.win_rate > 0.50 and self.profit_factor > 1.3:
            return "STRONG — robust out-of-sample performance"
        if self.deflated_sharpe > 0.5 and self.profit_factor > 1.1:
            return "ACCEPTABLE — modest edge, fragile"
        if self.profit_factor > 1.0:
            return "MARGINAL — barely profitable, likely noise"
        return "POOR — needs more training or data"


def _deflated_sharpe_ratio(sharpe: float, n_trades: int,
                            skew: float, kurt: float,
                            n_trials: int = 10) -> float:
    """
    Bailey & López de Prado (2014) Deflated Sharpe Ratio.
    Adjusts the observed Sharpe for selection bias (number of model variants tried)
    and non-normality of returns. Values > 1.0 are robust.

    n_trials: estimate of how many strategy variants were tested.
    """
    if n_trades < 5:
        return 0.0
    # Maximum expected Sharpe under H0 across n_trials independent strategies
    emc = 0.5772156649  # Euler-Mascheroni
    z_alpha = stats.norm.ppf(1 - 1.0 / max(n_trials, 1))
    z_alpha_minus = stats.norm.ppf(1 - 1.0 / max(n_trials * np.e, 1))
    expected_max_sr = ((1 - emc) * z_alpha + emc * z_alpha_minus)

    # Standard error of Sharpe accounting for skew/kurtosis
    sigma_sr = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2)
                        / max(n_trades - 1, 1))
    if sigma_sr <= 1e-9:
        return 0.0
    dsr_z = (sharpe - expected_max_sr) / sigma_sr
    return float(stats.norm.cdf(dsr_z))


def run_backtest(
    df: pd.DataFrame,
    signals: np.ndarray,
    initial_capital: float = 10_000.0,
    risk_per_trade: float = 0.01,
    sl_atr_mult: float = 2.0,
    tp_atr_mult: float = 4.0,
    confidence: Optional[np.ndarray] = None,
    min_confidence: float = 0.40,
    n_trials_for_dsr: int = 10,
    use_realistic_costs: bool = False,
    cost_model: Optional[RealisticCostModel] = None,
) -> BacktestResult:
    """
    Bar-by-bar trade simulation with realistic costs and intra-bar SL/TP fills.

    df:        DataFrame with columns ['open','high','low','close','atr',
                                       'timestamp' (optional)]
    signals:   integer array, 0=HOLD, 1=BUY, 2=SELL — same length as df
    confidence: optional array of probabilities (filter out low-confidence signals)
    use_realistic_costs: if True, uses session/volatility-dependent costs
    cost_model: optional RealisticCostModel instance (uses default if None)
    """
    assert len(signals) == len(df), "signals and df length mismatch"
    n = len(df)
    closes = df["close"].values
    highs  = df["high"].values  if "high"  in df.columns else closes
    lows   = df["low"].values   if "low"   in df.columns else closes
    atrs   = df["atr"].values   if "atr"   in df.columns else np.full(n, closes[0] * 0.002)

    # Forward 5-bar direction labels for accuracy metric
    fwd5 = pd.Series(closes).pct_change(5).shift(-5).fillna(0).values

    capital     = initial_capital
    equity_curve = [capital]
    timestamps   = []
    trades: List[Trade] = []

    position    = 0       # 0/1/-1
    entry_price = 0.0
    entry_bar   = 0
    sl_price    = 0.0
    tp_price    = 0.0
    pos_size    = 0.0     # in oz

    direction_correct = 0
    direction_total = 0

    # Setup cost model
    cost_model = cost_model or _DEFAULT_COST_MODEL if use_realistic_costs else None

    def _get_bar_costs(idx: int, vol_regime: str = "normal") -> Tuple[float, float]:
        """Returns (spread_usd, slippage_usd) per oz for bar idx."""
        if cost_model is None:
            return GOLD_SPREAD_USD, GOLD_SLIPPAGE_USD
        ts = df["timestamp"].iloc[idx] if "timestamp" in df.columns else pd.Timestamp.now()
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        spread = cost_model.compute_spread(ts, vol_regime)
        # slippage approximated as spread * 0.3 for quick per-bar estimate
        slippage = spread * 0.3
        return spread, slippage

    def _vol_regime(idx: int) -> str:
        if cost_model is None or "atr" not in df.columns:
            return "normal"
        atr = df["atr"].iloc[idx]
        atr_ma = df["atr"].rolling(50).mean().iloc[idx]
        return detect_volatility_regime(atr, atr_ma)

    for i in range(n):
        bar_high = highs[i]
        bar_low  = lows[i]
        bar_open = df["open"].values[i] if "open" in df.columns else closes[i]
        bar_close = closes[i]
        bar_atr   = max(atrs[i], 0.5)
        ts = df["timestamp"].iloc[i] if "timestamp" in df.columns else i
        vol_regime = _vol_regime(i)
        spread_usd, slip_usd = _get_bar_costs(i, vol_regime)
        leg_cost = (spread_usd + slip_usd)

        # ── Check intra-bar SL/TP fills first (assume worst case for the trader) ──
        if position != 0:
            hit_sl = False
            hit_tp = False
            if position == 1:
                if bar_low <= sl_price:
                    hit_sl = True
                elif bar_high >= tp_price:
                    hit_tp = True
            else:
                if bar_high >= sl_price:
                    hit_sl = True
                elif bar_low <= tp_price:
                    hit_tp = True

            exit_reason = None; exit_price = 0.0
            if hit_sl:
                exit_price = sl_price - position * (slip_usd)  # adverse slippage
                exit_reason = "sl"
            elif hit_tp:
                exit_price = tp_price + position * 0  # touch fill
                exit_reason = "tp"

            if exit_reason:
                pnl_dollar = position * (exit_price - entry_price) * pos_size
                # Spread + slippage cost on exit
                pnl_dollar -= leg_cost * pos_size
                pnl_pct = pnl_dollar / capital
                capital += pnl_dollar
                trades.append(Trade(
                    entry_time=entry_bar, exit_time=i,
                    direction=position, entry_price=entry_price,
                    exit_price=exit_price, size=pos_size,
                    pnl_dollar=pnl_dollar, pnl_pct=pnl_pct,
                    exit_reason=exit_reason, bars_held=i - entry_bar,
                ))
                position = 0

        # ── Direction-accuracy bookkeeping (uses signal at bar i) ──
        sig = int(signals[i])
        if sig in (1, 2) and i + 5 < n:
            direction_total += 1
            if (sig == 1 and fwd5[i] > 0) or (sig == 2 and fwd5[i] < 0):
                direction_correct += 1

        # ── Entry logic: open new position if signal disagrees with current pos ──
        # Confidence filter
        conf_ok = True
        if confidence is not None and i < len(confidence):
            conf_ok = confidence[i] >= min_confidence

        if conf_ok and sig in (1, 2):
            target_pos = 1 if sig == 1 else -1

            # Close opposite position first
            if position != 0 and position != target_pos:
                exit_price = bar_close - position * leg_cost
                pnl_dollar = position * (exit_price - entry_price) * pos_size
                pnl_dollar -= leg_cost * pos_size
                pnl_pct = pnl_dollar / capital
                capital += pnl_dollar
                trades.append(Trade(
                    entry_time=entry_bar, exit_time=i,
                    direction=position, entry_price=entry_price,
                    exit_price=exit_price, size=pos_size,
                    pnl_dollar=pnl_dollar, pnl_pct=pnl_pct,
                    exit_reason="signal_flip", bars_held=i - entry_bar,
                ))
                position = 0

            # Open new position
            if position == 0:
                # Risk-based sizing: 1% of capital risked at SL distance
                sl_dist = bar_atr * sl_atr_mult
                risk_dollar = capital * risk_per_trade
                pos_size = max(0.01, min(risk_dollar / sl_dist, capital * 0.5 / bar_close))
                # Round to 0.01 oz
                pos_size = round(pos_size, 2)

                entry_price = bar_close + target_pos * leg_cost
                if target_pos == 1:
                    sl_price = entry_price - sl_dist
                    tp_price = entry_price + bar_atr * tp_atr_mult
                else:
                    sl_price = entry_price + sl_dist
                    tp_price = entry_price - bar_atr * tp_atr_mult
                position = target_pos
                entry_bar = i

        # Mark-to-market equity
        mtm = capital
        if position != 0:
            mtm += position * (bar_close - entry_price) * pos_size
        equity_curve.append(mtm)
        timestamps.append(ts)

    # Close any open position at the last close
    if position != 0:
        spread_usd, slip_usd = _get_bar_costs(n - 1, _vol_regime(n - 1))
        leg_cost = spread_usd + slip_usd
        exit_price = closes[-1] - position * leg_cost
        pnl_dollar = position * (exit_price - entry_price) * pos_size
        pnl_dollar -= leg_cost * pos_size
        pnl_pct = pnl_dollar / capital
        capital += pnl_dollar
        trades.append(Trade(
            entry_time=entry_bar, exit_time=n - 1,
            direction=position, entry_price=entry_price,
            exit_price=exit_price, size=pos_size,
            pnl_dollar=pnl_dollar, pnl_pct=pnl_pct,
            exit_reason="eod", bars_held=n - 1 - entry_bar,
        ))

    # ── Compute metrics ──────────────────────────────────────────────────
    eq = np.array(equity_curve)
    rets = np.diff(eq) / eq[:-1]

    total_ret = (eq[-1] - eq[0]) / eq[0]
    if len(rets) > 0 and rets.std() > 1e-9:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(252 * 390))   # 1m bars/year approx
        downside = rets[rets < 0]
        sortino = float(rets.mean() / downside.std() * np.sqrt(252 * 390)) \
                   if len(downside) > 1 and downside.std() > 1e-9 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    # Drawdown
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / running_max
    max_dd = float(-dd.min()) if len(dd) > 0 else 0.0
    in_dd = dd < 0
    cur_dur, max_dur = 0, 0
    for v in in_dd:
        cur_dur = cur_dur + 1 if v else 0
        max_dur = max(max_dur, cur_dur)

    pnls = [t.pnl_dollar for t in trades]
    n_win = sum(1 for p in pnls if p > 0)
    n_loss = sum(1 for p in pnls if p <= 0)
    win_rate = n_win / max(len(trades), 1)
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    pf = gp / gl if gl > 1e-9 else (gp if gp > 0 else 0.0)
    avg_win  = (gp / n_win)  if n_win  > 0 else 0.0
    avg_loss = (-gl / n_loss) if n_loss > 0 else 0.0
    avg_pnl  = np.mean(pnls) if pnls else 0.0

    calmar = (total_ret / max_dd) if max_dd > 1e-9 else 0.0
    dir_acc = direction_correct / max(direction_total, 1)

    # Deflated Sharpe
    if len(pnls) >= 5:
        skew_v = float(stats.skew(pnls))
        kurt_v = float(stats.kurtosis(pnls, fisher=False))
        dsr = _deflated_sharpe_ratio(sharpe, len(pnls), skew_v, kurt_v,
                                      n_trials=n_trials_for_dsr)
    else:
        dsr = 0.0

    result = BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        timestamps=timestamps,
        total_return=total_ret,
        sharpe=sharpe,
        deflated_sharpe=dsr,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        max_dd_duration=max_dur,
        win_rate=win_rate,
        profit_factor=pf,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_trade_pnl=avg_pnl,
        n_trades=len(trades),
        n_winning=n_win,
        n_losing=n_loss,
        direction_accuracy=dir_acc,
        initial_capital=initial_capital,
        final_capital=capital,
    )

    # ── Validation sanity check ─────────────────────────────────────────
    if pnls:
        try:
            from services.backtest_validator import BacktestValidator
            validator = BacktestValidator()
            returns_series = pd.Series(pnls)
            report = validator.validate(
                backtest_returns=returns_series,
                n_trials=max(1, n_trials_for_dsr),
            )
            if report.errors:
                print("\n" + report.summary())
            elif report.warnings:
                print("\n[BACKTEST VALIDATION WARNINGS]")
                for w in report.warnings:
                    print(f"  ⚠️  {w}")
        except Exception as e:
            print(f"[Validation check skipped: {e}]")

    return result


def walk_forward_backtest(
    df: pd.DataFrame,
    signal_generator: Callable[[pd.DataFrame], np.ndarray],
    n_folds: int = 5,
    initial_capital: float = 10_000.0,
) -> Dict:
    """
    Walk-forward expanding-window backtest. For each fold, generate signals
    on the unseen out-of-sample window. Returns aggregated stats.
    """
    n = len(df)
    fold_size = n // (n_folds + 1)
    fold_results = []

    for k in range(n_folds):
        in_sample_end  = fold_size * (k + 1)
        out_of_sample_start = in_sample_end
        out_of_sample_end   = min(in_sample_end + fold_size, n)
        if out_of_sample_end - out_of_sample_start < 50:
            continue

        oos_df = df.iloc[out_of_sample_start:out_of_sample_end].reset_index(drop=True)
        signals = signal_generator(oos_df)
        if signals is None or len(signals) != len(oos_df):
            continue

        result = run_backtest(oos_df, signals, initial_capital=initial_capital)
        fold_results.append({
            "fold":             k + 1,
            "n_trades":         result.n_trades,
            "win_rate":         result.win_rate,
            "direction_acc":    result.direction_accuracy,
            "profit_factor":    result.profit_factor,
            "sharpe":           result.sharpe,
            "deflated_sharpe":  result.deflated_sharpe,
            "max_drawdown":     result.max_drawdown,
            "total_return":     result.total_return,
        })

    if not fold_results:
        return {"error": "No valid folds"}

    fold_df = pd.DataFrame(fold_results)
    return {
        "folds":            fold_results,
        "n_folds_run":      len(fold_results),
        "avg_win_rate":     fold_df["win_rate"].mean(),
        "avg_dir_acc":      fold_df["direction_acc"].mean(),
        "avg_profit_factor": fold_df["profit_factor"].mean(),
        "avg_sharpe":       fold_df["sharpe"].mean(),
        "avg_dsr":          fold_df["deflated_sharpe"].mean(),
        "avg_max_dd":       fold_df["max_drawdown"].mean(),
        "avg_total_return": fold_df["total_return"].mean(),
        "consistency":      fold_df["total_return"].apply(lambda x: x > 0).mean(),
    }


def _capture_backtest_to_memory(results: dict, strategy: str = "Gold RL"):
    """Capture backtest results to trading memory."""
    try:
        from services.trading_memory import get_memory
        memory = get_memory()
        if not memory._enabled:
            return
        memory.capture_backtest({
            "strategy": strategy,
            "asset": "XAUUSD",
            "period": f"{results.get('n_folds_run', 0)} folds",
            "sharpe": results.get("avg_sharpe", "N/A"),
            "max_drawdown": results.get("avg_max_dd", 0),
            "win_rate": results.get("avg_win_rate", 0)
        })
    except Exception:
        pass
