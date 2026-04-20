"""
Market Context Builder
======================
Assembles ALL platform data — technical indicators, financial models,
regime detection, risk metrics, Monte Carlo, and FinBERT-aggregated news —
into a single MarketContext object ready to be serialised for an AI prompt.

Architecture notes:
- All imports from app.py are LAZY (inside build_market_context()) to avoid
  circular imports at module-load time.
- All components run in parallel via ThreadPoolExecutor with an 8-second
  wall-clock deadline; missing components fall back to safe defaults.
- Each component section is wrapped in try/except so a single failure never
  aborts the entire build.
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Sub-dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TechnicalSignal:
    indicator: str     # e.g. "RSI (14)"
    signal: str        # "BUY" | "SELL" | "HOLD"
    strength: str      # "STRONG" | "MODERATE" | "WEAK"
    value: str         # human-readable, e.g. "28.4"


@dataclass
class HestonSnapshot:
    kappa: float       # Mean-reversion speed
    theta: float       # Long-run variance
    xi: float          # Vol-of-vol
    rho: float         # Price/vol correlation
    v0: float          # Initial variance (current)


@dataclass
class MarkovSnapshot:
    current_regime: str        # "LOW_VOL" | "NORMAL" | "HIGH_VOL"
    current_state: int
    next_regime: str
    next_regime_prob: float    # Probability of transitioning to next_regime
    stability: float           # P(stay in current regime)
    avg_return_in_regime: float


@dataclass
class GARCHSnapshot:
    current_volatility: float   # Annualised
    forecast_1day: float
    forecast_5day: float
    forecast_22day: float
    long_run_volatility: float
    half_life: float            # Days to revert to long-run mean


@dataclass
class FinBERTAggregation:
    dominant_sentiment: str     # "bullish" | "bearish" | "neutral"
    weighted_score: float       # -1.0 to +1.0
    avg_confidence: float
    bullish_count: int
    bearish_count: int
    neutral_count: int
    total_analyzed: int


@dataclass
class NewsItem:
    headline: str
    source: str
    sentiment_label: str
    sentiment_score: float
    confidence: float
    impact: str
    time_ago: str
    url: str


# ---------------------------------------------------------------------------
# Master dataclass
# ---------------------------------------------------------------------------

@dataclass
class MarketContext:
    # Identity
    symbol: str
    timestamp: float = field(default_factory=time.time)

    # Price
    current_price: float = 0.0
    price_change_pct: float = 0.0      # 1-day % change
    trend_direction: str = "UNKNOWN"   # "UP" | "DOWN" | "SIDEWAYS"

    # Technical signals (from generate_signals)
    technical_action: str = "HOLD"
    technical_confidence: float = 0.5
    signals: List[TechnicalSignal] = field(default_factory=list)

    # Unified recommendation (from get_unified_trading_recommendation)
    unified_action: str = "HOLD"
    unified_confidence: float = 0.5
    entry_zone: str = "N/A"
    stop_loss: str = "N/A"
    take_profit: str = "N/A"
    risk_reward: str = "N/A"
    volatility_regime: str = "NORMAL"
    session: str = "UNKNOWN"
    signal_breakdown: Dict[str, float] = field(default_factory=dict)

    # Heston stochastic vol
    heston: Optional[HestonSnapshot] = None

    # Markov regime
    markov: Optional[MarkovSnapshot] = None

    # GARCH forecast
    garch: Optional[GARCHSnapshot] = None

    # Risk metrics
    var_95: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Monte Carlo (30-day, 200 paths)
    mc_mean_price: float = 0.0
    mc_prob_up: float = 0.5
    mc_ci_95_lower: float = 0.0
    mc_ci_95_upper: float = 0.0

    # News + FinBERT
    news_items: List[NewsItem] = field(default_factory=list)
    finbert_aggregation: Optional[FinBERTAggregation] = None


# ---------------------------------------------------------------------------
# FinBERT aggregation helper
# ---------------------------------------------------------------------------

def _aggregate_finbert(results: List[Dict]) -> FinBERTAggregation:
    """
    Confidence-weighted aggregate of FinBERT per-headline results.

    High-confidence predictions dominate; tie between bullish/bearish
    resolves to neutral.
    """
    if not results:
        return FinBERTAggregation(
            dominant_sentiment="neutral",
            weighted_score=0.0,
            avg_confidence=0.0,
            bullish_count=0,
            bearish_count=0,
            neutral_count=0,
            total_analyzed=0,
        )

    total_weight = sum(r.get("confidence", 0.5) for r in results)
    if total_weight > 0:
        weighted_score = sum(
            r.get("score", 0.0) * r.get("confidence", 0.5) for r in results
        ) / total_weight
    else:
        weighted_score = 0.0

    avg_confidence = total_weight / len(results)

    counts: Dict[str, int] = {"bullish": 0, "bearish": 0, "neutral": 0}
    for r in results:
        s = r.get("sentiment", "neutral")
        counts[s] = counts.get(s, 0) + 1

    # Tiebreak toward neutral
    if counts["bullish"] > counts["bearish"]:
        dominant = "bullish"
    elif counts["bearish"] > counts["bullish"]:
        dominant = "bearish"
    else:
        dominant = "neutral"

    return FinBERTAggregation(
        dominant_sentiment=dominant,
        weighted_score=round(weighted_score, 4),
        avg_confidence=round(avg_confidence, 4),
        bullish_count=counts["bullish"],
        bearish_count=counts["bearish"],
        neutral_count=counts.get("neutral", 0),
        total_analyzed=len(results),
    )


# ---------------------------------------------------------------------------
# Component fetch functions (each runs in its own thread)
# ---------------------------------------------------------------------------

def _fetch_price(symbol: str, ctx: MarketContext) -> None:
    try:
        from app import fetch_yahoo_finance_data, get_current_price
        ctx.current_price = get_current_price(symbol)
        df = fetch_yahoo_finance_data(symbol, period="3mo", interval="1d")
        if df is not None and "close" in df.columns and len(df) > 2:
            closes = df["close"].dropna()
            if len(closes) >= 2:
                ctx.price_change_pct = float(
                    (closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2] * 100
                )
            returns_5d = closes.pct_change().dropna().tail(5).mean()
            if returns_5d > 0.001:
                ctx.trend_direction = "UP"
            elif returns_5d < -0.001:
                ctx.trend_direction = "DOWN"
            else:
                ctx.trend_direction = "SIDEWAYS"
    except Exception as exc:
        print(f"[MarketContext:{symbol}] price error: {exc}")


def _fetch_signals(symbol: str, ctx: MarketContext) -> None:
    try:
        from app import generate_signals
        action, confidence, raw_signals = generate_signals(symbol)
        ctx.technical_action = action
        ctx.technical_confidence = confidence
        ctx.signals = [
            TechnicalSignal(
                indicator=s["indicator"],
                signal=s["signal"],
                strength=s["strength"],
                value=s["value"],
            )
            for s in raw_signals
        ]
    except Exception as exc:
        print(f"[MarketContext:{symbol}] signals error: {exc}")


def _fetch_unified_rec(symbol: str, ctx: MarketContext) -> None:
    try:
        from app import get_unified_trading_recommendation
        rec = get_unified_trading_recommendation(symbol)
        ctx.unified_action = rec.get("action", "HOLD")
        ctx.unified_confidence = rec.get("confidence", 0.5)
        ctx.entry_zone = rec.get("entry_zone", "N/A")
        ctx.stop_loss = rec.get("stop_loss", "N/A")
        ctx.take_profit = rec.get("take_profit", "N/A")
        ctx.risk_reward = rec.get("risk_reward", "N/A")
        ctx.volatility_regime = rec.get("volatility_regime", "NORMAL")
        ctx.session = rec.get("session", "UNKNOWN")
        ctx.signal_breakdown = rec.get("signal_breakdown", {})
    except Exception as exc:
        print(f"[MarketContext:{symbol}] unified rec error: {exc}")


def _fetch_heston(symbol: str, ctx: MarketContext) -> None:
    try:
        from app import calculate_real_heston_params
        hp = calculate_real_heston_params(symbol)
        ctx.heston = HestonSnapshot(
            kappa=hp.kappa,
            theta=hp.theta,
            xi=hp.xi,
            rho=hp.rho,
            v0=hp.v0,
        )
    except Exception as exc:
        print(f"[MarketContext:{symbol}] heston error: {exc}")


def _fetch_monte_carlo(symbol: str, ctx: MarketContext) -> None:
    try:
        from app import run_monte_carlo_simulation
        mc = run_monte_carlo_simulation(symbol, days=30, n_paths=200)
        ctx.mc_mean_price = float(mc["mean_price"])
        ctx.mc_prob_up = float(mc["prob_up"])
        ctx.mc_ci_95_lower = float(mc["ci_95"][0])
        ctx.mc_ci_95_upper = float(mc["ci_95"][1])
    except Exception as exc:
        print(f"[MarketContext:{symbol}] monte carlo error: {exc}")


def _fetch_markov(symbol: str, ctx: MarketContext) -> None:
    try:
        from app import fetch_yahoo_finance_data
        from services.markov_model import MarkovRegimeModel

        df = fetch_yahoo_finance_data(symbol, period="2y", interval="1wk")
        if df is None or len(df) < 50:
            return
        prices = df["close"].dropna()
        returns = prices.pct_change().dropna()

        model = MarkovRegimeModel(symbol, n_regimes=3)
        results = model.fit(prices, returns)

        current_state = int(results["current_state"])
        tm = results["transition_matrix"]
        regime_stats = results.get("regime_stats", {})
        next_state_probs = results.get("next_state_probs")

        if next_state_probs is not None and len(next_state_probs) > 0:
            next_state_idx = int(np.argmax(next_state_probs))
            next_regime_prob = float(next_state_probs[next_state_idx])
        else:
            next_state_idx = current_state
            next_regime_prob = float(tm[current_state, current_state]) if tm is not None else 0.5

        next_regime_label = (
            regime_stats.get(next_state_idx, {}).get("label", "UNKNOWN")
            if regime_stats else "UNKNOWN"
        )
        stability = float(tm[current_state, current_state]) if tm is not None else 0.5
        avg_return = float(regime_stats.get(current_state, {}).get("avg_return", 0.0))

        ctx.markov = MarkovSnapshot(
            current_regime=results["current_regime"],
            current_state=current_state,
            next_regime=next_regime_label,
            next_regime_prob=next_regime_prob,
            stability=stability,
            avg_return_in_regime=avg_return,
        )
    except Exception as exc:
        print(f"[MarketContext:{symbol}] markov error: {exc}")


def _fetch_garch(symbol: str, ctx: MarketContext) -> None:
    try:
        from app import fetch_yahoo_finance_data
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from volatility_models import VolatilityModels

        df = fetch_yahoo_finance_data(symbol, period="3mo", interval="1d")
        if df is None or "close" not in df.columns or len(df) < 30:
            return

        returns_arr = df["close"].pct_change().dropna().values
        vm = VolatilityModels(returns_arr)
        vm.fit_garch(p=1, q=1)
        vf = vm.get_volatility_forecast()

        ctx.garch = GARCHSnapshot(
            current_volatility=float(vf.current_volatility),
            forecast_1day=float(vf.forecast_1day),
            forecast_5day=float(vf.forecast_5day),
            forecast_22day=float(vf.forecast_22day),
            long_run_volatility=float(vf.long_run_volatility),
            half_life=float(vf.half_life) if np.isfinite(vf.half_life) else 999.0,
        )
    except Exception as exc:
        print(f"[MarketContext:{symbol}] garch error: {exc}")


def _fetch_risk_metrics(symbol: str, ctx: MarketContext) -> None:
    try:
        from app import fetch_yahoo_finance_data
        from services.advanced_models import (
            calculate_expected_shortfall,
            calculate_sharpe_ratio,
            calculate_sortino_ratio,
            calculate_max_drawdown,
        )

        df = fetch_yahoo_finance_data(symbol, period="3mo", interval="1d")
        if df is None or "close" not in df.columns or len(df) < 30:
            return

        ret = df["close"].pct_change().dropna().values
        ctx.var_95 = float(np.percentile(ret, 5))
        ctx.expected_shortfall = float(calculate_expected_shortfall(ret, 0.95))
        equity = (1 + ret).cumprod()
        ctx.max_drawdown = float(calculate_max_drawdown(equity))
        ctx.sharpe_ratio = float(calculate_sharpe_ratio(ret))
        ctx.sortino_ratio = float(calculate_sortino_ratio(ret))
    except Exception as exc:
        print(f"[MarketContext:{symbol}] risk metrics error: {exc}")


def _fetch_news_and_sentiment(symbol: str, ctx: MarketContext) -> None:
    try:
        from services.ai_news import get_intelligent_news
        from services.local_ai_service import get_local_ai_service

        raw_news = get_intelligent_news(symbol, max_items=10)
        ctx.news_items = [
            NewsItem(
                headline=n.get("headline", ""),
                source=n.get("source", ""),
                sentiment_label=n.get("sentiment_label", "neutral"),
                sentiment_score=float(n.get("sentiment", 0.0)),
                confidence=float(n.get("confidence", 0.0)),
                impact=n.get("impact", "MEDIUM"),
                time_ago=n.get("time_ago", ""),
                url=n.get("url", ""),
            )
            for n in raw_news
            if n.get("headline")
        ]

        headlines = [ni.headline for ni in ctx.news_items]
        if headlines:
            ai_svc = get_local_ai_service()
            finbert_results = ai_svc.sentiment_analyzer.analyze_batch(headlines)
            ctx.finbert_aggregation = _aggregate_finbert(finbert_results)
    except Exception as exc:
        print(f"[MarketContext:{symbol}] news/finbert error: {exc}")


# ---------------------------------------------------------------------------
# Main builder — parallel with 8-second wall-clock deadline
# ---------------------------------------------------------------------------

def build_market_context(symbol: str) -> MarketContext:
    """
    Assemble all platform computations for *symbol* into a MarketContext.

    All components run in parallel (ThreadPoolExecutor).  The hard deadline is
    8 seconds — any component that hasn't finished by then is left at its safe
    default value.  Partial context is always better than no context.

    Args:
        symbol: Trading symbol, e.g. "XAUUSD"

    Returns:
        MarketContext populated with all available data.
    """
    ctx = MarketContext(symbol=symbol)

    tasks = {
        "price":        lambda: _fetch_price(symbol, ctx),
        "signals":      lambda: _fetch_signals(symbol, ctx),
        "unified_rec":  lambda: _fetch_unified_rec(symbol, ctx),
        "heston":       lambda: _fetch_heston(symbol, ctx),
        "monte_carlo":  lambda: _fetch_monte_carlo(symbol, ctx),
        "markov":       lambda: _fetch_markov(symbol, ctx),
        "garch":        lambda: _fetch_garch(symbol, ctx),
        "risk_metrics": lambda: _fetch_risk_metrics(symbol, ctx),
        "news":         lambda: _fetch_news_and_sentiment(symbol, ctx),
    }

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_to_name = {executor.submit(fn): name for name, fn in tasks.items()}
        try:
            for future in as_completed(future_to_name, timeout=25.0):
                name = future_to_name[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"[MarketContext:{symbol}] task '{name}' raised: {exc}")
        except TimeoutError:
            print(f"[MarketContext:{symbol}] 25s deadline reached — using partial context")

    return ctx


# ---------------------------------------------------------------------------
# Prompt formatter
# ---------------------------------------------------------------------------

def format_for_prompt(ctx: MarketContext) -> str:
    """
    Serialise a MarketContext into a structured plain-text context block
    suitable for prepending to an AI prompt.

    Target size: < 1 000 tokens (~4 000 characters).
    """
    lines: List[str] = [
        f"=== MARKET CONTEXT: {ctx.symbol} ===",
        f"Generated: {datetime.fromtimestamp(ctx.timestamp).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## PRICE",
        f"  Current price: {ctx.current_price:,.4f}",
        f"  1-day change:  {ctx.price_change_pct:+.2f}%",
        f"  Trend:         {ctx.trend_direction}",
        f"  Session:       {ctx.session}",
        "",
        f"## TECHNICAL SIGNALS ({len(ctx.signals)} indicators)",
    ]
    for s in ctx.signals:
        lines.append(
            f"  {s.indicator:<24} {s.signal:<6} [{s.strength:<8}]  value={s.value}"
        )

    lines += [
        "",
        "## UNIFIED RECOMMENDATION",
        f"  Action:        {ctx.unified_action}  (confidence {ctx.unified_confidence:.0%})",
        f"  Entry zone:    {ctx.entry_zone}",
        f"  Stop-loss:     {ctx.stop_loss}",
        f"  Take-profit:   {ctx.take_profit}",
        f"  Risk/Reward:   {ctx.risk_reward}",
        f"  Vol regime:    {ctx.volatility_regime}",
    ]
    if ctx.signal_breakdown:
        parts = "  ".join(
            f"{k}={v:.2f}" for k, v in ctx.signal_breakdown.items()
        )
        lines.append(f"  Signal scores: {parts}")

    lines += ["", "## HESTON STOCHASTIC VOLATILITY"]
    if ctx.heston:
        h = ctx.heston
        lines += [
            f"  kappa (mean-rev):  {h.kappa:.3f}",
            f"  theta (LR var):    {h.theta:.6f}  → LR vol {(h.theta ** 0.5) * 100:.1f}%",
            f"  xi    (vol-of-vol):{h.xi:.3f}",
            f"  rho   (price/vol): {h.rho:.3f}",
            f"  v0    (curr var):  {h.v0:.6f}  → current vol {(h.v0 ** 0.5) * 100:.1f}%",
        ]
    else:
        lines.append("  Not available")

    lines += ["", "## MARKOV REGIME DETECTION"]
    if ctx.markov:
        m = ctx.markov
        lines += [
            f"  Current regime:   {m.current_regime} (state #{m.current_state})",
            f"  Stability:        {m.stability:.0%} (prob of staying)",
            f"  Next regime:      {m.next_regime} ({m.next_regime_prob:.0%} prob)",
            f"  Avg return/period:{m.avg_return_in_regime:.4%}",
        ]
    else:
        lines.append("  Not available")

    lines += ["", "## GARCH VOLATILITY FORECAST (annualised)"]
    if ctx.garch:
        g = ctx.garch
        lines += [
            f"  Current:   {g.current_volatility:.1%}",
            f"  1-day fwd: {g.forecast_1day:.1%}",
            f"  5-day fwd: {g.forecast_5day:.1%}",
            f"  22-day fwd:{g.forecast_22day:.1%}",
            f"  Long-run:  {g.long_run_volatility:.1%}",
            f"  Half-life: {g.half_life:.1f} days",
        ]
    else:
        lines.append("  Not available")

    lines += [
        "",
        "## RISK METRICS (90-day daily returns)",
        f"  VaR 95% (1-day):    {abs(ctx.var_95):.2%}",
        f"  Expected Shortfall: {abs(ctx.expected_shortfall):.2%}",
        f"  Max Drawdown:       {ctx.max_drawdown:.2%}",
        f"  Sharpe Ratio:       {ctx.sharpe_ratio:.3f}",
        f"  Sortino Ratio:      {ctx.sortino_ratio:.3f}",
        "",
        "## MONTE CARLO (30-day, 200 paths)",
        f"  Mean price:   {ctx.mc_mean_price:,.4f}",
        f"  Prob(up):     {ctx.mc_prob_up:.0%}",
        f"  95% CI:       [{ctx.mc_ci_95_lower:,.2f} – {ctx.mc_ci_95_upper:,.2f}]",
    ]

    n_articles = ctx.finbert_aggregation.total_analyzed if ctx.finbert_aggregation else 0
    lines += ["", f"## FINBERT NEWS SENTIMENT ({n_articles} articles)"]
    if ctx.finbert_aggregation:
        fa = ctx.finbert_aggregation
        lines += [
            f"  Dominant:    {fa.dominant_sentiment}  (score {fa.weighted_score:+.3f})",
            f"  Confidence:  {fa.avg_confidence:.0%}",
            f"  Breakdown:   {fa.bullish_count} bullish / "
            f"{fa.bearish_count} bearish / {fa.neutral_count} neutral",
        ]
    else:
        lines.append("  Not available")

    lines += ["", "## TOP NEWS HEADLINES"]
    for i, ni in enumerate(ctx.news_items[:5], 1):
        label = ni.sentiment_label.upper()
        lines.append(
            f"  {i}. [{label:<8}] {ni.headline[:100]}"
            f"  ({ni.source}, {ni.time_ago})"
        )

    lines.append("\n=== END CONTEXT ===")
    return "\n".join(lines)


def format_for_ollama_prompt(ctx: MarketContext) -> str:
    """
    Compact (~120-token) context block for small local LLMs (e.g. llama3.2:1b).

    Only includes the most actionable data so the model can respond within 30s
    on CPU hardware.  Used instead of format_for_prompt() when Ollama is the
    backend, because sending a 600-token context to a 1B model causes timeouts.
    """
    regime = ctx.markov.current_regime if ctx.markov else "N/A"
    garch_1d = f"{ctx.garch.forecast_1day:.1%}" if ctx.garch else "N/A"
    fb = ctx.finbert_aggregation
    fb_str = f"{fb.dominant_sentiment} ({fb.weighted_score:+.2f})" if fb else "N/A"
    top_news = ctx.news_items[0].headline[:80] if ctx.news_items else "No news"

    # Pick the 3 strongest-signal indicators
    order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2}
    sorted_sigs = sorted(ctx.signals, key=lambda s: order.get(s.strength, 3))
    sig_lines = "\n".join(
        f"  {s.indicator}: {s.signal} [{s.strength}] {s.value}"
        for s in sorted_sigs[:3]
    )

    return (
        f"Symbol: {ctx.symbol} | Price: {ctx.current_price:,.2f} "
        f"({ctx.price_change_pct:+.2f}% today) | Trend: {ctx.trend_direction}\n"
        f"Recommendation: {ctx.unified_action} (confidence {ctx.unified_confidence:.0%})\n"
        f"Top signals:\n{sig_lines}\n"
        f"Regime: {regime} | Vol forecast 1d: {garch_1d}\n"
        f"FinBERT news sentiment: {fb_str}\n"
        f"Top news: {top_news}"
    )
