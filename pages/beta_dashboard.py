"""
Beta Dashboard — ML Model Trading Signals & Metrics + Regime Integration
========================================================================
Shows live BUY/SELL/HOLD signals from trained LGB/XGB/RF models
+ regime detection & prediction + regime-adjusted signals.
"""
import sys
import os
from pathlib import Path
import json
from datetime import datetime, timedelta
from dateutil.tz import tzlocal

import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from app import COLORS
from beta_testing.features import compute_1m_features
from beta_testing.models.lgb_model import Gold1mLightGBM
from beta_testing.models.xgb_model import Gold1mXGBoost
from beta_testing.models.rf_model import Gold1mRandomForest
import regime_integration as ri
from regime_integration_v2 import IntegratedTradingSystem
from sentiment.fetcher_expanded import ExpandedNewsFetcher
from sentiment.hardened_pretrade_booster import HardenedPreTradeBooster
from sentiment.sentiment_hardened import HardenedSentimentAnalyzer
from paper_trading.logger import (
    get_performance_summary,
    get_recent_trades_table,
    compute_equity_curve,
)
from execution.engine import ExecutionEngine

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[MT5] MetaTrader5 package not installed")

# Exness uses "XAUUSDm" (micro/cent account suffix).  Fallback cascade tries
# the most common Exness variants before giving up.
MT5_GOLD_SYMBOLS = ["XAUUSDm", "XAUUSD", "GOLD"]

# ── Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "data" / "beta_testing" / "processed" / "models"
DATA_PATH = PROJECT_ROOT / "data" / "beta_testing" / "processed" / "gold_2025_2026.csv"
RESULTS_CSV = MODELS_DIR / "final_results_v9.csv"

HORIZONS = [20, 60]
MODEL_NAMES = ["lgb", "xgb", "rf"]
MODEL_LABELS = {"lgb": "LightGBM", "xgb": "XGBoost", "rf": "RandomForest"}

# Regime colors
REGIME_COLORS = {
    "STRONG_TREND_UP": COLORS["success"],
    "STRONG_TREND_DOWN": COLORS["danger"],
    "GRIND_UP": "#2ed573",
    "CHOPPY_RANGE": COLORS["warning"],
    "HIGH_VOL_CHAOS": "#ff3838",
    "LOW_VOL_DRIFT": "#a4b0be",
    "UNKNOWN": COLORS["text_secondary"],
}


# ── Model cache ─────────────────────────────────────────────────────────
class ModelCache:
    def __init__(self):
        self.models = {}
        self.feature_cols = {}
        self.results = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        for h in HORIZONS:
            prefix = f"beta_h{h}"
            self.models[h] = {}
            for name in MODEL_NAMES:
                # XGB models use .ubj (native format), others use .pkl
                ext = "ubj" if name == "xgb" else "pkl"
                path = MODELS_DIR / f"{prefix}_{name}.{ext}"
                if not path.exists():
                    continue
                if name == "lgb":
                    m = Gold1mLightGBM()
                elif name == "xgb":
                    m = Gold1mXGBoost()
                else:
                    m = Gold1mRandomForest()
                try:
                    m.load(str(path))
                    self.models[h][name] = m
                except Exception as e:
                    print(f"[BetaDashboard] Failed to load {path}: {e}")

            feat_path = MODELS_DIR / f"{prefix}_features.json"
            if feat_path.exists():
                with open(feat_path) as f:
                    self.feature_cols[h] = json.load(f)

            res_path = MODELS_DIR / f"{prefix}_results.json"
            if res_path.exists():
                with open(res_path) as f:
                    self.results[h] = json.load(f)

        self._loaded = True

    def is_ready(self):
        return any(self.models.get(h, {}) for h in HORIZONS)


_model_cache = ModelCache()

# ── Microstructure + Risk Manager Integration ───────────────────────────
_trading_system = IntegratedTradingSystem(account_balance=10000.0)
_execution_engine = ExecutionEngine()


# ── Data cache (live data with CSV fallback) ────────────────────────────
_df_cache = None
_df_cache_time = None
_df_cache_source = None  # "mt5" | "csv"
CACHE_TTL_LIVE = 60      # Normal cache for live MT5 data
CACHE_TTL_FALLBACK = 10  # Aggressive retry for CSV fallback

def _load_df():
    """Load live gold data from MT5 broker, with yfinance + CSV fallback."""
    global _df_cache, _df_cache_time, _df_cache_source

    print(f"[_load_df] Cache: source={_df_cache_source}, age={(datetime.now()-_df_cache_time).total_seconds() if _df_cache_time else 'None'}s")

    # Check cache — source-aware TTL
    if _df_cache is not None and _df_cache_time is not None:
        ttl = CACHE_TTL_LIVE if _df_cache_source == "mt5" else CACHE_TTL_FALLBACK
        if (datetime.now() - _df_cache_time).total_seconds() < ttl:
            print(f"[_load_df] Returning cached data (source={_df_cache_source}, ttl={ttl}s)")
            return _df_cache

    # 1. Try MT5 broker data (primary — real-time from Exness)
    if MT5_AVAILABLE:
        selected_symbol = None
        try:
            if not mt5.initialize():
                raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

            terminal_info = mt5.terminal_info()
            if terminal_info:
                print(f"[MT5] Connected to terminal: {terminal_info.name}, build: {terminal_info.build}")

            for sym in MT5_GOLD_SYMBOLS:
                if mt5.symbol_select(sym, True):
                    rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M1, 0, 500)
                    if rates is not None and len(rates) > 0:
                        selected_symbol = sym
                        print(f"[MT5] Selected symbol: {sym}")
                        break

            if selected_symbol is None:
                raise RuntimeError(f"None of {MT5_GOLD_SYMBOLS} available in Market Watch")

            if rates is None or len(rates) < 200:
                raise RuntimeError("Not enough bars")

            df_live = pd.DataFrame(rates)
            # Use datetime.fromtimestamp to respect local timezone (not UTC)
            df_live['date'] = pd.to_datetime([datetime.fromtimestamp(t) for t in df_live['time']])
            df_live = df_live.rename(columns={'tick_volume': 'volume'})
            df_live = df_live[['date', 'open', 'high', 'low', 'close', 'volume']]
            df_live = df_live.sort_values("date").set_index("date")

            last_bar_time = df_live.index[-1]
            age_sec = (datetime.now() - last_bar_time).total_seconds()
            if age_sec < 120:
                _df_cache = df_live
                _df_cache_time = datetime.now()
                _df_cache_source = "mt5"
                print(f"[MT5] Loaded {len(df_live)} live bars via {selected_symbol}. Last: {last_bar_time}")
                mt5.shutdown()
                return _df_cache
            else:
                mt5.shutdown()
                print("\n" + "="*70)
                print("WARNING: MT5 returned stale M1 bars!")
                print(f"  Last bar age: {age_sec/60:.0f} minutes")
                print("  Likely cause: mt5.initialize() connected to the WRONG terminal.")
                print("  FIX:")
                print("    1. Close ALL MetaTrader 5 terminals")
                print("    2. Open ONLY ONE terminal (your Exness account)")
                print("    3. Open the XAUUSDm M1 chart")
                print("    4. Wait 1 minute for M1 bars to build")
                print("    5. Restart the dashboard")
                print("="*70 + "\n")
                raise RuntimeError("MT5 data stale")

        except Exception as e:
            print(f"[MT5] Error: {e}")
            try:
                mt5.shutdown()
            except:
                pass

    # 2. Fallback to yfinance (GC=F futures)
    try:
        ticker = yf.Ticker("GC=F")
        df_live = ticker.history(period="5d", interval="1m")

        if not df_live.empty:
            df_live = df_live.reset_index()
            df_live = df_live.rename(columns={
                'Datetime': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            df_live['date'] = pd.to_datetime(df_live['date']).dt.tz_convert(tzlocal()).dt.tz_localize(None)
            df_live = df_live.sort_values("date").reset_index(drop=True)

            # Clean: drop zero volume, forward-fill gaps, drop NaN
            df_live = df_live[df_live['volume'] > 0]
            df_live[['open', 'high', 'low', 'close']] = df_live[['open', 'high', 'low', 'close']].ffill()
            df_live = df_live.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

            last_bar_time = df_live['date'].iloc[-1]
            if (datetime.now() - last_bar_time).total_seconds() < 120:
                df_live = df_live.set_index("date").sort_index()
                _df_cache = df_live
                _df_cache_time = datetime.now()
                print(f"[LiveData] Loaded {len(df_live)} live bars. Last: {last_bar_time}")
                return _df_cache
            else:
                print(f"[LiveData] Stale data (last bar: {last_bar_time}), falling back to CSV")
    except Exception as e:
        print(f"[LiveData] Error fetching yfinance: {e}")

    # 3. Final fallback to CSV
    if _df_cache is None and DATA_PATH.exists():
        print(f"[CSV] Loading fallback: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        df = df.drop(columns=[c for c in ["is_original", "minutes_since_last_bar"] if c in df.columns], errors="ignore")
        _df_cache = df
        _df_cache_time = datetime.now()
        _df_cache_source = "csv"
        print(f"[CSV] Loaded {len(df)} fallback rows. Last: {df.index[-1]}")
        return _df_cache

    return _df_cache


# ── Signal generation ───────────────────────────────────────────────────
def _generate_live_signals():
    """Generate live signals from the most recent data bar."""
    _model_cache.load()
    df = _load_df()
    if not _model_cache.is_ready() or df is None:
        return None, None

    # Only need last ~5000 rows for feature rolling windows (max MA=200 + buffers)
    df_tail = df.tail(5000)
    features = compute_1m_features(df_tail)

    signals = {}
    for h in HORIZONS:
        if h not in _model_cache.models or h not in _model_cache.feature_cols:
            continue

        target_cols = [c for c in features.columns if c.startswith("target_")]
        X_live = features.drop(columns=target_cols).iloc[[-1]]
        feat_cols = _model_cache.feature_cols[h]
        X_live = X_live.reindex(columns=feat_cols, fill_value=0)

        preds = {}
        for name, model in _model_cache.models[h].items():
            try:
                prob = float(model.predict(X_live)[0])
                preds[name] = prob
            except Exception as e:
                print(f"[BetaDashboard] Prediction error {name} H={h}: {e}")
                preds[name] = 0.5

        if preds:
            avg_prob = np.mean(list(preds.values()))
            action = "BUY" if avg_prob > 0.65 else "SELL" if avg_prob < 0.35 else "HOLD"
            confidence = abs(avg_prob - 0.5) * 2
            signals[h] = {
                "action": action,
                "confidence": confidence,
                "raw_prob": avg_prob,
                "preds": preds,
                "price": float(df["close"].iloc[-1]),
            }

    return signals, df


# ── Regime integration ──────────────────────────────────────────────────
def _get_regime_prediction(df: pd.DataFrame, raw_signal: dict = None):
    """Get regime prediction via regime_integration bridge.
    If raw_signal is provided, skips redundant ML recomputation."""
    try:
        return ri.predict(df, raw_signal=raw_signal)
    except Exception as e:
        print(f"[BetaDashboard] Regime integration error: {e}")
        return None


# ── UI helpers ──────────────────────────────────────────────────────────
def _card(title, body):
    return dbc.Card([
        dbc.CardHeader(
            html.Span(title, style={"fontWeight": "bold", "color": COLORS["text"],
                       "fontSize": "11px", "letterSpacing": "1px"}),
            style={"backgroundColor": COLORS["surface"],
                   "borderBottom": f"1px solid {COLORS['border']}",
                   "padding": "9px 14px"}
        ),
        dbc.CardBody(body, style={"backgroundColor": COLORS["surface"], "padding": "12px"}),
    ], style={"backgroundColor": COLORS["surface"],
              "border": f"1px solid {COLORS['border']}",
              "borderRadius": "6px", "marginBottom": "12px"})


def _stat(label, value_id, color=None, sub=""):
    return html.Div([
        html.Div(label, style={"color": COLORS["text_secondary"], "fontSize": "9px",
                                "letterSpacing": "0.8px", "textTransform": "uppercase"}),
        html.Div("--", id=value_id, style={"color": color or COLORS["text"], "fontSize": "20px",
                                "fontWeight": "bold", "lineHeight": "1.2"}),
        html.Div(sub, style={"color": COLORS["text_secondary"], "fontSize": "9px", "marginTop": "2px"}),
    ], style={"padding": "8px 12px", "borderRight": f"1px solid {COLORS['border']}", "minWidth": "100px"})


def _stat_static(label, value, color=None, sub=""):
    """Static stat display (no callback id)."""
    return html.Div([
        html.Div(label, style={"color": COLORS["text_secondary"], "fontSize": "9px",
                                "letterSpacing": "0.8px", "textTransform": "uppercase"}),
        html.Div(value, style={"color": color or COLORS["text"], "fontSize": "20px",
                                "fontWeight": "bold", "lineHeight": "1.2"}),
        html.Div(sub, style={"color": COLORS["text_secondary"], "fontSize": "9px", "marginTop": "2px"}),
    ], style={"padding": "8px 12px", "borderRight": f"1px solid {COLORS['border']}", "minWidth": "100px"})


def _signal_badge(action, confidence):
    if action == "BUY":
        col = COLORS["success"]
        bg = "#00ff8820"
        icon = "📈"
    elif action == "SELL":
        col = COLORS["danger"]
        bg = "#ff475720"
        icon = "📉"
    else:
        col = COLORS["warning"]
        bg = "#ffa50220"
        icon = "⏸️"

    return html.Div([
        html.Div("ML SIGNAL", style={"color": COLORS["text_secondary"], "fontSize": "9px",
                                       "letterSpacing": "0.8px", "textTransform": "uppercase"}),
        html.Div(f"{icon} {action}", style={"fontSize": "32px", "fontWeight": "bold",
                                              "color": col, "lineHeight": "1"}),
        html.Div(f"Confidence: {confidence:.1%}", style={"color": col, "fontSize": "10px",
                                                            "marginTop": "4px"}),
    ], style={"textAlign": "center", "padding": "10px"})


def _regime_badge(regime, confidence):
    col = REGIME_COLORS.get(regime, COLORS["text_secondary"])
    return html.Div([
        html.Div("REGIME", style={"color": COLORS["text_secondary"], "fontSize": "9px",
                                    "letterSpacing": "0.8px", "textTransform": "uppercase"}),
        html.Div(regime.replace("_", " "), style={"fontSize": "24px", "fontWeight": "bold",
                                                     "color": col, "lineHeight": "1"}),
        html.Div(f"Confidence: {confidence:.1%}", style={"color": col, "fontSize": "10px",
                                                            "marginTop": "4px"}),
    ], style={"textAlign": "center", "padding": "10px"})


# ── Layout ──────────────────────────────────────────────────────────────
layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": COLORS["background"], "minHeight": "100vh", "padding": "14px"},
    children=[
        html.H2("Beta Dashboard", style={"color": COLORS["accent"], "marginBottom": "4px"}),
        html.P("ML + Regime-Aware Trading Signals — 343K+ XAU/USD 1M bars (2025-2026)",
               style={"color": COLORS["text_secondary"], "fontSize": "13px"}),

        # ── Live Signal Banner ────────────────────────────────────────
        _card(
            "🟢 LIVE ML SIGNAL  ·  LGB + XGB + RF ensemble",
            html.Div([
                dbc.Row([
                    dbc.Col(html.Div(id="beta-signal-badge"), width=3),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(_stat("PRICE", "beta-price"), width=3),
                            dbc.Col(_stat("H=20 SIGNAL", "beta-h20-action", COLORS["accent"]), width=3),
                            dbc.Col(_stat("H=20 CONF", "beta-h20-conf", COLORS["info"]), width=3),
                            dbc.Col(_stat("H=60 SIGNAL", "beta-h60-action", COLORS["accent"]), width=3),
                        ]),
                        dbc.Row([
                            dbc.Col(_stat("H=60 CONF", "beta-h60-conf", COLORS["info"]), width=2),
                            dbc.Col(_stat("LGB PROB", "beta-lgb-prob", COLORS["info"]), width=2),
                            dbc.Col(_stat("XGB PROB", "beta-xgb-prob", COLORS["info"]), width=2),
                            dbc.Col(_stat("RF PROB", "beta-rf-prob", COLORS["info"]), width=2),
                            dbc.Col(_stat("LAST UPDATE", "beta-last-update", COLORS["text"]), width=4),
                        ], className="mt-1"),
                    ], width=9),
                ], align="center"),
                html.Div(id="beta-signal-status",
                         children="Loading models...",
                         style={"color": COLORS["text_secondary"], "fontSize": "10px",
                                "marginTop": "8px", "textAlign": "center"}),
            ])
        ),

        # ── Regime Panel ──────────────────────────────────────────────
        _card(
            "📊 MARKET REGIME  ·  HMM Detection + 20m Prediction",
            html.Div([
                dbc.Row([
                    dbc.Col(html.Div(id="beta-regime-badge"), width=3),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(_stat("CURRENT REGIME", "beta-regime-current", COLORS["accent"]), width=3),
                            dbc.Col(_stat("REGIME CONF", "beta-regime-conf", COLORS["info"]), width=3),
                            dbc.Col(_stat("PREDICTED (20m)", "beta-regime-predicted", COLORS["warning"]), width=3),
                            dbc.Col(_stat("PRED CONF", "beta-regime-pred-conf", COLORS["info"]), width=3),
                        ]),
                        dbc.Row([
                            dbc.Col(_stat("TRADING STATUS", "beta-trading-status", COLORS["success"]), width=3),
                            dbc.Col(_stat("POSITION SIZE", "beta-position-multiplier", COLORS["info"]), width=3),
                            dbc.Col(_stat("TRANSITION WARN", "beta-transition-warn", COLORS["warning"]), width=3),
                            dbc.Col(_stat("REGIME REASON", "beta-regime-reason", COLORS["text_secondary"]), width=3),
                        ], className="mt-1"),
                    ], width=9),
                ], align="center"),
            ])
        ),

        # ── 15M Primary Bias ──────────────────────────────────────────
        _card(
            "📡 15M PRIMARY BIAS  ·  Ensemble Directional Signal",
            html.Div([
                dbc.Row([
                    dbc.Col(_stat("15M DIRECTION", "beta-15m-direction", COLORS["accent"]), width=3),
                    dbc.Col(_stat("15M STRENGTH", "beta-15m-strength", COLORS["info"]), width=3),
                    dbc.Col(_stat("WEIGHTED CONVICTION", "beta-15m-agreement", COLORS["warning"]), width=3),
                    dbc.Col(_stat("15M REGIME", "beta-15m-regime", COLORS["text_secondary"]), width=3),
                ]),
                html.Div(id="beta-15m-raw-probs",
                         style={"color": COLORS["text_secondary"], "fontSize": "10px",
                                "marginTop": "8px", "textAlign": "center"}),
            ])
        ),

        # ── 1M Execution Signal ───────────────────────────────────────
        _card(
            "⚡ 1M EXECUTION SIGNAL  ·  Microstructure Entry Timing",
            html.Div([
                dbc.Row([
                    dbc.Col(_stat("EXECUTION", "beta-1m-decision", COLORS["accent"]), width=3),
                    dbc.Col(_stat("ENTRY QUALITY", "beta-1m-quality", COLORS["info"]), width=3),
                    dbc.Col(_stat("STOP LOSS", "beta-stop-loss", COLORS["danger"]), width=2),
                    dbc.Col(_stat("TAKE PROFIT", "beta-take-profit", COLORS["success"]), width=2),
                    dbc.Col(_stat("R:R", "beta-risk-reward", COLORS["info"]), width=2),
                ]),
                html.Div(id="beta-1m-reason",
                         children="Waiting for execution signal...",
                         style={"color": COLORS["text_secondary"], "fontSize": "11px",
                                "marginTop": "10px", "textAlign": "center"}),
            ])
        ),

        # ── Position State Machine ────────────────────────────────────
        _card(
            "🎯 POSITION STATE  ·  Open Trade Tracking",
            html.Div([
                dbc.Row([
                    dbc.Col(_stat("STATE", "beta-position-state", COLORS["accent"]), width=3),
                    dbc.Col(_stat("UNREALIZED P&L", "beta-position-pnl", COLORS["info"]), width=3),
                    dbc.Col(_stat("HOLD TIME", "beta-position-hold", COLORS["warning"]), width=3),
                    dbc.Col(_stat("LAST EXIT", "beta-last-exit", COLORS["text_secondary"]), width=3),
                ]),
            ])
        ),

        # ── Manual Execution Panel ────────────────────────────────────
        _card(
            "⚡ MANUAL EXECUTION  ·  MT5 Order Preparation",
            html.Div([
                dbc.Row([
                    dbc.Col(_stat("MT5 STATUS", "beta-exec-status", COLORS["accent"]), width=3),
                    dbc.Col(_stat("SYMBOL", "beta-exec-symbol", COLORS["info"]), width=3),
                    dbc.Col(_stat("SPREAD", "beta-exec-spread", COLORS["warning"]), width=3),
                    dbc.Col(_stat("FREE MARGIN", "beta-exec-margin", COLORS["success"]), width=3),
                ]),
                html.Hr(style={"borderColor": COLORS["border"], "margin": "12px 0"}),
                html.Div(id="beta-exec-prepared",
                         children="No active signal - waiting for setup...",
                         style={"color": COLORS["text_secondary"], "fontSize": "12px",
                                "textAlign": "center"}),
            ])
        ),

        _card(
            "🏦 BROKER POSITIONS  ·  MT5 Open Positions",
            html.Div(id="beta-broker-positions", children="No open positions")
        ),

        # ── Sentiment Validation Panel ────────────────────────────────
        _card(
            "📰 SENTIMENT ANALYSIS  ·  5-Layer Hardened Defense",
            html.Div([
                dbc.Row([
                    dbc.Col(_stat("SENTIMENT SCORE", "beta-sentiment-score", COLORS["info"]), width=2),
                    dbc.Col(_stat("DIRECTION", "beta-sentiment-dir", COLORS["accent"]), width=2),
                    dbc.Col(_stat("CONFIDENCE", "beta-sentiment-conf", COLORS["info"]), width=2),
                    dbc.Col(_stat("SOURCES", "beta-sentiment-sources", COLORS["text"]), width=2),
                    dbc.Col(_stat("VALID", "beta-sentiment-valid", COLORS["success"]), width=2),
                    dbc.Col(_stat("BOOST", "beta-sentiment-boost", COLORS["warning"]), width=2),
                ]),
                dbc.Row([
                    dbc.Col(_stat("L1 SOURCE", "beta-l1-source", COLORS["text_secondary"]), width=2),
                    dbc.Col(_stat("L2 FAKE GUARD", "beta-l2-fake", COLORS["text_secondary"]), width=2),
                    dbc.Col(_stat("L3 CONSENSUS", "beta-l3-consensus", COLORS["text_secondary"]), width=2),
                    dbc.Col(_stat("L4 PRICE", "beta-l4-price", COLORS["text_secondary"]), width=2),
                    dbc.Col(_stat("L5 BLACK SWAN", "beta-l5-swan", COLORS["text_secondary"]), width=2),
                    dbc.Col(_stat("COOLDOWN", "beta-cooldown", COLORS["danger"]), width=2),
                ], className="mt-1"),
                html.Div(id="beta-sentiment-reason",
                         children="Waiting for sentiment analysis...",
                         style={"color": COLORS["text_secondary"], "fontSize": "11px",
                                "marginTop": "10px", "textAlign": "center"}),
            ])
        ),

        # ── Microstructure Filter Panel ───────────────────────────────
        _card(
            "🔬 MICROSTRUCTURE FILTER  ·  OFI + VPIN + Entropy",
            html.Div([
                dbc.Row([
                    dbc.Col(_stat("OFI PROXY", "beta-micro-ofi", COLORS["accent"]), width=2),
                    dbc.Col(_stat("VPIN", "beta-micro-vpin", COLORS["warning"]), width=2),
                    dbc.Col(_stat("ENTROPY", "beta-micro-entropy", COLORS["info"]), width=2),
                    dbc.Col(_stat("HFT ACTIVITY", "beta-micro-hft", COLORS["text_secondary"]), width=2),
                    dbc.Col(_stat("QUALITY", "beta-micro-quality", COLORS["success"]), width=2),
                ]),
                html.Div(id="beta-micro-reason",
                         children="Microstructure analysis initializing...",
                         style={"color": COLORS["text_secondary"], "fontSize": "11px",
                                "marginTop": "10px", "textAlign": "center"}),
            ])
        ),

        # ── Risk Manager Panel ────────────────────────────────────────
        _card(
            "🛡️ RISK MANAGER  ·  Live P&L + Drawdown Monitoring",
            html.Div([
                dbc.Row([
                    dbc.Col(_stat("BALANCE", "beta-risk-balance", COLORS["success"]), width=2),
                    dbc.Col(_stat("DAILY PnL", "beta-risk-daily-pnl", COLORS["info"]), width=2),
                    dbc.Col(_stat("WIN RATE", "beta-risk-winrate", COLORS["accent"]), width=2),
                    dbc.Col(_stat("MAX DD", "beta-risk-maxdd", COLORS["danger"]), width=2),
                    dbc.Col(_stat("CONSEC LOSS", "beta-risk-consec", COLORS["warning"]), width=2),
                    dbc.Col(_stat("HALTED", "beta-risk-halted", COLORS["danger"]), width=2),
                ]),
                html.Div(id="beta-risk-reason",
                         children="Risk manager initializing...",
                         style={"color": COLORS["text_secondary"], "fontSize": "11px",
                                "marginTop": "10px", "textAlign": "center"}),
            ])
        ),

        # ── Paper Trading Performance ────────────────────────────────
        _card(
            "📊 PAPER TRADING  ·  Live Performance Analytics",
            html.Div([
                dbc.Row([
                    dbc.Col(_stat("TOTAL TRADES", "beta-paper-trades", COLORS["accent"]), width=2),
                    dbc.Col(_stat("WIN RATE", "beta-paper-winrate", COLORS["success"]), width=2),
                    dbc.Col(_stat("PROFIT FACTOR", "beta-paper-pf", COLORS["info"]), width=2),
                    dbc.Col(_stat("AVG WIN", "beta-paper-avgwin", COLORS["success"]), width=2),
                    dbc.Col(_stat("AVG LOSS", "beta-paper-avgloss", COLORS["danger"]), width=2),
                    dbc.Col(_stat("SHARPE", "beta-paper-sharpe", COLORS["warning"]), width=2),
                ]),
                html.Div(id="beta-paper-summary",
                         style={"color": COLORS["text_secondary"], "fontSize": "11px",
                                "marginTop": "10px", "textAlign": "center"}),
            ])
        ),

        _card(
            "📈 EQUITY CURVE  ·  Account Balance Over Time",
            dcc.Graph(id="beta-equity-chart", config={"displayModeBar": False},
                      style={"height": "280px"})
        ),

        _card(
            "📋 RECENT TRADES  ·  Last 10 Paper Trades",
            html.Div(id="beta-recent-trades", children=[])
        ),

        # ── Per-Model Cards ───────────────────────────────────────────
        html.H4("Per-Model Breakdown", style={"color": COLORS["accent"], "marginTop": "20px", "marginBottom": "12px"}),
        dbc.Row(id="beta-model-cards", className="g-3"),

        # ── Performance Metrics ───────────────────────────────────────
        html.H4("Model Performance (Test Set)", style={"color": COLORS["accent"], "marginTop": "20px", "marginBottom": "12px"}),
        dbc.Row(id="beta-metrics-row", className="g-3"),

        # ── Confidence Filtering (from V9 Kaggle results) ─────────────
        html.H4("V9 Kaggle Results — Confidence Filtering", style={"color": COLORS["accent"], "marginTop": "20px", "marginBottom": "12px"}),
        html.P("Full 6.79M row dataset results. Only trade when model confidence exceeds threshold.",
               style={"color": COLORS["text_secondary"], "fontSize": "12px"}),
        html.Div(id="beta-conf-table"),
        dcc.Graph(id="beta-acc-chart", config={"displayModeBar": False}),

        dcc.Interval(id="beta-dashboard-interval", interval=5_000),
    ],
)


# ── Callback ────────────────────────────────────────────────────────────
@callback(
    # ML signal outputs
    Output("beta-signal-badge", "children"),
    Output("beta-price", "children"),
    Output("beta-h20-action", "children"),
    Output("beta-h20-conf", "children"),
    Output("beta-h60-action", "children"),
    Output("beta-h60-conf", "children"),
    Output("beta-lgb-prob", "children"),
    Output("beta-xgb-prob", "children"),
    Output("beta-rf-prob", "children"),
    Output("beta-last-update", "children"),
    Output("beta-signal-status", "children"),
    # Regime outputs
    Output("beta-regime-badge", "children"),
    Output("beta-regime-current", "children"),
    Output("beta-regime-conf", "children"),
    Output("beta-regime-predicted", "children"),
    Output("beta-regime-pred-conf", "children"),
    Output("beta-trading-status", "children"),
    Output("beta-position-multiplier", "children"),
    Output("beta-transition-warn", "children"),
    Output("beta-regime-reason", "children"),
    # 15M Primary Bias outputs
    Output("beta-15m-direction", "children"),
    Output("beta-15m-strength", "children"),
    Output("beta-15m-agreement", "children"),
    Output("beta-15m-regime", "children"),
    Output("beta-15m-raw-probs", "children"),
    # 1M Execution signal outputs
    Output("beta-1m-decision", "children"),
    Output("beta-1m-quality", "children"),
    Output("beta-stop-loss", "children"),
    Output("beta-take-profit", "children"),
    Output("beta-risk-reward", "children"),
    Output("beta-1m-reason", "children"),
    # Position state outputs
    Output("beta-position-state", "children"),
    Output("beta-position-pnl", "children"),
    Output("beta-position-hold", "children"),
    Output("beta-last-exit", "children"),
    # Execution outputs
    Output("beta-exec-status", "children"),
    Output("beta-exec-symbol", "children"),
    Output("beta-exec-spread", "children"),
    Output("beta-exec-margin", "children"),
    Output("beta-exec-prepared", "children"),
    Output("beta-broker-positions", "children"),
    # Sentiment outputs
    Output("beta-sentiment-score", "children"),
    Output("beta-sentiment-dir", "children"),
    Output("beta-sentiment-conf", "children"),
    Output("beta-sentiment-sources", "children"),
    Output("beta-sentiment-valid", "children"),
    Output("beta-sentiment-boost", "children"),
    Output("beta-l1-source", "children"),
    Output("beta-l2-fake", "children"),
    Output("beta-l3-consensus", "children"),
    Output("beta-l4-price", "children"),
    Output("beta-l5-swan", "children"),
    Output("beta-cooldown", "children"),
    Output("beta-sentiment-reason", "children"),
    # Microstructure outputs
    Output("beta-micro-ofi", "children"),
    Output("beta-micro-vpin", "children"),
    Output("beta-micro-entropy", "children"),
    Output("beta-micro-hft", "children"),
    Output("beta-micro-quality", "children"),
    # Risk manager outputs
    Output("beta-risk-balance", "children"),
    Output("beta-risk-daily-pnl", "children"),
    Output("beta-risk-winrate", "children"),
    Output("beta-risk-maxdd", "children"),
    Output("beta-risk-consec", "children"),
    Output("beta-risk-halted", "children"),
    # Paper trading outputs
    Output("beta-paper-trades", "children"),
    Output("beta-paper-winrate", "children"),
    Output("beta-paper-pf", "children"),
    Output("beta-paper-avgwin", "children"),
    Output("beta-paper-avgloss", "children"),
    Output("beta-paper-sharpe", "children"),
    Output("beta-paper-summary", "children"),
    Output("beta-equity-chart", "figure"),
    Output("beta-recent-trades", "children"),
    # Existing outputs
    Output("beta-model-cards", "children"),
    Output("beta-metrics-row", "children"),
    Output("beta-conf-table", "children"),
    Output("beta-acc-chart", "figure"),
    Input("beta-dashboard-interval", "n_intervals"),
    prevent_initial_call=False,
)
def refresh_beta_dashboard(_n_intervals):
    _model_cache.load()
    signals, df = _generate_live_signals()

    # ── Data freshness gate ─────────────────────────────────────
    is_live = False
    if df is not None and len(df) > 0:
        age_sec = (datetime.now() - df.index[-1]).total_seconds()
        is_live = age_sec < 120
    else:
        age_sec = float('inf')

    if not is_live:
        banner = html.Div(
            "🔴 STALE DATA — TRADING BLOCKED (≥120s old)",
            style={"color": "#FF4136", "fontWeight": "bold", "fontSize": "14px", "textAlign": "center"}
        )
        # 54 outputs: blocked placeholder for every component
        no_data = html.Span("—", style={"color": COLORS["text_secondary"]})
        blocked = (
            # ML signals (11)
            no_data, no_data, no_data, no_data, no_data, no_data, no_data,
            no_data, no_data, no_data, banner,
            # Regime (9)
            no_data, no_data, no_data, no_data, no_data, no_data, no_data, no_data, no_data,
            # 15M Bias (5)
            no_data, no_data, no_data, no_data, no_data,
            # 1M Execution (6)
            no_data, no_data, no_data, no_data, no_data, no_data,
            # Position state (4)
            no_data, no_data, no_data, no_data,
            # Execution (6)
            no_data, no_data, no_data, no_data, no_data, no_data,
            # Sentiment (13)
            no_data, no_data, no_data, no_data, no_data, no_data, no_data,
            no_data, no_data, no_data, no_data, no_data, no_data,
            # Microstructure (5)
            no_data, no_data, no_data, no_data, no_data,
            # Risk manager (6)
            no_data, no_data, no_data, no_data, no_data, no_data,
            # Paper trading (9)
            no_data, no_data, no_data, no_data, no_data, no_data, no_data, go.Figure(), [],
            # Existing (4)
            [], [], [], go.Figure()
        )
        return blocked

    # Data is live — green banner injected into beta-signal-status later
    live_banner = html.Div(
        "✅ LIVE — MT5 Exness (< 2 min)",
        style={"color": "#2ECC40", "fontWeight": "bold", "fontSize": "14px", "textAlign": "center"}
    )
    
    # Build raw_signal for regime integration to skip redundant ML recomputation
    raw_signal = None
    if signals:
        primary = signals.get(60, signals.get(20, {}))
        individual_probs = {}
        for h, sig in signals.items():
            for name, prob in sig.get("preds", {}).items():
                individual_probs[f"{name}_h{h}"] = prob
        raw_signal = {
            "action": primary.get("action", "HOLD"),
            "confidence": primary.get("confidence", 0),
            "raw_prob": primary.get("raw_prob", 0.5),
            "individual_probs": individual_probs,
            "price": primary.get("price", 0),
        }
    
    regime_pred = _get_regime_prediction(df, raw_signal=raw_signal) if df is not None else None

    # ── HYBRID TWO-STAGE PIPELINE ───────────────────────────────
    signal_15m = None
    signal_v2 = None
    if df is not None and _trading_system is not None:
        try:
            # Stage 1: 15m directional bias (from 15m ensemble models)
            signal_15m = _trading_system._predict_15m(df)

            # Stage 2: 1m execution signal (microstructure entry timing)
            signal_v2 = _trading_system.get_1m_execution_signal(
                df_1m=df,
                signal_15m=signal_15m,
                current_time=datetime.now(),
                live_price=df["close"].iloc[-1],
            )
        except Exception as e:
            print(f"[BetaDashboard] Hybrid pipeline error: {e}")

    # Empty states
    empty = dbc.Col(
        html.Div("—", style={"color": COLORS["text_secondary"], "fontSize": "20px", "fontWeight": "bold"}),
        width=2
    )
    empty_fig = go.Figure().update_layout(
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"],
    )

    if signals is None or not _model_cache.is_ready():
        return (
            # ML signals
            _signal_badge("HOLD", 0),
            "—", "—", "—", "—", "—", "—", "—", "—", "—",
            "Models not loaded. Run training script first.",
            # Regime
            _regime_badge("UNKNOWN", 0),
            "—", "—", "—", "—", "—", "—", "—", "—",
            # 15M Bias
            "—", "—", "—", "—", "—",
            # 1M Execution
            "—", "—", "—", "—", "—",
            "Regime system not ready. Training in progress...",
            # Position state
            "—", "—", "—", "—",
            # Execution
            "—", "—", "—", "—", "—", "—",
            # Sentiment
            "—", "—", "—", "—", "—", "—",
            "—", "—", "—", "—", "—", "—",
            "Sentiment system initializing...",
            # Microstructure
            "—", "—", "—", "—", "—",
            # Risk manager
            "—", "—", "—", "—", "—", "—",
            # Paper trading
            "—", "—", "—", "—", "—", "—", "—", empty_fig, [],
            # Existing
            [empty] * 3,
            [empty] * 4,
            dbc.Alert("No V9 results found.", color="warning"),
            empty_fig,
        )

    # ── ML Signals ──────────────────────────────────────────────
    primary = signals.get(60, signals.get(20, {}))
    action = primary.get("action", "HOLD")
    conf = primary.get("confidence", 0)
    price = primary.get("price", 0)

    h20 = signals.get(20, {})
    h60 = signals.get(60, {})

    h20_preds = h20.get("preds", {})
    h60_preds = h60.get("preds", {})
    # Bug 6 fix: Add H=20/H=60 labels to prob strings
    lgb_str = html.Span([
        html.Small("H=20: ", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
        f"{h20_preds.get('lgb', 0):.3f} ",
        html.Small("| H=60: ", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
        f"{h60_preds.get('lgb', 0):.3f}",
    ])
    xgb_str = html.Span([
        html.Small("H=20: ", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
        f"{h20_preds.get('xgb', 0):.3f} ",
        html.Small("| H=60: ", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
        f"{h60_preds.get('xgb', 0):.3f}",
    ])
    rf_str = html.Span([
        html.Small("H=20: ", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
        f"{h20_preds.get('rf', 0):.3f} ",
        html.Small("| H=60: ", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
        f"{h60_preds.get('rf', 0):.3f}",
    ])

    # Data freshness banner
    last_bar_time = df.index[-1]
    status = html.Div([
        live_banner,
        html.Div(
            f"Last bar: {last_bar_time.strftime('%Y-%m-%d %H:%M')}  •  Refresh: {datetime.now().strftime('%H:%M:%S')}",
            style={'color': COLORS['text_secondary'], 'fontSize': '10px', 'textAlign': 'center', 'marginTop': '4px'}
        ),
    ])

    # ── Regime Data ─────────────────────────────────────────────
    if regime_pred:
        current_regime = regime_pred.get("current_regime", "UNKNOWN")
        regime_conf = regime_pred.get("regime_confidence", 0)
        predicted_regime = regime_pred.get("predicted_regime", "UNKNOWN")
        pred_conf = regime_pred.get("prediction_confidence", 0)
        trading_status = regime_pred.get("trading_status", "UNKNOWN")
        # Bug 2 fix: Show regime multiplier (0.25x), not risk-adjusted final_position_size (0.02x)
        regime_multiplier = regime_pred.get("regime_multiplier", 0)
        position_size = regime_multiplier
        transition_warning = regime_pred.get("regime_transition_warning", False)
        regime_reason = regime_pred.get("regime_reason", "")

        raw_action = regime_pred.get("raw_action", "HOLD")
        raw_conf = regime_pred.get("raw_confidence", 0)
        final_action = regime_pred.get("final_action", "HOLD")
        final_conf = regime_pred.get("final_confidence", 0)
        stop_loss = regime_pred.get("stop_loss")
        take_profit = regime_pred.get("take_profit")
        risk_reward = regime_pred.get("risk_reward", 1.5)
        reason = regime_pred.get("reason", "")

        regime_badge = _regime_badge(current_regime, regime_conf)
        regime_current = current_regime.replace("_", " ")
        regime_conf_str = f"{regime_conf:.1%}"
        regime_pred_str = predicted_regime.replace("_", " ")
        regime_pred_conf_str = f"{pred_conf:.1%}"
        trading_status_str = trading_status
        position_mult_str = f"{position_size:.2f}x"
        transition_warn_str = "⚠️ YES" if transition_warning else "✅ No"
        regime_reason_str = regime_reason[:40]

        raw_signal_str = f"{raw_action} @ {raw_conf:.1%}"
        adjusted_signal_str = f"{final_action} @ {final_conf:.1%}"
        stop_str = f"{stop_loss:.2f}" if stop_loss else "—"
        tp_str = f"{take_profit:.2f}" if take_profit else "—"
        # Bug 4 fix: Color R:R < 1.0 red with warning
        if risk_reward < 1.0:
            rr_str = html.Span([
                f"{risk_reward:.2f} ",
                html.Small("⚠️ Risk>Reward", style={"color": COLORS["danger"], "fontSize": "10px"}),
            ], style={"color": COLORS["danger"]})
        else:
            rr_str = f"{risk_reward:.2f}"
        reason_str = reason if reason else "No adjustment needed"
    else:
        regime_badge = _regime_badge("UNKNOWN", 0)
        regime_current = regime_conf_str = regime_pred_str = regime_pred_conf_str = "—"
        trading_status_str = position_mult_str = transition_warn_str = regime_reason_str = "—"
        raw_signal_str = adjusted_signal_str = stop_str = tp_str = rr_str = "—"
        reason_str = "Regime models not loaded yet"

    # Hybrid pipeline display values
    # 15M Bias outputs
    bias_direction = signal_15m.get('direction', 'HOLD') if signal_15m else 'HOLD'
    bias_strength = signal_15m.get('strength', 0.0) if signal_15m else 0.0
    bias_agreement = signal_15m.get('model_agreement', 0.0) if signal_15m else 0.0
    weighted_conviction = signal_15m.get('weighted_conviction', 0.0) if signal_15m else 0.0
    bias_regime = signal_15m.get('regime', 'UNKNOWN') if signal_15m else 'UNKNOWN'

    # Weighted conviction display with color coding
    if weighted_conviction > 0.50:
        conv_color = COLORS["success"]
    elif weighted_conviction > 0.30:
        conv_color = COLORS["warning"]
    else:
        conv_color = COLORS["danger"]
    bias_agreement = html.Span(f"{weighted_conviction:.1%}", style={"color": conv_color, "fontWeight": "bold"})

    # Raw probabilities tooltip
    raw_probs = signal_15m.get('raw_probs', {}) if signal_15m else {}
    raw_probs_str = ""
    if raw_probs:
        parts = []
        for name, p in raw_probs.items():
            dir_str = "BUY" if p > 0.5 else "SELL"
            parts.append(f"{name.upper()}: {p:.1%} {dir_str}")
        raw_probs_str = " | ".join(parts)
    else:
        raw_probs_str = "Model probabilities not available"

    # 1M Execution outputs
    decision_1m = "—"
    quality_1m = "—"
    stop_str = "—"
    tp_str = "—"
    rr_str = "—"
    reason_1m = "Waiting for execution signal..."

    if signal_v2 is not None:
        v2_action = signal_v2.final_decision
        # Map v2 decisions to display labels
        if v2_action == 'OPEN_LONG':
            decision_1m = html.Span("BUY", style={"color": COLORS["success"], "fontWeight": "bold"})
        elif v2_action == 'OPEN_SHORT':
            decision_1m = html.Span("SELL", style={"color": COLORS["danger"], "fontWeight": "bold"})
        elif v2_action == 'BLOCKED':
            decision_1m = html.Span("BLOCKED", style={"color": COLORS["warning"], "fontWeight": "bold"})
        else:
            decision_1m = html.Span("HOLD", style={"color": COLORS["text_secondary"]})

        quality_1m = f"{signal_v2.entry_quality_1m:.2f}"
        stop_loss = signal_v2.stop_loss if signal_v2.stop_loss > 0 else None
        take_profit = signal_v2.take_profit if signal_v2.take_profit > 0 else None
        risk_reward = signal_v2.risk_reward_ratio
        stop_str = f"{stop_loss:.2f}" if stop_loss else "—"
        tp_str = f"{take_profit:.2f}" if take_profit else "—"
        if risk_reward < 1.0:
            rr_str = html.Span([
                f"{risk_reward:.2f} ",
                html.Small("⚠️ Risk>Reward", style={"color": COLORS["danger"], "fontSize": "10px"}),
            ], style={"color": COLORS["danger"]})
        else:
            rr_str = f"{risk_reward:.2f}"
        reason_1m = signal_v2.decision_reason if signal_v2.decision_reason else "No adjustment needed"

        # Override regime display with v2's actual state
        position_mult_str = f"{signal_v2.regime_position_multiplier:.2f}x"
        if _trading_system.risk_manager and _trading_system.risk_manager.trading_halted:
            trading_status_str = "HALTED"
        elif _trading_system.risk_manager and _trading_system.risk_manager.open_trade is not None:
            trading_status_str = "TRADE OPEN"
        elif v2_action == 'BLOCKED':
            trading_status_str = "BLOCKED"
        else:
            trading_status_str = "ACTIVE"

    # Position state machine display values
    position_state_str = "IDLE"
    position_pnl_str = "—"
    position_hold_str = "—"
    last_exit_str = "—"
    if signal_v2 is not None:
        state = signal_v2.position_state
        if state == 'LONG':
            position_state_str = html.Span("LONG", style={"color": COLORS["success"], "fontWeight": "bold"})
        elif state == 'SHORT':
            position_state_str = html.Span("SHORT", style={"color": COLORS["danger"], "fontWeight": "bold"})
        else:
            position_state_str = html.Span("IDLE", style={"color": COLORS["text_secondary"]})

        if state in ['LONG', 'SHORT']:
            pnl_color = COLORS["success"] if signal_v2.unrealized_pnl >= 0 else COLORS["danger"]
            position_pnl_str = html.Span(
                f"${signal_v2.unrealized_pnl:+.2f}",
                style={"color": pnl_color, "fontWeight": "bold"}
            )
            position_hold_str = f"{signal_v2.position_hold_time_min:.1f} min"

        if signal_v2.last_exit_reason:
            exit_color = COLORS["success"] if signal_v2.last_exit_pnl >= 0 else COLORS["danger"]
            exit_explanations = {
                'SL': 'Stop loss hit — price moved against the position.',
                'TP': 'Take profit hit — target reached.',
                'MICRO_REVERSE': 'Order flow flipped against position. Early exit to avoid larger loss.',
                'TOXIC_FLOW': 'VPIN spike detected — adverse selection likely.',
                'TIMEOUT': 'Max hold time (2 hours) reached.',
                'TRAILING_STOP': 'Trailing stop triggered after price pulled back from highs.',
                '15M_FLIP': '15m bias reversed. Position closed to align with new bias.',
            }
            expl = exit_explanations.get(signal_v2.last_exit_reason, 'Position closed.')
            last_exit_str = html.Div([
                html.Span([
                    html.Span(f"{signal_v2.last_exit_reason} ", style={"color": COLORS["text"], "fontWeight": "bold"}),
                    html.Span(f"${signal_v2.last_exit_pnl:+.2f}", style={"color": exit_color, "fontWeight": "bold"}),
                ]),
                html.Div(expl, style={"color": COLORS["text_secondary"], "fontSize": "9px", "marginTop": "2px"}),
            ])

    # ── Execution Engine ────────────────────────────────────────
    exec_status_str = html.Span("MT5 Offline", style={"color": COLORS["danger"]})
    exec_symbol_str = "—"
    exec_spread_str = "—"
    exec_margin_str = "—"
    exec_prepared_str = "No active signal - waiting for setup..."
    broker_positions_str = "No open positions"

    try:
        if MT5_AVAILABLE and mt5.terminal_info() and mt5.terminal_info().connected:
            exec_status_str = html.Span("✅ Connected", style={"color": COLORS["success"]})
            symbol = _execution_engine.get_symbol()
            exec_symbol_str = symbol if symbol else "—"

            if symbol:
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    spread = (tick.ask - tick.bid) / 0.01
                    exec_spread_str = html.Span(
                        f"{spread:.1f} pips",
                        style={"color": COLORS["success"] if spread <= 5.0 else COLORS["danger"]}
                    )

            acc = _execution_engine._get_account_info()
            exec_margin_str = f"${acc['free_margin']:,.2f} ({acc['free_margin']/acc['balance']*100:.0f}%)" if acc['balance'] > 0 else "—"

            # Prepare order if signal fired
            if signal_v2 is not None and signal_v2.final_decision in ['OPEN_LONG', 'OPEN_SHORT']:
                direction = 'BUY' if signal_v2.final_decision == 'OPEN_LONG' else 'SELL'
                prepared = _execution_engine.prepare_order(
                    direction=direction,
                    volume=signal_v2.position_size_lots,
                    sl=signal_v2.stop_loss,
                    tp=signal_v2.take_profit,
                    comment="AchillesHybrid"
                )
                if prepared.validated:
                    exec_prepared_str = html.Div([
                        html.Div(f"READY: {prepared.direction} {prepared.volume} lots @ {prepared.symbol}",
                                 style={"color": COLORS["success"], "fontWeight": "bold", "fontSize": "14px"}),
                        html.Div(f"SL: {prepared.sl:.2f}  |  TP: {prepared.tp:.2f}",
                                 style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                        html.Div(f"Checks: spread={prepared.pre_trade_checks.get('spread_pips', 0):.1f} pips, "
                                 f"margin={prepared.pre_trade_checks.get('free_margin_pct', 0):.1%}",
                                 style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                    ])
                else:
                    exec_prepared_str = html.Div([
                        html.Div(f"BLOCKED: {prepared.validation_reason}",
                                 style={"color": COLORS["danger"], "fontWeight": "bold", "fontSize": "13px"}),
                    ])

            # Broker positions
            positions = _execution_engine.get_open_positions()
            if positions:
                pos_rows = []
                for pos in positions:
                    pnl_color = COLORS["success"] if pos.current_profit >= 0 else COLORS["danger"]
                    pos_rows.append(html.Tr([
                        html.Td(f"#{pos.ticket}", style={"fontSize": "11px"}),
                        html.Td(f"{pos.direction}", style={"fontSize": "11px"}),
                        html.Td(f"{pos.volume}", style={"fontSize": "11px"}),
                        html.Td(f"{pos.open_price:.2f}", style={"fontSize": "11px"}),
                        html.Td(f"${pos.current_profit:+.2f}", style={"fontSize": "11px", "color": pnl_color, "fontWeight": "bold"}),
                        html.Td(f"SL:{pos.sl:.2f} TP:{pos.tp:.2f}", style={"fontSize": "10px"}),
                    ]))
                broker_positions_str = dbc.Table(
                    [html.Thead(html.Tr([
                        html.Th("Ticket", style={"fontSize": "10px"}),
                        html.Th("Dir", style={"fontSize": "10px"}),
                        html.Th("Size", style={"fontSize": "10px"}),
                        html.Th("Price", style={"fontSize": "10px"}),
                        html.Th("PnL", style={"fontSize": "10px"}),
                        html.Th("SL/TP", style={"fontSize": "10px"}),
                    ])), html.Tbody(pos_rows)],
                    size="sm", bordered=False, striped=True, hover=True,
                    style={"color": COLORS["text"], "marginBottom": "0"}
                )
            else:
                broker_positions_str = html.Div("No open positions", style={"color": COLORS["text_secondary"], "textAlign": "center"})
        else:
            exec_status_str = html.Span("MT5 Offline", style={"color": COLORS["danger"]})
            exec_prepared_str = html.Div("Connect to MetaTrader 5 to enable execution.", style={"color": COLORS["text_secondary"], "fontSize": "12px"})
    except Exception as e:
        print(f"[BetaDashboard] Execution engine error: {e}")

    # ── Sentiment Analysis ──────────────────────────────────────
    fetcher = ExpandedNewsFetcher()
    headlines_df = fetcher.fetch_all(use_demo=True)

    booster = HardenedPreTradeBooster()
    analyzer = HardenedSentimentAnalyzer()

    # Process headlines through fake news guard
    analyzed = booster.analyze_headlines_only(headlines_df)

    # Build trade signal for sentiment boost
    trade_signal = {
        'action': final_action,
        'confidence': final_conf,
        'price': price,
    }

    # Apply sentiment boost
    boosted = booster.apply(trade_signal, analyzed, df)

    # Extract sentiment results for display
    sentiment_score = boosted.get('sentiment_raw_score', 0)
    sentiment_valid = boosted.get('sentiment_valid', False)
    sentiment_reason = boosted.get('sentiment_reason', 'No analysis')
    sentiment_sources = boosted.get('sentiment_sources', 0)
    sentiment_boost = boosted.get('sentiment_boost', 1.0)
    sentiment_blocked = boosted.get('sentiment_blocked', False)
    sentiment_adjusted = boosted.get('sentiment_adjusted', False)
    trigger_type = boosted.get('trigger_type', '')
    l1_headlines = boosted.get('l1_headlines', 0)
    l2_real_headlines = boosted.get('l2_real_headlines', 0)

    # Update final signal with sentiment
    if sentiment_valid and sentiment_adjusted:
        final_action = boosted['action']
        final_conf = boosted['confidence']
        adjusted_signal_str = f"{final_action} @ {final_conf:.1%}"

    # Layer status — show actual pipeline counts
    l1_status = f"L1: {l1_headlines} fetched"
    l2_status = f"L2: {l2_real_headlines} real"
    l3_status = f"L3: {sentiment_sources} sources"
    l4_status = "PASS" if sentiment_valid else "FAIL"
    if sentiment_valid:
        l5_status = "PASS"
    elif trigger_type == 'extreme_score':
        l5_status = "EXTREME"
    elif trigger_type == 'cluster':
        l5_status = "CLUSTER"
    else:
        l5_status = "FAIL"
    cooldown_status = "ACTIVE" if "cooldown" in sentiment_reason.lower() else "OFF"

    sentiment_dir = "POS" if sentiment_score > 0.1 else "NEG" if sentiment_score < -0.1 else "NEUT"
    sentiment_conf_str = f"{abs(sentiment_score):.2f}"
    sentiment_valid_str = "YES" if sentiment_valid else "NO"
    sentiment_boost_str = f"{sentiment_boost:.2f}x"
    if sentiment_blocked:
        sentiment_boost_str = "BLOCKED"
    
    cluster_count = boosted.get('cluster_count', 0)

    # Format sentiment reason based on trigger type (Bug 1 fix)
    if trigger_type == 'extreme_score':
        sentiment_reason_display = f"Extreme sentiment score ({abs(sentiment_score):.2f}) — possible manipulation"
    elif trigger_type == 'cluster':
        sentiment_reason_display = f"Headline cluster: {cluster_count} headlines in 2 min — possible coordinated campaign"
    else:
        sentiment_reason_display = sentiment_reason

    # ── Microstructure + Risk Manager ───────────────────────────
    micro_ofi_str = "—"
    micro_vpin_str = "—"
    micro_entropy_str = "—"
    micro_hft_str = "—"
    micro_quality_str = "—"
    risk_balance_str = "—"
    risk_daily_pnl_str = "—"
    risk_winrate_str = "—"
    risk_maxdd_str = "—"
    risk_consec_str = "—"
    risk_halted_str = "—"

    if df is not None and _trading_system is not None:
        try:
            micro = _trading_system._analyze_microstructure(df)
            micro_ofi_str = f"{micro['ofi_proxy']:.2f}"
            micro_vpin_str = html.Span([
                f"{micro['vpin_proxy']:.2f} ",
                html.Small("✅" if micro['vpin_proxy'] < 0.6 else "❌", style={"fontSize": "10px"}),
            ])
            micro_entropy_str = html.Span([
                f"{micro['sign_entropy']:.2f} ",
                html.Small("✅" if micro['sign_entropy'] < 0.7 else "❌", style={"fontSize": "10px"}),
            ])
            micro_hft_str = f"{micro['hft_activity']:.2f}"
            micro_quality_str = f"{micro['quality_score']:.2f}"

            risk_data = _trading_system.get_dashboard_data()
            risk_balance_str = f"${risk_data['account_balance']:,.2f}"
            risk_daily_pnl_str = f"${risk_data['daily_pnl']:,.2f}"
            risk_winrate_str = f"{risk_data['win_rate']:.1%}"
            risk_maxdd_str = f"{risk_data['max_drawdown_pct']:.2%}"
            risk_consec_str = f"{risk_data['consecutive_losses']}/3"
            if risk_data['trading_halted']:
                risk_halted_str = html.Span([
                    "YES ",
                    html.Small(risk_data['halt_reason'], style={"color": COLORS["danger"], "fontSize": "9px"}),
                ], style={"color": COLORS["danger"]})
            else:
                risk_halted_str = html.Span("NO", style={"color": COLORS["success"]})
        except Exception as e:
            print(f"[BetaDashboard] Microstructure/Risk error: {e}")

    # ── Paper Trading Analytics ───────────────────────────────
    paper_trades_str = "—"
    paper_winrate_str = "—"
    paper_pf_str = "—"
    paper_avgwin_str = "—"
    paper_avgloss_str = "—"
    paper_sharpe_str = "—"
    paper_summary_str = "No trades yet."
    equity_fig = go.Figure().update_layout(
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"], margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
    )
    recent_trades_table = html.Div("No trades recorded yet.", style={"color": COLORS["text_secondary"], "textAlign": "center"})

    try:
        perf = get_performance_summary(initial_balance=10000.0)
        if perf["total_trades"] > 0:
            paper_trades_str = str(perf["total_trades"])
            paper_winrate_str = f"{perf['win_rate']:.1%}"
            paper_pf_str = f"{perf['profit_factor']:.2f}" if perf['profit_factor'] != float('inf') else "∞"
            paper_avgwin_str = f"${perf['avg_win']:.2f}"
            paper_avgloss_str = f"${perf['avg_loss']:.2f}"
            paper_sharpe_str = f"{perf['sharpe_annual']:.2f}"
            paper_summary_str = (
                f"Total P&L: ${perf['total_pnl']:+.2f}  |  "
                f"Best: ${perf['best_trade']:.2f}  |  "
                f"Worst: ${perf['worst_trade']:.2f}  |  "
                f"Max Win Streak: {perf['max_consecutive_wins']}  |  "
                f"Max Loss Streak: {perf['max_consecutive_losses']}"
            )

            # Equity curve
            eq_df = compute_equity_curve(initial_balance=10000.0)
            if not eq_df.empty:
                equity_fig = go.Figure()
                equity_fig.add_trace(go.Scatter(
                    x=eq_df["time"], y=eq_df["equity"],
                    mode="lines+markers", name="Equity",
                    line=dict(color=COLORS["accent"], width=2),
                    marker=dict(size=4),
                ))
                equity_fig.add_hline(y=10000, line_dash="dash", line_color="gray", opacity=0.5)
                equity_fig.update_layout(
                    paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["surface"],
                    font_color=COLORS["text"], margin=dict(l=40, r=20, t=30, b=40),
                    xaxis=dict(gridcolor=COLORS.get("grid", "#333")),
                    yaxis=dict(gridcolor=COLORS.get("grid", "#333"), tickprefix="$"),
                    showlegend=False,
                )

            # Recent trades table
            recent = get_recent_trades_table(limit=10)
            if not recent.empty:
                table_header = html.Thead(html.Tr([
                    html.Th("Dir", style={"fontSize": "10px"}),
                    html.Th("Entry", style={"fontSize": "10px"}),
                    html.Th("Exit", style={"fontSize": "10px"}),
                    html.Th("Price", style={"fontSize": "10px"}),
                    html.Th("Size", style={"fontSize": "10px"}),
                    html.Th("PnL", style={"fontSize": "10px"}),
                    html.Th("Hold", style={"fontSize": "10px"}),
                    html.Th("Reason", style={"fontSize": "10px"}),
                ]))
                rows = []
                for _, row in recent.iterrows():
                    pnl_color = COLORS["success"] if "+" in str(row["pnl_fmt"]) else COLORS["danger"]
                    rows.append(html.Tr([
                        html.Td(row["direction_icon"], style={"fontSize": "11px"}),
                        html.Td(row["entry_time"], style={"fontSize": "11px"}),
                        html.Td(row["exit_time"], style={"fontSize": "11px"}),
                        html.Td(f"{row['entry_price']:.2f} → {row['exit_price']:.2f}", style={"fontSize": "11px"}),
                        html.Td(f"{row['size_lots']}", style={"fontSize": "11px"}),
                        html.Td(row["pnl_fmt"], style={"fontSize": "11px", "color": pnl_color, "fontWeight": "bold"}),
                        html.Td(row["hold_fmt"], style={"fontSize": "11px"}),
                        html.Td(row["exit_reason"], style={"fontSize": "10px"}),
                    ]))
                recent_trades_table = dbc.Table(
                    [table_header, html.Tbody(rows)],
                    size="sm", bordered=False, striped=True, hover=True,
                    style={"color": COLORS["text"], "marginBottom": "0"}
                )
    except Exception as e:
        print(f"[BetaDashboard] Paper trading analytics error: {e}")

    # ── Per-Model Cards ─────────────────────────────────────────
    model_cards = []
    for h in HORIZONS:
        if h not in signals:
            continue
        sig = signals[h]
        for name, prob in sig.get("preds", {}).items():
            label = MODEL_LABELS.get(name, name)
            paction = "BUY" if prob > 0.55 else "SELL" if prob < 0.45 else "HOLD"
            pcol = COLORS["success"] if paction == "BUY" else COLORS["danger"] if paction == "SELL" else COLORS["warning"]
            model_cards.append(
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H6(f"{label} H={h}", style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
                            html.H4(paction, style={"color": pcol, "fontWeight": "bold"}),
                            html.Div(f"Prob: {prob:.3f}", style={"color": COLORS["text"], "fontSize": "12px"}),
                            html.Div(f"Conf: {abs(prob-0.5)*2:.1%}", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
                        ])
                    ], style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}"}),
                    width=2,
                )
            )

    # ── Metrics ─────────────────────────────────────────────────
    metrics = []
    for h in HORIZONS:
        if h not in _model_cache.results:
            continue
        res = _model_cache.results[h]
        best_acc = max((res.get(m, {}).get("accuracy", 0) for m in MODEL_NAMES), default=0)
        best_auc = max((res.get(m, {}).get("auc", 0) for m in MODEL_NAMES), default=0)
        best_conf = max((res.get(m, {}).get("conf_acc", 0) for m in MODEL_NAMES), default=0)
        # Bug 5 fix: Use _stat_static for metrics (values are computed, not callback-updated)
        metrics.extend([
            dbc.Col(_stat_static(f"H={h} Best Acc", f"{best_acc:.2%}", COLORS["success"]), width=3),
            dbc.Col(_stat_static(f"H={h} Best AUC", f"{best_auc:.4f}", COLORS["info"]), width=3),
            dbc.Col(_stat_static(f"H={h} Best Conf@0.6", f"{best_conf:.2%}", COLORS["accent"]), width=3),
            dbc.Col(_stat_static(f"H={h} Ensemble", f"{res.get('ensemble', {}).get('accuracy', 0):.2%}", COLORS["info"]), width=3),
        ])

    # ── V9 Results ──────────────────────────────────────────────
    df_v9 = pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists() else pd.DataFrame()
    if not df_v9.empty:
        table_rows = []
        for _, row in df_v9.sort_values("conf_acc", ascending=False).head(16).iterrows():
            color = "success" if row["conf_acc"] >= 0.60 else "info" if row["conf_acc"] >= 0.55 else "secondary"
            table_rows.append(
                html.Tr([
                    html.Td(row["model"], style={"color": COLORS["text"]}),
                    html.Td(f"H={int(row['horizon'])}", style={"color": COLORS["text"]}),
                    html.Td(f"{row['conf_thresh']:.2f}", style={"color": COLORS["text"]}),
                    html.Td(dbc.Badge(f"{row['conf_acc']:.2%}", color=color), style={"color": COLORS["text"]}),
                    html.Td(f"{row['conf_pct']:.2%}", style={"color": COLORS["text"]}),
                    html.Td(f"{row['overall_acc']:.2%}", style={"color": COLORS["text_secondary"]}),
                    html.Td(f"{row['overall_auc']:.4f}", style={"color": COLORS["text_secondary"]}),
                ])
            )
        conf_table = dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("Model", style={"color": COLORS["accent"]}),
                        html.Th("Horizon", style={"color": COLORS["accent"]}),
                        html.Th("Threshold", style={"color": COLORS["accent"]}),
                        html.Th("Conf Acc", style={"color": COLORS["accent"]}),
                        html.Th("Coverage", style={"color": COLORS["accent"]}),
                        html.Th("Overall Acc", style={"color": COLORS["accent"]}),
                        html.Th("AUC", style={"color": COLORS["accent"]}),
                    ])
                ),
                html.Tbody(table_rows),
            ],
            bordered=True, hover=True, size="sm",
            style={"color": COLORS["text"], "backgroundColor": COLORS["surface"]},
        )

        fig = go.Figure()
        for (horizon, model), group in df_v9.groupby(["horizon", "model"]):
            fig.add_trace(go.Scatter(
                x=group["conf_thresh"],
                y=group["conf_acc"],
                mode="lines+markers",
                name=f"{model} H={horizon}",
                line=dict(width=2),
            ))
        fig.update_layout(
            title="Confident Accuracy vs Threshold (V9 Full Dataset)",
            xaxis_title="Confidence Threshold",
            yaxis_title="Accuracy",
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["surface"],
            font_color=COLORS["text"],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(tickformat=".0%", gridcolor=COLORS.get("grid", "#333")),
            xaxis=dict(gridcolor=COLORS.get("grid", "#333")),
            margin=dict(l=40, r=20, t=60, b=40),
        )
    else:
        conf_table = dbc.Alert("V9 results not found.", color="warning")
        fig = go.Figure().update_layout(
            paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
            font_color=COLORS["text"],
        )

    return (
        # ML signals
        _signal_badge(action, conf),
        f"{price:,.2f}",
        h20.get("action", "—"),
        f"{h20.get('confidence', 0):.1%}",
        h60.get("action", "—"),
        f"{h60.get('confidence', 0):.1%}",
        lgb_str,
        xgb_str,
        rf_str,
        datetime.now().strftime("%H:%M:%S"),
        status,
        # Regime
        regime_badge,
        regime_current,
        regime_conf_str,
        regime_pred_str,
        regime_pred_conf_str,
        trading_status_str,
        position_mult_str,
        transition_warn_str,
        regime_reason_str,
        # 15M Primary Bias
        bias_direction,
        f"{bias_strength:.1%}",
        bias_agreement,
        bias_regime.replace("_", " ") if bias_regime != "UNKNOWN" else "—",
        raw_probs_str,
        # 1M Execution Signal
        decision_1m,
        quality_1m,
        stop_str,
        tp_str,
        rr_str,
        reason_1m,
        # Position State
        position_state_str,
        position_pnl_str,
        position_hold_str,
        last_exit_str,
        # Execution
        exec_status_str,
        exec_symbol_str,
        exec_spread_str,
        exec_margin_str,
        exec_prepared_str,
        broker_positions_str,
        # Sentiment
        f"{sentiment_score:+.2f}",
        sentiment_dir,
        sentiment_conf_str,
        str(sentiment_sources),
        sentiment_valid_str,
        sentiment_boost_str,
        l1_status,
        l2_status,
        l3_status,
        l4_status,
        l5_status,
        cooldown_status,
        sentiment_reason_display,
        # Microstructure
        micro_ofi_str,
        micro_vpin_str,
        micro_entropy_str,
        micro_hft_str,
        micro_quality_str,
        # Risk manager
        risk_balance_str,
        risk_daily_pnl_str,
        risk_winrate_str,
        risk_maxdd_str,
        risk_consec_str,
        risk_halted_str,
        # Paper trading
        paper_trades_str,
        paper_winrate_str,
        paper_pf_str,
        paper_avgwin_str,
        paper_avgloss_str,
        paper_sharpe_str,
        paper_summary_str,
        equity_fig,
        recent_trades_table,
        # Existing
        model_cards,
        metrics,
        conf_table,
        fig,
    )
