"""
Precision Strategy Page — 1-minute multi-asset trading system
==============================================================
Dedicated tab for the 4-layer precision system (Microstructure cleaning →
Flow toxicity → ML signal → Risk management).

Supports XAU/USD, BTC/USD, EUR/USD, GBP/USD with per-asset configs.
Live BUY/SELL signal panel + Train + Backtest buttons.
"""

from __future__ import annotations

import os
import threading
import time as _time
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State, no_update
from datetime import datetime

# ── Lazy imports of the trading system to keep page-load light ───────────────
from services.precision_trading_system import (
    PrecisionTradingSystem,
    Asset,
    ASSET_CONFIGS,
    XGBOOST_AVAILABLE,
    PYKALMAN_AVAILABLE,
    HMMLEARN_AVAILABLE,
)

# ── Colour palette (matches main app) ────────────────────────────────────────
C = {
    "bg":       "#000000",
    "surface":  "#0a0a0a",
    "surf_lt":  "#121212",
    "border":   "#222222",
    "accent":   "#00ff88",
    "danger":   "#ff4757",
    "warn":     "#ffa502",
    "info":     "#00d4ff",
    "text":     "#ffffff",
    "muted":    "#888888",
    "purple":   "#a855f7",
}

# ── Per-symbol singleton state ────────────────────────────────────────────────
# Each system instance is keyed by (asset, model_type). Loaded lazily on
# first use; trained models persisted to data/precision_<ASSET>_<MODEL>.pkl.
_systems: Dict[str, PrecisionTradingSystem] = {}
_systems_lock = threading.Lock()
_training_state: Dict[str, Dict[str, Any]] = {}   # per-key progress dict

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)


def _save_path(asset: str, model: str) -> str:
    return os.path.join(DATA_DIR, f"precision_{asset}_{model}.pkl")


def _get_system(asset: str, model: str) -> PrecisionTradingSystem:
    """Return (or lazy-create) a PrecisionTradingSystem for this asset+model."""
    key = f"{asset}_{model}"
    with _systems_lock:
        if key not in _systems:
            sys = PrecisionTradingSystem(
                asset=Asset(asset),
                model_type=model,           # 'xgboost' or 'lorentzian'
                use_hmm=HMMLEARN_AVAILABLE,
            )
            # Restore from disk if available
            sp = _save_path(asset, model)
            if os.path.exists(sp):
                try:
                    sys.load(sp)
                    print(f"[Precision] Loaded {asset}/{model} from disk")
                except Exception as e:
                    print(f"[Precision] Load failed: {e}")
            _systems[key] = sys
        return _systems[key]


# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers (mirrors smma_strategy aesthetic)
# ─────────────────────────────────────────────────────────────────────────────

def _card(title, body, right=None):
    hdr = [html.Span(title, style={"fontWeight": "bold", "color": C["text"],
                                    "fontSize": "11px", "letterSpacing": "1px"})]
    if right is not None:
        hdr.append(right)
    return dbc.Card([
        dbc.CardHeader(hdr, style={"backgroundColor": C["surface"],
                                    "borderBottom": f"1px solid {C['border']}",
                                    "padding": "9px 14px",
                                    "display": "flex",
                                    "justifyContent": "space-between",
                                    "alignItems": "center"}),
        dbc.CardBody(body, style={"backgroundColor": C["surf_lt"],
                                   "padding": "12px"}),
    ], style={"backgroundColor": C["surface"],
              "border": f"1px solid {C['border']}",
              "borderRadius": "6px", "marginBottom": "12px"})


def _stat(label, value_id, color=None, sub=""):
    return html.Div([
        html.Div(label, style={"color": C["muted"], "fontSize": "9px",
                                "letterSpacing": "0.8px",
                                "textTransform": "uppercase"}),
        html.Div("--", id=value_id,
                 style={"color": color or C["text"], "fontSize": "20px",
                        "fontWeight": "bold", "lineHeight": "1.2"}),
        html.Div(sub, style={"color": C["muted"], "fontSize": "9px",
                              "marginTop": "2px"}),
    ], style={"padding": "8px 12px",
              "borderRight": f"1px solid {C['border']}",
              "minWidth": "100px"})


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

DEPS_NOTE = []
if not XGBOOST_AVAILABLE:
    DEPS_NOTE.append("xgboost missing")
if not PYKALMAN_AVAILABLE:
    DEPS_NOTE.append("pykalman missing")
if not HMMLEARN_AVAILABLE:
    DEPS_NOTE.append("hmmlearn missing")
DEPS_BANNER = (
    f"⚠️  Optional deps unavailable: {', '.join(DEPS_NOTE)} — system will "
    f"use fallbacks. Run `pip install -r requirements.txt` to enable all features."
    if DEPS_NOTE else None
)


layout = dbc.Container(fluid=True, style={"backgroundColor": C["bg"],
                                            "minHeight": "100vh",
                                            "padding": "14px"}, children=[

    # Optional warning banner
    dbc.Alert(DEPS_BANNER, color="warning", style={"fontSize": "11px"},
                dismissable=True) if DEPS_BANNER else html.Div(),

    # ── Top bar ───────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.H5([
                html.Span("🎯", style={"marginRight": "6px",
                                        "color": C["accent"]}),
                "Precision Strategy",
                html.Span("  ·  Microstructure → VPIN → ML → Risk",
                            style={"fontSize": "12px", "color": C["muted"],
                                    "marginLeft": "8px"}),
            ], style={"color": C["text"], "margin": 0}),
            html.P(
                "1-min multi-asset framework: Smart Price + Kalman + VPIN toxicity gate "
                "+ XGBoost/Lorentzian + HMM regime + ATR-based 3-tier TP",
                style={"color": C["muted"], "fontSize": "10px",
                        "margin": "3px 0 0 0"},
            ),
        ], width=6),

        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.Label("Symbol", style={"color": C["muted"],
                                "fontSize": "10px", "display": "block",
                                "marginBottom": "3px"}),
                    dcc.Dropdown(
                        id="prec-symbol",
                        options=[{"label": s, "value": s} for s in
                                  ("XAUUSD", "BTCUSD", "EURUSD", "GBPUSD")],
                        value="XAUUSD", clearable=False,
                        style={"width": "110px"},
                    ),
                ], width="auto"),
                dbc.Col([
                    html.Label("Model", style={"color": C["muted"],
                                "fontSize": "10px", "display": "block",
                                "marginBottom": "3px"}),
                    dcc.Dropdown(
                        id="prec-model",
                        options=[
                            {"label": "XGBoost",     "value": "xgboost"},
                            {"label": "Lorentzian",  "value": "lorentzian"},
                        ],
                        value="xgboost", clearable=False,
                        style={"width": "120px"},
                    ),
                ], width="auto"),
                dbc.Col([
                    html.Label(" ", style={"display": "block",
                                "fontSize": "10px", "marginBottom": "3px"}),
                    dbc.Button("🧠 Train", id="prec-train-btn",
                                color="warning", size="sm",
                                style={"fontSize": "11px",
                                        "marginRight": "5px"}),
                ], width="auto"),
                dbc.Col([
                    html.Label(" ", style={"display": "block",
                                "fontSize": "10px", "marginBottom": "3px"}),
                    dbc.Button("📊 Backtest", id="prec-backtest-btn",
                                color="info", size="sm",
                                style={"fontSize": "11px",
                                        "marginRight": "5px"}),
                ], width="auto"),
                dbc.Col([
                    html.Label(" ", style={"display": "block",
                                "fontSize": "10px", "marginBottom": "3px"}),
                    dbc.Button("🔁 Walk-Forward", id="prec-wfbacktest-btn",
                                color="success", size="sm",
                                style={"fontSize": "11px"}),
                ], width="auto"),
            ], align="end"),
        ], width=6, style={"textAlign": "right"}),
    ], className="mb-3", align="center"),

    # ── Live signal banner ─────────────────────────────────────────────────
    _card(
        "🟢 LIVE SIGNAL  ·  1-min precision",
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div("SIGNAL", style={"color": C["muted"],
                                "fontSize": "9px", "letterSpacing": "0.8px",
                                "textTransform": "uppercase"}),
                    html.Div(id="prec-signal-badge", children="---",
                                style={"fontSize": "32px", "fontWeight": "bold",
                                        "color": C["muted"], "lineHeight": "1"}),
                    html.Div(id="prec-signal-confidence",
                                children="Confidence: --",
                                style={"color": C["muted"], "fontSize": "10px",
                                        "marginTop": "4px"}),
                ], width=3),
                dbc.Col([
                    dbc.Row([
                        dbc.Col(_stat("ENTRY",       "prec-entry"),  width=3),
                        dbc.Col(_stat("STOP LOSS",   "prec-sl",     C["danger"]), width=3),
                        dbc.Col(_stat("TP1",         "prec-tp1",    C["accent"]), width=2),
                        dbc.Col(_stat("TP2",         "prec-tp2",    C["accent"]), width=2),
                        dbc.Col(_stat("TP3",         "prec-tp3",    C["accent"]), width=2),
                    ]),
                    dbc.Row([
                        dbc.Col(_stat("LOT SIZE",  "prec-lot",       C["info"]), width=3),
                        dbc.Col(_stat("VPIN",      "prec-vpin",      C["warn"]), width=3),
                        dbc.Col(_stat("REGIME",    "prec-regime",    C["purple"]), width=3),
                        dbc.Col(_stat("ATR",       "prec-atr",       C["text"]), width=3),
                    ], className="mt-1"),
                    dbc.Row([
                        dbc.Col(_stat("MTF SCORE",   "prec-mtf-score",   C["info"]),   width=3),
                        dbc.Col(_stat("CONFLUENCE",  "prec-mtf-conf",    C["accent"]), width=3),
                        dbc.Col(_stat("CVD DIV",     "prec-cvd-div",     C["purple"]), width=2),
                        dbc.Col(_stat("VAL AREA",    "prec-in-va",       C["text"]),   width=2),
                        dbc.Col(_stat("SIZING",      "prec-sizing",      C["warn"]),   width=2),
                    ], className="mt-1"),
                ], width=9),
            ], align="center"),
            html.Div(id="prec-signal-status",
                        children="Click 'Train' to begin.",
                        style={"color": C["muted"], "fontSize": "10px",
                                "marginTop": "8px",
                                "textAlign": "center"}),
            dbc.Progress(id="prec-progress", value=0, max=100,
                            color="warning", striped=True, animated=True,
                            style={"height": "3px", "marginTop": "6px"}),
        ]),
    ),

    # ── Backtest results ───────────────────────────────────────────────────
    _card(
        "📊 BACKTEST RESULTS  ·  out-of-sample",
        dcc.Loading(
            id="prec-bt-loading", type="circle", color=C["info"],
            children=html.Div([
                dbc.Row([
                    dbc.Col(_stat("TOTAL TRADES",   "prec-bt-trades",      C["text"]),    width=2),
                    dbc.Col(_stat("WIN RATE",        "prec-bt-winrate",     C["accent"]),  width=2),
                    dbc.Col(_stat("PROFIT FACTOR",   "prec-bt-pf",          C["info"]),    width=2),
                    dbc.Col(_stat("SHARPE",          "prec-bt-sharpe",      C["info"]),    width=2),
                    dbc.Col(_stat("MAX DD",          "prec-bt-mdd",         C["warn"]),    width=2),
                    dbc.Col(_stat("TOTAL RETURN",    "prec-bt-return",      C["accent"]),  width=2),
                ]),
                html.Div(id="prec-bt-status",
                            children="No backtest run yet. Click 'Backtest' after training.",
                            style={"color": C["muted"], "fontSize": "10px",
                                    "marginTop": "8px",
                                    "textAlign": "center"}),
                dcc.Graph(id="prec-equity-chart",
                            figure=go.Figure(layout=dict(
                                paper_bgcolor=C["bg"], plot_bgcolor=C["surface"],
                                font=dict(color=C["text"], size=9),
                                height=240,
                                margin=dict(l=8, r=8, t=18, b=8),
                                annotations=[dict(text="Equity curve will appear after backtest",
                                                    x=0.5, y=0.5, showarrow=False,
                                                    font=dict(color=C["muted"], size=11))],
                                xaxis=dict(visible=False), yaxis=dict(visible=False),
                            )),
                            config={"displayModeBar": False},
                            style={"height": "240px"}),
            ])
        ),
    ),

    # ── Walk-Forward results ───────────────────────────────────────────────
    _card(
        "🔁 WALK-FORWARD  ·  rolling out-of-sample · intra-bar fills",
        dcc.Loading(
            id="prec-wf-loading", type="circle", color=C["accent"],
            children=html.Div([
                dbc.Row([
                    dbc.Col(_stat("FOLDS",          "prec-wf-folds",      C["text"]),    width=2),
                    dbc.Col(_stat("TOTAL TRADES",   "prec-wf-trades",     C["text"]),    width=2),
                    dbc.Col(_stat("WIN RATE",       "prec-wf-winrate",    C["accent"]),  width=2),
                    dbc.Col(_stat("PRECISION L",    "prec-wf-prec-long",  C["accent"]),  width=2),
                    dbc.Col(_stat("PRECISION S",    "prec-wf-prec-short", C["danger"]),  width=2),
                    dbc.Col(_stat("SHARPE",         "prec-wf-sharpe",     C["info"]),    width=2),
                ]),
                dbc.Row([
                    dbc.Col(_stat("PROFIT FACTOR",  "prec-wf-pf",         C["info"]),    width=2),
                    dbc.Col(_stat("MAX DD",         "prec-wf-mdd",        C["warn"]),    width=2),
                    dbc.Col(_stat("ECE",            "prec-wf-ece",        C["purple"],
                                    sub="lower=better, <0.05=calibrated"),               width=3),
                    dbc.Col(_stat("p-VALUE",        "prec-wf-pvalue",     C["purple"],
                                    sub="<0.05 = stat. significant"),                    width=3),
                    dbc.Col(_stat("TOTAL RETURN",   "prec-wf-return",     C["accent"]),  width=2),
                ], className="mt-1"),
                html.Div(id="prec-wf-status",
                            children="No walk-forward run yet. Click 'Walk-Forward' for rigorous OOS metrics (~30-60s).",
                            style={"color": C["muted"], "fontSize": "10px",
                                    "marginTop": "8px",
                                    "textAlign": "center"}),
                dcc.Graph(id="prec-wf-equity-chart",
                            figure=go.Figure(layout=dict(
                                paper_bgcolor=C["bg"], plot_bgcolor=C["surface"],
                                font=dict(color=C["text"], size=9),
                                height=240,
                                margin=dict(l=8, r=8, t=18, b=8),
                                annotations=[dict(text="Walk-forward equity curve will appear here",
                                                    x=0.5, y=0.5, showarrow=False,
                                                    font=dict(color=C["muted"], size=11))],
                                xaxis=dict(visible=False), yaxis=dict(visible=False),
                            )),
                            config={"displayModeBar": False},
                            style={"height": "240px"}),
                html.Div(id="prec-wf-fold-table",
                            style={"marginTop": "8px",
                                    "fontSize": "10px",
                                    "color": C["muted"]}),
            ])
        ),
    ),

    # ── Architecture description (collapsible reference) ───────────────────
    dbc.Accordion([
        dbc.AccordionItem(
            html.Div([
                html.P([
                    html.Strong("Layer 1 — Microstructure cleaning:  "),
                    "Smart Price (volume-weighted mid), Kalman filter, "
                    "Spread filter (block when spread > 30% ATR)."
                ], style={"fontSize": "11px", "color": C["text"]}),
                html.P([
                    html.Strong("Layer 2 — Market structure (v3):  "),
                    "Lightweight FVG (Fair-Value Gap) detection; nearer FVG edge "
                    "is used as a tighter stop when present and aligned with the "
                    "trade direction."
                ], style={"fontSize": "11px", "color": C["text"]}),
                html.P([
                    html.Strong("Layer 3 — Flow & toxicity (v3):  "),
                    "VPIN (1-50-50 standard) + CVD with bullish/bearish divergence "
                    "+ Volume Profile (POC, Value Area, distance-to-POC) + "
                    "absorption detector + signed volume + realized volatility."
                ], style={"fontSize": "11px", "color": C["text"]}),
                html.P([
                    html.Strong("Layer 4 — MTF confluence (v3):  "),
                    "1m / 5m / 15m EMA trend alignment, weighted 0.5/0.3/0.2 "
                    "into a confluence score in [-1, 1]. Strongly opposing MTF "
                    "(±0.6) gates trades."
                ], style={"fontSize": "11px", "color": C["text"]}),
                html.P([
                    html.Strong("Layer 5 — ML signal (v3):  "),
                    "XGBoost on 35+ microstructure features OR Lorentzian KNN. "
                    "Probabilities calibrated via Isotonic Regression with "
                    "TimeSeriesSplit (out-of-sample) — fixes XGBoost overconfidence. "
                    "HMM regime adapts strategy in trending/MR regimes."
                ], style={"fontSize": "11px", "color": C["text"]}),
                html.P([
                    html.Strong("Layer 6 — Risk + execution (v3):  "),
                    "ATR-based stops (refined by FVG when tighter) + 3-tier TP. "
                    "Quarter-Kelly position sizing kicks in after ≥20 closed "
                    "trades; until then, fixed 1% risk. London/NY session filter."
                ], style={"fontSize": "11px", "color": C["text"]}),
                html.P([
                    html.Strong("Walk-forward backtest (v3):  "),
                    "5-fold anchored walk-forward with intra-bar OHLC-ordered "
                    "fill simulation (worst-case path per direction). Reports "
                    "ECE (calibration error), p-value (one-sample t-test of "
                    "trade PnL), per-direction precision."
                ], style={"fontSize": "11px", "color": C["text"]}),
            ]),
            title="ℹ️  Architecture (6 layers · v3)", item_id="arch",
        )
    ], start_collapsed=True, flush=True,
        style={"backgroundColor": C["surface"], "marginTop": "10px"}),

    # ── Stores + interval ─────────────────────────────────────────────────
    dcc.Store(id="prec-train-trigger", data=0),
    dcc.Store(id="prec-bt-trigger", data=0),
    dcc.Store(id="prec-wfbt-trigger", data=0),
    html.Div(id="prec-train-output", style={"display": "none"}),
    html.Div(id="prec-bt-output", style={"display": "none"}),
    html.Div(id="prec-wfbt-output", style={"display": "none"}),
    dcc.Interval(id="prec-interval", interval=15_000, n_intervals=0),
])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_recent_data(symbol: str, period: str = "5d", interval: str = "1m") -> pd.DataFrame:
    """Fetch recent 1-min OHLCV data via app.py's existing helper."""
    from app import fetch_yahoo_finance_data
    df = fetch_yahoo_finance_data(symbol, period=period, interval=interval)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df


def _fetch_training_data(symbol: str) -> pd.DataFrame:
    """Fetch ~30 days of 1-min data for training (yfinance hard limit)."""
    from app import fetch_yahoo_finance_data
    # yfinance allows max 7 days at 1m granularity per request, so use 7d / 1m
    # for the training set. Caller (training thread) can stitch chunks if it
    # wants more — for now 7 days = ~7×1440 = ~10k bars which is enough for
    # the XGBoost classifier + a 70/30 split.
    df = fetch_yahoo_finance_data(symbol, period="7d", interval="1m")
    if df is None or df.empty:
        # Fall back to 15-min if 1m unavailable (weekend / illiquid hour)
        df = fetch_yahoo_finance_data(symbol, period="60d", interval="15m")
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Callback 1: Live signal — runs every 15s
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("prec-signal-badge",     "children"),
    Output("prec-signal-badge",     "style"),
    Output("prec-signal-confidence","children"),
    Output("prec-entry",            "children"),
    Output("prec-sl",               "children"),
    Output("prec-tp1",              "children"),
    Output("prec-tp2",              "children"),
    Output("prec-tp3",              "children"),
    Output("prec-lot",              "children"),
    Output("prec-vpin",             "children"),
    Output("prec-regime",           "children"),
    Output("prec-atr",              "children"),
    Output("prec-mtf-score",        "children"),
    Output("prec-mtf-conf",         "children"),
    Output("prec-cvd-div",          "children"),
    Output("prec-in-va",            "children"),
    Output("prec-sizing",           "children"),
    Output("prec-signal-status",    "children"),
    Output("prec-progress",         "value"),
    Input("prec-interval",          "n_intervals"),
    Input("prec-symbol",            "value"),
    Input("prec-model",             "value"),
    Input("prec-train-trigger",     "data"),
    prevent_initial_call=False,
)
def update_live_signal(_n, symbol, model, _trig):
    DASH_DASH = "--"
    n_dashes_v3 = 14  # 9 base stats + 5 v3 stats
    if not symbol or not model:
        return ("---", {"fontSize": "32px", "color": C["muted"], "fontWeight": "bold", "lineHeight": "1"},
                "Select symbol",) + (DASH_DASH,) * n_dashes_v3 + ("Pick a symbol + model", 0)

    sys = _get_system(symbol, model)
    key = f"{symbol}_{model}"
    state = _training_state.get(key, {})

    # Training in progress?
    if state.get("status") == "training":
        prog = int(state.get("progress", 0))
        return ("⏳", {"fontSize": "32px", "color": C["warn"], "fontWeight": "bold", "lineHeight": "1"},
                f"Training {symbol}/{model}…",) + (DASH_DASH,) * n_dashes_v3 + (state.get("message", "Training…"), prog)

    # Untrained?
    if not sys.is_trained:
        return ("---", {"fontSize": "32px", "color": C["muted"], "fontWeight": "bold", "lineHeight": "1"},
                "Model not trained",) + (DASH_DASH,) * n_dashes_v3 + ("Click 'Train' to begin (~30-60 s)", 0)

    # Generate live signal
    try:
        df = _fetch_recent_data(symbol)
        if df is None or df.empty or len(df) < 100:
            return (no_update,) * (3 + n_dashes_v3) + ("Waiting for data…", no_update)
        sig = sys.generate_live_signal(df)
    except Exception as e:
        return ("ERR", {"fontSize": "32px", "color": C["danger"], "fontWeight": "bold", "lineHeight": "1"},
                str(e)[:60],) + (DASH_DASH,) * n_dashes_v3 + (f"Signal error: {type(e).__name__}", 0)

    action = sig["action"]
    # Color the badge by direction
    if "BUY" in action.upper():
        col = C["accent"] if action.startswith("BUY") else "#7fff7f"
    elif "SELL" in action.upper():
        col = C["danger"] if action.startswith("SELL") else "#ff7575"
    else:
        col = C["warn"]
    # Mute the colour when blocked
    if "BLOCK" in action.upper() or "SPREAD" in action.upper():
        col = C["muted"]

    badge_style = {"fontSize": "30px", "color": col, "fontWeight": "bold", "lineHeight": "1"}
    conf_text = f"Confidence: {sig['confidence']:.1%}"

    # Pretty-print v3 fields
    mtf_score_str = f"{sig.get('mtf_score', 0):+.2f}"
    mtf_conf_str = sig.get("mtf_confluence", "neutral").upper().replace("_", " ")
    cvd_d = sig.get("cvd_div", 0)
    cvd_div_str = "BULL ↑" if cvd_d > 0 else "BEAR ↓" if cvd_d < 0 else "—"
    in_va_str = "INSIDE" if sig.get("in_value_area", True) else "OUTSIDE"
    sizing_str = sig.get("sizing_method", "fixed").upper()

    return (
        action, badge_style, conf_text,
        f"{sig['entry']:,.5f}",
        f"{sig['sl']:,.5f}",
        f"{sig['tp1']:,.5f}",
        f"{sig['tp2']:,.5f}",
        f"{sig['tp3']:,.5f}",
        f"{sig['lot_size']:.4f}",
        f"{sig['vpin']:.2f}",
        sig["regime"].upper(),
        f"{sig['atr']:,.5f}",
        mtf_score_str,
        mtf_conf_str,
        cvd_div_str,
        in_va_str,
        sizing_str,
        f"Live ✓  •  Last update: {datetime.now().strftime('%H:%M:%S')}",
        100,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Callback 2: Train button → daemon thread
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("prec-train-output", "children"),
    Output("prec-train-trigger", "data"),
    Input("prec-train-btn", "n_clicks"),
    State("prec-symbol", "value"),
    State("prec-model", "value"),
    State("prec-train-trigger", "data"),
    prevent_initial_call=True,
)
def start_training(n_clicks, symbol, model, trig):
    if not n_clicks:
        return no_update, no_update
    key = f"{symbol}_{model}"
    if _training_state.get(key, {}).get("status") == "training":
        return "Already training", trig

    _training_state[key] = {
        "status": "training",
        "progress": 0,
        "message": f"Fetching data for {symbol}…",
    }

    def _train_bg():
        try:
            df = _fetch_training_data(symbol)
            if df is None or df.empty or len(df) < 500:
                _training_state[key] = {
                    "status": "error",
                    "progress": 0,
                    "message": f"Insufficient data ({len(df) if df is not None else 0} bars)",
                }
                return
            _training_state[key].update({
                "progress": 30,
                "message": f"Training {model} on {len(df)} bars…",
            })
            sys = _get_system(symbol, model)
            sys.train(df)
            sys.save(_save_path(symbol, model))
            _training_state[key] = {
                "status": "ready",
                "progress": 100,
                "message": f"Trained on {len(df)} bars",
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            _training_state[key] = {
                "status": "error",
                "progress": 0,
                "message": f"Training error: {type(e).__name__}: {e}",
            }

    threading.Thread(target=_train_bg, daemon=True,
                      name=f"PrecTrain-{key}").start()
    return f"Training started for {symbol}/{model}", (trig or 0) + 1


# ─────────────────────────────────────────────────────────────────────────────
# Callback 3: Backtest button → daemon thread
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("prec-bt-output",      "children"),
    Output("prec-bt-trigger",     "data"),
    Output("prec-bt-trades",      "children"),
    Output("prec-bt-winrate",     "children"),
    Output("prec-bt-pf",          "children"),
    Output("prec-bt-sharpe",      "children"),
    Output("prec-bt-mdd",         "children"),
    Output("prec-bt-return",      "children"),
    Output("prec-bt-status",      "children"),
    Output("prec-equity-chart",   "figure"),
    Input("prec-backtest-btn",    "n_clicks"),
    State("prec-symbol",          "value"),
    State("prec-model",           "value"),
    State("prec-bt-trigger",      "data"),
    prevent_initial_call=True,
)
def run_backtest(n_clicks, symbol, model, trig):
    if not n_clicks:
        return (no_update,)*10
    sys = _get_system(symbol, model)

    try:
        # Fetch fresh data and backtest
        df = _fetch_training_data(symbol)
        if df is None or df.empty or len(df) < 500:
            empty_fig = go.Figure(layout=dict(
                paper_bgcolor=C["bg"], plot_bgcolor=C["surface"],
                font=dict(color=C["text"]), height=240,
                annotations=[dict(text=f"Not enough data ({0 if df is None else len(df)} bars)",
                                    x=0.5, y=0.5, showarrow=False,
                                    font=dict(color=C["danger"], size=12))],
                xaxis=dict(visible=False), yaxis=dict(visible=False),
            ))
            return ("err", trig or 0, "0", "--", "--", "--", "--", "--",
                    f"❌ Insufficient data ({0 if df is None else len(df)} bars)",
                    empty_fig)

        result = sys.backtest(df, train_pct=0.7)
        if "error" in result:
            empty_fig = go.Figure(layout=dict(paper_bgcolor=C["bg"],
                                                plot_bgcolor=C["surface"],
                                                height=240,
                                                annotations=[dict(text=result["error"],
                                                                    x=0.5, y=0.5,
                                                                    showarrow=False,
                                                                    font=dict(color=C["danger"],
                                                                                size=11))]))
            return ("err", trig or 0, "0", "--", "--", "--", "--", "--",
                    f"❌ {result['error']}", empty_fig)

        # Build equity curve figure
        eq = result.get("equity_curve") or []
        ts = result.get("timestamps")  or list(range(len(eq)))
        fig = go.Figure()
        if eq:
            initial = result.get("initial_equity", 10_000)
            colors = [C["accent"] if v >= initial else C["danger"] for v in eq]
            # Single line + filled area
            fig.add_trace(go.Scatter(
                x=ts, y=eq, mode="lines",
                line=dict(color=C["accent"], width=2),
                fill="tozeroy",
                fillcolor="rgba(0,255,136,0.07)",
                name="Equity",
            ))
            fig.add_hline(y=initial, line=dict(color=C["muted"], width=1, dash="dot"),
                            annotation_text=f"start ${initial:,.0f}",
                            annotation_font=dict(color=C["muted"], size=9))

        fig.update_layout(
            paper_bgcolor=C["bg"], plot_bgcolor=C["surface"],
            font=dict(color=C["text"], size=9),
            height=240,
            margin=dict(l=8, r=8, t=18, b=8),
            xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
            yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"],
                        side="right",
                        title=dict(text="Equity ($)", font=dict(size=9))),
            showlegend=False,
        )

        return (
            "done",
            (trig or 0) + 1,
            f"{result.get('total_trades', 0)}",
            f"{result.get('win_rate', 0):.1%}",
            f"{result.get('profit_factor', 0):.2f}" if result.get("profit_factor", 0) != float("inf") else "∞",
            f"{result.get('sharpe', 0):.2f}",
            f"${result.get('max_drawdown', 0):,.0f}",
            f"{result.get('total_return', 0):.2%}",
            (f"✅ Backtest complete  •  train: {result.get('train_bars', 0):,} bars  "
                f"•  test (unseen): {result.get('test_bars', 0):,} bars"),
            fig,
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        empty_fig = go.Figure(layout=dict(paper_bgcolor=C["bg"],
                                            plot_bgcolor=C["surface"],
                                            height=240))
        return ("err", trig or 0, "0", "--", "--", "--", "--", "--",
                f"❌ {type(e).__name__}: {e}", empty_fig)


# ─────────────────────────────────────────────────────────────────────────────
# Callback 4: Walk-Forward backtest button
# ─────────────────────────────────────────────────────────────────────────────

def _empty_wf_fig(msg: str, color: Optional[str] = None) -> go.Figure:
    return go.Figure(layout=dict(
        paper_bgcolor=C["bg"], plot_bgcolor=C["surface"],
        font=dict(color=C["text"]), height=240,
        margin=dict(l=8, r=8, t=18, b=8),
        annotations=[dict(text=msg, x=0.5, y=0.5, showarrow=False,
                            font=dict(color=color or C["muted"], size=11))],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    ))


def _format_pf(pf) -> str:
    try:
        return "∞" if pf == float("inf") else f"{float(pf):.2f}"
    except Exception:
        return "--"


def _build_fold_table(per_fold):
    """Compact per-fold breakdown for walk-forward results."""
    if not per_fold:
        return html.Div("No folds completed.",
                        style={"color": C["muted"], "fontSize": "10px",
                                "textAlign": "center"})
    header = html.Tr([
        html.Th(h, style={"padding": "4px 8px", "color": C["muted"],
                            "borderBottom": f"1px solid {C['border']}",
                            "textTransform": "uppercase", "fontSize": "9px"})
        for h in ("Fold", "Train", "Test", "Trades", "Win %",
                    "Sharpe", "PF", "Return")
    ])
    rows = []
    for f in per_fold:
        ret = f.get("total_return", 0)
        ret_color = C["accent"] if ret >= 0 else C["danger"]
        rows.append(html.Tr([
            html.Td(f.get("fold", 0),         style={"padding": "3px 8px"}),
            html.Td(f"{f.get('train_bars', 0):,}", style={"padding": "3px 8px"}),
            html.Td(f"{f.get('test_bars', 0):,}",  style={"padding": "3px 8px"}),
            html.Td(f.get("trades", 0),       style={"padding": "3px 8px"}),
            html.Td(f"{f.get('win_rate', 0):.1%}",   style={"padding": "3px 8px"}),
            html.Td(f"{f.get('sharpe', 0):.2f}",     style={"padding": "3px 8px"}),
            html.Td(_format_pf(f.get("profit_factor", 0)),
                    style={"padding": "3px 8px"}),
            html.Td(f"{ret:+.2%}",            style={"padding": "3px 8px",
                                                        "color": ret_color,
                                                        "fontWeight": "bold"}),
        ]))
    return html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={"width": "100%", "color": C["text"], "fontSize": "10px",
                "fontFamily": "monospace",
                "borderCollapse": "collapse", "marginTop": "6px"},
    )


@callback(
    Output("prec-wfbt-output",        "children"),
    Output("prec-wfbt-trigger",       "data"),
    Output("prec-wf-folds",           "children"),
    Output("prec-wf-trades",          "children"),
    Output("prec-wf-winrate",         "children"),
    Output("prec-wf-prec-long",       "children"),
    Output("prec-wf-prec-short",      "children"),
    Output("prec-wf-sharpe",          "children"),
    Output("prec-wf-pf",              "children"),
    Output("prec-wf-mdd",             "children"),
    Output("prec-wf-ece",             "children"),
    Output("prec-wf-pvalue",          "children"),
    Output("prec-wf-return",          "children"),
    Output("prec-wf-status",          "children"),
    Output("prec-wf-equity-chart",    "figure"),
    Output("prec-wf-fold-table",      "children"),
    Input("prec-wfbacktest-btn",      "n_clicks"),
    State("prec-symbol",              "value"),
    State("prec-model",               "value"),
    State("prec-wfbt-trigger",        "data"),
    prevent_initial_call=True,
)
def run_walk_forward(n_clicks, symbol, model, trig):
    if not n_clicks:
        return (no_update,) * 16
    sys = _get_system(symbol, model)
    try:
        df = _fetch_training_data(symbol)
        if df is None or df.empty or len(df) < 1000:
            n = 0 if df is None else len(df)
            empty = _empty_wf_fig(f"Walk-forward needs ≥ 1000 bars (got {n})",
                                   C["danger"])
            return ("err", trig or 0,
                    "0","0","--","--","--","--","--","--","--","--","--",
                    f"❌ Insufficient data ({n} bars)",
                    empty, "")

        result = sys.walk_forward_backtest(df, n_splits=5,
                                            test_size_pct=0.10,
                                            intra_bar_fills=True)
        if "error" in result:
            empty = _empty_wf_fig(result["error"], C["danger"])
            return ("err", trig or 0,
                    "0","0","--","--","--","--","--","--","--","--","--",
                    f"❌ {result['error']}",
                    empty, "")

        eq = result.get("equity_curve") or []
        ts = result.get("timestamps") or list(range(len(eq)))
        fig = go.Figure()
        if eq:
            initial = result.get("initial_equity", 10_000)
            fig.add_trace(go.Scatter(
                x=ts, y=eq, mode="lines",
                line=dict(color=C["accent"], width=2),
                fill="tozeroy",
                fillcolor="rgba(0,255,136,0.07)",
                name="Equity",
            ))
            fig.add_hline(y=initial, line=dict(color=C["muted"], width=1, dash="dot"),
                            annotation_text=f"start ${initial:,.0f}",
                            annotation_font=dict(color=C["muted"], size=9))
        fig.update_layout(
            paper_bgcolor=C["bg"], plot_bgcolor=C["surface"],
            font=dict(color=C["text"], size=9),
            height=240,
            margin=dict(l=8, r=8, t=18, b=8),
            xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
            yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"],
                        side="right",
                        title=dict(text="Equity ($)", font=dict(size=9))),
            showlegend=False,
        )

        ece = result.get("ece", 0)
        pval = result.get("p_value")
        pval_str = f"{pval:.4f}" if pval is not None else "n/a"
        sig_marker = " ✓" if pval is not None and pval < 0.05 else ""

        return (
            "done",
            (trig or 0) + 1,
            f"{result.get('n_splits', 0)}",
            f"{result.get('total_trades', 0)}",
            f"{result.get('win_rate', 0):.1%}",
            f"{result.get('precision_long', 0):.1%}",
            f"{result.get('precision_short', 0):.1%}",
            f"{result.get('sharpe', 0):.2f}",
            _format_pf(result.get("profit_factor", 0)),
            f"${result.get('max_drawdown', 0):,.0f}",
            f"{ece:.4f}",
            pval_str + sig_marker,
            f"{result.get('total_return', 0):+.2%}",
            (f"✅ Walk-forward complete  •  {result.get('n_splits', 0)} folds  "
                f"•  trained up to {result.get('train_bars', 0):,} bars  "
                f"•  total OOS test bars: {result.get('test_bars', 0):,}"
                + (f"  •  statistically significant at p<0.05" if sig_marker else "")),
            fig,
            _build_fold_table(result.get("per_fold", [])),
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        empty = _empty_wf_fig(f"❌ {type(e).__name__}: {e}", C["danger"])
        return ("err", trig or 0,
                "0","0","--","--","--","--","--","--","--","--","--",
                f"❌ {type(e).__name__}: {e}",
                empty, "")

