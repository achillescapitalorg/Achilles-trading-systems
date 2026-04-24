"""
XAUUSD SMMA Strategy Page — with Synthetic Order Book, Momentum & Volatility
=============================================================================

Strategy layers (in order of priority):
  1. Trend  — SMMA-75 anchor + stack alignment (3>9>40>75) + Hurst exponent
  2. Entry  — price touches / crosses any SMMA level while trend confirmed
  3. Momentum filter — MFI not extreme, volume delta confirms direction,
                       cumulative delta trending with signal
  4. Volatility filter — ATR regime (expanding = valid, contracting = skip),
                         price inside Keltner channel bands = mean-revert risk

Synthetic Order Book:
  Real L2 order book data is unavailable for OTC gold via public APIs.
  Volume Profile (Price-At-Volume) built from 1m OHLCV bars is the
  professional substitute: the Point of Control (POC) and Value Area
  (High/Low) identify where the market has accepted the most volume —
  equivalent to dense order clusters on a real order book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State, Patch, no_update
from datetime import datetime

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

SMMA_COL = {"smma3": C["accent"], "smma9": C["info"],
            "smma40": C["warn"],  "smma75": C["danger"]}
SIG_COL  = {"BUY": C["accent"], "SELL": C["danger"], "HOLD": C["warn"]}


def _categorical_x(timestamps) -> pd.Series:
    """Gap-free categorical x-axis labels for Plotly charts."""
    ts = pd.to_datetime(timestamps)
    fmt = "%m/%d %H:%M" if len(ts) > 1 and (ts.iloc[-1] - ts.iloc[-2]).total_seconds() < 86400 else "%b %d"
    return ts.dt.strftime(fmt)

TF_PERIOD = {"1m": "5d", "5m": "5d", "15m": "5d"}

# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────

def _card(title, body, right=None, body_style=None):
    hdr = [html.Span(title, style={"fontWeight": "bold", "color": C["text"],
                                   "fontSize": "11px", "letterSpacing": "1px"})]
    if right is not None:
        hdr.append(right)
    return dbc.Card([
        dbc.CardHeader(hdr, style={"backgroundColor": C["surface"],
                                   "borderBottom": f"1px solid {C['border']}",
                                   "padding": "9px 14px", "display": "flex",
                                   "justifyContent": "space-between",
                                   "alignItems": "center"}),
        dbc.CardBody(body, style={**{"backgroundColor": C["surf_lt"],
                                     "padding": "12px"},
                                  **(body_style or {})}),
    ], style={"backgroundColor": C["surface"],
              "border": f"1px solid {C['border']}",
              "borderRadius": "6px", "marginBottom": "12px"})


def _stat(label, value, col=None, sub=None):
    return html.Div([
        html.Div(label, style={"color": C["muted"], "fontSize": "9px",
                               "letterSpacing": "0.8px", "textTransform": "uppercase"}),
        html.Div(value, style={"color": col or C["text"], "fontSize": "18px",
                               "fontWeight": "bold", "lineHeight": "1.2"}),
        html.Div(sub or "", style={"color": C["muted"], "fontSize": "9px",
                                   "marginTop": "2px"}),
    ], style={"padding": "10px 14px",
              "borderRight": f"1px solid {C['border']}",
              "minWidth": "110px"})


# ─────────────────────────────────────────────────────────────────────────────
# Page layout
# ─────────────────────────────────────────────────────────────────────────────

layout = dbc.Container(fluid=True, style={"backgroundColor": C["bg"],
                                          "minHeight": "100vh",
                                          "padding": "14px"}, children=[

    # ── Top bar ───────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.H5([
                html.Span("⚡", style={"marginRight": "6px", "color": C["accent"]}),
                "XAUUSD  SMMA Strategy",
                html.Span("  3 · 9 · 40 · 75", style={"fontSize": "13px",
                           "color": C["muted"], "marginLeft": "8px"}),
            ], style={"color": C["text"], "margin": 0}),
            html.P("Volume-Profile order book · Momentum (MFI / Delta) · Volatility (ATR / Keltner)",
                   style={"color": C["muted"], "fontSize": "10px", "margin": "3px 0 0 0"}),
        ], width=7),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.Label("Timeframe", style={"color": C["muted"], "fontSize": "10px",
                               "display": "block", "marginBottom": "3px"}),
                    dcc.Dropdown(id="smma-tf-selector",
                                 options=[{"label": t, "value": t}
                                          for t in ("1m", "5m", "15m")],
                                 value="1m", clearable=False,
                                 style={"width": "85px"}),
                ], width="auto"),
                dbc.Col([
                    html.Label("Symbol", style={"color": C["muted"], "fontSize": "10px",
                               "display": "block", "marginBottom": "3px"}),
                    dcc.Dropdown(id="smma-symbol-selector",
                                 options=[{"label": s, "value": s}
                                          for s in ("XAUUSD", "BTCUSD", "EURUSD")],
                                 value="XAUUSD", clearable=False,
                                 style={"width": "110px"}),
                ], width="auto"),
                dbc.Col([
                    html.Label("\u00a0", style={"display": "block", "fontSize": "10px",
                               "marginBottom": "3px"}),
                    dbc.Button("↻ Refresh", id="smma-refresh-btn",
                               color="success", size="sm",
                               style={"fontSize": "11px"}),
                ], width="auto"),
            ], align="end"),
        ], width=5, style={"textAlign": "right"}),
    ], className="mb-3", align="center"),

    # Suppress all loading spinners for every callback output on this page
    dcc.Loading(
        display="hide",
        target_components={
            "smma-chart": "figure",
            "smma-orderbook-chart": "figure",
            "smma-delta-chart": "figure",
            "smma-stat-strip": "children",
            "smma-live-price": "children",
            "smma-verification-panel": "children",
            "smma-mtf-hurst": "children",
            "smma-signal-log": "children",
        },
    ),

    # ── Stat strip ────────────────────────────────────────────────────────────
    html.Div(id="smma-stat-strip",
             style={"display": "flex", "flexWrap": "wrap",
                    "backgroundColor": C["surface"],
                    "border": f"1px solid {C['border']}",
                    "borderRadius": "6px",
                    "marginBottom": "12px",
                    "overflowX": "auto"}),

    # ── Main chart row ────────────────────────────────────────────────────────
    dbc.Row([
        # Price + indicators chart
        dbc.Col([
            _card("📈 PRICE  ·  SMMA  ·  VWAP  ·  KELTNER",
                  dcc.Graph(id="smma-chart",
                      figure=go.Figure(layout=dict(
                          paper_bgcolor=C["bg"], plot_bgcolor=C["surface"],
                          font=dict(color=C["text"]),
                          annotations=[dict(text="Loading…", x=0.5, y=0.5,
                                            showarrow=False,
                                            font=dict(color=C["muted"],size=12))],
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                      )),
                      config={
                          "scrollZoom": True,
                          "responsive": False,
                          "displayModeBar": True,
                          "displaylogo": False,
                          "modeBarButtonsToRemove": [
                              "lasso2d", "select2d", "autoScale2d", "toggleSpikelines",
                          ],
                          "modeBarButtonsToAdd": ["drawline", "eraseshape"],
                          "doubleClick": "reset",
                          "showTips": False,
                      },
                      style={"height": "520px"}),
                  right=html.Div(id="smma-live-price",
                                 style={"fontSize": "14px", "fontWeight": "bold",
                                        "color": C["accent"]})),
        ], width=8),

        # Right column: order book + verification
        dbc.Col([
            _card("📚 SYNTHETIC ORDER BOOK  (Volume Profile)",
                  dcc.Graph(id="smma-orderbook-chart",
                      figure=go.Figure(layout=dict(
                          paper_bgcolor=C["bg"], plot_bgcolor=C["surface"],
                          font=dict(color=C["text"]),
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                      )),
                      config={"displayModeBar": False},
                      style={"height": "260px"})),
            _card("🔍 TREND + MOMENTUM + VOLATILITY",
                  html.Div(id="smma-verification-panel")),
        ], width=4),
    ], className="mb-2"),

    # ── Bottom row: Delta + Hurst MTF ────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            _card("🌊 VOLUME DELTA  (Buy − Sell Pressure)",
                  dcc.Graph(id="smma-delta-chart",
                      figure=go.Figure(layout=dict(
                          paper_bgcolor=C["bg"], plot_bgcolor=C["surface"],
                          font=dict(color=C["text"]),
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                      )),
                      config={"displayModeBar": False},
                      style={"height": "180px"})),
        ], width=6),
        dbc.Col([
            _card("〰 MULTI-TIMEFRAME HURST  (1m · 5m · 15m)",
                  html.Div(id="smma-mtf-hurst")),
        ], width=3),
        dbc.Col([
            _card("📋 SIGNAL LOG",
                  html.Div(id="smma-signal-log",
                           style={"maxHeight": "180px", "overflowY": "auto",
                                  "fontFamily": "monospace",
                                  "fontSize": "10px"})),
        ], width=3),
    ]),

    dcc.Interval(id="smma-interval", interval=15_000, n_intervals=0),
])


# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python calculation helpers (self-contained, no circular import)
# ─────────────────────────────────────────────────────────────────────────────

def _smma(prices: pd.Series, period: int):
    prices = prices.dropna()
    if len(prices) < period:
        return None
    return prices.ewm(alpha=1.0 / period, adjust=False).mean()


def _atr_wilder(df: pd.DataFrame, period=14):
    if len(df) < period + 1:
        return None
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _vwap_session(df: pd.DataFrame):
    if len(df) < 2:
        return None
    tp    = (df["high"] + df["low"] + df["close"]) / 3
    # Use index (DatetimeIndex) or "timestamp" column, whichever exists
    if "timestamp" in df.columns:
        dates = pd.to_datetime(df["timestamp"]).dt.date
    else:
        dates = pd.to_datetime(df.index).date
    tp_vol = tp * df["volume"]
    cvtp  = tp_vol.groupby(dates).cumsum()
    cvol  = df["volume"].groupby(dates).cumsum()
    v     = cvtp / cvol.replace(0, np.nan)
    v.index = df.index
    return v


def _mfi(df: pd.DataFrame, period=14):
    if len(df) < period + 1:
        return None
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    rmf = tp * df["volume"]
    pos = rmf.where(tp > tp.shift(), 0.0)
    neg = rmf.where(tp < tp.shift(), 0.0)
    mfr = pos.rolling(period).sum() / neg.rolling(period).sum().replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def _keltner(df: pd.DataFrame, ema_p=20, atr_p=10, mult=2.0):
    if len(df) < max(ema_p, atr_p) + 5:
        return None
    ema = df["close"].ewm(span=ema_p, adjust=False).mean()
    atr = _atr_wilder(df, atr_p)
    if atr is None:
        return None
    return {"upper": ema + mult * atr, "middle": ema, "lower": ema - mult * atr}


def _delta(df: pd.DataFrame):
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    buy  = df["volume"] * (df["close"] - df["low"])  / hl
    sell = df["volume"] * (df["high"]  - df["close"]) / hl
    buy  = buy.fillna(df["volume"] / 2)
    sell = sell.fillna(df["volume"] / 2)
    d    = buy - sell
    return pd.DataFrame({"buy": buy, "sell": sell,
                          "delta": d, "cum": d.cumsum()}, index=df.index)


def _volume_profile(df: pd.DataFrame, n=35):
    if df is None or len(df) < 10:
        return None
    lo_g, hi_g = float(df["low"].min()), float(df["high"].max())
    if hi_g <= lo_g:
        return None
    bsz   = (hi_g - lo_g) / n
    levels = np.linspace(lo_g + bsz / 2, hi_g - bsz / 2, n)
    vols   = np.zeros(n)
    for _, r in df.iterrows():
        lo, hi, vol = r["low"], r["high"], r["volume"]
        if hi == lo:
            idx = min(int((r["close"] - lo_g) / bsz), n - 1)
            vols[idx] += vol
        else:
            i0 = max(0,   int((lo - lo_g) / bsz))
            i1 = min(n-1, int((hi - lo_g) / bsz))
            span = i1 - i0 + 1
            vols[i0:i1+1] += vol / span
    poc_i   = int(np.argmax(vols))
    poc_p   = float(levels[poc_i])
    poc_v   = float(vols[poc_i])
    # Value area (70%)
    total   = vols.sum()
    target  = total * 0.70
    acc     = vols[poc_i]
    li, hi_i = poc_i, poc_i
    while acc < target and (li > 0 or hi_i < n - 1):
        al = vols[li-1]  if li  > 0   else 0
        ah = vols[hi_i+1] if hi_i < n-1 else 0
        if al >= ah and li > 0:
            li -= 1; acc += al
        elif hi_i < n - 1:
            hi_i += 1; acc += ah
        else:
            li -= 1; acc += al
    return {"levels": levels, "volumes": vols,
            "poc_price": poc_p, "poc_volume": poc_v,
            "vah": float(levels[hi_i]), "val": float(levels[li]),
            "va_pct": float(acc / total)}


def _hurst(returns, min_lag=5, max_lag=None):
    n = len(returns)
    if max_lag is None:
        max_lag = max(min_lag + 5, n // 4)
    max_lag = min(max_lag, n // 2)
    if max_lag <= min_lag or n < 2 * max_lag:
        return 0.5, 0.0
    rs_vals, n_vals = [], []
    for lag in range(min_lag, max_lag + 1):
        chunks = n // lag
        if chunks < 3:
            continue
        crs = []
        for i in range(chunks):
            c = returns[i*lag:(i+1)*lag]
            if len(c) < 2:
                continue
            cd = np.cumsum(c - np.mean(c))
            R  = cd.max() - cd.min()
            S  = np.std(c, ddof=1)
            if S > 1e-10:
                crs.append(R / S)
        if len(crs) >= 2:
            rs_vals.append(np.mean(crs))
            n_vals.append(lag)
    if len(rs_vals) < 3:
        return 0.5, 0.0
    ln = np.log(np.array(n_vals))
    lr = np.log(np.array(rs_vals) + 1e-10)
    c  = np.polyfit(ln, lr, 1)
    H  = float(np.clip(c[0], 0.1, 0.9))
    res = np.abs(lr - (c[0]*ln + c[1]))
    conf = max(0.0, 1.0 - min(res.max() / 2.0, 1.0))
    return H, round(conf, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Core strategy engine  (full-pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def _run_strategy(df: pd.DataFrame):
    """
    Run the full SMMA strategy pipeline on a DataFrame of OHLCV bars.
    Returns a rich result dict used by all chart/panel builders.
    """
    out = dict(
        signal="HOLD", strength="WEAK", trend="SIDEWAYS",
        smma3=None, smma9=None, smma40=None, smma75=None,
        vwap=None, keltner=None,
        atr=None, atr_pct=None, atr_regime="NORMAL",
        mfi=None, delta_df=None, cum_delta=None,
        hurst=0.5, hurst_conf=0.0,
        touch_level=None, vp=None,
        momentum_ok=False, volatility_ok=False,
        details=[],
    )

    if df is None or len(df) < 80:
        out["details"].append("Need ≥80 bars")
        return out

    closes = df["close"].dropna()

    # ── SMMAss ────────────────────────────────────────────────────────────────
    s = {p: _smma(closes, p) for p in (3, 9, 40, 75)}
    if any(v is None for v in s.values()):
        out["details"].append("SMMA computation failed")
        return out

    out["smma3"]  = s[3]
    out["smma9"]  = s[9]
    out["smma40"] = s[40]
    out["smma75"] = s[75]
    v3, v9, v40, v75 = (float(s[p].iloc[-1]) for p in (3, 9, 40, 75))
    price = float(closes.iloc[-1])

    # ── VWAP ──────────────────────────────────────────────────────────────────
    out["vwap"] = _vwap_session(df)

    # ── Keltner ───────────────────────────────────────────────────────────────
    out["keltner"] = _keltner(df)

    # ── ATR + volatility regime ───────────────────────────────────────────────
    atr_series = _atr_wilder(df, 14)
    if atr_series is not None:
        atr_val  = float(atr_series.iloc[-1])
        atr_mean = float(atr_series.tail(50).mean())
        atr_pct  = atr_val / price * 100
        out["atr"]     = atr_val
        out["atr_pct"] = round(atr_pct, 3)
        # Volatility expanding (ATR > 1.1× mean) = momentum environment
        if atr_val > atr_mean * 1.1:
            out["atr_regime"] = "EXPANDING"
            out["volatility_ok"] = True
        elif atr_val < atr_mean * 0.8:
            out["atr_regime"] = "CONTRACTING"
        else:
            out["atr_regime"] = "NORMAL"
            out["volatility_ok"] = True   # normal vol is also OK for entries

    # ── MFI ───────────────────────────────────────────────────────────────────
    mfi_series = _mfi(df, 14)
    if mfi_series is not None:
        out["mfi"] = float(mfi_series.iloc[-1])

    # ── Volume Delta ──────────────────────────────────────────────────────────
    delta_df = _delta(df)
    out["delta_df"]  = delta_df
    out["cum_delta"] = float(delta_df["cum"].iloc[-1]) if delta_df is not None else 0.0

    # ── Volume Profile ────────────────────────────────────────────────────────
    out["vp"] = _volume_profile(df)

    # ── Hurst exponent ────────────────────────────────────────────────────────
    returns = closes.pct_change().dropna().values
    H, Hc   = _hurst(returns, min_lag=5, max_lag=min(40, len(returns)//3))
    out["hurst"]      = H
    out["hurst_conf"] = Hc

    # ──────────────────────────────────────────────────────────────────────────
    # Trend verification  (3 mechanisms)
    # ──────────────────────────────────────────────────────────────────────────
    aligned_up   = v3 > v9 > v40 > v75
    aligned_down = v3 < v9 < v40 < v75
    price_above  = price > v75

    stack_trend = ("UP" if aligned_up else "DOWN" if aligned_down else "MIXED")
    out["details"].append(f"SMMA stack: {stack_trend}")
    out["details"].append(f"Price vs SMMA-75: {'ABOVE' if price_above else 'BELOW'}")
    out["details"].append(f"Hurst H={H:.3f} conf={Hc:.0%} "
                          f"({'trending' if H>0.55 else 'mean-rev' if H<0.45 else 'random'})")

    if aligned_up and price_above:
        trend = "UP"
    elif aligned_down and not price_above:
        trend = "DOWN"
    else:
        trend = "SIDEWAYS"
    out["trend"] = trend

    # ──────────────────────────────────────────────────────────────────────────
    # Momentum check
    # ──────────────────────────────────────────────────────────────────────────
    mfi_val = out["mfi"]
    cum_d   = out["cum_delta"]

    if trend == "UP":
        # MFI not overbought (<80), delta positive = buyers in control
        mfi_ok    = mfi_val is None or mfi_val < 80
        delta_ok  = cum_d > 0
        out["momentum_ok"] = mfi_ok and delta_ok
        _mfi_str = f"{mfi_val:.1f}" if mfi_val is not None else "N/A"
        out["details"].append(
            f"Momentum: MFI={_mfi_str} "
            f"ΔCum={cum_d:+,.0f} → {'OK' if out['momentum_ok'] else 'WEAK'}"
        )
    elif trend == "DOWN":
        mfi_ok    = mfi_val is None or mfi_val > 20
        delta_ok  = cum_d < 0
        out["momentum_ok"] = mfi_ok and delta_ok
        _mfi_str = f"{mfi_val:.1f}" if mfi_val is not None else "N/A"
        out["details"].append(
            f"Momentum: MFI={_mfi_str} "
            f"ΔCum={cum_d:+,.0f} → {'OK' if out['momentum_ok'] else 'WEAK'}"
        )
    else:
        out["momentum_ok"] = False

    out["details"].append(f"Volatility: ATR={out['atr_pct']:.3f}% ({out['atr_regime']})")

    # ──────────────────────────────────────────────────────────────────────────
    # Touch / crossover detection
    # ──────────────────────────────────────────────────────────────────────────
    tol = 0.0015   # 0.15% tolerance
    touches = {}
    for lbl, val in [("SMMA-3", v3), ("SMMA-9", v9),
                     ("SMMA-40", v40), ("SMMA-75", v75)]:
        if abs(price - val) / val <= tol:
            touches[lbl] = val

    # Also catch recent bar crossover
    if len(closes) >= 3:
        prev = float(closes.iloc[-2])
        for lbl, sv, val in [("SMMA-3", s[3], v3), ("SMMA-9", s[9], v9),
                              ("SMMA-40", s[40], v40), ("SMMA-75", s[75], v75)]:
            pv = float(sv.iloc[-2])
            if (prev < pv and price >= val) or (prev > pv and price <= val):
                touches[f"{lbl}✕"] = val

    # ──────────────────────────────────────────────────────────────────────────
    # Final signal  (all three gates must pass: trend + momentum + volatility)
    # ──────────────────────────────────────────────────────────────────────────
    if touches and trend in ("UP", "DOWN"):
        sig   = "BUY" if trend == "UP" else "SELL"
        touch = list(touches.keys())[0]

        # Strength: heavier SMMA = more significant support/resistance
        if any("SMMA-40" in k or "SMMA-75" in k for k in touches):
            raw_strength = "STRONG"
        elif any("SMMA-9" in k for k in touches):
            raw_strength = "MODERATE"
        else:
            raw_strength = "WEAK"

        # Downgrade if momentum or volatility gates fail
        if not out["momentum_ok"] and not out["volatility_ok"]:
            raw_strength = "WEAK"
            out["details"].append("Signal downgraded: momentum + volatility both weak")
        elif not out["momentum_ok"]:
            if raw_strength == "STRONG":
                raw_strength = "MODERATE"
            out["details"].append("Signal downgraded: momentum not confirmed")
        elif out["atr_regime"] == "CONTRACTING":
            if raw_strength == "STRONG":
                raw_strength = "MODERATE"
            out["details"].append("Caution: volatility contracting")

        out["signal"]      = sig
        out["strength"]    = raw_strength
        out["touch_level"] = touch
        out["details"].append(f"→ {sig} [{raw_strength}] at {touch}")

    elif not touches and trend in ("UP", "DOWN"):
        out["details"].append(f"Trend={trend}, waiting for pullback to SMMA level")
    else:
        out["details"].append("Mixed trend — stand aside")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure builders
# ─────────────────────────────────────────────────────────────────────────────

def _fig_price(df, r, symbol, tf):
    """5-subplot price chart: candles, SMMA, VWAP, Keltner, MFI, volume."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.60, 0.20, 0.20],
        vertical_spacing=0.02,
        subplot_titles=("", "MFI (14)", "Volume"),
    )

    ts = df["timestamp"] if "timestamp" in df.columns else df.index
    ts_str = _categorical_x(ts)

    # Row 1 — Candlestick
    fig.add_trace(go.Candlestick(
        x=ts_str, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Price",
        increasing_line_color=C["accent"],  decreasing_line_color=C["danger"],
        increasing_fillcolor=C["accent"],   decreasing_fillcolor=C["danger"],
        line=dict(width=1),
    ), row=1, col=1)

    # SMMA lines — Scattergl uses WebGL for much faster rendering
    for key, period, label in [("smma3",3,"SMMA 3"), ("smma9",9,"SMMA 9"),
                                ("smma40",40,"SMMA 40"), ("smma75",75,"SMMA 75")]:
        s = r.get(key)
        if s is not None:
            lw = 2.2 if period in (40, 75) else 1.4
            fig.add_trace(go.Scattergl(x=ts_str, y=s, name=label, mode="lines",
                                       line=dict(color=SMMA_COL[key], width=lw),
                                       opacity=0.9), row=1, col=1)

    # VWAP
    if r.get("vwap") is not None:
        fig.add_trace(go.Scattergl(x=ts_str, y=r["vwap"], name="VWAP", mode="lines",
                                   line=dict(color=C["purple"], width=1.5, dash="dot"),
                                   opacity=0.8), row=1, col=1)

    # Keltner
    kc = r.get("keltner")
    if kc is not None:
        for band, lbl, dash in [("upper", "KC Up", "dash"),
                                 ("lower", "KC Lo", "dash"),
                                 ("middle","KC Mid","dot")]:
            fig.add_trace(go.Scattergl(x=ts_str, y=kc[band], name=lbl, mode="lines",
                                       line=dict(color=C["info"], width=1, dash=dash),
                                       opacity=0.45, showlegend=(band=="middle")),
                          row=1, col=1)

    # Volume Profile levels as horizontal lines
    vp = r.get("vp")
    if vp:
        for lvl, col, name in [(vp["poc_price"], C["warn"],    "POC"),
                                (vp["vah"],       C["accent"],  "VAH"),
                                (vp["val"],       C["danger"],  "VAL")]:
            fig.add_hline(y=lvl, line=dict(color=col, width=1, dash="dot"),
                          annotation_text=name,
                          annotation_font=dict(color=col, size=9),
                          row=1, col=1)

    # Signal marker
    sig = r["signal"]
    if sig in ("BUY", "SELL"):
        lr_idx = len(df) - 1
        lr = df.iloc[-1]
        my = lr["low"] * 0.9997 if sig == "BUY" else lr["high"] * 1.0003
        fig.add_trace(go.Scattergl(
            x=[ts_str.iloc[lr_idx]], y=[my],
            mode="markers+text",
            marker=dict(symbol="triangle-up" if sig=="BUY" else "triangle-down",
                        size=16, color=SIG_COL[sig]),
            text=[f"  {sig}"], textposition="middle right",
            textfont=dict(color=SIG_COL[sig], size=10),
            name=f"{sig} Signal",
        ), row=1, col=1)

    # Row 2 — MFI
    mfi_s = _mfi(df, 14)
    if mfi_s is not None:
        fig.add_trace(go.Scattergl(x=ts_str, y=mfi_s, name="MFI", mode="lines",
                                   line=dict(color=C["info"], width=1.5)), row=2, col=1)
        for lvl, col in [(80, C["danger"]), (20, C["accent"]), (50, C["muted"])]:
            fig.add_hline(y=lvl, line=dict(color=col, width=0.8, dash="dot"), row=2, col=1)

    # Row 3 — Volume bars
    vcol = [C["accent"] if c >= o else C["danger"]
            for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=ts_str, y=df["volume"], name="Volume",
                         marker_color=vcol, opacity=0.7), row=3, col=1)

    # Crosshair spike lines — TradingView style
    spike_style = dict(showspikes=True, spikecolor=C["muted"],
                       spikethickness=1, spikedash="solid", spikemode="across",
                       spikesnap="cursor")
    y_spike = dict(showspikes=True, spikecolor=C["muted"],
                   spikethickness=1, spikedash="dot", spikesnap="cursor")

    fig.update_layout(
        paper_bgcolor=C["bg"], plot_bgcolor=C["surface"],
        font=dict(color=C["text"], size=9),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=8)),
        xaxis_rangeslider_visible=False,
        margin=dict(l=6, r=6, t=24, b=6),
        uirevision=f"{symbol}_{tf}",
        # TradingView defaults: pan by default, crosshair hover
        dragmode="pan",
        hovermode="x unified",
        hoverdistance=20,
    )
    for axis in ("xaxis", "xaxis2", "xaxis3"):
        fig.update_layout(**{axis: dict(gridcolor=C["border"],
                                        zerolinecolor=C["border"],
                                        **spike_style)})
    for axis in ("yaxis", "yaxis2", "yaxis3"):
        fig.update_layout(**{axis: dict(gridcolor=C["border"],
                                        zerolinecolor=C["border"],
                                        **y_spike)})
    for row_n in (1, 2, 3):
        fig.update_yaxes(side="right", row=row_n, col=1)
    fig.update_xaxes(nticks=10)
    return fig


def _fig_orderbook(df, r, price, symbol="", tf=""):
    """Horizontal Volume Profile — synthetic order book."""
    vp = r.get("vp")
    fig = go.Figure(layout=dict(
        paper_bgcolor=C["bg"], plot_bgcolor=C["surface"],
        font=dict(color=C["text"], size=9),
        margin=dict(l=6, r=6, t=6, b=6),
    ))
    if vp is None:
        fig.add_annotation(text="Insufficient data", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=C["muted"]))
        return fig

    levels  = vp["levels"]
    volumes = vp["volumes"]
    poc_p   = vp["poc_price"]
    vah     = vp["vah"]
    val_    = vp["val"]
    max_v   = volumes.max() or 1

    # Colour each bucket: red=sell zone (above POC), green=buy zone (below POC)
    # Highlight value area in brighter shade
    bar_colors = []
    for lv, vol in zip(levels, volumes):
        in_va = val_ <= lv <= vah
        above = lv > poc_p
        if in_va:
            bar_colors.append("rgba(255,71,87,0.75)"  if above
                              else "rgba(0,255,136,0.75)")
        else:
            bar_colors.append("rgba(255,71,87,0.35)"  if above
                              else "rgba(0,255,136,0.35)")

    fig.add_trace(go.Bar(
        y=levels,
        x=volumes,
        orientation="h",
        marker_color=bar_colors,
        name="Volume",
        showlegend=False,
    ))

    # POC line
    fig.add_hline(y=poc_p, line=dict(color=C["warn"], width=2),
                  annotation_text=f"POC {poc_p:,.2f}",
                  annotation_font=dict(color=C["warn"], size=8))
    # VAH / VAL
    fig.add_hline(y=vah, line=dict(color=C["accent"], width=1, dash="dot"),
                  annotation_text=f"VAH {vah:,.2f}",
                  annotation_font=dict(color=C["accent"], size=8))
    fig.add_hline(y=val_, line=dict(color=C["danger"], width=1, dash="dot"),
                  annotation_text=f"VAL {val_:,.2f}",
                  annotation_font=dict(color=C["danger"], size=8))
    # Current price
    fig.add_hline(y=price, line=dict(color=C["info"], width=1.5),
                  annotation_text=f"▶ {price:,.2f}",
                  annotation_font=dict(color=C["info"], size=9))

    fig.update_layout(
        xaxis=dict(title="Volume", gridcolor=C["border"], showticklabels=False),
        yaxis=dict(title="Price", gridcolor=C["border"], side="right"),
        xaxis_rangeslider_visible=False,
        uirevision=f"{symbol}_{tf}",
    )
    return fig


def _fig_delta(df, symbol="", tf=""):
    """Bar chart of per-bar volume delta with cumulative line."""
    delta_df = _delta(df)
    if delta_df is None:
        return go.Figure(layout=dict(paper_bgcolor=C["bg"],
                                     plot_bgcolor=C["surface"]))
    ts_str = _categorical_x(df["timestamp"] if "timestamp" in df.columns else df.index)
    dcol = [C["accent"] if v >= 0 else C["danger"] for v in delta_df["delta"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=ts_str, y=delta_df["delta"], name="Delta",
                         marker_color=dcol, opacity=0.8), secondary_y=False)
    fig.add_trace(go.Scattergl(x=ts_str, y=delta_df["cum"], name="Cum Delta",
                               line=dict(color=C["info"], width=1.5)),
                  secondary_y=True)
    fig.add_hline(y=0, line=dict(color=C["muted"], width=0.8))
    fig.update_layout(
        paper_bgcolor=C["bg"], plot_bgcolor=C["surface"],
        font=dict(color=C["text"], size=9),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=8)),
        margin=dict(l=6, r=6, t=6, b=6),
        xaxis=dict(gridcolor=C["border"]),
        yaxis=dict(gridcolor=C["border"], side="right"),
        yaxis2=dict(gridcolor=C["border"], overlaying="y", side="left",
                    showgrid=False),
        uirevision=f"{symbol}_{tf}",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Verification panel builder
# ─────────────────────────────────────────────────────────────────────────────

def _verification_panel(r, price):
    v3  = float(r["smma3"].iloc[-1])  if r["smma3"]  is not None else None
    v9  = float(r["smma9"].iloc[-1])  if r["smma9"]  is not None else None
    v40 = float(r["smma40"].iloc[-1]) if r["smma40"] is not None else None
    v75 = float(r["smma75"].iloc[-1]) if r["smma75"] is not None else None

    def row(lbl, val, col, note=""):
        return dbc.Row([
            dbc.Col(html.Span(lbl, style={"color": C["muted"], "fontSize": "9px",
                                          "textTransform": "uppercase"}), width=5),
            dbc.Col(html.Span(val, style={"color": col, "fontSize": "10px",
                                          "fontWeight": "bold"}), width=4),
            dbc.Col(html.Span(note, style={"color": C["muted"], "fontSize": "9px"}), width=3),
        ], className="mb-1", align="center")

    # SMMA stack
    au = v3 and v9 and v40 and v75 and (v3 > v9 > v40 > v75)
    ad = v3 and v9 and v40 and v75 and (v3 < v9 < v40 < v75)
    stk_col = C["accent"] if au else C["danger"] if ad else C["warn"]
    stk_txt = "3>9>40>75 ✓" if au else "3<9<40<75 ✓" if ad else "Mixed"

    # Price vs SMMA-75
    p75_txt = "ABOVE ▲" if (v75 and price > v75) else "BELOW ▼"
    p75_col = C["accent"] if (v75 and price > v75) else C["danger"]

    # Hurst
    H, Hc = r["hurst"], r["hurst_conf"]
    h_col = C["accent"] if H > 0.55 else C["info"] if H < 0.45 else C["warn"]
    h_lbl = "Trending" if H > 0.55 else "Mean-rev" if H < 0.45 else "Random"

    # MFI
    mfi = r["mfi"]
    mfi_col = C["danger"] if mfi and mfi > 80 else C["accent"] if mfi and mfi < 20 else C["text"]

    # ATR
    atr_col = C["accent"] if r["atr_regime"] == "EXPANDING" else (
              C["warn"]   if r["atr_regime"] == "CONTRACTING" else C["text"])

    # Volume Profile
    vp = r.get("vp")
    poc_rel = ""
    if vp:
        poc_rel = f"POC {vp['poc_price']:,.2f}"

    # Momentum gate
    mom_col  = C["accent"] if r["momentum_ok"]  else C["danger"]
    vol_col  = C["accent"] if r["volatility_ok"] else C["danger"]

    sections = [
        html.Div("─── TREND ──────────────────", style={"color": C["muted"],
                 "fontSize": "9px", "marginBottom": "4px"}),
        row("SMMA Stack",   stk_txt,          stk_col),
        row("vs SMMA-75",   p75_txt,          p75_col),
        row("Hurst H",      f"{H:.3f} {h_lbl}", h_col, f"conf {Hc:.0%}"),
        row("Trend",        r["trend"],
            C["accent"] if r["trend"]=="UP" else C["danger"] if r["trend"]=="DOWN" else C["warn"]),
        html.Div("─── MOMENTUM ───────────────", style={"color": C["muted"],
                 "fontSize": "9px", "margin": "6px 0 4px 0"}),
        row("MFI (14)",     f"{mfi:.1f}" if mfi else "N/A", mfi_col,
            "OB>80/OS<20"),
        row("Cum Delta",    f"{r['cum_delta']:+,.0f}" if r['cum_delta'] else "N/A",
            C["accent"] if (r['cum_delta'] or 0) > 0 else C["danger"]),
        row("Mom Gate",     "PASS ✓" if r["momentum_ok"] else "FAIL ✗", mom_col),
        html.Div("─── VOLATILITY ─────────────", style={"color": C["muted"],
                 "fontSize": "9px", "margin": "6px 0 4px 0"}),
        row("ATR (14)",     f"{r['atr']:.2f}" if r["atr"] else "N/A", atr_col,
            f"{r['atr_pct']:.2f}%"),
        row("ATR Regime",   r["atr_regime"],  atr_col),
        row("Vol Gate",     "PASS ✓" if r["volatility_ok"] else "FAIL ✗", vol_col),
        html.Div("─── ORDER BOOK ─────────────", style={"color": C["muted"],
                 "fontSize": "9px", "margin": "6px 0 4px 0"}),
        row("POC",          poc_rel or "N/A", C["warn"]),
        row("Value Area",   f"{vp['val']:,.2f}–{vp['vah']:,.2f}" if vp else "N/A",
            C["info"]),
        row("VA Coverage",  f"{vp['va_pct']:.0%}" if vp else "N/A", C["muted"]),
    ]
    return html.Div(sections)


_last_data_hash: dict = {}   # {f"{symbol}_{tf}": (last_close, last_ts)}

# ─────────────────────────────────────────────────────────────────────────────
# Main callback
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    [
        Output("smma-chart",             "figure"),
        Output("smma-orderbook-chart",   "figure"),
        Output("smma-delta-chart",       "figure"),
        Output("smma-stat-strip",        "children"),
        Output("smma-live-price",        "children"),
        Output("smma-verification-panel","children"),
        Output("smma-mtf-hurst",         "children"),
        Output("smma-signal-log",        "children"),
    ],
    [
        Input("smma-refresh-btn",     "n_clicks"),
        Input("smma-interval",        "n_intervals"),
        Input("smma-tf-selector",     "value"),
        Input("smma-symbol-selector", "value"),
    ],
    prevent_initial_call=False,
)
def update_smma_strategy(n_clicks, n_intervals, timeframe, symbol):
    from app import fetch_yahoo_finance_data, get_current_price
    from dash import callback_context

    symbol    = symbol    or "XAUUSD"
    timeframe = timeframe or "1m"
    period    = TF_PERIOD.get(timeframe, "1d")
    cache_key = f"{symbol}_{timeframe}"

    # Fetch data
    df = fetch_yahoo_finance_data(symbol, period=period, interval=timeframe)

    blank_fig = go.Figure(layout=dict(
        paper_bgcolor=C["bg"], plot_bgcolor=C["surface"],
        font=dict(color=C["text"]),
        annotations=[dict(text="Insufficient data", x=0.5, y=0.5,
                          showarrow=False, font=dict(color=C["muted"]))],
    ))

    if df is None or len(df) < 80:
        err = html.Div("Insufficient data — try a wider period.",
                       style={"color": C["danger"], "fontSize": "11px"})
        return blank_fig, blank_fig, blank_fig, [], "N/A", err, err, err

    # Skip full pipeline rebuild when triggered by interval and data hasn't changed
    triggered = callback_context.triggered_id if callback_context.triggered else None
    last_close = float(df["close"].iloc[-1])
    last_ts    = str(df["timestamp"].iloc[-1])
    data_sig   = (last_close, last_ts)
    if triggered == "smma-refresh-btn":
        # Force rebuild: clear the saved hash so interval also re-renders after manual refresh
        _last_data_hash.pop(cache_key, None)
    elif triggered == "smma-interval" and _last_data_hash.get(cache_key) == data_sig:
        return no_update
    _last_data_hash[cache_key] = data_sig

    # Cap data to 400 candles for chart performance — enough for all indicators (SMMA-75 needs 75)
    if len(df) > 400:
        df = df.iloc[-400:].reset_index(drop=True)

    # Run full strategy pipeline
    r = _run_strategy(df)

    try:
        price = get_current_price(symbol)
    except Exception:
        price = float(df["close"].iloc[-1])

    sig   = r["signal"]
    sc    = SIG_COL.get(sig, C["text"])
    tc    = (C["accent"] if r["trend"] == "UP" else
             C["danger"] if r["trend"] == "DOWN" else C["warn"])

    # ── Figures ───────────────────────────────────────────────────────────────
    fig_price     = _fig_price(df, r, symbol, timeframe)
    fig_orderbook = _fig_orderbook(df, r, price, symbol, timeframe)
    fig_delta     = _fig_delta(df, symbol, timeframe)

    # ── Stat strip ────────────────────────────────────────────────────────────
    v3  = float(r["smma3"].iloc[-1])  if r["smma3"]  is not None else 0
    v9  = float(r["smma9"].iloc[-1])  if r["smma9"]  is not None else 0
    v40 = float(r["smma40"].iloc[-1]) if r["smma40"] is not None else 0
    v75 = float(r["smma75"].iloc[-1]) if r["smma75"] is not None else 0

    mfi_txt = f"{r['mfi']:.0f}" if r["mfi"] is not None else "N/A"
    atr_txt = f"{r['atr']:.2f} ({r['atr_pct']:.2f}%)" if r["atr"] else "N/A"
    poc_txt = f"{r['vp']['poc_price']:,.2f}" if r["vp"] else "N/A"

    stat_strip = [
        _stat("SIGNAL",     sig,    sc,    r["strength"]),
        _stat("TREND",      r["trend"], tc, "SMMA+Hurst"),
        _stat("HURST",      f"{r['hurst']:.3f}", C["accent"] if r["hurst"]>0.55 else C["warn"],
              f"conf {r['hurst_conf']:.0%}"),
        _stat("MFI (14)",   mfi_txt,
              C["danger"] if r["mfi"] and r["mfi"]>80 else
              C["accent"] if r["mfi"] and r["mfi"]<20 else C["text"], "Vol-wtd RSI"),
        _stat("ATR (14)",   atr_txt, C["accent"] if r["atr_regime"]=="EXPANDING" else C["warn"],
              r["atr_regime"]),
        _stat("POC",        poc_txt, C["warn"],    "Order anchor"),
        _stat("SMMA-3",     f"{v3:,.2f}",  SMMA_COL["smma3"],  "Fast"),
        _stat("SMMA-75",    f"{v75:,.2f}", SMMA_COL["smma75"], "Anchor"),
    ]

    # ── Live price ────────────────────────────────────────────────────────────
    price_html = f"{symbol}  {price:,.3f}  |  {timeframe}"

    # ── Verification panel ────────────────────────────────────────────────────
    verif = _verification_panel(r, price)

    # ── Multi-TF Hurst ────────────────────────────────────────────────────────
    mtf_rows = []
    for tf, tp in [("1m","1d"), ("5m","5d"), ("15m","5d")]:
        try:
            d2 = fetch_yahoo_finance_data(symbol, period=tp, interval=tf)
            if d2 is not None and len(d2) >= 30:
                ret2 = d2["close"].pct_change().dropna().values
                H2, Hc2 = _hurst(ret2, min_lag=5, max_lag=min(40, len(ret2)//3))
            else:
                H2, Hc2 = 0.5, 0.0
        except Exception:
            H2, Hc2 = 0.5, 0.0

        hc2 = (C["accent"] if H2>0.55 else C["info"] if H2<0.45 else C["warn"])
        ht2 = ("TRENDING" if H2>0.55 else "MEAN-REV" if H2<0.45 else "NEUTRAL")
        cur = tf == timeframe
        mtf_rows.append(dbc.Row([
            dbc.Col(html.Span(f"{'►' if cur else ' '} {tf}",
                              style={"color": C["accent"] if cur else C["muted"],
                                     "fontSize": "10px", "fontWeight": "bold"}), width=2),
            dbc.Col(html.Span(f"H={H2:.3f}",
                              style={"color": hc2, "fontSize": "10px",
                                     "fontWeight": "bold"}), width=3),
            dbc.Col(dbc.Progress(value=int(H2*100),
                                 color="success" if H2>0.55 else
                                       "info" if H2<0.45 else "warning",
                                 style={"height": "5px"}), width=4),
            dbc.Col(html.Span(ht2, style={"color": hc2, "fontSize": "9px"}), width=3),
        ], className="mb-2", align="center"))

    mtf_html = html.Div([
        dbc.Row([
            dbc.Col(html.Span("TF",     style={"color":C["muted"],"fontSize":"9px"}), width=2),
            dbc.Col(html.Span("H",      style={"color":C["muted"],"fontSize":"9px"}), width=3),
            dbc.Col(html.Span("Scale",  style={"color":C["muted"],"fontSize":"9px"}), width=4),
            dbc.Col(html.Span("Regime", style={"color":C["muted"],"fontSize":"9px"}), width=3),
        ], className="mb-1"),
        html.Hr(style={"borderColor":C["border"],"margin":"4px 0 8px 0"}),
        *mtf_rows,
    ])

    # ── Signal log ────────────────────────────────────────────────────────────
    ts_now = datetime.now().strftime("%H:%M:%S")
    log_lines = [f"[{ts_now}] {symbol} {timeframe}"]
    log_lines += [f"  {d}" for d in r.get("details", [])]

    def _lc(line):
        if "BUY" in line:    return C["accent"]
        if "SELL" in line:   return C["danger"]
        if "FAIL" in line:   return C["warn"]
        return C["muted"]

    log_html = html.Div([
        html.Div(l, style={"color": _lc(l), "marginBottom": "1px"})
        for l in log_lines
    ])

    return (fig_price, fig_orderbook, fig_delta,
            stat_strip, price_html,
            verif, mtf_html, log_html)
