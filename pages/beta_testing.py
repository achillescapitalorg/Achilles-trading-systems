"""
Beta Testing Page — Data Pipeline + Model Outcomes Dashboard
=============================================================
Shows data status, model metrics, predictions, and ensemble signals.
"""
import sys
import os
import json
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

from app import COLORS
from beta_testing import BetaDataLoader, KaggleGoldDownloader, DukascopyGoldDownloader
from beta_testing.data_validator import validate_gold_data
from beta_testing.config import PROCESSED_DIR

MODEL_DIR = PROCESSED_DIR / "models"
RESULTS_FILE = MODEL_DIR / "gold_1m_full_results.json"

# ── Singleton loader ─────────────────────────────────────────────────────────
_loader = BetaDataLoader()


# ── Helper: status card for data source ──────────────────────────────────────
def _source_card(source: str):
    is_kaggle = source == "kaggle"
    title = "Kaggle Dataset" if is_kaggle else "Dukascopy Tick Data"
    icon = "Data" if is_kaggle else "Tick"
    desc = (
        "21.5 years of 1-minute OHLCV (2004-2026). Free, public domain."
        if is_kaggle
        else "Institutional tick-level data with bid/ask. Free from Swiss ECN."
    )
    btn_id = "beta-dl-kaggle" if is_kaggle else "beta-dl-dukascopy"
    store_id = f"beta-{source}-status"

    return dbc.Card(
        [
            dbc.CardHeader(
                html.H5(f"{icon} {title}", style={"color": COLORS["accent"], "margin": 0}),
                style={"backgroundColor": COLORS["surface"], "borderBottom": f"1px solid {COLORS['border']}"},
            ),
            dbc.CardBody(
                [
                    html.P(desc, style={"color": COLORS["text"], "fontSize": "12px"}),
                    html.Div(id=store_id, children=_status_badge(False, "Checking...")),
                    html.Hr(style={"borderColor": COLORS["border"]}),
                    dbc.Button(
                        "Check / Download Instructions",
                        id=btn_id,
                        color="primary",
                        size="sm",
                        style={"width": "100%"},
                    ),
                ]
            ),
        ],
        style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}"},
    )


def _status_badge(available: bool, detail: str):
    color = "success" if available else "warning"
    text = "READY" if available else "NOT READY"
    return dbc.Badge(
        f"{text} — {detail}",
        color=color,
        className="p-2",
        style={"fontSize": "12px", "width": "100%", "textAlign": "left"},
    )


def _metric_card(title, value, subtitle, color="accent"):
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H6(title, style={"color": COLORS["text"], "marginBottom": "4px"}),
                    html.H3(value, style={"color": COLORS[color], "margin": 0}),
                    html.Small(subtitle, style={"color": COLORS["text"]}),
                ]
            )
        ],
        style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}"},
    )


def _load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


# ── Layout ───────────────────────────────────────────────────────────────────
layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": COLORS["background"], "minHeight": "100vh", "padding": "20px"},
    children=[
        html.H2(
            "Beta Testing — Gold 1M Data & Models",
            style={"color": COLORS["accent"], "marginBottom": "20px"},
        ),

        # ── Section 1: Data Pipeline ─────────────────────────────────────────
        html.H4("Data Pipeline Status", style={"color": COLORS["accent"], "marginTop": "10px"}),
        dbc.Row(
            [
                dbc.Col(_source_card("kaggle"), width=6),
                dbc.Col(_source_card("dukascopy"), width=6),
            ],
            className="g-3",
            style={"marginTop": "10px"},
        ),
        dbc.Row(
            dbc.Col(
                dbc.ButtonGroup(
                    [
                        dbc.Button("Refresh Status", id="beta-refresh-btn", color="secondary", size="sm"),
                        dbc.Button("Validate Data", id="beta-validate-btn", color="info", size="sm"),
                    ]
                ),
                width="auto",
            ),
            style={"marginTop": "20px"},
        ),
        dcc.Loading(
            id="beta-loading",
            type="circle",
            color=COLORS["accent"],
            children=html.Div(id="beta-validation-output", style={"marginTop": "20px"}),
        ),

        html.Hr(style={"borderColor": COLORS["border"], "marginTop": "30px"}),

        # ── Section 2: Model Outcomes ────────────────────────────────────────
        html.H4("Model Training Outcomes", style={"color": COLORS["accent"], "marginTop": "10px"}),
        dbc.Alert(
            id="model-status-banner",
            color="warning",
            children="Models not yet trained. Training in progress on full 21.5-year dataset...",
        ),

        html.H5("Accuracy", style={"color": COLORS["text"], "marginTop": "15px"}),
        dbc.Row(
            [
                dbc.Col(_metric_card("LightGBM", "—", "Test accuracy", "info"), width=3),
                dbc.Col(_metric_card("XGBoost", "—", "Test accuracy", "info"), width=3),
                dbc.Col(_metric_card("Random Forest", "—", "Test accuracy", "info"), width=3),
                dbc.Col(_metric_card("Ensemble", "—", "Weighted 50/30/20", "accent"), width=3),
            ],
            className="g-3",
            id="model-acc-row",
        ),

        html.H5("AUC Score", style={"color": COLORS["text"], "marginTop": "15px"}),
        dbc.Row(
            [
                dbc.Col(_metric_card("LightGBM", "—", "ROC-AUC", "info"), width=3),
                dbc.Col(_metric_card("XGBoost", "—", "ROC-AUC", "info"), width=3),
                dbc.Col(_metric_card("Random Forest", "—", "ROC-AUC", "info"), width=3),
                dbc.Col(_metric_card("Ensemble", "—", "ROC-AUC", "accent"), width=3),
            ],
            className="g-3",
            id="model-auc-row",
        ),

        html.H5("LogLoss", style={"color": COLORS["text"], "marginTop": "15px"}),
        dbc.Row(
            [
                dbc.Col(_metric_card("LightGBM", "—", "Cross-entropy", "text"), width=3),
                dbc.Col(_metric_card("XGBoost", "—", "Cross-entropy", "text"), width=3),
                dbc.Col(_metric_card("Random Forest", "—", "Cross-entropy", "text"), width=3),
                dbc.Col(_metric_card("Baseline", "—", "Random guess", "text"), width=3),
            ],
            className="g-3",
            id="model-loss-row",
        ),

        html.H5("Top Feature Importance (LightGBM)", style={"color": COLORS["text"], "marginTop": "20px"}),
        html.Div(id="model-feature-output"),

        # Intervals
        dcc.Interval(id="beta-interval", interval=10_000),
        dcc.Interval(id="model-interval", interval=30_000),
    ],
)


# ── Callbacks: Data Pipeline ─────────────────────────────────────────────────

def _fmt_status(d: dict) -> html.Div:
    avail = d.get("available", False)
    if avail:
        size = d.get("size_mb", 0)
        rows = d.get("rows", "—")
        detail = f"{size} MB | {rows} rows"
    else:
        detail = "File not found. Click button for instructions."
    return _status_badge(avail, detail)


@callback(
    Output("beta-kaggle-status", "children"),
    Output("beta-dukascopy-status", "children"),
    Input("beta-interval", "n_intervals"),
    Input("beta-refresh-btn", "n_clicks"),
    prevent_initial_call=False,
)
def refresh_status(_n_intervals, _n_clicks):
    k = _loader.kaggle.get_status()
    d = _loader.dukascopy.get_status()
    if k["available"]:
        try:
            df = _loader.load_kaggle()
            k["rows"] = f"{len(df):,}"
        except Exception:
            k["rows"] = "unknown"
    if d["available"]:
        try:
            df = _loader.load_dukascopy()
            d["rows"] = f"{len(df):,}"
        except Exception:
            d["rows"] = "unknown"
    return _fmt_status(k), _fmt_status(d)


@callback(
    Output("beta-validation-output", "children"),
    Input("beta-validate-btn", "n_clicks"),
    prevent_initial_call=True,
)
def run_validation(_n_clicks):
    try:
        df = _loader.load_unified()
        if df is None:
            return dbc.Alert("No data available. Please download data first.", color="warning")
        report = validate_gold_data(df)
        rows = []
        for key, value in report.items():
            if key == "yearly_coverage":
                val_str = " | ".join([f"{yr}: {cnt:,}" for yr, cnt in list(value.items())[:5]])
                if len(value) > 5:
                    val_str += f" ... ({len(value) - 5} more years)"
            elif isinstance(value, float):
                val_str = f"{value:.2f}"
            else:
                val_str = str(value)
            rows.append(html.Tr([html.Td(key, style={"color": COLORS["text"]}), html.Td(val_str, style={"color": COLORS["accent"]})]))
        score = report.get("quality_score", 0)
        rating = report.get("quality_rating", "UNKNOWN")
        score_color = "success" if score >= 90 else "info" if score >= 70 else "warning" if score >= 50 else "danger"
        return html.Div(
            [
                dbc.Alert(f"Quality Score: {score}/100 — {rating}", color=score_color, style={"fontWeight": "bold"}),
                dbc.Table(
                    [html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])), html.Tbody(rows)],
                    bordered=True, hover=True, size="sm",
                    style={"color": COLORS["text"], "backgroundColor": COLORS["surface"]},
                ),
            ]
        )
    except Exception as exc:
        return dbc.Alert(f"Validation error: {exc}", color="danger")


@callback(
    Output("beta-validation-output", "children", allow_duplicate=True),
    Input("beta-dl-kaggle", "n_clicks"),
    prevent_initial_call=True,
)
def kaggle_instructions(_n_clicks):
    return dbc.Alert(
        [
            html.H6("Kaggle Download Instructions"),
            html.Ol(
                [
                    html.Li("Create a free account at https://www.kaggle.com"),
                    html.Li(
                        "Visit: ",
                        html.A(
                            "XAU/USD Gold Price Historical Data",
                            href="https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024",
                            target="_blank",
                        ),
                    ),
                    html.Li("Click Download and extract the ZIP"),
                    html.Li(f"Place XAU_1m_data.csv in: {_loader.kaggle.output_dir}"),
                ]
            ),
        ],
        color="info",
        dismissable=True,
    )


@callback(
    Output("beta-validation-output", "children", allow_duplicate=True),
    Input("beta-dl-dukascopy", "n_clicks"),
    prevent_initial_call=True,
)
def dukascopy_instructions(_n_clicks):
    return dbc.Alert(
        [
            html.H6("Dukascopy Download Instructions"),
            html.P("Dukascopy provides free tick-level data via HTTP download."),
            html.Pre(
                "from beta_testing import DukascopyGoldDownloader\n\n"
                "dl = DukascopyGoldDownloader()\n"
                "ticks = dl.download_range(start=datetime(2020,1,1), end=datetime(2024,1,1))\n"
                "dl.build_1m_csv(ticks)",
                style={"backgroundColor": "#1a1a1a", "padding": "10px", "borderRadius": "4px"},
            ),
        ],
        color="info",
        dismissable=True,
    )


# ── Callbacks: Model Outcomes ────────────────────────────────────────────────

@callback(
    Output("model-status-banner", "children"),
    Output("model-status-banner", "color"),
    Output("model-acc-row", "children"),
    Output("model-auc-row", "children"),
    Output("model-loss-row", "children"),
    Input("model-interval", "n_intervals"),
    prevent_initial_call=False,
)
def refresh_model_metrics(_n_intervals):
    results = _load_results()

    if not results:
        empty = [dbc.Col(_metric_card("—", "—", "Waiting...", "text"), width=3) for _ in range(4)]
        return (
            "Training in progress on full 21.5-year dataset. This may take 1-2 hours.",
            "warning",
            empty, empty, empty,
        )

    banner = f"Models trained on full dataset. Test period: {results.get('test_period', '2022-2026')}"
    banner_color = "success"

    acc_row = [
        dbc.Col(_metric_card("LightGBM", f"{results.get('lgb', {}).get('accuracy', 0):.2%}", "Test accuracy", "info"), width=3),
        dbc.Col(_metric_card("XGBoost", f"{results.get('xgb', {}).get('accuracy', 0):.2%}", "Test accuracy", "info"), width=3),
        dbc.Col(_metric_card("Random Forest", f"{results.get('rf', {}).get('accuracy', 0):.2%}", "Test accuracy", "info"), width=3),
        dbc.Col(_metric_card("Ensemble", f"{results.get('ensemble', {}).get('accuracy', 0):.2%}", "Weighted 50/30/20", "accent"), width=3),
    ]

    auc_row = [
        dbc.Col(_metric_card("LightGBM", f"{results.get('lgb', {}).get('auc', 0):.4f}", "ROC-AUC", "info"), width=3),
        dbc.Col(_metric_card("XGBoost", f"{results.get('xgb', {}).get('auc', 0):.4f}", "ROC-AUC", "info"), width=3),
        dbc.Col(_metric_card("Random Forest", f"{results.get('rf', {}).get('auc', 0):.4f}", "ROC-AUC", "info"), width=3),
        dbc.Col(_metric_card("Ensemble", f"{results.get('ensemble', {}).get('auc', 0):.4f}", "ROC-AUC", "accent"), width=3),
    ]

    loss_row = [
        dbc.Col(_metric_card("LightGBM", f"{results.get('lgb', {}).get('logloss', 0):.4f}", "LogLoss", "text"), width=3),
        dbc.Col(_metric_card("XGBoost", f"{results.get('xgb', {}).get('logloss', 0):.4f}", "LogLoss", "text"), width=3),
        dbc.Col(_metric_card("Random Forest", f"{results.get('rf', {}).get('logloss', 0):.4f}", "LogLoss", "text"), width=3),
        dbc.Col(_metric_card("Baseline", "0.693", "Random guess", "text"), width=3),
    ]

    return banner, banner_color, acc_row, auc_row, loss_row


@callback(
    Output("model-feature-output", "children"),
    Input("model-interval", "n_intervals"),
    prevent_initial_call=False,
)
def refresh_features(_n_intervals):
    try:
        from beta_testing.models import Gold1mLightGBM
        model = Gold1mLightGBM()
        model_path = MODEL_DIR / "gold_1m_full_lgb.pkl"
        if not model_path.exists():
            return dbc.Alert("Feature importance will appear after training completes.", color="secondary")
        model.load(str(model_path))
        imp = model.get_importance().head(15)

        rows = []
        for feat, score in imp.items():
            pct = score / imp.sum() * 100
            rows.append(
                html.Tr([
                    html.Td(feat, style={"color": COLORS["text"]}),
                    html.Td(f"{score:.0f}", style={"color": COLORS["accent"]}),
                    html.Td(
                        dbc.Progress(value=pct, color="success", style={"height": "8px"}),
                        style={"width": "150px"},
                    ),
                ])
            )

        return dbc.Table(
            [html.Thead(html.Tr([html.Th("Feature"), html.Th("Importance"), html.Th("")]))]
            + [html.Tbody(rows)],
            bordered=True, size="sm",
            style={"color": COLORS["text"], "backgroundColor": COLORS["surface"]},
        )
    except Exception as exc:
        return dbc.Alert(f"Feature loading error: {exc}", color="warning")
