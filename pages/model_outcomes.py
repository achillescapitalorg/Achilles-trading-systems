"""
Model Outcomes Dashboard
========================
Shows trained model metrics, predictions, and ensemble signals.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import json
from pathlib import Path

from app import COLORS
from beta_testing.config import PROCESSED_DIR

MODEL_DIR = PROCESSED_DIR / "models"
RESULTS_FILE = MODEL_DIR / "gold_1m_full_results.json"
FEATURES_FILE = MODEL_DIR / "gold_1m_full_features.json"


def _load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


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
        style={
            "backgroundColor": COLORS["surface"],
            "border": f"1px solid {COLORS['border']}",
        },
    )


layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": COLORS["background"], "minHeight": "100vh", "padding": "20px"},
    children=[
        html.H2(
            "Beta Testing — Model Outcomes",
            style={"color": COLORS["accent"], "marginBottom": "20px"},
        ),
        html.P(
            "Trained model performance on 1-minute gold data (full 21.5-year dataset).",
            style={"color": COLORS["text"]},
        ),

        # Status banner
        dbc.Alert(
            id="model-status-banner",
            color="warning",
            children="Models not yet trained. Training in progress...",
        ),

        # ── Model Metrics Cards ──────────────────────────────────────────────
        html.H4("Performance Metrics", style={"color": COLORS["accent"], "marginTop": "20px"}),
        dbc.Row(
            [
                dbc.Col(_metric_card("LightGBM AUC", "—", "Directional accuracy", "info"), width=3),
                dbc.Col(_metric_card("XGBoost AUC", "—", "Directional accuracy", "info"), width=3),
                dbc.Col(_metric_card("Random Forest AUC", "—", "Directional accuracy", "info"), width=3),
                dbc.Col(_metric_card("Ensemble AUC", "—", "Weighted ensemble", "accent"), width=3),
            ],
            className="g-3",
            id="model-auc-row",
        ),
        dbc.Row(
            [
                dbc.Col(_metric_card("LGB Accuracy", "—", "Test set", "text"), width=3),
                dbc.Col(_metric_card("XGB Accuracy", "—", "Test set", "text"), width=3),
                dbc.Col(_metric_card("RF Accuracy", "—", "Test set", "text"), width=3),
                dbc.Col(_metric_card("Ensemble Accuracy", "—", "Test set", "accent"), width=3),
            ],
            className="g-3",
            style={"marginTop": "10px"},
            id="model-acc-row",
        ),

        # ── Signal Generator ─────────────────────────────────────────────────
        html.H4("Live Signal Generator", style={"color": COLORS["accent"], "marginTop": "30px"}),
        dbc.Row(
            dbc.Col(
                dbc.Button(
                    "Refresh Signal",
                    id="model-refresh-signal",
                    color="primary",
                    size="sm",
                ),
                width="auto",
            ),
            style={"marginTop": "10px"},
        ),
        html.Div(id="model-signal-output", style={"marginTop": "15px"}),

        # ── Feature Importance ───────────────────────────────────────────────
        html.H4("Top Features", style={"color": COLORS["accent"], "marginTop": "30px"}),
        html.Div(id="model-feature-output"),

        # Auto-refresh every 30s
        dcc.Interval(id="model-interval", interval=30_000),
    ],
)


@callback(
    Output("model-status-banner", "children"),
    Output("model-status-banner", "color"),
    Output("model-auc-row", "children"),
    Output("model-acc-row", "children"),
    Input("model-interval", "n_intervals"),
    Input("model-refresh-signal", "n_clicks"),
    prevent_initial_call=False,
)
def refresh_metrics(_n_intervals, _n_clicks):
    results = _load_results()

    if not results:
        return (
            "Models not yet trained. Check back soon.",
            "warning",
            [dbc.Col(_metric_card("—", "—", "Waiting for training...", "text"), width=3) for _ in range(4)],
            [dbc.Col(_metric_card("—", "—", "Waiting for training...", "text"), width=3) for _ in range(4)],
        )

    banner = "Models trained successfully on full dataset."
    banner_color = "success"

    auc_row = [
        dbc.Col(_metric_card("LightGBM AUC", f"{results.get('lgb', {}).get('auc', 0):.4f}", "Test set", "info"), width=3),
        dbc.Col(_metric_card("XGBoost AUC", f"{results.get('xgb', {}).get('auc', 0):.4f}", "Test set", "info"), width=3),
        dbc.Col(_metric_card("Random Forest AUC", f"{results.get('rf', {}).get('auc', 0):.4f}", "Test set", "info"), width=3),
        dbc.Col(_metric_card("Ensemble AUC", f"{results.get('ensemble', {}).get('auc', 0):.4f}", "Weighted 50/30/20", "accent"), width=3),
    ]

    acc_row = [
        dbc.Col(_metric_card("LGB Accuracy", f"{results.get('lgb', {}).get('accuracy', 0):.2%}", "Test set", "text"), width=3),
        dbc.Col(_metric_card("XGB Accuracy", f"{results.get('xgb', {}).get('accuracy', 0):.2%}", "Test set", "text"), width=3),
        dbc.Col(_metric_card("RF Accuracy", f"{results.get('rf', {}).get('accuracy', 0):.2%}", "Test set", "text"), width=3),
        dbc.Col(_metric_card("Ensemble Accuracy", f"{results.get('ensemble', {}).get('accuracy', 0):.2%}", "Test set", "accent"), width=3),
    ]

    return banner, banner_color, auc_row, acc_row


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
            return dbc.Alert("Feature importance not available yet.", color="secondary")
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
            bordered=True,
            size="sm",
            style={"color": COLORS["text"], "backgroundColor": COLORS["surface"]},
        )
    except Exception as exc:
        return dbc.Alert(f"Feature loading error: {exc}", color="warning")
