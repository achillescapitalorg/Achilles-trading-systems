"""
Train Beta Models Locally
=========================
Trains LightGBM, XGBoost, and Random Forest on local gold_2025_2026.csv
for use in the Beta Dashboard live signal display.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from beta_testing.features import compute_1m_features
from beta_testing.models.lgb_model import Gold1mLightGBM
from beta_testing.models.xgb_model import Gold1mXGBoost
from beta_testing.models.rf_model import Gold1mRandomForest

DATA_PATH = Path("data/beta_testing/processed/gold_2025_2026.csv")
SAVE_DIR = Path("data/beta_testing/processed/models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data(df: pd.DataFrame, horizon: int):
    """Prepare features and binary target for a given horizon."""
    print(f"\n=== Preparing data for H={horizon} ===", flush=True)
    features = compute_1m_features(df)

    # Add H=60 targets if not present
    if horizon == 60 and 'target_dir_60' not in features.columns:
        close = df['close']
        features['target_ret_60'] = close.pct_change(60).shift(-60)
        features['target_dir_60'] = np.sign(features['target_ret_60'])

    target_col = f'target_dir_{horizon}'
    if target_col not in features.columns:
        raise ValueError(f"Target column {target_col} not found. Available: {[c for c in features.columns if 'target' in c]}")

    target_cols = [c for c in features.columns if c.startswith('target_')]
    y = features[target_col].values
    X = features.drop(columns=target_cols)

    # Binary target: 1 = up, 0 = down (random assignment for zeros)
    y_binary = np.where(y > 0, 1, 0)
    zero_mask = y == 0
    if zero_mask.sum() > 0:
        y_binary[zero_mask] = np.random.RandomState(42).randint(0, 2, size=zero_mask.sum())

    # Temporal split: 80/10/10
    n = len(X)
    split1 = int(n * 0.80)
    split2 = int(n * 0.90)

    X_train, y_train = X.iloc[:split1], y_binary[:split1]
    X_val, y_val = X.iloc[split1:split2], y_binary[split1:split2]
    X_test, y_test = X.iloc[split2:], y_binary[split2:]

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}", flush=True)
    print(f"Train up-ratio: {y_train.mean():.3f} | Test up-ratio: {y_test.mean():.3f}", flush=True)

    return X_train, y_train, X_val, y_val, X_test, y_test, list(X.columns)


def evaluate_model(name, model, X_test, y_test):
    """Compute full metrics for a model."""
    preds_proba = model.predict(X_test)
    preds = (preds_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_proba)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    # Confident predictions (threshold 0.6)
    conf_mask = np.abs(preds_proba - 0.5) > 0.1  # > 0.6 or < 0.4
    if conf_mask.sum() > 0:
        conf_acc = accuracy_score(y_test[conf_mask], preds[conf_mask])
        conf_pct = conf_mask.mean()
    else:
        conf_acc = 0.0
        conf_pct = 0.0

    results = {
        "accuracy": float(acc),
        "auc": float(auc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "conf_acc": float(conf_acc),
        "conf_pct": float(conf_pct),
    }

    print(f"\n{name}:")
    print(f"  Accuracy:  {acc:.4f}", flush=True)
    print(f"  AUC:       {auc:.4f}", flush=True)
    print(f"  Precision: {prec:.4f}", flush=True)
    print(f"  Recall:    {rec:.4f}", flush=True)
    print(f"  F1:        {f1:.4f}", flush=True)
    print(f"  Conf@0.6:  {conf_acc:.4f} ({conf_pct:.2%} coverage)", flush=True)

    return results


def train_and_save(horizon: int):
    """Train all models for a horizon and save."""
    print(f"\n{'='*60}")
    print(f"TRAINING HORIZON = {horizon}")
    print(f"{'='*60}")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.set_index("date").sort_index()

    # Keep only original-ish bars for cleaner training
    df = df.drop(columns=[col for col in ["is_original", "minutes_since_last_bar"] if col in df.columns], errors="ignore")

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_data(df, horizon)

    models = {}
    results = {}

    # LightGBM — faster local params
    print("\n--- Training LightGBM ---", flush=True)
    lgb = Gold1mLightGBM(params={
        'learning_rate': 0.05,
        'num_leaves': 15,
        'max_depth': 4,
        'min_data_in_leaf': 500,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
    })
    lgb.fit(X_train, y_train, X_val, y_val, num_boost_round=500)
    models["lgb"] = lgb
    results["lgb"] = evaluate_model("LightGBM", lgb, X_test, y_test)

    # XGBoost — faster local params
    print("\n--- Training XGBoost ---", flush=True)
    xgb = Gold1mXGBoost(params={
        'learning_rate': 0.05,
        'max_depth': 4,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
    })
    xgb.fit(X_train, y_train, X_val, y_val)
    models["xgb"] = xgb
    results["xgb"] = evaluate_model("XGBoost", xgb, X_test, y_test)

    # Random Forest
    print("\n--- Training Random Forest ---", flush=True)
    rf = Gold1mRandomForest()
    rf.fit(X_train, y_train)
    models["rf"] = rf
    results["rf"] = evaluate_model("RandomForest", rf, X_test, y_test)

    # Ensemble (simple average)
    print("\n--- Ensemble (50/30/20) ---")
    ens_proba = 0.5 * models["lgb"].predict(X_test) + 0.3 * models["xgb"].predict(X_test) + 0.2 * models["rf"].predict(X_test)
    ens_preds = (ens_proba > 0.5).astype(int)
    ens_acc = accuracy_score(y_test, ens_preds)
    ens_auc = roc_auc_score(y_test, ens_proba)
    results["ensemble"] = {"accuracy": float(ens_acc), "auc": float(ens_auc)}
    print(f"  Accuracy: {ens_acc:.4f}")
    print(f"  AUC:      {ens_auc:.4f}")

    # Save models
    prefix = f"beta_h{horizon}"
    for name, model in models.items():
        path = SAVE_DIR / f"{prefix}_{name}.pkl"
        model.save(str(path))
        print(f"  Saved {name} -> {path}")

    # Save results
    results_path = SAVE_DIR / f"{prefix}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results -> {results_path}")

    # Save feature columns
    feat_path = SAVE_DIR / f"{prefix}_features.json"
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f)

    # Save last 500 rows of test data for quick signal generation
    last_data = X_test.tail(500).copy()
    last_data["actual"] = y_test[-500:]
    last_data.to_csv(SAVE_DIR / f"{prefix}_last500.csv")

    return models, results


if __name__ == "__main__":
    for h in [20, 60]:
        train_and_save(h)
    print("\n✅ All models trained and saved!")
