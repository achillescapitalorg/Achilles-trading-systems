"""
Retrain LGB/XGB/RF with cross-asset features.
Target: binary direction BUT only train on moves > 0.1% (filter noise).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from sklearn.metrics import accuracy_score

from beta_testing.features import compute_1m_features
from beta_testing.models.lgb_model import Gold1mLightGBM
from beta_testing.models.xgb_model import Gold1mXGBoost
from beta_testing.models.rf_model import Gold1mRandomForest

# ─── CONFIG ───
DATA_PATH = Path("data/beta_testing/processed/gold_2025_2026.csv")
MODEL_DIR = Path("data/beta_testing/processed/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [20, 60]
THRESHOLD = 0.001  # 0.1%

# ─── LOAD & FEATURES ───
print("Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.set_index("date").sort_index()
print(f"Loaded {len(df):,} rows")

print("Computing features...")
features = compute_1m_features(df)
print(f"Features: {features.shape[1]} columns")

# ─── TRAIN / VAL / TEST SPLIT ───
n = len(features)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# ─── TRAINING LOOP ───
for h in HORIZONS:
    print(f"\n{'='*60}")
    print(f"Training H={h}")
    print(f"{'='*60}")

    ret_col = f"target_ret_{h}"
    if ret_col not in features.columns:
        print(f"  Target {ret_col} not found, skipping")
        continue

    # Drop NaN
    valid = features.dropna(subset=[ret_col]).copy()
    feat_cols = [c for c in valid.columns if not c.startswith("target")]
    X = valid[feat_cols]
    ret = valid[ret_col]

    # Binary target: 1=up, 0=down
    y_binary = (ret > 0).astype(int)

    # Filter: only train on bars where |return| > threshold (strong moves)
    strong_mask = ret.abs() > THRESHOLD

    X_strong = X[strong_mask]
    y_strong = y_binary[strong_mask]

    # Temporal split on STRONG bars only
    train_mask = y_strong.index <= y_strong.index[train_end] if len(y_strong) > train_end else pd.Series(True, index=y_strong.index)
    # Use quantile-based split on the strong subset
    n_strong = len(y_strong)
    strong_train_end = int(n_strong * 0.80)
    strong_val_end = int(n_strong * 0.90)

    X_train = X_strong.iloc[:strong_train_end]
    y_train = y_strong.iloc[:strong_train_end]
    X_val = X_strong.iloc[strong_train_end:strong_val_end]
    y_val = y_strong.iloc[strong_train_end:strong_val_end]
    # Test on ALL bars (including noise) to see real performance
    X_test = X.iloc[val_end:]
    y_test = y_binary.iloc[val_end:]

    print(f"  Strong train: {len(X_train):,}")
    print(f"  Strong val:   {len(X_val):,}")
    print(f"  Test (all):   {len(X_test):,}")
    print(f"  Train up-ratio: {y_train.mean():.2%}")

    # ─── LGB ───
    print(f"  Training LGB...")
    lgb = Gold1mLightGBM()
    lgb.fit(X_train, y_train, X_val, y_val, num_boost_round=500)
    lgb_proba = lgb.model.predict(X_test)
    lgb_pred = (lgb_proba > 0.5).astype(int)
    lgb_acc = accuracy_score(y_test, lgb_pred)
    # Confident accuracy (only trade when |prob-0.5| > 0.1)
    conf_mask = np.abs(lgb_proba - 0.5) > 0.1
    lgb_conf_acc = accuracy_score(y_test[conf_mask], lgb_pred[conf_mask]) if conf_mask.sum() > 100 else 0
    print(f"    LGB acc: {lgb_acc:.2%} | Conf@0.1: {lgb_conf_acc:.2%} ({conf_mask.mean():.1%} coverage)")

    # Save as dict wrapper
    import joblib
    joblib.dump({'model': lgb.model, 'params': lgb.params, 'features': feat_cols}, MODEL_DIR / f"beta_h{h}_lgb.pkl")

    # ─── XGB ───
    print(f"  Training XGB...")
    xgb = Gold1mXGBoost()
    xgb.fit(X_train, y_train, X_val, y_val)
    xgb_proba = xgb.predict(X_test)
    xgb_pred = (xgb_proba > 0.5).astype(int)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    conf_mask = np.abs(xgb_proba - 0.5) > 0.1
    xgb_conf_acc = accuracy_score(y_test[conf_mask], xgb_pred[conf_mask]) if conf_mask.sum() > 100 else 0
    print(f"    XGB acc: {xgb_acc:.2%} | Conf@0.1: {xgb_conf_acc:.2%} ({conf_mask.mean():.1%} coverage)")
    xgb.save(MODEL_DIR / f"beta_h{h}_xgb.ubj")

    # ─── RF ───
    print(f"  Training RF...")
    rf = Gold1mRandomForest()
    rf.fit(X_train, y_train)
    rf_proba = rf.predict(X_test)
    rf_pred = (rf_proba > 0.5).astype(int)
    rf_acc = accuracy_score(y_test, rf_pred)
    conf_mask = np.abs(rf_proba - 0.5) > 0.1
    rf_conf_acc = accuracy_score(y_test[conf_mask], rf_pred[conf_mask]) if conf_mask.sum() > 100 else 0
    print(f"    RF acc: {rf_acc:.2%} | Conf@0.1: {rf_conf_acc:.2%} ({conf_mask.mean():.1%} coverage)")
    rf.save(MODEL_DIR / f"beta_h{h}_rf.pkl")

    # ─── Ensemble ───
    ens_proba = (lgb_proba + xgb_proba + rf_proba) / 3.0
    ens_pred = (ens_proba > 0.5).astype(int)
    ens_acc = accuracy_score(y_test, ens_pred)
    conf_mask = np.abs(ens_proba - 0.5) > 0.1
    ens_conf_acc = accuracy_score(y_test[conf_mask], ens_pred[conf_mask]) if conf_mask.sum() > 100 else 0
    print(f"    Ensemble acc: {ens_acc:.2%} | Conf@0.1: {ens_conf_acc:.2%} ({conf_mask.mean():.1%} coverage)")

    # Save feature list
    with open(MODEL_DIR / f"beta_h{h}_features.json", "w") as f:
        json.dump(feat_cols, f)

    # Save results
    results = {
        "horizon": h,
        "threshold": THRESHOLD,
        "n_features": len(feat_cols),
        "lgb": {"accuracy": float(lgb_acc), "conf_acc": float(lgb_conf_acc)},
        "xgb": {"accuracy": float(xgb_acc), "conf_acc": float(xgb_conf_acc)},
        "rf": {"accuracy": float(rf_acc), "conf_acc": float(rf_conf_acc)},
        "ensemble": {"accuracy": float(ens_acc), "conf_acc": float(ens_conf_acc)},
    }
    with open(MODEL_DIR / f"beta_h{h}_results.json", "w") as f:
        json.dump(results, f, indent=2)

print(f"\nDone. Models saved to {MODEL_DIR}")
