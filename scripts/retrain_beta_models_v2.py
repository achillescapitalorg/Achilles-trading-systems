"""
Retrain Beta Models v2
======================
Fixed training pipeline with:
- Time-decay sample weighting (recent data matters more)
- Noise filtering: only train on |return| > 0.1%
- No random zero-bar assignment
- Deeper models (max_depth=10, more leaves)
- Class weights for gold's upward drift
- 30-day test set (not 2 weeks) for robust validation

Run: source venv/Scripts/activate && python scripts/retrain_beta_models_v2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from beta_testing.features import compute_1m_features
from beta_testing.models.lgb_model import Gold1mLightGBM
from beta_testing.models.xgb_model import Gold1mXGBoost
from beta_testing.models.rf_model import Gold1mRandomForest

# ─── CONFIG ───
DATA_PATH = Path("data/beta_testing/processed/gold_unified_16m.csv")
SAVE_DIR = Path("data/beta_testing/processed/models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [20, 60]
NOISE_THRESHOLD = 0.001  # 0.1%
TIME_DECAY_HALF_LIFE_DAYS = 60  # Recent data weighted 2x more than 60-day-old


def compute_time_weights(dates: pd.DatetimeIndex, half_life_days: float = 60.0) -> np.ndarray:
    """Exponential decay weights. Recent = 1.0, half_life ago = 0.5"""
    now = dates.max()
    days_ago = (now - dates).total_seconds().values / 86400.0
    weights = np.exp(-np.log(2) * days_ago / half_life_days)
    return weights


def prepare_data(df: pd.DataFrame, horizon: int):
    """Prepare features, targets, and time-decay weights."""
    print(f"\n{'='*60}")
    print(f"Preparing data for H={horizon}")
    print(f"{'='*60}")

    features = compute_1m_features(df)

    # Ensure target exists
    ret_col = f'target_ret_{horizon}'
    dir_col = f'target_dir_{horizon}'
    if ret_col not in features.columns:
        close = df['close']
        features[ret_col] = close.pct_change(horizon).shift(-horizon)
        features[dir_col] = np.sign(features[ret_col])

    target_cols = [c for c in features.columns if c.startswith('target_')]
    y_ret = features[ret_col]
    y_dir = features[dir_col]

    X = features.drop(columns=target_cols)
    feat_cols = list(X.columns)

    # Binary target: 1=up, 0=down
    y_binary = np.where(y_ret > 0, 1, 0)

    # ─── CRITICAL FIX: Filter noise ───
    # Only train on bars where |return| > threshold
    strong_mask = y_ret.abs() > NOISE_THRESHOLD
    n_total = len(features)
    n_strong = strong_mask.sum()
    print(f"  Total bars: {n_total:,}")
    print(f"  Strong moves (|ret| > {NOISE_THRESHOLD*100:.2f}%): {n_strong:,} ({n_strong/n_total:.1%})")
    print(f"  Noise bars excluded: {n_total - n_strong:,}")

    # Temporal split: last 30 days = test, 30 days before = val, rest = train
    last_date = features.index.max()
    test_cutoff = last_date - timedelta(days=30)
    val_cutoff = test_cutoff - timedelta(days=30)

    train_mask = features.index < val_cutoff
    val_mask = (features.index >= val_cutoff) & (features.index < test_cutoff)
    test_mask = features.index >= test_cutoff

    # For training: use ONLY strong moves
    train_strong_mask = train_mask & strong_mask

    X_train = X[train_strong_mask]
    y_train = y_binary[train_strong_mask]
    train_dates = features.index[train_strong_mask]

    X_val = X[val_mask]
    y_val = y_binary[val_mask]

    X_test = X[test_mask]
    y_test = y_binary[test_mask]
    y_test_ret = y_ret[test_mask]  # For analysis

    # Time-decay weights for training set
    sample_weights = compute_time_weights(train_dates, TIME_DECAY_HALF_LIFE_DAYS)
    # Normalize so mean weight = 1
    sample_weights = sample_weights * (len(sample_weights) / sample_weights.sum())

    print(f"  Train: {len(X_train):,} (strong moves only, time-weighted)")
    print(f"  Val:   {len(X_val):,}")
    print(f"  Test:  {len(X_test):,}")
    print(f"  Train up-ratio: {y_train.mean():.2%}")
    print(f"  Weight range: {sample_weights.min():.3f} - {sample_weights.max():.3f}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feat_cols, sample_weights, y_test_ret


def evaluate_model(name, model, X_test, y_test, y_test_ret):
    """Compute full metrics."""
    preds_proba = model.predict(X_test)
    preds = (preds_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_proba)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    # Confident predictions (|prob - 0.5| > 0.15  => prob > 0.65 or < 0.35)
    conf_mask = np.abs(preds_proba - 0.5) > 0.15
    if conf_mask.sum() > 0:
        conf_acc = accuracy_score(y_test[conf_mask], preds[conf_mask])
        conf_pct = conf_mask.mean()
    else:
        conf_acc = 0.0
        conf_pct = 0.0

    # Strong-move-only confident accuracy
    strong_mask = y_test_ret.abs() > NOISE_THRESHOLD
    if strong_mask.sum() > 0:
        strong_acc = accuracy_score(y_test[strong_mask], preds[strong_mask])
    else:
        strong_acc = 0.0

    results = {
        "accuracy": float(acc),
        "auc": float(auc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "conf_acc": float(conf_acc),
        "conf_pct": float(conf_pct),
        "strong_acc": float(strong_acc),
    }

    print(f"\n{name}:")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  AUC:          {auc:.4f}")
    print(f"  Precision:    {prec:.4f}")
    print(f"  Recall:       {rec:.4f}")
    print(f"  F1:           {f1:.4f}")
    print(f"  Conf@0.65:    {conf_acc:.4f} ({conf_pct:.2%} coverage)")
    print(f"  Strong Acc:   {strong_acc:.4f}")

    return results


def train_and_save(horizon: int):
    """Train all models for a horizon and save."""
    print(f"\n{'='*60}")
    print(f"TRAINING HORIZON = {horizon}")
    print(f"{'='*60}")

    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run preprocess_and_combine.py first.")
        return None, None

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()
    print(f"Loaded unified data: {len(df):,} rows, {df.index.min()} to {df.index.max()}")

    X_train, y_train, X_val, y_val, X_test, y_test, feat_cols, sample_weights, y_test_ret = prepare_data(df, horizon)

    # Compute class weight for imbalance (gold tends to drift up)
    pos_ratio = y_train.mean()
    scale_pos_weight = (1 - pos_ratio) / (pos_ratio + 1e-9)
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    models = {}
    results = {}

    # ─── LightGBM ───
    print("\n--- Training LightGBM ---")
    lgb = Gold1mLightGBM(params={
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'max_depth': 10,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight,
        'verbose': -1,
    })
    lgb.fit(X_train, y_train, X_val, y_val, sample_weight=sample_weights, num_boost_round=1000)
    models["lgb"] = lgb
    results["lgb"] = evaluate_model("LightGBM", lgb, X_test, y_test, y_test_ret)

    # ─── XGBoost ───
    print("\n--- Training XGBoost ---")
    xgb = Gold1mXGBoost(params={
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.05,
        'max_depth': 10,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'scale_pos_weight': scale_pos_weight,
    })
    xgb.fit(X_train, y_train, X_val, y_val, sample_weight=sample_weights)
    models["xgb"] = xgb
    results["xgb"] = evaluate_model("XGBoost", xgb, X_test, y_test, y_test_ret)

    # ─── Random Forest ───
    print("\n--- Training Random Forest ---")
    rf = Gold1mRandomForest(params={
        'n_estimators': 500,
        'max_depth': 20,
        'min_samples_leaf': 50,
        'class_weight': 'balanced_subsample',
        'n_jobs': -1,
    })
    rf.fit(X_train, y_train, sample_weight=sample_weights)
    models["rf"] = rf
    results["rf"] = evaluate_model("RandomForest", rf, X_test, y_test, y_test_ret)

    # ─── Ensemble (weighted average) ───
    print("\n--- Ensemble (50/30/20) ---")
    ens_proba = (
        0.5 * models["lgb"].predict(X_test) +
        0.3 * models["xgb"].predict(X_test) +
        0.2 * models["rf"].predict(X_test)
    )
    ens_preds = (ens_proba > 0.5).astype(int)
    ens_acc = accuracy_score(y_test, ens_preds)
    ens_auc = roc_auc_score(y_test, ens_proba)
    conf_mask = np.abs(ens_proba - 0.5) > 0.15
    if conf_mask.sum() > 0:
        ens_conf_acc = accuracy_score(y_test[conf_mask], ens_preds[conf_mask])
        ens_conf_pct = conf_mask.mean()
    else:
        ens_conf_acc = 0.0
        ens_conf_pct = 0.0

    results["ensemble"] = {
        "accuracy": float(ens_acc),
        "auc": float(ens_auc),
        "conf_acc": float(ens_conf_acc),
        "conf_pct": float(ens_conf_pct),
    }
    print(f"  Accuracy:     {ens_acc:.4f}")
    print(f"  AUC:          {ens_auc:.4f}")
    print(f"  Conf@0.65:    {ens_conf_acc:.4f} ({ens_conf_pct:.2%} coverage)")

    # ─── Save ───
    prefix = f"beta_h{horizon}"
    for name, model in models.items():
        path = SAVE_DIR / f"{prefix}_{name}.pkl"
        model.save(str(path))
        print(f"  Saved {name} -> {path}")

    # Feature columns
    with open(SAVE_DIR / f"{prefix}_features.json", "w") as f:
        json.dump(feat_cols, f)

    # Results
    with open(SAVE_DIR / f"{prefix}_results_v2.json", "w") as f:
        json.dump(results, f, indent=2)

    return models, results


def main():
    print("=" * 60)
    print("Beta Model Retraining v2")
    print(f"Started: {datetime.now()}")
    print("=" * 60)

    all_results = {}
    for h in HORIZONS:
        models, results = train_and_save(h)
        if results:
            all_results[h] = results

    print(f"\n{'='*60}")
    print("Retraining Complete")
    print(f"Finished: {datetime.now()}")
    print(f"{'='*60}")

    # Summary
    print("\n--- SUMMARY ---")
    for h, res in all_results.items():
        ens = res.get("ensemble", {})
        print(f"H{h}: Acc={ens.get('accuracy', 0):.2%} AUC={ens.get('auc', 0):.3f} Conf@0.65={ens.get('conf_acc', 0):.2%} ({ens.get('conf_pct', 0):.1%} coverage)")


if __name__ == "__main__":
    main()
