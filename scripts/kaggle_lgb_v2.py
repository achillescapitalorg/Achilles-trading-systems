"""
================================================================================
Gold 1-Minute LightGBM Trainer V2
================================================================================
Enhanced version with:
  - Volatility-filtered target (drops uncertain/noise labels)
  - Microstructure features (VWAP, price location, ADX, gaps, etc.)
  - Mutual-information feature selection (top 30)
  - SMOTE class balancing
  - Aggressive hyperparameters optimized for AUC
  - Threshold tuning for max F1 / precision-recall trade-off
  - Full classification reports on train/val/test

Usage on Kaggle:
  1. Upload your dataset (XAU_1m_data.csv or XAU_1m_data_cleaned.csv)
  2. Copy-paste each CELL block into a Kaggle notebook cell
  3. Run all cells
================================================================================
"""

# ==============================================================================
# CELL 1: Imports
# ==============================================================================
import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    precision_score, recall_score, f1_score,
    precision_recall_curve, classification_report,
    confusion_matrix
)
from sklearn.feature_selection import mutual_info_classif

# Optional: SMOTE (install if not available)
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("WARNING: imbalanced-learn not installed. Install via: pip install imbalanced-learn")

warnings.filterwarnings('ignore')

print("=" * 60)
print("Gold 1m LightGBM Trainer V2")
print("=" * 60)

# ==============================================================================
# CELL 2: Configuration
# ==============================================================================
# Tweak these knobs based on your dataset size and runtime budget
CONFIG = {
    'horizon': 20,                    # Bars ahead to predict (was 5, now 20 for more signal)
    'feature_top_k': 30,              # Keep top-K features by mutual information
    'smote_balance': 0.85,            # SMOTE target ratio (1.0 = perfect balance)
    'use_smote': True,
    'drop_neutral_threshold': True,   # Drop labels where |return| < vol_threshold
    'vol_multiplier': 0.5,            # Label threshold = vol_multiplier * rolling_vol
    'early_stopping_rounds': 150,
    'num_boost_round': 5000,
    'random_state': 42,
}

print("Config:", CONFIG)

# ==============================================================================
# CELL 3: Auto-discover CSV path (Kaggle / Local)
# ==============================================================================
IS_KAGGLE = os.path.exists('/kaggle')

if IS_KAGGLE:
    # Try to find any CSV in /kaggle/input
    csv_files = glob.glob('/kaggle/input/**/*.csv', recursive=True)
    # Prefer cleaned, then raw XAU files, then any CSV
    preferred = [f for f in csv_files if 'XAU' in f.upper() or 'cleaned' in f.lower()]
    csv_path = preferred[0] if preferred else (csv_files[0] if csv_files else None)
else:
    # Local fallback
    csv_path = 'data/beta_testing/kaggle_raw/XAU_1m_data_cleaned.csv'
    if not os.path.exists(csv_path):
        csv_path = 'data/beta_testing/kaggle_raw/XAU_1m_data.csv'

if not csv_path or not os.path.exists(csv_path):
    raise FileNotFoundError(f"No CSV found. Searched: {csv_files if IS_KAGGLE else csv_path}")

print(f"Loading: {csv_path}")

# ==============================================================================
# CELL 4: Load & Clean Data
# ==============================================================================
# Try auto-detect separator
with open(csv_path, 'r') as f:
    first_line = f.readline()
    sep = ';' if ';' in first_line else (',' if ',' in first_line else None)

df = pd.read_csv(csv_path, sep=sep, engine='python')
print(f"Raw shape: {df.shape}")

# Standardize columns
df.columns = [c.strip().lower() for c in df.columns]

# Rename common variants
rename_map = {
    'timestamp': 'datetime',
    'date': 'datetime',
    'time': 'datetime',
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'volume': 'volume',
}
for old, new in rename_map.items():
    if old in df.columns and new not in df.columns:
        df.rename(columns={old: new}, inplace=True)

# Parse datetime
dt_col = 'datetime' if 'datetime' in df.columns else df.columns[0]
if not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
    df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')

df = df.sort_values(dt_col).reset_index(drop=True)
df.set_index(dt_col, inplace=True)

# Ensure required columns exist
required = ['open', 'high', 'low', 'close']
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Fill volume if missing
if 'volume' not in df.columns:
    df['volume'] = 1.0

# Drop rows with NaN in OHLC
df = df.dropna(subset=required)
print(f"After cleaning: {df.shape}")

# ==============================================================================
# CELL 5: Feature Engineering
# ==============================================================================
print("\nEngineering features...")

close = df['close']
high = df['high']
low = df['low']
open_ = df['open']
volume = df['volume']
returns = close.pct_change()

f = pd.DataFrame(index=df.index)

# --- Price returns ---
f['ret_1'] = returns
f['ret_5'] = close.pct_change(5)
f['ret_10'] = close.pct_change(10)
f['ret_20'] = close.pct_change(20)
f['ret_50'] = close.pct_change(50)

# --- EMA distances ---
for span in [5, 10, 20, 50]:
    ema = close.ewm(span=span, adjust=False).mean()
    f[f'ema_{span}'] = (close - ema) / (ema + 1e-10)

# --- RSI ---
for period in [7, 14, 21]:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    f[f'rsi_{period}'] = 100 - 100 / (1 + rs)

# --- MACD ---
macd = close.ewm(span=8, adjust=False).mean() - close.ewm(span=17, adjust=False).mean()
signal = macd.ewm(span=9, adjust=False).mean()
f['macd'] = macd
f['macd_signal'] = signal
f['macd_hist'] = macd - signal

# --- ATR ---
for period in [7, 14, 21]:
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    f[f'atr_{period}'] = tr.ewm(span=period, adjust=False).mean()
    f[f'atr_ratio_{period}'] = f[f'atr_{period}'] / (close + 1e-10)

# --- Bollinger Bands ---
for window in [20, 50]:
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    f[f'bb_pos_{window}'] = (close - ma) / (std + 1e-10)
    f[f'bb_width_{window}'] = std / (ma + 1e-10)

# --- Volume features ---
f['vol_rel_5'] = volume / (volume.rolling(5).mean() + 1e-10)
f['vol_rel_20'] = volume / (volume.rolling(20).mean() + 1e-10)
f['dollar_vol'] = (close * volume) / (close * volume).rolling(20).mean()

# --- Temporal ---
f['hour'] = df.index.hour / 23.0
f['day_of_week'] = df.index.dayofweek / 6.0
f['month'] = (df.index.month - 1) / 11.0

# --- Microstructure features ---
# VWAP deviation
typical = (high + low + close) / 3
vwap = (typical * volume).cumsum() / (volume.cumsum() + 1e-10)
f['vwap_dev'] = (close - vwap) / (vwap + 1e-10)

# Price location within bar
f['close_loc'] = (close - low) / (high - low + 1e-10)

# Price location within recent range
for window in [10, 20, 50]:
    hh = high.rolling(window).max()
    ll = low.rolling(window).min()
    f[f'price_loc_{window}'] = (close - ll) / (hh - ll + 1e-10)

# Bar momentum
f['body'] = (close - open_) / (high - low + 1e-10)

# Gap from previous close
f['gap'] = (open_ - close.shift(1)) / close.shift(1)

# Bars since high/low (inverse — 0 = just hit)
is_high_20 = (close == high.rolling(20).max())
is_low_20 = (close == low.rolling(20).min())
f['bars_since_high'] = (~is_high_20).iloc[::-1].cumsum().iloc[::-1]
f['bars_since_low'] = (~is_low_20).iloc[::-1].cumsum().iloc[::-1]

# Realized volatility at multiple scales
for window in [10, 30, 60]:
    f[f'realized_vol_{window}'] = returns.rolling(window).std() * np.sqrt(window)

# ADX proxy
for window in [14, 30]:
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr = tr.rolling(window).mean()
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    plus_di = 100 * plus_dm.rolling(window).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.rolling(window).mean() / (atr + 1e-10)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    f[f'adx_{window}'] = dx.rolling(window).mean()

# --- Target engineering ---
H = CONFIG['horizon']
ret = close.pct_change(H).shift(-H)
f[f'target_ret_{H}'] = ret

if CONFIG['drop_neutral_threshold']:
    rolling_vol = ret.rolling(1000).std()
    threshold = CONFIG['vol_multiplier'] * rolling_vol
    f[f'target_dir_{H}'] = np.where(
        abs(ret) > threshold,
        np.sign(ret),
        0  # Neutral / uncertain
    )
else:
    f[f'target_dir_{H}'] = np.sign(ret)

# Cleanup
target_cols = [c for c in f.columns if c.startswith('target_')]
feature_cols = [c for c in f.columns if not c.startswith('target_')]

f = f.replace([np.inf, -np.inf], np.nan)
f[feature_cols] = f[feature_cols].ffill().fillna(0)

# Drop rows with NaN targets or early-window NaNs
f = f.dropna(subset=target_cols)
print(f"Features shape: {f.shape}")

# ==============================================================================
# CELL 6: Prepare Train / Val / Test Splits (Chronological)
# ==============================================================================
target_col = f'target_dir_{H}'
y_all = f[target_col].values
X_all = f[feature_cols]

# Drop neutral labels (0) if using volatility filter
if CONFIG['drop_neutral_threshold']:
    mask = y_all != 0
    X_all = X_all.iloc[mask]
    y_all = y_all[mask]
    # Remap {-1, 1} -> {0, 1} for binary classification
    y_all = ((y_all > 0).astype(int))
    print(f"After dropping neutrals: {len(y_all)} samples, class dist: {np.bincount(y_all)}")

# Temporal split: 70 / 15 / 15
n = len(X_all)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, y_train = X_all.iloc[:train_end], y_all[:train_end]
X_val, y_val = X_all.iloc[train_end:val_end], y_all[train_end:val_end]
X_test, y_test = X_all.iloc[val_end:], y_all[val_end:]

print(f"\nTrain: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
print(f"Train class dist: {np.bincount(y_train)}")

# ==============================================================================
# CELL 7: Feature Selection (Mutual Information)
# ==============================================================================
print("\nSelecting features via Mutual Information...")

# Use a sample for speed if train set is huge
sample_size = min(200000, len(X_train))
idx = np.random.choice(len(X_train), sample_size, replace=False)
mi_scores = mutual_info_classif(
    X_train.iloc[idx], y_train[idx],
    random_state=CONFIG['random_state'],
    n_neighbors=5,
    discrete_features=False
)

mi_df = pd.DataFrame({'feature': X_train.columns, 'mi': mi_scores})
mi_df = mi_df.sort_values('mi', ascending=False).reset_index(drop=True)

# Keep top K
top_k = min(CONFIG['feature_top_k'], len(mi_df))
selected_features = mi_df.head(top_k)['feature'].tolist()

X_train = X_train[selected_features]
X_val = X_val[selected_features]
X_test = X_test[selected_features]

print(f"Selected top {top_k} features:")
print(mi_df.head(10).to_string(index=False))

# ==============================================================================
# CELL 8: SMOTE Class Balancing (Optional)
# ==============================================================================
if CONFIG['use_smote'] and HAS_SMOTE:
    print("\nApplying SMOTE...")
    smote = SMOTE(
        sampling_strategy=CONFIG['smote_balance'],
        random_state=CONFIG['random_state'],
        k_neighbors=5
    )
    # SMOTE on a sample if too large (memory limit on Kaggle)
    if len(X_train) > 500000:
        smote_idx = np.random.choice(len(X_train), 500000, replace=False)
        X_train_sm, y_train_sm = X_train.iloc[smote_idx], y_train[smote_idx]
    else:
        X_train_sm, y_train_sm = X_train, y_train

    X_train, y_train = smote.fit_resample(X_train_sm, y_train_sm)
    print(f"After SMOTE: {len(y_train)} samples, class dist: {np.bincount(y_train)}")
else:
    print("\nSkipping SMOTE.")

# ==============================================================================
# CELL 9: Train LightGBM
# ==============================================================================
print("\n" + "=" * 60)
print("TRAINING LIGHTGBM")
print("=" * 60)

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'max_depth': 7,
    'min_data_in_leaf': 100,
    'learning_rate': 0.02,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 3,
    'lambda_l1': 0.1,
    'lambda_l2': 1.0,
    'max_bin': 255,
    'verbose': -1,
    'seed': CONFIG['random_state'],
    'deterministic': True,
    'scale_pos_weight': (y_train == 0).sum() / max((y_train == 1).sum(), 1),
}

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=CONFIG['num_boost_round'],
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(CONFIG['early_stopping_rounds'], verbose=True),
        lgb.log_evaluation(period=100)
    ]
)

print(f"\n[LGB] Best iteration: {model.best_iteration}")
print(f"[LGB] Best validation score: {model.best_score['valid']['auc']:.6f}")

# ==============================================================================
# CELL 10: Threshold Optimization (Maximize F1)
# ==============================================================================
print("\n" + "=" * 60)
print("THRESHOLD OPTIMIZATION")
print("=" * 60)

val_probs = model.predict(X_val, num_iteration=model.best_iteration)

# Try multiple thresholds
thresholds = np.arange(0.30, 0.71, 0.01)
best_f1 = 0
best_thresh = 0.5
best_metrics = {}

for t in thresholds:
    preds = (val_probs > t).astype(int)
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t
        best_metrics = {
            'threshold': t,
            'accuracy': accuracy_score(y_val, preds),
            'precision': precision_score(y_val, preds, zero_division=0),
            'recall': recall_score(y_val, preds, zero_division=0),
            'f1': f1,
            'auc': roc_auc_score(y_val, val_probs),
        }

print(f"\nOPTIMAL THRESHOLD: {best_thresh:.3f}")
print(f"  Accuracy:  {best_metrics['accuracy']:.4f}")
print(f"  Precision: {best_metrics['precision']:.4f}")
print(f"  Recall:    {best_metrics['recall']:.4f}")
print(f"  F1:        {best_metrics['f1']:.4f}")
print(f"  AUC:       {best_metrics['auc']:.4f}")

# ==============================================================================
# CELL 11: Final Evaluation on All Splits
# ==============================================================================
print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

def evaluate_split(name, X, y, model, threshold):
    probs = model.predict(X, num_iteration=model.best_iteration)
    preds = (probs > threshold).astype(int)

    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    ll = log_loss(y, probs)

    print(f"\n[{name}]  n={len(y):,}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  LogLoss:   {ll:.4f}")
    print(f"  CM:        {confusion_matrix(y, preds).tolist()}")
    return {
        'accuracy': acc, 'auc': auc, 'precision': prec,
        'recall': rec, 'f1': f1, 'logloss': ll
    }

results = {
    'train': evaluate_split('TRAIN', X_train, y_train, model, best_thresh),
    'val':   evaluate_split('VAL',   X_val,   y_val,   model, best_thresh),
    'test':  evaluate_split('TEST',  X_test,  y_test,  model, best_thresh),
}

# ==============================================================================
# CELL 12: Feature Importance
# ==============================================================================
print("\n" + "=" * 60)
print("TOP 15 FEATURES (by gain)")
print("=" * 60)

imp = model.feature_importance(importance_type='gain')
imp_df = pd.DataFrame({'feature': selected_features, 'gain': imp})
imp_df = imp_df.sort_values('gain', ascending=False)
print(imp_df.head(15).to_string(index=False))

# ==============================================================================
# CELL 13: Save Model
# ==============================================================================
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

output = {
    'model': model,
    'features': selected_features,
    'threshold': best_thresh,
    'config': CONFIG,
    'metrics': results,
    'feature_importance': imp_df,
    'mi_scores': mi_df.head(top_k),
}

save_path = 'gold_1m_lgb_v2.pkl'
joblib.dump(output, save_path)
print(f"Saved to: {save_path}  ({os.path.getsize(save_path)/1024/1024:.1f} MB)")

# Also save feature names as text for easy reference
with open('selected_features_v2.txt', 'w') as f_out:
    f_out.write('\n'.join(selected_features))
print("Saved: selected_features_v2.txt")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
