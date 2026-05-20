"""
Gold 1-Minute: Full Training of All 3 Models (V3)
=================================================
Models: LightGBM, XGBoost, Random Forest
Strategy: Advanced features + confidence filtering + meta-labeling
Dataset: Full 6.7M rows on Kaggle
"""

import os
import sys
import glob
import json
import warnings
import time
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
np.random.seed(42)

print("=" * 70)
print("GOLD 1-MINUTE: FULL MODEL TRAINING V3")
print("=" * 70)
print(f"Start: {datetime.now().isoformat()}")

# =============================================================================
# CONFIG
# =============================================================================
HORIZON = 20
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
FEATURE_TOP_K = 40
EARLY_STOP = 200
MAX_ROUNDS = 5000
CONFIDENCE_THRESHOLD = 0.60  # Only trade when model confidence > 60%

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/6] Loading data...")
csv_files = glob.glob('/kaggle/input/**/*.csv', recursive=True)
preferred = [f for f in csv_files if 'xau' in f.lower() or 'gold' in f.lower()]
csv_path = preferred[0] if preferred else csv_files[0]
print(f"  File: {csv_path}")

with open(csv_path, 'r') as fh:
    first = fh.readline()
    sep = ';' if ';' in first else (',' if ',' in first else None)

df = pd.read_csv(csv_path, sep=sep, engine='python')
print(f"  Raw shape: {df.shape}")

df.columns = [c.strip().lower() for c in df.columns]
rename = {'timestamp': 'datetime', 'date': 'datetime', 'time': 'datetime'}
for o, n in rename.items():
    if o in df.columns and n not in df.columns:
        df.rename(columns={o: n}, inplace=True)

dt_col = 'datetime' if 'datetime' in df.columns else df.columns[0]
if not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
    df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')

df = df.sort_values(dt_col).reset_index(drop=True)
df.set_index(dt_col, inplace=True)

for c in ['open', 'high', 'low', 'close']:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")
if 'volume' not in df.columns:
    df['volume'] = 1.0

df = df.dropna(subset=['open', 'high', 'low', 'close'])
print(f"  Clean shape: {df.shape}")

# =============================================================================
# ADVANCED FEATURE ENGINEERING
# =============================================================================
print("\n[2/6] Engineering advanced features...")

close = df['close']
high = df['high']
low = df['low']
open_ = df['open']
volume = df['volume']
returns = close.pct_change()

F = pd.DataFrame(index=df.index)

# --- Basic returns ---
F['ret_1'] = returns
for w in [3, 5, 10, 20, 50]:
    F[f'ret_{w}'] = close.pct_change(w)

# --- EMA & WMA distances ---
for span in [5, 10, 20, 50]:
    ema = close.ewm(span=span, adjust=False).mean()
    F[f'ema_{span}'] = (close - ema) / (ema + 1e-10)
    
for window in [5, 10, 20, 50]:
    wma = close.rolling(window).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)
    F[f'wma_{window}'] = (close - wma) / (wma + 1e-10)

# --- RSI ---
for period in [7, 14, 21]:
    d = close.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/period, adjust=False).mean()
    al = l.ewm(alpha=1/period, adjust=False).mean()
    rs = ag / (al + 1e-10)
    F[f'rsi_{period}'] = 100 - 100 / (1 + rs)

# --- Stochastic Oscillator ---
for k_period in [14, 21]:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    F[f'stoch_k_{k_period}'] = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    F[f'stoch_d_{k_period}'] = F[f'stoch_k_{k_period}'].rolling(3).mean()

# --- Williams %R ---
for w in [14, 21]:
    hh = high.rolling(w).max()
    ll = low.rolling(w).min()
    F[f'williams_r_{w}'] = -100 * (hh - close) / (hh - ll + 1e-10)

# --- CCI (Commodity Channel Index) ---
for w in [20, 50]:
    tp = (high + low + close) / 3
    ma_tp = tp.rolling(w).mean()
    md_tp = tp.rolling(w).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    F[f'cci_{w}'] = (tp - ma_tp) / (0.015 * md_tp + 1e-10)

# --- MACD ---
macd = close.ewm(span=8, adjust=False).mean() - close.ewm(span=17, adjust=False).mean()
sig = macd.ewm(span=9, adjust=False).mean()
F['macd'] = macd
F['macd_signal'] = sig
F['macd_hist'] = macd - sig

# --- ATR ---
for p in [7, 14, 21]:
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    F[f'atr_{p}'] = tr.ewm(span=p, adjust=False).mean()
    F[f'atr_ratio_{p}'] = F[f'atr_{p}'] / (close + 1e-10)

# --- Bollinger Bands ---
for w in [20, 50]:
    ma = close.rolling(w).mean()
    std = close.rolling(w).std()
    F[f'bb_pos_{w}'] = (close - ma) / (std + 1e-10)
    F[f'bb_width_{w}'] = std / (ma + 1e-10)

# --- Ichimoku Cloud ---
tenkan_sen = (high.rolling(9).max() + low.rolling(9).min()) / 2
kijun_sen = (high.rolling(26).max() + low.rolling(26).min()) / 2
senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
senkou_span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
F['tenkan_sen'] = (close - tenkan_sen) / (tenkan_sen + 1e-10)
F['kijun_sen'] = (close - kijun_sen) / (kijun_sen + 1e-10)
F['senkou_a'] = (close - senkou_span_a) / (senkou_span_a + 1e-10)
F['senkou_b'] = (close - senkou_span_b) / (senkou_span_b + 1e-10)
F['ichimoku_cloud'] = (senkou_span_a - senkou_span_b) / (senkou_span_b + 1e-10)

# --- ADX ---
for w in [14, 30]:
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr = tr.rolling(w).mean()
    pdm = (high - high.shift(1)).clip(lower=0)
    mdm = (low.shift(1) - low).clip(lower=0)
    pdi = 100 * pdm.rolling(w).mean() / (atr + 1e-10)
    mdi = 100 * mdm.rolling(w).mean() / (atr + 1e-10)
    dx = 100 * abs(pdi - mdi) / (pdi + mdi + 1e-10)
    F[f'adx_{w}'] = dx.rolling(w).mean()

# --- Volume features ---
F['vol_rel_5'] = volume / (volume.rolling(5).mean() + 1e-10)
F['vol_rel_20'] = volume / (volume.rolling(20).mean() + 1e-10)
F['vol_rel_50'] = volume / (volume.rolling(50).mean() + 1e-10)
F['dollar_vol'] = (close * volume) / ((close * volume).rolling(20).mean() + 1e-10)
F['obv'] = (np.sign(close.diff()) * volume).cumsum()
F['obv_ema'] = F['obv'].ewm(span=20, adjust=False).mean()
F['obv_ratio'] = F['obv'] / (F['obv_ema'] + 1e-10)

# --- Temporal features ---
F['hour'] = df.index.hour / 23.0
F['dow'] = df.index.dayofweek / 6.0
F['month'] = (df.index.month - 1) / 11.0
F['is_london'] = ((df.index.hour >= 8) & (df.index.hour <= 16)).astype(float)
F['is_ny'] = ((df.index.hour >= 13) & (df.index.hour <= 21)).astype(float)
F['is_asia'] = ((df.index.hour >= 0) & (df.index.hour <= 7)).astype(float)

# --- Microstructure ---
typical = (high + low + close) / 3
vwap = (typical * volume).cumsum() / (volume.cumsum() + 1e-10)
F['vwap_dev'] = (close - vwap) / (vwap + 1e-10)
F['close_loc'] = (close - low) / (high - low + 1e-10)

for w in [10, 20, 50]:
    hh = high.rolling(w).max()
    ll = low.rolling(w).min()
    F[f'price_loc_{w}'] = (close - ll) / (hh - ll + 1e-10)

F['body'] = (close - open_) / (high - low + 1e-10)
F['gap'] = (open_ - close.shift(1)) / close.shift(1)
F['upper_shadow'] = (high - np.maximum(close, open_)) / (high - low + 1e-10)
F['lower_shadow'] = (np.minimum(close, open_) - low) / (high - low + 1e-10)

is_h20 = (close == high.rolling(20).max())
is_l20 = (close == low.rolling(20).min())
F['bars_since_high'] = (~is_h20).iloc[::-1].cumsum().iloc[::-1]
F['bars_since_low'] = (~is_l20).iloc[::-1].cumsum().iloc[::-1]

# --- Volatility features ---
for w in [10, 30, 60]:
    F[f'realized_vol_{w}'] = returns.rolling(w).std() * np.sqrt(w)
    F[f'realized_var_{w}'] = returns.rolling(w).var() * w

F['vol_regime'] = (F['realized_vol_30'] > F['realized_vol_30'].rolling(100).mean()).astype(float)

# --- Interaction features ---
F['rsi_macd'] = F['rsi_14'] * F['macd_hist']
F['vol_price_loc'] = F['vol_rel_20'] * F['price_loc_20']
F['atr_ret'] = F['atr_14'] * F['ret_1']
F['gap_body'] = F['gap'] * F['body']
F['adx_rsi'] = F['adx_14'] * F['rsi_14']
F['bb_vol'] = F['bb_pos_20'] * F['vol_rel_20']
F['ichimoku_tenkan_kijun'] = F['tenkan_sen'] * F['kijun_sen']
F['cci_rsi'] = F['cci_20'] * F['rsi_14']
F['stoch_rsi'] = F['stoch_k_14'] * F['rsi_14']

# --- Lagged features (autocorrelation) ---
for lag in [1, 2, 3, 5]:
    F[f'ret_1_lag_{lag}'] = F['ret_1'].shift(lag)
    F[f'vol_rel_20_lag_{lag}'] = F['vol_rel_20'].shift(lag)

print(f"  Total features: {len(F.columns)}")

# =============================================================================
# TARGET ENGINEERING
# =============================================================================
print("\n[3/6] Building targets...")

ret = close.pct_change(HORIZON).shift(-HORIZON)
F['target_ret'] = ret

# Direction target
F['target_dir'] = np.sign(ret)

# Volatility-filtered target
rolling_vol = ret.rolling(1000).std()
F['target_vol'] = np.where(abs(ret) > 0.5 * rolling_vol, np.sign(ret), 0)

# Cleanup
target_cols = ['target_ret', 'target_dir', 'target_vol']
feature_cols = [c for c in F.columns if c not in target_cols]

F = F.replace([np.inf, -np.inf], np.nan)
F[feature_cols] = F[feature_cols].ffill().fillna(0)
F = F.dropna(subset=target_cols)

# Drop neutral for vol target
mask = F['target_vol'] != 0
X_all = F[feature_cols].loc[mask]
y_all = (F['target_vol'].loc[mask] > 0).astype(int)

print(f"  Samples after filtering: {len(y_all)}")
print(f"  Class distribution: {np.bincount(y_all)}")

# Temporal split
n = len(X_all)
train_end = int(n * TRAIN_FRAC)
val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

X_train, y_train = X_all.iloc[:train_end], y_all[:train_end]
X_val, y_val = X_all.iloc[train_end:val_end], y_all[train_end:val_end]
X_test, y_test = X_all.iloc[val_end:], y_all[val_end:]

print(f"  Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")

# =============================================================================
# FEATURE SELECTION
# =============================================================================
print("\n[4/6] Selecting features via Mutual Information...")

sample_size = min(300000, len(X_train))
idx = np.random.choice(len(X_train), sample_size, replace=False)
mi_scores = mutual_info_classif(
    X_train.iloc[idx], y_train[idx],
    random_state=42, n_neighbors=5, discrete_features=False
)

mi_df = pd.DataFrame({'feature': X_train.columns, 'mi': mi_scores})
mi_df = mi_df.sort_values('mi', ascending=False).reset_index(drop=True)

top_k = min(FEATURE_TOP_K, len(mi_df))
selected_features = mi_df.head(top_k)['feature'].tolist()

X_train = X_train[selected_features]
X_val = X_val[selected_features]
X_test = X_test[selected_features]

print(f"  Selected top {top_k} features:")
for _, row in mi_df.head(10).iterrows():
    print(f"    {row['feature']}: {row['mi']:.4f}")

# =============================================================================
# MODEL TRAINING
# =============================================================================
print("\n[5/6] Training models...")

results = {}
scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

# --- Model 1: LightGBM ---
print("\n  >>> Training LightGBM...")
lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 127, 'max_depth': 10, 'min_data_in_leaf': 50,
    'learning_rate': 0.03, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 3,
    'lambda_l1': 0.05, 'lambda_l2': 0.5, 'max_bin': 255,
    'verbose': -1, 'seed': 42, 'deterministic': True,
    'scale_pos_weight': scale_pos,
}

dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

lgb_model = lgb.train(
    lgb_params, dtrain, num_boost_round=MAX_ROUNDS,
    valid_sets=[dtrain, dval], valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)]
)

print(f"      Best iter: {lgb_model.best_iteration}  Val AUC: {lgb_model.best_score['valid']['auc']:.5f}")

# --- Model 2: XGBoost ---
print("\n  >>> Training XGBoost...")
xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'max_depth': 8, 'learning_rate': 0.03, 'n_estimators': 2000,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 1.0,
    'min_child_weight': 50, 'gamma': 0.1,
    'tree_method': 'hist', 'random_state': 42, 'n_jobs': -1,
    'scale_pos_weight': scale_pos,
}

xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=EARLY_STOP,
    verbose=False
)
print(f"      Best iter: {xgb_model.best_iteration}  Val AUC: {xgb_model.best_score:.5f}")

# --- Model 3: Random Forest ---
print("\n  >>> Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=500, max_depth=15, min_samples_leaf=100,
    max_features='sqrt', class_weight='balanced',
    random_state=42, n_jobs=-1, verbose=0
)
rf_model.fit(X_train, y_train)
print(f"      Trained {len(rf_model.estimators_)} trees")

# =============================================================================
# EVALUATION
# =============================================================================
print("\n[6/6] Evaluating all models...")

def evaluate_model(name, model, X_tr, y_tr, X_v, y_v, X_te, y_te, is_lgb=False, is_xgb=False):
    def _probs(X):
        if is_lgb:
            return model.predict(X, num_iteration=model.best_iteration)
        elif is_xgb:
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict_proba(X)[:, 1]
    
    def _metrics(X, y, split):
        probs = _probs(X)
        preds = (probs > 0.5).astype(int)
        
        # Confidence-filtered metrics (only predict when confident)
        conf_mask = (probs > CONFIDENCE_THRESHOLD) | (probs < (1 - CONFIDENCE_THRESHOLD))
        if conf_mask.sum() > 100:
            conf_preds = (probs[conf_mask] > 0.5).astype(int)
            conf_acc = accuracy_score(y[conf_mask], conf_preds)
            conf_pct = conf_mask.mean()
        else:
            conf_acc = 0
            conf_pct = 0
        
        return {
            f'{split}_acc': accuracy_score(y, preds),
            f'{split}_auc': roc_auc_score(y, probs),
            f'{split}_prec': precision_score(y, preds, zero_division=0),
            f'{split}_rec': recall_score(y, preds, zero_division=0),
            f'{split}_f1': f1_score(y, preds, zero_division=0),
            f'{split}_logloss': log_loss(y, probs),
            f'{split}_conf_acc': conf_acc,
            f'{split}_conf_pct': conf_pct,
        }
    
    m = {}
    m.update(_metrics(X_tr, y_tr, 'train'))
    m.update(_metrics(X_v, y_v, 'val'))
    m.update(_metrics(X_te, y_te, 'test'))
    m['name'] = name
    return m

results['lgb'] = evaluate_model('LightGBM', lgb_model, X_train, y_train, X_val, y_val, X_test, y_test, is_lgb=True)
results['xgb'] = evaluate_model('XGBoost', xgb_model, X_train, y_train, X_val, y_val, X_test, y_test, is_xgb=True)
results['rf'] = evaluate_model('RandomForest', rf_model, X_train, y_train, X_val, y_val, X_test, y_test)

# =============================================================================
# ENSEMBLE
# =============================================================================
print("\n  >>> Evaluating Ensemble (mean of probabilities)...")

lgb_probs = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
rf_probs = rf_model.predict_proba(X_test)[:, 1]

ensemble_probs = (lgb_probs + xgb_probs + rf_probs) / 3
ensemble_preds = (ensemble_probs > 0.5).astype(int)

conf_mask = (ensemble_probs > CONFIDENCE_THRESHOLD) | (ensemble_probs < (1 - CONFIDENCE_THRESHOLD))
if conf_mask.sum() > 100:
    conf_acc = accuracy_score(y_test[conf_mask], (ensemble_probs[conf_mask] > 0.5).astype(int))
    conf_pct = conf_mask.mean()
else:
    conf_acc = 0
    conf_pct = 0

results['ensemble'] = {
    'name': 'Ensemble',
    'test_acc': accuracy_score(y_test, ensemble_preds),
    'test_auc': roc_auc_score(y_test, ensemble_probs),
    'test_prec': precision_score(y_test, ensemble_preds, zero_division=0),
    'test_rec': recall_score(y_test, ensemble_preds, zero_division=0),
    'test_f1': f1_score(y_test, ensemble_preds, zero_division=0),
    'test_logloss': log_loss(y_test, ensemble_probs),
    'test_conf_acc': conf_acc,
    'test_conf_pct': conf_pct,
}

# =============================================================================
# PRINT RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

for name, res in results.items():
    print(f"\n[{res['name']}]")
    for k, v in res.items():
        if k != 'name' and isinstance(v, float):
            print(f"  {k}: {v:.4f}")

# =============================================================================
# SAVE EVERYTHING
# =============================================================================
print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

output = {
    'lgb_model': lgb_model,
    'xgb_model': xgb_model,
    'rf_model': rf_model,
    'features': selected_features,
    'mi_scores': mi_df,
    'results': results,
    'config': {
        'horizon': HORIZON,
        'feature_top_k': FEATURE_TOP_K,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
    }
}

joblib.dump(output, 'all_models_v3.pkl')
print(f"  Saved: all_models_v3.pkl ({os.path.getsize('all_models_v3.pkl')/1024/1024:.1f} MB)")

results_df = pd.DataFrame([{k: v for k, v in r.items()} for r in results.values()])
results_df.to_csv('model_results_v3.csv', index=False)
print(f"  Saved: model_results_v3.csv")

with open('model_config_v3.json', 'w') as f:
    json.dump({
        'best_model': 'ensemble',
        'best_test_auc': results['ensemble']['test_auc'],
        'best_test_acc': results['ensemble']['test_acc'],
        'best_test_conf_acc': results['ensemble']['test_conf_acc'],
        'best_test_conf_pct': results['ensemble']['test_conf_pct'],
        'n_features': len(selected_features),
        'horizon': HORIZON,
    }, f, indent=2)
print(f"  Saved: model_config_v3.json")

print("\n" + "=" * 70)
print(f"Done: {datetime.now().isoformat()}")
print("=" * 70)
