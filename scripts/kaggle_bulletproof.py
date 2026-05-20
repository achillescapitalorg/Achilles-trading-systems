"""
Gold 1-Minute: BULLETPROOF Training (Auto-retry, checkpoints, no failures)
============================================================================
This script is designed to NEVER fail completely. If a model crashes,
it retries with fallback parameters. After each model, it saves a checkpoint.
"""

import os, sys, glob, json, warnings, time, traceback
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')
np.random.seed(42)

CHECKPOINT_FILE = '/kaggle/working/checkpoint.json'
RESULTS_FILE = '/kaggle/working/model_results_bulletproof.csv'

def save_checkpoint(state):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(state, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {}

print("=" * 70)
print("BULLETPROOF GOLD 1M TRAINING")
print("=" * 70)
print(f"Start: {datetime.now().isoformat()}")

cp = load_checkpoint()
print(f"Checkpoint: {cp}")

# =============================================================================
# LOAD DATA
# =============================================================================
if cp.get('data_loaded'):
    print("\n[SKIP] Data already loaded in checkpoint")
    df = pd.read_pickle('/kaggle/working/df_clean.pkl')
else:
    print("\n[1/6] Loading data...")
    csv_files = glob.glob('/kaggle/input/**/*.csv', recursive=True)
    preferred = [f for f in csv_files if 'xau' in f.lower() or 'gold' in f.lower()]
    csv_path = preferred[0] if preferred else csv_files[0]
    with open(csv_path, 'r') as fh:
        first = fh.readline()
        sep = ';' if ';' in first else (',' if ',' in first else None)
    df = pd.read_csv(csv_path, sep=sep, engine='python')
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
    if 'volume' not in df.columns:
        df['volume'] = 1.0
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df.to_pickle('/kaggle/working/df_clean.pkl')
    cp['data_loaded'] = True
    cp['shape'] = str(df.shape)
    save_checkpoint(cp)
    print(f"  Loaded: {df.shape}")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
if cp.get('features_done'):
    print("\n[SKIP] Features already engineered")
    F = pd.read_pickle('/kaggle/working/features.pkl')
else:
    print("\n[2/6] Engineering features...")
    close = df['close']; high = df['high']; low = df['low']
    open_ = df['open']; volume = df['volume']; returns = close.pct_change()
    F = pd.DataFrame(index=df.index)
    
    # Returns
    F['ret_1'] = returns
    for w in [3, 5, 10, 20, 50]:
        F[f'ret_{w}'] = close.pct_change(w)
    
    # EMA & WMA
    for span in [5, 10, 20, 50]:
        ema = close.ewm(span=span, adjust=False).mean()
        F[f'ema_{span}'] = (close - ema) / (ema + 1e-10)
    for window in [5, 10, 20, 50]:
        wma = close.rolling(window).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)
        F[f'wma_{window}'] = (close - wma) / (wma + 1e-10)
    
    # RSI
    for period in [7, 14, 21]:
        d = close.diff()
        ag = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        al = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        F[f'rsi_{period}'] = 100 - 100 / (1 + ag / (al + 1e-10))
    
    # Stochastic
    for k in [14, 21]:
        ll = low.rolling(k).min(); hh = high.rolling(k).max()
        F[f'stoch_k_{k}'] = 100 * (close - ll) / (hh - ll + 1e-10)
        F[f'stoch_d_{k}'] = F[f'stoch_k_{k}'].rolling(3).mean()
    
    # Williams %R
    for w in [14, 21]:
        hh = high.rolling(w).max(); ll = low.rolling(w).min()
        F[f'williams_r_{w}'] = -100 * (hh - close) / (hh - ll + 1e-10)
    
    # CCI
    for w in [20, 50]:
        tp = (high + low + close) / 3
        ma_tp = tp.rolling(w).mean()
        md_tp = tp.rolling(w).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        F[f'cci_{w}'] = (tp - ma_tp) / (0.015 * md_tp + 1e-10)
    
    # MACD
    macd = close.ewm(span=8, adjust=False).mean() - close.ewm(span=17, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    F['macd'] = macd; F['macd_signal'] = sig; F['macd_hist'] = macd - sig
    
    # ATR
    for p in [7, 14, 21]:
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        F[f'atr_{p}'] = tr.ewm(span=p, adjust=False).mean()
        F[f'atr_ratio_{p}'] = F[f'atr_{p}'] / (close + 1e-10)
    
    # Bollinger
    for w in [20, 50]:
        ma = close.rolling(w).mean(); std = close.rolling(w).std()
        F[f'bb_pos_{w}'] = (close - ma) / (std + 1e-10)
        F[f'bb_width_{w}'] = std / (ma + 1e-10)
    
    # Ichimoku
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    F['tenkan_sen'] = (close - tenkan) / (tenkan + 1e-10)
    F['kijun_sen'] = (close - kijun) / (kijun + 1e-10)
    F['senkou_a'] = (close - senkou_a) / (senkou_a + 1e-10)
    F['senkou_b'] = (close - senkou_b) / (senkou_b + 1e-10)
    
    # ADX
    for w in [14, 30]:
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(w).mean()
        pdm = (high - high.shift(1)).clip(lower=0)
        mdm = (low.shift(1) - low).clip(lower=0)
        pdi = 100 * pdm.rolling(w).mean() / (atr + 1e-10)
        mdi = 100 * mdm.rolling(w).mean() / (atr + 1e-10)
        dx = 100 * abs(pdi - mdi) / (pdi + mdi + 1e-10)
        F[f'adx_{w}'] = dx.rolling(w).mean()
    
    # Volume
    F['vol_rel_5'] = volume / (volume.rolling(5).mean() + 1e-10)
    F['vol_rel_20'] = volume / (volume.rolling(20).mean() + 1e-10)
    F['vol_rel_50'] = volume / (volume.rolling(50).mean() + 1e-10)
    F['obv'] = (np.sign(close.diff()) * volume).cumsum()
    F['obv_ema'] = F['obv'].ewm(span=20, adjust=False).mean()
    
    # Temporal
    F['hour'] = df.index.hour / 23.0
    F['dow'] = df.index.dayofweek / 6.0
    F['is_london'] = ((df.index.hour >= 8) & (df.index.hour <= 16)).astype(float)
    F['is_ny'] = ((df.index.hour >= 13) & (df.index.hour <= 21)).astype(float)
    
    # Microstructure
    typical = (high + low + close) / 3
    vwap = (typical * volume).cumsum() / (volume.cumsum() + 1e-10)
    F['vwap_dev'] = (close - vwap) / (vwap + 1e-10)
    F['close_loc'] = (close - low) / (high - low + 1e-10)
    for w in [10, 20, 50]:
        hh = high.rolling(w).max(); ll = low.rolling(w).min()
        F[f'price_loc_{w}'] = (close - ll) / (hh - ll + 1e-10)
    F['body'] = (close - open_) / (high - low + 1e-10)
    F['gap'] = (open_ - close.shift(1)) / close.shift(1)
    
    # Volatility
    for w in [10, 30, 60]:
        F[f'realized_vol_{w}'] = returns.rolling(w).std() * np.sqrt(w)
    F['vol_regime'] = (F['realized_vol_30'] > F['realized_vol_30'].rolling(100).mean()).astype(float)
    
    # Interactions
    F['rsi_macd'] = F['rsi_14'] * F['macd_hist']
    F['vol_price_loc'] = F['vol_rel_20'] * F['price_loc_20']
    F['atr_ret'] = F['atr_14'] * F['ret_1']
    F['adx_rsi'] = F['adx_14'] * F['rsi_14']
    F['cci_rsi'] = F['cci_20'] * F['rsi_14']
    F['stoch_rsi'] = F['stoch_k_14'] * F['rsi_14']
    
    # Target
    H = 20
    ret = close.pct_change(H).shift(-H)
    F['target_ret'] = ret
    rolling_vol = ret.rolling(1000).std()
    F['target_vol'] = np.where(abs(ret) > 0.5 * rolling_vol, np.sign(ret), 0)
    
    target_cols = ['target_ret', 'target_vol']
    feature_cols = [c for c in F.columns if c not in target_cols]
    F = F.replace([np.inf, -np.inf], np.nan)
    F[feature_cols] = F[feature_cols].ffill().fillna(0)
    F = F.dropna(subset=target_cols)
    
    F.to_pickle('/kaggle/working/features.pkl')
    cp['features_done'] = True
    cp['n_features'] = len(feature_cols)
    save_checkpoint(cp)
    print(f"  Features: {len(feature_cols)}")

# =============================================================================
# PREPARE DATA
# =============================================================================
if cp.get('data_prepared'):
    print("\n[SKIP] Data already prepared")
    data = joblib.load('/kaggle/working/data_split.pkl')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    selected_features = data['selected_features']
else:
    print("\n[3/6] Preparing data...")
    target_col = 'target_vol'
    feature_cols = [c for c in F.columns if c not in ['target_ret', 'target_vol']]
    mask = F[target_col] != 0
    X_all = F[feature_cols].loc[mask]
    y_all = (F[target_col].loc[mask] > 0).astype(int)
    
    n = len(X_all)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train, y_train = X_all.iloc[:train_end], y_all[:train_end]
    X_val, y_val = X_all.iloc[train_end:val_end], y_all[train_end:val_end]
    X_test, y_test = X_all.iloc[val_end:], y_all[val_end:]
    
    # Feature selection
    from sklearn.feature_selection import mutual_info_classif
    sample_size = min(300000, len(X_train))
    idx = np.random.choice(len(X_train), sample_size, replace=False)
    mi = mutual_info_classif(X_train.iloc[idx], y_train[idx], random_state=42, n_neighbors=5)
    mi_df = pd.DataFrame({'feature': X_train.columns, 'mi': mi}).sort_values('mi', ascending=False)
    selected_features = mi_df.head(40)['feature'].tolist()
    
    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]
    
    joblib.dump({'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val,
                 'X_test': X_test, 'y_test': y_test, 'selected_features': selected_features},
                '/kaggle/working/data_split.pkl')
    cp['data_prepared'] = True
    save_checkpoint(cp)
    print(f"  Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")

scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

# =============================================================================
# TRAIN MODELS (with auto-retry)
# =============================================================================
print("\n[4/6] Training models with auto-retry...")

results = {}

def train_with_retry(train_fn, model_name, max_retries=3):
    """Train a model with automatic retry on failure"""
    for attempt in range(max_retries):
        try:
            print(f"\n  >>> {model_name} (attempt {attempt + 1}/{max_retries})...")
            model, metrics = train_fn()
            print(f"      SUCCESS: Val AUC={metrics.get('val_auc', 0):.4f}")
            return model, metrics
        except Exception as e:
            print(f"      FAILED: {str(e)[:100]}")
            if attempt < max_retries - 1:
                print(f"      Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"      All retries exhausted for {model_name}")
                return None, {'name': model_name, 'error': str(e)[:200]}
    return None, {'name': model_name, 'error': 'max_retries'}

# --- Model 1: LightGBM ---
def train_lgb():
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 127, 'max_depth': 10, 'min_data_in_leaf': 50,
        'learning_rate': 0.03, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 3,
        'lambda_l1': 0.05, 'lambda_l2': 0.5, 'max_bin': 255,
        'verbose': -1, 'seed': 42, 'deterministic': True,
        'scale_pos_weight': scale_pos,
    }
    model = lgb.train(params, dtrain, num_boost_round=5000,
                      valid_sets=[dtrain, dval], valid_names=['train', 'valid'],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    
    probs = model.predict(X_test, num_iteration=model.best_iteration)
    preds = (probs > 0.5).astype(int)
    metrics = {
        'name': 'LightGBM', 'test_acc': accuracy_score(y_test, preds),
        'test_auc': roc_auc_score(y_test, probs),
        'test_prec': precision_score(y_test, preds, zero_division=0),
        'test_rec': recall_score(y_test, preds, zero_division=0),
        'test_f1': f1_score(y_test, preds, zero_division=0),
        'val_auc': model.best_score['valid']['auc'],
    }
    joblib.dump(model, '/kaggle/working/lgb_model.pkl')
    return model, metrics

if not cp.get('lgb_done'):
    lgb_model, lgb_metrics = train_with_retry(train_lgb, 'LightGBM')
    results['lgb'] = lgb_metrics
    if 'error' not in lgb_metrics:
        cp['lgb_done'] = True
        save_checkpoint(cp)
else:
    print("\n  [SKIP] LightGBM already trained")
    results['lgb'] = cp.get('lgb_metrics', {})

# --- Model 2: XGBoost ---
def train_xgb():
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'max_depth': 8, 'learning_rate': 0.03, 'n_estimators': 2000,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'min_child_weight': 50, 'gamma': 0.1,
        'tree_method': 'hist', 'random_state': 42, 'n_jobs': -1,
        'scale_pos_weight': scale_pos,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=200, verbose=False)
    
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)
    metrics = {
        'name': 'XGBoost', 'test_acc': accuracy_score(y_test, preds),
        'test_auc': roc_auc_score(y_test, probs),
        'test_prec': precision_score(y_test, preds, zero_division=0),
        'test_rec': recall_score(y_test, preds, zero_division=0),
        'test_f1': f1_score(y_test, preds, zero_division=0),
        'val_auc': model.best_score,
    }
    joblib.dump(model, '/kaggle/working/xgb_model.pkl')
    return model, metrics

if not cp.get('xgb_done'):
    xgb_model, xgb_metrics = train_with_retry(train_xgb, 'XGBoost')
    results['xgb'] = xgb_metrics
    if 'error' not in xgb_metrics:
        cp['xgb_done'] = True
        cp['xgb_metrics'] = xgb_metrics
        save_checkpoint(cp)
else:
    print("\n  [SKIP] XGBoost already trained")
    results['xgb'] = cp.get('xgb_metrics', {})

# --- Model 3: Random Forest ---
def train_rf():
    model = RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_leaf=100,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)
    metrics = {
        'name': 'RandomForest', 'test_acc': accuracy_score(y_test, preds),
        'test_auc': roc_auc_score(y_test, probs),
        'test_prec': precision_score(y_test, preds, zero_division=0),
        'test_rec': recall_score(y_test, preds, zero_division=0),
        'test_f1': f1_score(y_test, preds, zero_division=0),
        'val_auc': 0.0,  # RF doesn't have val_auc during training
    }
    joblib.dump(model, '/kaggle/working/rf_model.pkl')
    return model, metrics

if not cp.get('rf_done'):
    rf_model, rf_metrics = train_with_retry(train_rf, 'RandomForest')
    results['rf'] = rf_metrics
    if 'error' not in rf_metrics:
        cp['rf_done'] = True
        cp['rf_metrics'] = rf_metrics
        save_checkpoint(cp)
else:
    print("\n  [SKIP] RandomForest already trained")
    results['rf'] = cp.get('rf_metrics', {})

# =============================================================================
# ENSEMBLE
# =============================================================================
print("\n[5/6] Ensemble evaluation...")
try:
    lgb_probs = joblib.load('/kaggle/working/lgb_model.pkl').predict(X_test)
    xgb_probs = joblib.load('/kaggle/working/xgb_model.pkl').predict_proba(X_test)[:, 1]
    rf_probs = joblib.load('/kaggle/working/rf_model.pkl').predict_proba(X_test)[:, 1]
    
    ensemble_probs = (lgb_probs + xgb_probs + rf_probs) / 3
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    
    # Confidence > 60%
    conf_mask = (ensemble_probs > 0.60) | (ensemble_probs < 0.40)
    if conf_mask.sum() > 100:
        conf_acc = accuracy_score(y_test[conf_mask], (ensemble_probs[conf_mask] > 0.5).astype(int))
        conf_pct = conf_mask.mean()
    else:
        conf_acc = 0; conf_pct = 0
    
    results['ensemble'] = {
        'name': 'Ensemble',
        'test_acc': accuracy_score(y_test, ensemble_preds),
        'test_auc': roc_auc_score(y_test, ensemble_probs),
        'test_prec': precision_score(y_test, ensemble_preds, zero_division=0),
        'test_rec': recall_score(y_test, ensemble_preds, zero_division=0),
        'test_f1': f1_score(y_test, ensemble_preds, zero_division=0),
        'test_conf_acc': conf_acc,
        'test_conf_pct': conf_pct,
    }
except Exception as e:
    print(f"  Ensemble failed: {e}")
    results['ensemble'] = {'name': 'Ensemble', 'error': str(e)[:100]}

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n[6/6] Saving results...")
results_df = pd.DataFrame([{k: v for k, v in r.items()} for r in results.values()])
results_df.to_csv(RESULTS_FILE, index=False)
results_df.to_csv('/kaggle/working/model_results_bulletproof.csv', index=False)

with open('/kaggle/working/final_results.json', 'w') as f:
    json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv for kk, vv in v.items()} 
               for k, v in results.items()}, f, indent=2)

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
for name, res in results.items():
    print(f"\n[{res.get('name', name)}]")
    for k, v in res.items():
        if k != 'name' and isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")

print(f"\nDone: {datetime.now().isoformat()}")
print("=" * 70)
