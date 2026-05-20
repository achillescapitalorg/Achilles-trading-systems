"""
Gold 1-Minute: V9 — ALL DATA, no filtering
============================================
Trains on complete 6.7M rows with raw direction targets.
No volatility filtering — every bar is up or down.
"""

import os, sys, glob, json, warnings, time
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, log_loss
from sklearn.feature_selection import mutual_info_classif

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except:
    HAS_SMOTE = False

warnings.filterwarnings('ignore')
np.random.seed(42)

CHECKPOINT_FILE = '/kaggle/working/checkpoint_v9.json'

def save_cp(state):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(state, f)

def load_cp():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {}

print("=" * 70)
print("V9: ALL DATA — NO FILTERING")
print("=" * 70)
print(f"Start: {datetime.now().isoformat()}")

cp = load_cp()
print(f"Checkpoint: {cp}")

# =============================================================================
# LOAD DATA
# =============================================================================
if cp.get('data_loaded'):
    print("\n[SKIP] Data loaded")
    df = pd.read_pickle('/kaggle/working/df_v9.pkl')
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
    df.to_pickle('/kaggle/working/df_v9.pkl')
    cp['data_loaded'] = True
    cp['shape'] = str(df.shape)
    save_cp(cp)
    print(f"  Loaded: {df.shape}")

# =============================================================================
# FEATURES
# =============================================================================
if cp.get('features_done'):
    print("\n[SKIP] Features done")
    F = pd.read_pickle('/kaggle/working/features_v9.pkl')
else:
    print("\n[2/6] Engineering features...")
    close = df['close']; high = df['high']; low = df['low']
    open_ = df['open']; volume = df['volume']; returns = close.pct_change()
    F = pd.DataFrame(index=df.index)
    
    F['ret_1'] = returns
    for w in [3, 5, 10, 20, 50, 100]:
        F[f'ret_{w}'] = close.pct_change(w)
    
    for span in [5, 10, 20, 50]:
        ema = close.ewm(span=span, adjust=False).mean()
        F[f'ema_{span}'] = (close - ema) / (ema + 1e-10)
    for window in [5, 10, 20, 50]:
        wma = close.rolling(window).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)
        F[f'wma_{window}'] = (close - wma) / (wma + 1e-10)
    
    for period in [7, 14, 21]:
        d = close.diff()
        ag = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        al = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        F[f'rsi_{period}'] = 100 - 100 / (1 + ag / (al + 1e-10))
    
    for k in [14, 21]:
        ll = low.rolling(k).min(); hh = high.rolling(k).max()
        F[f'stoch_k_{k}'] = 100 * (close - ll) / (hh - ll + 1e-10)
        F[f'stoch_d_{k}'] = F[f'stoch_k_{k}'].rolling(3).mean()
    
    for w in [14, 21]:
        hh = high.rolling(w).max(); ll = low.rolling(w).min()
        F[f'williams_r_{w}'] = -100 * (hh - close) / (hh - ll + 1e-10)
    
    for w in [20, 50]:
        tp = (high + low + close) / 3
        ma_tp = tp.rolling(w).mean()
        md_tp = tp.rolling(w).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        F[f'cci_{w}'] = (tp - ma_tp) / (0.015 * md_tp + 1e-10)
    
    macd = close.ewm(span=8, adjust=False).mean() - close.ewm(span=17, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    F['macd'] = macd; F['macd_signal'] = sig; F['macd_hist'] = macd - sig
    
    for p in [7, 14, 21]:
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        F[f'atr_{p}'] = tr.ewm(span=p, adjust=False).mean()
        F[f'atr_ratio_{p}'] = F[f'atr_{p}'] / (close + 1e-10)
    
    for w in [20, 50]:
        ma = close.rolling(w).mean(); std = close.rolling(w).std()
        F[f'bb_pos_{w}'] = (close - ma) / (std + 1e-10)
        F[f'bb_width_{w}'] = std / (ma + 1e-10)
    
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    F['tenkan_sen'] = (close - tenkan) / (tenkan + 1e-10)
    F['kijun_sen'] = (close - kijun) / (kijun + 1e-10)
    F['senkou_a'] = (close - senkou_a) / (senkou_a + 1e-10)
    F['senkou_b'] = (close - senkou_b) / (senkou_b + 1e-10)
    
    for w in [14, 30]:
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(w).mean()
        pdm = (high - high.shift(1)).clip(lower=0)
        mdm = (low.shift(1) - low).clip(lower=0)
        pdi = 100 * pdm.rolling(w).mean() / (atr + 1e-10)
        mdi = 100 * mdm.rolling(w).mean() / (atr + 1e-10)
        dx = 100 * abs(pdi - mdi) / (pdi + mdi + 1e-10)
        F[f'adx_{w}'] = dx.rolling(w).mean()
    
    F['vol_rel_5'] = volume / (volume.rolling(5).mean() + 1e-10)
    F['vol_rel_20'] = volume / (volume.rolling(20).mean() + 1e-10)
    F['vol_rel_50'] = volume / (volume.rolling(50).mean() + 1e-10)
    F['obv'] = (np.sign(close.diff()) * volume).cumsum()
    F['obv_ema'] = F['obv'].ewm(span=20, adjust=False).mean()
    
    F['hour'] = df.index.hour / 23.0
    F['dow'] = df.index.dayofweek / 6.0
    F['is_london'] = ((df.index.hour >= 8) & (df.index.hour <= 16)).astype(float)
    F['is_ny'] = ((df.index.hour >= 13) & (df.index.hour <= 21)).astype(float)
    
    typical = (high + low + close) / 3
    vwap = (typical * volume).cumsum() / (volume.cumsum() + 1e-10)
    F['vwap_dev'] = (close - vwap) / (vwap + 1e-10)
    F['close_loc'] = (close - low) / (high - low + 1e-10)
    for w in [10, 20, 50]:
        hh = high.rolling(w).max(); ll = low.rolling(w).min()
        F[f'price_loc_{w}'] = (close - ll) / (hh - ll + 1e-10)
    F['body'] = (close - open_) / (high - low + 1e-10)
    F['gap'] = (open_ - close.shift(1)) / close.shift(1)
    
    for w in [10, 30, 60]:
        F[f'realized_vol_{w}'] = returns.rolling(w).std() * np.sqrt(w)
    F['vol_regime'] = (F['realized_vol_30'] > F['realized_vol_30'].rolling(100).mean()).astype(float)
    
    F['rsi_macd'] = F['rsi_14'] * F['macd_hist']
    F['vol_price_loc'] = F['vol_rel_20'] * F['price_loc_20']
    F['atr_ret'] = F['atr_14'] * F['ret_1']
    F['adx_rsi'] = F['adx_14'] * F['rsi_14']
    F['cci_rsi'] = F['cci_20'] * F['rsi_14']
    F['stoch_rsi'] = F['stoch_k_14'] * F['rsi_14']
    
    for lag in [1, 2, 3, 5]:
        F[f'ret_1_lag_{lag}'] = F['ret_1'].shift(lag)
    
    # RAW DIRECTION TARGET — NO FILTERING, ALL ROWS
    for H in [20, 60]:
        ret = close.pct_change(H).shift(-H)
        F[f'target_{H}'] = (ret > 0).astype(int)  # 1 = up, 0 = down
    
    feature_cols = [c for c in F.columns if not c.startswith('target_')]
    F = F.replace([np.inf, -np.inf], np.nan)
    F[feature_cols] = F[feature_cols].ffill().fillna(0)
    # Only drop NaN targets (last H rows), keep ALL other rows
    F = F.dropna(subset=[c for c in F.columns if c.startswith('target_')])
    
    F.to_pickle('/kaggle/working/features_v9.pkl')
    cp['features_done'] = True
    cp['n_features'] = len(feature_cols)
    cp['n_rows'] = len(F)
    save_cp(cp)
    print(f"  Features: {len(feature_cols)} | Rows: {len(F):,}")

# =============================================================================
# TRAIN
# =============================================================================
print("\n[3/6] Training on ALL data...")

all_results = []

for HORIZON in [20, 60]:
    print(f"\n{'='*70}")
    print(f"HORIZON = {HORIZON}")
    print(f"{'='*70}")
    
    target_col = f'target_{HORIZON}'
    feature_cols = [c for c in F.columns if not c.startswith('target_')]
    X_all = F[feature_cols]
    y_all = F[target_col].values
    
    n = len(X_all)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train, y_train = X_all.iloc[:train_end], y_all[:train_end]
    X_val, y_val = X_all.iloc[train_end:val_end], y_all[train_end:val_end]
    X_test, y_test = X_all.iloc[val_end:], y_all[val_end:]
    
    print(f"  Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")
    print(f"  Class balance: {np.bincount(y_train)}")
    
    # Feature selection
    sample_size = min(300000, len(X_train))
    idx = np.random.choice(len(X_train), sample_size, replace=False)
    mi = mutual_info_classif(X_train.iloc[idx], y_train[idx], random_state=42, n_neighbors=5)
    mi_df = pd.DataFrame({'feature': X_train.columns, 'mi': mi}).sort_values('mi', ascending=False)
    selected = mi_df.head(40)['feature'].tolist()
    
    X_train = X_train[selected]
    X_val = X_val[selected]
    X_test = X_test[selected]
    
    # SMOTE
    if HAS_SMOTE:
        print("  SMOTE...")
        smote = SMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=5)
        if len(X_train) > 800000:
            sm_idx = np.random.choice(len(X_train), 800000, replace=False)
            X_train_sm, y_train_sm = X_train.iloc[sm_idx], y_train[sm_idx]
        else:
            X_train_sm, y_train_sm = X_train, y_train
        X_train, y_train = smote.fit_resample(X_train_sm, y_train_sm)
        print(f"  After SMOTE: {np.bincount(y_train)}")
    
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    # LightGBM
    print(f"  >>> LightGBM H={HORIZON}...")
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
    lgb_model = lgb.train(lgb_params, dtrain, num_boost_round=5000,
                          valid_sets=[dtrain, dval], valid_names=['train', 'valid'],
                          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    lgb_probs = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    
    # XGBoost
    print(f"  >>> XGBoost H={HORIZON}...")
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
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # Random Forest
    print(f"  >>> RandomForest H={HORIZON}...")
    rf_model = RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_leaf=100,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    for name, probs in [('LGB', lgb_probs), ('XGB', xgb_probs), ('RF', rf_probs)]:
        preds = (probs > 0.5).astype(int)
        for thresh in [0.55, 0.60, 0.65, 0.70]:
            conf_mask = (probs > thresh) | (probs < (1 - thresh))
            if conf_mask.sum() > 100:
                conf_acc = accuracy_score(y_test[conf_mask], (probs[conf_mask] > 0.5).astype(int))
                conf_pct = conf_mask.mean()
            else:
                conf_acc = 0; conf_pct = 0
            all_results.append({
                'horizon': HORIZON, 'model': name, 'conf_thresh': thresh,
                'conf_acc': conf_acc, 'conf_pct': conf_pct,
                'overall_acc': accuracy_score(y_test, preds),
                'overall_auc': roc_auc_score(y_test, probs),
                'overall_f1': f1_score(y_test, preds, zero_division=0),
            })
    
    # Ensemble
    ens_probs = (lgb_probs + xgb_probs + rf_probs) / 3
    ens_preds = (ens_probs > 0.5).astype(int)
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        conf_mask = (ens_probs > thresh) | (ens_probs < (1 - thresh))
        if conf_mask.sum() > 100:
            conf_acc = accuracy_score(y_test[conf_mask], (ens_probs[conf_mask] > 0.5).astype(int))
            conf_pct = conf_mask.mean()
        else:
            conf_acc = 0; conf_pct = 0
        all_results.append({
            'horizon': HORIZON, 'model': 'ENSEMBLE', 'conf_thresh': thresh,
            'conf_acc': conf_acc, 'conf_pct': conf_pct,
            'overall_acc': accuracy_score(y_test, ens_preds),
            'overall_auc': roc_auc_score(y_test, ens_probs),
            'overall_f1': f1_score(y_test, ens_preds, zero_division=0),
        })
    
    joblib.dump({'lgb': lgb_model, 'xgb': xgb_model, 'rf': rf_model, 'features': selected},
                f'/kaggle/working/models_h{HORIZON}_v9.pkl')

# Save
print("\n[4/6] Saving...")
results_df = pd.DataFrame(all_results)
results_df.to_csv('/kaggle/working/final_results_v9.csv', index=False)

best = results_df.loc[results_df['conf_acc'].idxmax()]
print("\n" + "=" * 70)
print("BEST CONFIG")
print("=" * 70)
print(f"  {best['model']} H={best['horizon']} thresh={best['conf_thresh']}")
print(f"  Confident Accuracy: {best['conf_acc']:.4f} ({best['conf_acc']*100:.2f}%)")
print(f"  Coverage: {best['conf_pct']:.2%}")
print(f"  Overall Accuracy: {best['overall_acc']:.4f}")

print("\nTOP 10 BY CONFIDENT ACCURACY")
for _, row in results_df.sort_values('conf_acc', ascending=False).head(10).iterrows():
    print(f"  {row['model']:10} H={row['horizon']:2} thresh={row['conf_thresh']:.2f}  conf_acc={row['conf_acc']:.4f}  coverage={row['conf_pct']:.2%}")

print(f"\nDone: {datetime.now().isoformat()}")
print("=" * 70)
