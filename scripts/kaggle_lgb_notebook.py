"""
===========================================================================
KAGGLE NOTEBOOK: LightGBM Gold 1M Model Training
===========================================================================
INSTRUCTIONS:
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. On the right panel, click "Add Input" → Search "XAUUSD Gold Price Historical Data"
   → Select the dataset by Novandra Anugrah → Click "Add"
4. Copy-paste each "Cell" below into the notebook
5. Click "Run All"
6. When finished, download `gold_1m_lgb.pkl` from the Output panel
7. Place it in: Achilles-trading-systems/data/beta_testing/processed/models/
===========================================================================
"""

# =============================================================================
# CELL 1: Imports
# =============================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import joblib
import warnings
warnings.filterwarnings('ignore')
print("Imports ready")

# =============================================================================
# CELL 2: Load Dataset (auto-detects Kaggle input path)
# =============================================================================
import os

# Kaggle input path (auto-set when you add the dataset to the notebook)
kaggle_path = '/kaggle/input/xauusd-gold-price-historical-data-2004-2024/XAU_1m_data.csv'
local_path = 'XAU_1m_data.csv'

csv_path = kaggle_path if os.path.exists(kaggle_path) else local_path

print(f"Loading from: {csv_path}")
df = pd.read_csv(csv_path, sep=None, engine='python')
df.columns = [c.lower().strip() for c in df.columns]
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').sort_index()

print(f"COMPLETE DATASET: {len(df):,} rows")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# =============================================================================
# CELL 3: Clean & Preprocess
# =============================================================================
print("\nCleaning data (filling small gaps, removing weekends)...")

df_full = df.resample('1min').asfreq()
df_full[['open','high','low','close','volume']] = df_full[['open','high','low','close','volume']].ffill(limit=4)
df_clean = df_full[df_full.index.to_series().diff() <= pd.Timedelta(minutes=5)].copy()
df_clean = df_clean.dropna(subset=['open','high','low','close'])
df_clean['minutes_since_last_bar'] = 1.0

print(f"Cleaned: {len(df_clean):,} rows")
print(f"Added bars: {len(df_clean) - len(df):,}")

# =============================================================================
# CELL 4: Feature Engineering (~50 features)
# =============================================================================
print("\nComputing features...")

f = pd.DataFrame(index=df_clean.index)
close = df_clean['close']
high = df_clean['high']
low = df_clean['low']
open_p = df_clean['open']
volume = df_clean['volume']

# Price returns
returns = close.pct_change()
f['ret_1'] = returns
f['ret_5'] = close.pct_change(5)
f['ret_10'] = close.pct_change(10)
f['ret_20'] = close.pct_change(20)
f['sign_ret_1'] = np.sign(returns)
f['abs_ret_1'] = returns.abs()

# Momentum: EMAs
for span in [5, 10, 20, 50]:
    ema = close.ewm(span=span).mean()
    f[f'ema_{span}'] = (close - ema) / (ema + 1e-10)
    f[f'ema_slope_{span}'] = ema.diff(span) / (ema + 1e-10)

f['ema_5_10_cross'] = f['ema_5'] - f['ema_10']
f['ema_10_20_cross'] = f['ema_10'] - f['ema_20']

# RSI
for period in [7, 14, 21]:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period).mean()
    rs = gain / (loss + 1e-10)
    f[f'rsi_{period}'] = 100 - (100 / (1 + rs))

# MACD
ema_8 = close.ewm(span=8).mean()
ema_17 = close.ewm(span=17).mean()
macd = ema_8 - ema_17
signal = macd.ewm(span=5).mean()
f['macd'] = macd / (close * 0.001 + 1e-10)
f['macd_signal'] = signal / (close * 0.001 + 1e-10)
f['macd_hist'] = f['macd'] - f['macd_signal']
f['macd_hist_slope'] = f['macd_hist'].diff(3)

# Volatility
tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
for period in [7, 14, 30, 50]:
    atr = tr.rolling(period).mean()
    f[f'atr_{period}'] = atr / (close + 1e-10)
f['atr_ratio'] = f['atr_14'] / (f['atr_50'] + 1e-10)

# Bollinger
bb_ma = close.rolling(20).mean()
bb_std = close.rolling(20).std()
f['bb_position'] = (close - bb_ma) / (2 * bb_std + 1e-10)
f['bb_width'] = bb_std / (bb_ma + 1e-10)
f['bb_squeeze'] = f['bb_width'] / (f['bb_width'].rolling(50).max() + 1e-10)

# Realized vol
log_ret = np.log(close / close.shift())
f['realized_vol_10'] = log_ret.rolling(10).std()
f['realized_vol_30'] = log_ret.rolling(30).std()
f['vol_ratio'] = f['realized_vol_10'] / (f['realized_vol_30'] + 1e-10)

# Mean reversion
for period in [20, 50, 100]:
    ma = close.rolling(period).mean()
    f[f'dist_ma_{period}'] = (close - ma) / (tr.rolling(14).mean() + 1e-10)

lowest_14 = low.rolling(14).min()
highest_14 = high.rolling(14).max()
f['stoch_k'] = 100 * (close - lowest_14) / (highest_14 - lowest_14 + 1e-10)
f['stoch_d'] = f['stoch_k'].rolling(3).mean()

tp = (high + low + close) / 3
cci_ma = tp.rolling(20).mean()
cci_std = tp.rolling(20).std()
f['cci'] = (tp - cci_ma) / (0.015 * cci_std + 1e-10)

# Volume
vol_ma_20 = volume.rolling(20).mean()
f['volume_ratio'] = volume / (vol_ma_20 + 1e-10)
f['volume_ratio_ma'] = f['volume_ratio'].rolling(5).mean()
obv = (np.sign(close.diff()) * volume).cumsum()
f['obv_slope'] = obv.diff(10) / (obv.abs() + 1e-10)

# Microstructure
f['bar_range'] = (high - low) / (close + 1e-10)
f['bar_body'] = (close - open_p).abs() / (high - low + 1e-10)
f['bar_direction'] = np.sign(close - open_p)

f['consecutive_up'] = (returns > 0).astype(int).groupby(((returns <= 0)).astype(int).cumsum()).cumsum()
f['consecutive_down'] = (returns < 0).astype(int).groupby(((returns >= 0)).astype(int).cumsum()).cumsum()

# Temporal
f['hour'] = df_clean.index.hour
f['minute'] = df_clean.index.minute
f['is_london'] = ((df_clean.index.hour >= 8) & (df_clean.index.hour < 17)).astype(float)
f['is_ny'] = ((df_clean.index.hour >= 13) & (df_clean.index.hour < 22)).astype(float)
f['is_overlap'] = ((df_clean.index.hour >= 13) & (df_clean.index.hour < 17)).astype(float)

# Target
horizon = 5
f['target_ret_5'] = close.pct_change(horizon).shift(-horizon)
f['target_dir_5'] = np.sign(f['target_ret_5'])

# Clean
f = f.replace([np.inf, -np.inf], np.nan)
f = f.ffill().fillna(0)

print(f"Features computed: {len(f.columns)} columns")

# =============================================================================
# CELL 5: Prepare Train/Val/Test Splits
# =============================================================================
print("\nPreparing splits...")

target_cols = [c for c in f.columns if c.startswith('target_')]
y = f['target_dir_5'].values
y_binary = np.where(y > 0, 1, 0)
zero_mask = y == 0
if zero_mask.sum() > 0:
    y_binary[zero_mask] = np.random.RandomState(42).randint(0, 2, size=zero_mask.sum())

X = f.drop(columns=target_cols)

# Temporal split (chronological — no shuffle)
split = int(len(X) * 0.85)
split2 = int(split * 0.85)
X_train, y_train = X.iloc[:split2], y_binary[:split2]
X_val, y_val = X.iloc[split2:split], y_binary[split2:split]
X_test, y_test = X.iloc[split:], y_binary[split:]

print(f"Train: {len(X_train):,} ({100*len(X_train)/len(X):.0f}%)")
print(f"Val:   {len(X_val):,} ({100*len(X_val)/len(X):.0f}%)")
print(f"Test:  {len(X_test):,} ({100*len(X_test)/len(X):.0f}%)")
print(f"Train up-ratio: {y_train.mean():.3f}")

# =============================================================================
# CELL 6: Train LightGBM (Primary Model)
# =============================================================================
print("\n" + "="*60)
print("TRAINING LIGHTGBM")
print("="*60)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': 5,
    'min_data_in_leaf': 300,
    'learning_rate': 0.01,
    'feature_fraction': 0.65,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'lambda_l1': 0.5,
    'lambda_l2': 2.0,
    'max_bin': 127,
    'verbose': -1,
    'seed': 42,
    'deterministic': True,
}

n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
params['scale_pos_weight'] = n_neg / max(n_pos, 1)

train_data = lgb.Dataset(X_train, y_train)
val_data = lgb.Dataset(X_val, y_val, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=3000,
    valid_sets=[train_data, val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=200, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)

print(f"\n[LGB] Best iteration: {model.best_iteration}")
print(f"[LGB] Best validation AUC: {model.best_score['valid_1']['auc']:.4f}")

# =============================================================================
# CELL 7: Evaluate on Test Set
# =============================================================================
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

preds = model.predict(X_test, num_iteration=model.best_iteration)
pred_labels = (preds > 0.5).astype(int)

acc = accuracy_score(y_test, pred_labels)
auc = roc_auc_score(y_test, preds)
loss = log_loss(y_test, preds.clip(1e-6, 1 - 1e-6))

print(f"Accuracy:     {acc:.4f}  (baseline: {max(y_test.mean(), 1-y_test.mean()):.4f})")
print(f"AUC:          {auc:.4f}")
print(f"LogLoss:      {loss:.4f}")

# =============================================================================
# CELL 8: Feature Importance
# =============================================================================
print("\n" + "="*60)
print("TOP 15 FEATURES")
print("="*60)

importance = pd.Series(
    model.feature_importance(importance_type='gain'),
    index=X.columns
).sort_values(ascending=False)

for i, (feat, score) in enumerate(importance.head(15).items(), 1):
    print(f"{i:2d}. {feat:25s} {score:>10.0f}")

# =============================================================================
# CELL 9: Save Model
# =============================================================================
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

joblib.dump({
    'model': model,
    'params': params,
    'features': list(X.columns),
    'best_iteration': model.best_iteration,
    'metrics': {
        'accuracy': float(acc),
        'auc': float(auc),
        'logloss': float(loss),
        'baseline': float(max(y_test.mean(), 1-y_test.mean()))
    }
}, 'gold_1m_lgb.pkl')

print("\n✅ Saved: gold_1m_lgb.pkl")
print("\nNEXT STEPS:")
print("1. Download 'gold_1m_lgb.pkl' from the Output panel (right side)")
print("2. Place it in your project: data/beta_testing/processed/models/")
print("3. Tell me when it's done — then I'll give you the XGBoost notebook")
