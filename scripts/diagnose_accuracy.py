"""
MODEL ACCURACY DIAGNOSTICS
Finds why ensemble is stuck at 51% / 0.52 AUC
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ─── CONFIG ───
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
BETA_DIR = Path("data/beta_testing/processed")

# Try to find files
def find_file(name, candidates):
    for c in candidates:
        if c.exists():
            return c
    return None

csv_path = find_file("csv", [
    DATA_DIR / "gold_2025_2026.csv",
    BETA_DIR / "gold_2025_2026.csv",
    Path("XAU_1m_data.csv")
])

model_paths = {
    'lgb_h20': find_file("lgb_h20", [MODEL_DIR / "gold_1m_lgb.pkl", MODEL_DIR / "beta_h20_lgb.pkl", BETA_DIR / "models" / "beta_h20_lgb.pkl"]),
    'xgb_h20': find_file("xgb_h20", [MODEL_DIR / "gold_1m_xgb.pkl", MODEL_DIR / "beta_h20_xgb.pkl", BETA_DIR / "models" / "beta_h20_xgb.ubj"]),
    'rf_h20': find_file("rf_h20", [MODEL_DIR / "gold_1m_rf.pkl", MODEL_DIR / "beta_h20_rf.pkl", BETA_DIR / "models" / "beta_h20_rf.pkl"]),
}

print("=" * 70)
print("MODEL ACCURACY DIAGNOSTICS")
print("=" * 70)

# ─── 1. LOAD DATA ───
print("\n[1] Loading data...")
if not csv_path:
    print("  FAIL CSV not found")
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"  OK Loaded {len(df):,} rows")

# ─── 2. LOAD MODELS ───
print("\n[2] Loading models...")
models = {}
for name, path in model_paths.items():
    if not path:
        print(f"  FAIL {name}: not found")
        continue
    try:
        m = joblib.load(path)
        # Handle dict wrapper
        if isinstance(m, dict) and 'model' in m:
            m = m['model']
        models[name] = m
        print(f"  OK {name}: loaded")
    except Exception as e:
        print(f"  FAIL {name}: {e}")

if len(models) < 2:
    print("  Not enough models loaded to continue")
    sys.exit(1)

# ─── 3. BUILD FEATURES & TARGET ───
print("\n[3] Building features...")

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from beta_testing.features import compute_1m_features
    features = compute_1m_features(df)
    print(f"  OK Features computed: {features.shape}")
except Exception as e:
    print(f"  FAIL Feature import failed: {e}")
    print("  Trying inline basic features...")
    
    # Fallback basic features
    close = df['close']
    features = pd.DataFrame(index=df.index)
    features['ret_1'] = close.pct_change()
    features['ret_5'] = close.pct_change(5)
    features['ret_10'] = close.pct_change(10)
    features['ema_5'] = (close - close.ewm(5).mean()) / close
    features['ema_20'] = (close - close.ewm(20).mean()) / close
    features['rsi_14'] = 50  # placeholder
    features['atr_14'] = (df['high'] - df['low']).rolling(14).mean() / close
    features['macd'] = close.ewm(8).mean() - close.ewm(17).mean()
    features['bb_pos'] = (close - close.rolling(20).mean()) / close.rolling(20).std()
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    print(f"  WARN Using fallback features: {features.shape}")

# Target: 5-bar direction
horizon = 5
features['target'] = (features['ret_1'].shift(-horizon) > 0).astype(int)

# Clean
feat_cols = [c for c in features.columns if not c.startswith('target')]
X = features[feat_cols].fillna(0)
y = features['target']

# Temporal split: train 70%, val 15%, test 15%
n = len(X)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"  Test up-ratio: {y_test.mean():.3f}")

# ─── 4. SHAP IMPORTANCE ANALYSIS ───
print("\n[4] SHAP Importance Analysis...")
print("  (This may take 1-2 minutes)")

shap_df = None
try:
    import shap
    
    # Use a sample for speed
    sample_size = min(2000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    
    # LGB SHAP
    if 'lgb_h20' in models:
        lgb_model = models['lgb_h20']
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, shap_values might be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        importance = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Top 20 SHAP features:")
        for _, row in shap_df.head(20).iterrows():
            bar = "*" * int(row['importance'] / shap_df['importance'].max() * 30)
            print(f"    {row['importance']:.4f} {bar} {row['feature']}")
        
        # Critical check
        top_imp = shap_df['importance'].iloc[0]
        if top_imp < 0.05:
            print(f"\n  CRITICAL: Top feature importance is only {top_imp:.4f}")
            print(f"     Features have almost no predictive power.")
            print(f"     You need entirely new features (cross-asset, macro).")
        elif top_imp < 0.10:
            print(f"\n  WARNING: Top feature importance is {top_imp:.4f}")
            print(f"     Weak signal. Consider adding cross-asset features.")
        else:
            print(f"\n  OK Top feature importance is {top_imp:.4f} -- decent signal")
        
        # Save plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig('data/shap_summary.png', dpi=150, bbox_inches='tight')
        print(f"  Saved SHAP plot to data/shap_summary.png")
        
        shap_df.to_json('data/shap_importance.json', orient='records')
        
except ImportError:
    print("  WARN shap not installed. Run: pip install shap")
    print("  Skipping SHAP analysis...")
except Exception as e:
    print(f"  FAIL SHAP failed: {e}")

# ─── 5. FEATURE-TARGET CORRELATION ───
print("\n[5] Feature-Target Correlation...")

corrs = X_train.corrwith(y_train).abs().sort_values(ascending=False)
print(f"\n  Top 20 feature-target correlations:")
for feat, corr in corrs.head(20).items():
    bar = "*" * int(corr / corrs.max() * 30)
    print(f"    {corr:.4f} {bar} {feat}")

max_corr = corrs.max()
if max_corr < 0.03:
    print(f"\n  CRITICAL: Max feature-target correlation is {max_corr:.4f}")
    print(f"     No feature predicts the target. The signal doesn't exist in this data.")
elif max_corr < 0.05:
    print(f"\n  WARNING: Max correlation is {max_corr:.4f}")
    print(f"     Very weak signal. Need stronger features.")
else:
    print(f"\n  OK Max correlation is {max_corr:.4f} -- some signal exists")

corrs.to_json('data/feature_correlations.json')

# ─── 6. LOOK-AHEAD BIAS TEST ───
print("\n[6] Look-Ahead Bias Test (Shuffle Test)...")

# If features are forward-only, shuffling rows shouldn't change feature distributions
original_means = X_test.mean()
shuffled = X_test.sample(frac=1, random_state=99).reset_index(drop=True)
shuffled_means = shuffled.mean()

mean_diff = (original_means - shuffled_means).abs().max()
print(f"  Max feature mean difference after shuffle: {mean_diff:.6f}")

if mean_diff > 0.01:
    print(f"  LOOK-AHEAD BIAS DETECTED!")
    print(f"     Features change when shuffled -- they contain future information.")
    # Find which features changed
    changed = (original_means - shuffled_means).abs()
    bad_feats = changed[changed > 0.01].sort_values(ascending=False)
    print(f"     Biased features: {list(bad_feats.head(5).index)}")
else:
    print(f"  OK No look-ahead bias detected")

# ─── 7. REGIME-SPECIFIC ACCURACY ───
print("\n[7] Regime-Specific Accuracy...")

# Simple regime labels based on volatility
returns = df['close'].pct_change().fillna(0)
vol = returns.rolling(50).std() * np.sqrt(252 * 24 * 60)
trend = (df['close'] > df['close'].ewm(20).mean()).astype(int)

regime_labels = pd.Series('NORMAL', index=df.index)
regime_labels[vol > vol.quantile(0.8)] = 'HIGH_VOL'
regime_labels[(vol < vol.quantile(0.3)) & (trend == 1)] = 'TREND_UP'
regime_labels[(vol < vol.quantile(0.3)) & (trend == 0)] = 'TREND_DOWN'

# Align with test set
test_regimes = regime_labels.iloc[val_end:]

if 'lgb_h20' in models:
    from sklearn.metrics import accuracy_score
    lgb_preds = models['lgb_h20'].predict(X_test)
    
    print(f"\n  Accuracy by regime (LGB H=20):")
    for regime in test_regimes.unique():
        mask = test_regimes == regime
        if mask.sum() < 100:
            continue
        acc = accuracy_score(y_test[mask], (lgb_preds[mask] > 0.5).astype(int))
        baseline = max(y_test[mask].mean(), 1 - y_test[mask].mean())
        print(f"    {regime:12s}: {acc:.2%} (baseline: {baseline:.2%}, n={mask.sum()})")

# ─── 8. TARGET ANALYSIS ───
print("\n[8] Target Variable Analysis...")

future_returns = features['ret_1'].shift(-horizon).dropna()
print(f"  Future return distribution (horizon={horizon}):")
print(f"    Mean: {future_returns.mean():.6f}")
print(f"    Std:  {future_returns.std():.4f}")
print(f"    Skew: {future_returns.skew():.3f}")
print(f"    >0:   {(future_returns > 0).mean():.2%}")
print(f"    <-0.1%: {(future_returns < -0.001).mean():.2%}")
print(f"    >+0.1%: {(future_returns > 0.001).mean():.2%}")
print(f"    |ret|<0.1%: {(future_returns.abs() < 0.001).mean():.2%}")

# Check autocorrelation of returns
autocorr = future_returns.autocorr(lag=1)
print(f"    Autocorr(lag=1): {autocorr:.4f}")
if abs(autocorr) < 0.01:
    print(f"    Returns are essentially uncorrelated -- unpredictable")

# ─── 9. MODEL AGREEMENT ANALYSIS ───
print("\n[9] Model Agreement Analysis...")

if len(models) >= 2:
    preds = {}
    for name, m in models.items():
        preds[name] = m.predict(X_test)
    
    pred_df = pd.DataFrame(preds)
    pred_df['ensemble'] = pred_df.mean(axis=1)
    pred_df['y'] = y_test.values
    
    # When all models agree, is accuracy higher?
    pred_df['agreement'] = (
        (pred_df['lgb_h20'] > 0.5).astype(int) +
        (pred_df['xgb_h20'] > 0.5).astype(int) +
        (pred_df['rf_h20'] > 0.5).astype(int)
    )
    
    print(f"\n  Accuracy by model agreement:")
    for agree in [0, 1, 2, 3]:
        mask = pred_df['agreement'] == agree
        if mask.sum() < 50:
            continue
        acc = (pred_df.loc[mask, 'ensemble'] > 0.5).astype(int).eq(pred_df.loc[mask, 'y']).mean()
        print(f"    {agree}/3 agree: {acc:.2%} (n={mask.sum()})")

# ─── 10. SUMMARY & RECOMMENDATIONS ───
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

results = {
    'max_shap_importance': float(shap_df['importance'].iloc[0]) if shap_df is not None else None,
    'max_feature_correlation': float(max_corr),
    'lookahead_bias_detected': mean_diff > 0.01,
    'future_return_autocorr': float(autocorr) if 'autocorr' in locals() else None,
    'noise_pct': float((future_returns.abs() < 0.001).mean()) if 'future_returns' in locals() else None,
}

print(f"\nKey Metrics:")
for k, v in results.items():
    if v is not None:
        print(f"  {k}: {v:.4f}")

print(f"\nINTERPRETATION:")

if results.get('max_shap_importance', 0) < 0.05:
    print(f"  FEATURES ARE USELESS")
    print(f"     Your 50 technical indicators contain no predictive signal.")
    print(f"     You MUST add cross-asset features (DXY, yields, VIX, etc.)")
    print(f"     OR switch to a completely different approach (order flow, microstructure).")

if results.get('noise_pct', 0) > 0.5:
    print(f"  TARGET IS MOSTLY NOISE")
    print(f"     {results['noise_pct']:.1%} of 5-bar returns are within +/-0.1%.")
    print(f"     Predicting direction is hopeless -- the movement is random.")
    print(f"     Switch to predicting ONLY moves > 0.2% (strong signals).")

if results.get('lookahead_bias_detected', False):
    print(f"  LOOK-AHEAD BIAS CONFIRMED")
    print(f"     Some features use future data. This inflates training accuracy")
    print(f"     but gives random results on test set.")
    print(f"     Audit every feature calculation for .shift(1) usage.")

if results.get('future_return_autocorr', 0) < 0.01:
    print(f"  NO TEMPORAL STRUCTURE")
    print(f"     5-bar returns have zero autocorrelation.")
    print(f"     The past does not predict the future at this horizon.")
    print(f"     Try longer horizons (15m, 1h) or different targets.")

print(f"\nNEXT STEPS (in order):")
print(f"  1. If features useless -> Add DXY, yields, VIX, silver, oil, SPX")
print(f"  2. If target noisy -> Use 3-class target (strong up / flat / strong down)")
print(f"  3. If lookahead bias -> Fix features, retrain, retest")
print(f"  4. If no temporal structure -> Abandon 1m direction, try return magnitude or 15m")

# Ensure data dir exists
Path("data").mkdir(exist_ok=True)
with open('data/diagnostic_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to data/diagnostic_results.json")
