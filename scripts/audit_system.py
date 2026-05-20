#!/usr/bin/env python3
"""
SYSTEM AUDIT — Performance + Model Integrity + Hardcoding Detection
Run: python scripts/audit_system.py
"""

import os
import sys
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np

print("=" * 70)
print("SYSTEM AUDIT — Performance + Model Integrity + Hardcoding Detection")
print("=" * 70)

# ─── CONFIG: Adjust paths if your structure is different ───
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REGIME_DIR = Path("data/regime_cache")
BETA_DIR = Path("data/beta_testing/processed")  # Kimi's structure

# Try multiple possible paths
possible_csv = [
    DATA_DIR / "gold_2025_2026.csv",
    BETA_DIR / "gold_2025_2026.csv",
    Path("XAU_1m_data.csv"),
]

possible_models = {
    'LGB_H20': [MODEL_DIR / "beta_h20_lgb.pkl", BETA_DIR / "models" / "beta_h20_lgb.pkl"],
    'XGB_H20': [MODEL_DIR / "beta_h20_xgb.ubj", BETA_DIR / "models" / "beta_h20_xgb.ubj"],
    'RF_H20': [MODEL_DIR / "beta_h20_rf.pkl", BETA_DIR / "models" / "beta_h20_rf.pkl"],
    'LGB_H60': [MODEL_DIR / "beta_h60_lgb.pkl", BETA_DIR / "models" / "beta_h60_lgb.pkl"],
    'XGB_H60': [MODEL_DIR / "beta_h60_xgb.ubj", BETA_DIR / "models" / "beta_h60_xgb.ubj"],
    'RF_H60': [MODEL_DIR / "beta_h60_rf.pkl", BETA_DIR / "models" / "beta_h60_rf.pkl"],
    'HMM': [MODEL_DIR / "regime_models_hmm.pkl", BETA_DIR / "models" / "regime_models_hmm.pkl"],
    'PREDICTOR': [MODEL_DIR / "regime_models_predictor.pkl", BETA_DIR / "models" / "regime_models_predictor.pkl"],
}

possible_regime_feat = [
    REGIME_DIR / "regime_features_precomputed.parquet",
    BETA_DIR / "regime_cache" / "regime_features_precomputed.parquet",
]

# ─── 1. FILE DISCOVERY ───
print("\n[1] FILE DISCOVERY")
print("-" * 50)

csv_path = None
for p in possible_csv:
    if p.exists():
        csv_path = p
        print(f"  OK CSV found: {p} ({p.stat().st_size/1e6:.1f} MB)")
        break
if not csv_path:
    print(f"  FAIL CSV NOT FOUND. Tried: {possible_csv}")

model_paths = {}
for name, paths in possible_models.items():
    for p in paths:
        if p.exists():
            model_paths[name] = p
            print(f"  OK {name} model: {p} ({p.stat().st_size/1e6:.1f} MB)")
            break
    if name not in model_paths:
        print(f"  FAIL {name} model NOT FOUND")

feat_path = None
for p in possible_regime_feat:
    if p.exists():
        feat_path = p
        print(f"  OK Regime features: {p} ({p.stat().st_size/1e6:.1f} MB)")
        break
if not feat_path:
    print(f"  FAIL Regime features NOT FOUND")

# ─── 2. MODEL LOAD TIME ───
print("\n[2] MODEL LOAD TIME")
print("-" * 50)

loaded_models = {}

def load_model(name, path):
    t0 = time.time()
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        if name.startswith('LGB'):
            import joblib
            d = joblib.load(path)
            m = d.get('model', d)
            t1 = time.time()
            print(f"  OK {name}: {(t1-t0)*1000:.0f}ms")
            return m
        elif name.startswith('XGB'):
            import xgboost as xgb
            m = xgb.XGBClassifier()
            m.load_model(str(path))
            t1 = time.time()
            print(f"  OK {name}: {(t1-t0)*1000:.0f}ms")
            return m
        elif name.startswith('RF'):
            import joblib
            m = joblib.load(path)
            t1 = time.time()
            print(f"  OK {name}: {(t1-t0)*1000:.0f}ms")
            return m
        else:
            import joblib
            m = joblib.load(path)
            t1 = time.time()
            print(f"  OK {name}: {(t1-t0)*1000:.0f}ms")
            return m
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        return None

for name, path in model_paths.items():
    loaded_models[name] = load_model(name, path)

# ─── 3. MODEL INTEGRITY ───
print("\n[3] MODEL INTEGRITY")
print("-" * 50)

# Check any loaded LGB model
lgb_models = {k: v for k, v in loaded_models.items() if k.startswith('LGB') and v is not None}
if lgb_models:
    for name, m in lgb_models.items():
        # LightGBM Booster objects
        attrs = [a for a in ['predict', 'feature_name', 'num_trees', 'best_iteration'] if hasattr(m, a)]
        print(f"  {name} attributes: {attrs}")
        if 'feature_name' in attrs:
            try:
                n_feat = len(m.feature_name())
                print(f"  {name} features: {n_feat}")
            except:
                pass
        if 'num_trees' in attrs:
            try:
                print(f"  {name} trees: {m.num_trees()}")
            except:
                pass
        elif hasattr(m, 'num_boosted_rounds'):
            try:
                print(f"  {name} trees: {m.num_boosted_rounds()}")
            except:
                pass

xgb_models = {k: v for k, v in loaded_models.items() if k.startswith('XGB') and v is not None}
if xgb_models:
    for name, m in xgb_models.items():
        if hasattr(m, 'get_booster'):
            print(f"  {name}: Valid booster, rounds={m.get_booster().num_boosted_rounds()}")
        elif hasattr(m, 'predict'):
            print(f"  WARN {name}: Has predict but no get_booster() — might be sklearn wrapper")
        else:
            print(f"  FAIL {name}: No predict method! Type={type(m)}")

rf_models = {k: v for k, v in loaded_models.items() if k.startswith('RF') and v is not None}
if rf_models:
    for name, m in rf_models.items():
        from sklearn.ensemble import RandomForestClassifier
        if isinstance(m, RandomForestClassifier):
            print(f"  {name}: Valid RandomForest, trees={m.n_estimators}, features={m.n_features_in_}")
        else:
            print(f"  WARN {name}: Type={type(m)}, not RandomForestClassifier")

# ─── 4. HARDCODING DETECTION ───
print("\n[4] HARDCODING DETECTION")
print("-" * 50)
print("  Testing if models produce different outputs for different inputs...")

results = {}

ml_models = {k: v for k, v in loaded_models.items() if k in ['LGB_H20', 'XGB_H20', 'RF_H20', 'LGB_H60', 'XGB_H60', 'RF_H60'] and v is not None}
if ml_models:
    np.random.seed(42)
    
    # Try to infer feature count from first LGB model
    n_features = 50  # default guess
    first_lgb = next((v for k, v in ml_models.items() if k.startswith('LGB')), None)
    if first_lgb and hasattr(first_lgb, 'feature_name'):
        try:
            n_features = len(first_lgb.feature_name())
        except:
            pass
    
    X1 = np.random.randn(1, n_features)
    X2 = np.random.randn(1, n_features) * 3 + 10  # Very different
    
    for name, model in ml_models.items():
        try:
            # Use predict_proba for sklearn classifiers, predict for LGB Booster
            if name.startswith('LGB'):
                p1 = model.predict(X1)
                p2 = model.predict(X2)
            elif name.startswith('XGB'):
                p1 = model.predict_proba(X1)[:, 1]
                p2 = model.predict_proba(X2)[:, 1]
            elif name.startswith('RF'):
                p1 = model.predict_proba(X1)[:, 1]
                p2 = model.predict_proba(X2)[:, 1]
            else:
                p1 = model.predict(X1)
                p2 = model.predict(X2)
            # Handle both single value and array
            v1 = float(p1[0]) if hasattr(p1, '__len__') else float(p1)
            v2 = float(p2[0]) if hasattr(p2, '__len__') else float(p2)
            diff = abs(v1 - v2)
            results[name] = {'v1': v1, 'v2': v2, 'diff': diff, 'hardcoded': diff < 0.001}
            status = "HARDCODED" if diff < 0.001 else "OK"
            print(f"  {name}: input1={v1:.4f}, input2={v2:.4f}, diff={diff:.6f} {status}")
        except Exception as e:
            print(f"  FAIL {name} predict failed: {e}")
            results[name] = {'error': str(e)}
    
    if any(v.get('hardcoded', False) for v in results.values()):
        print(f"\n  CRITICAL: At least one model is HARDCODED!")
        print(f"       It returns the same prediction regardless of input.")
        print(f"       The dashboard is showing fake signals.")
else:
    print(f"  WARN Skipping — no ML models loaded")

# ─── 5. STARTUP TIME SIMULATION ───
print("\n[5] STARTUP TIME SIMULATION")
print("-" * 50)
print("  Simulating what happens when Dash app starts...")

t_start = time.time()

# Load all models (already done above)
# Load CSV tail
if csv_path:
    t0 = time.time()
    df = pd.read_csv(csv_path)
    df = df.tail(1000)
    t1 = time.time()
    print(f"  CSV tail load: {(t1-t0)*1000:.0f}ms")

# Load regime features
if feat_path:
    t0 = time.time()
    feat = pd.read_parquet(feat_path).tail(500)
    t1 = time.time()
    print(f"  Regime feat load: {(t1-t0)*1000:.0f}ms")

t_end = time.time()
total = t_end - t_start
print(f"\n  TOTAL simulated startup: {total*1000:.0f}ms ({total:.1f}s)")

if total > 5:
    print(f"  WARNING: Startup is {total:.1f}s. Should be <2s.")
    print(f"     If dashboard takes 2-3 min, models are likely loaded INSIDE callbacks.")
else:
    print(f"  OK Startup is fast. The 2-3 min delay is in the callback, not startup.")

# ─── 6. CALLBACK SIMULATION ───
print("\n[6] CALLBACK SIMULATION")
print("-" * 50)
print("  Simulating one dashboard callback cycle...")

csv_load_ms = 0
if csv_path:
    t0 = time.time()
    df = pd.read_csv(csv_path)
    t1 = time.time()
    csv_load_ms = (t1-t0)*1000
    print(f"  Full CSV read: {csv_load_ms:.0f}ms ({len(df):,} rows) PROBLEM if >500ms")
    
    # If reading full CSV, that's the bug
    if csv_load_ms > 500:
        print(f"  CRITICAL: Reading full 343K rows inside callback!")
        print(f"       Fix: Only read last 1000 rows, or use precomputed features.")
    
    # Simulate feature compute
    features = None
    try:
        from beta_testing.features import compute_1m_features
        t0 = time.time()
        features = compute_1m_features(df.tail(1000))
        t1 = time.time()
        print(f"  Feature compute (1000 rows): {(t1-t0)*1000:.0f}ms")
    except Exception as e:
        print(f"  WARN Feature compute: {e}")
    
    # Simulate prediction
    first_lgb = next((k for k in loaded_models if k.startswith('LGB') and loaded_models[k] is not None), None)
    if first_lgb and features is not None:
        try:
            X = features.iloc[[-1]].drop(columns=[c for c in features.columns if c.startswith('target_')])
            t0 = time.time()
            pred = loaded_models[first_lgb].predict(X)
            t1 = time.time()
            print(f"  {first_lgb} predict: {(t1-t0)*1000:.0f}ms")
        except Exception as e:
            print(f"  WARN {first_lgb} predict: {e}")

# ─── 7. CHECK FOR CALLBACK MODEL LOADING ───
print("\n[7] CHECKING FOR CALLBACK MODEL LOADING (Static Analysis)")
print("-" * 50)

# Search dashboard files for model loading inside callbacks
dashboard_files = list(Path(".").rglob("*.py"))
callback_issues = []

for f in dashboard_files:
    if 'dashboard' in f.name.lower() or 'beta' in f.name.lower():
        try:
            content = f.read_text()
            # Check if joblib.load or model load is inside a callback function
            lines = content.split('\n')
            in_callback = False
            for i, line in enumerate(lines):
                if '@callback' in line or 'def update_' in line:
                    in_callback = True
                    callback_start = i
                if in_callback:
                    if 'joblib.load' in line:
                        callback_issues.append(f"{f}:{i+1} — model loaded inside callback!")
                    if '.load(' in line and 'model' in line.lower() and 'joblib' in line.lower():
                        callback_issues.append(f"{f}:{i+1} — model loaded inside callback!")
                    if 'pd.read_csv' in line and 'gold' in line.lower():
                        callback_issues.append(f"{f}:{i+1} — full CSV read inside callback!")
                    # Heuristic: end of function (dedented line that's not empty/comment)
                    if line.strip() and not line.startswith(' ') and not line.startswith('#'):
                        if i > callback_start + 1:
                            in_callback = False
        except:
            pass

if callback_issues:
    print(f"  FOUND MODEL/CSV LOADING INSIDE CALLBACKS:")
    for issue in callback_issues:
        print(f"     {issue}")
else:
    print(f"  OK No obvious model loading inside callbacks found")

# ─── SUMMARY ───
print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

critical = []
warnings = []

if not csv_path:
    critical.append("Gold CSV not found")
ml_model_count = sum(1 for k in loaded_models if k.startswith(('LGB', 'XGB', 'RF')) and loaded_models[k] is not None)
if ml_model_count < 3:
    critical.append(f"Only {ml_model_count}/6 ML models found")
if results and any(v.get('hardcoded', False) for v in results.values()):
    critical.append("Models are HARDCODED — predictions are fake")
if csv_load_ms > 500:
    critical.append("Full CSV read inside callback — 2-3 min delay source")
if callback_issues:
    critical.append("Models loaded inside callbacks")

if critical:
    print(f"\nCRITICAL ISSUES ({len(critical)}):")
    for c in critical:
        print(f"   * {c}")
else:
    print(f"\nOK No critical issues found")

if warnings:
    print(f"\nWarnings:")
    for w in warnings:
        print(f"   * {w}")

print(f"\nRECOMMENDATIONS:")
if any("Full CSV read inside callback" in str(c) for c in critical):
    print(f"   1. Move pd.read_csv() outside callback — load once at module level")
    print(f"   2. Only pass last 1000 rows to callback via dcc.Store or global")
if any("Models are HARDCODED" in str(c) for c in critical):
    print(f"   1. Retrain models from scratch — current ones are broken")
    print(f"   2. Verify with: model.predict(X1) != model.predict(X2)")
if any("Models loaded inside callbacks" in str(c) for c in critical):
    print(f"   1. Move all joblib.load() calls to module top-level (global scope)")
    print(f"   2. Callbacks should only call model.predict(), not model.load()")
