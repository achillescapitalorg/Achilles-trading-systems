import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
from beta_testing.pipeline import Gold1mTrainingPipeline

print("Loading pre-filtered 2025-2026 data...")
df = pd.read_csv('data/beta_testing/processed/gold_2025_2026.csv', index_col=0, parse_dates=True)
print(f"Using 2025-2026: {len(df):,} rows")

pipeline = Gold1mTrainingPipeline(horizon=5)
X_train, y_train, X_val, y_val, X_test, y_test = pipeline.prepare_data(df)

print('\n=== Training LightGBM ===')
from beta_testing.models import Gold1mLightGBM
lgb = Gold1mLightGBM()
lgb.fit(X_train, y_train, X_val, y_val, num_boost_round=200)
pipeline.models['lgb'] = lgb

print('\n=== Training XGBoost ===')
from beta_testing.models import Gold1mXGBoost
xgb = Gold1mXGBoost()
xgb.params['n_estimators'] = 100
xgb.fit(X_train, y_train, X_val, y_val)
pipeline.models['xgb'] = xgb

print('\n=== Training Random Forest ===')
from beta_testing.models import Gold1mRandomForest
rf = Gold1mRandomForest()
rf.model.set_params(n_estimators=20)
rf.fit(X_train, y_train)
pipeline.models['rf'] = rf

print('\n=== Evaluating ===')
results = pipeline.evaluate(X_test, y_test)
pipeline.save_all(prefix='gold_1m_2025_2026')
print('\n=== TRAINING COMPLETE ===')
