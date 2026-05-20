import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
from beta_testing.pipeline import Gold1mTrainingPipeline
from beta_testing import BetaDataLoader

print("Loading data...")
loader = BetaDataLoader()
df = pd.read_csv(loader.kaggle.output_dir / 'XAU_1m_data_cleaned.csv', index_col=0, parse_dates=True)
print(f'FULL DATA: {len(df):,} rows | {df.index.min()} to {df.index.max()}')

pipeline = Gold1mTrainingPipeline(horizon=5)
X_train, y_train, X_val, y_val, X_test, y_test = pipeline.prepare_data(df)

print('\n=== Training LightGBM ===')
from beta_testing.models import Gold1mLightGBM
lgb = Gold1mLightGBM()
lgb.fit(X_train, y_train, X_val, y_val)
pipeline.models['lgb'] = lgb

print('\n=== Training XGBoost ===')
from beta_testing.models import Gold1mXGBoost
xgb = Gold1mXGBoost()
xgb.params['n_estimators'] = 500  # Reduced from 3000 for speed on 5M rows
xgb.fit(X_train, y_train, X_val, y_val)
pipeline.models['xgb'] = xgb

print('\n=== Training Random Forest ===')
from beta_testing.models import Gold1mRandomForest
rf = Gold1mRandomForest()
rf.model.set_params(n_estimators=50)
rf.fit(X_train, y_train)
pipeline.models['rf'] = rf

print('\n=== Evaluating ===')
results = pipeline.evaluate(X_test, y_test)
pipeline.save_all(prefix='gold_1m_full')
print('\n=== TRAINING COMPLETE ===')
