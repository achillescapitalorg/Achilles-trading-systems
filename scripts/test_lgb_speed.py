import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import pandas as pd
from beta_testing.features import compute_1m_features
from beta_testing.models import Gold1mLightGBM

print("Loading small data...")
df = pd.read_csv('data/beta_testing/processed/gold_2025_2026.csv', index_col=0, parse_dates=True)
df = df.head(10000)
print(f"Rows: {len(df):,}")

print("Computing features...")
f = compute_1m_features(df)
target_cols = [c for c in f.columns if c.startswith('target_')]
y = (f['target_dir_5'] > 0).astype(int)
X = f.drop(columns=target_cols)
X = X.iloc[100:]
y = y.iloc[100:]

print(f"Train shape: {X.shape}")

print("\nTraining LightGBM...")
t0 = time.time()
model = Gold1mLightGBM()
model.fit(X, y, X, y, num_boost_round=50)
t1 = time.time()
print(f"Done in {t1-t0:.1f}s")
