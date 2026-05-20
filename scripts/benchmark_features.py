import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import pandas as pd
from beta_testing import BetaDataLoader
from beta_testing.features import compute_1m_features

loader = BetaDataLoader()
df = pd.read_csv(loader.kaggle.output_dir / 'XAU_1m_data_cleaned.csv', index_col=0, parse_dates=True)
df = df[df.index >= '2025-01-01']
print(f'Rows: {len(df):,}')

t0 = time.time()
print('Computing features...')
f = compute_1m_features(df)
t1 = time.time()
print(f'Done in {t1-t0:.1f}s | Features: {len(f.columns)}')
