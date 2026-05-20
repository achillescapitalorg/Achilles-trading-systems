import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from beta_testing import BetaDataLoader

loader = BetaDataLoader()
df = pd.read_csv(loader.kaggle.output_dir / 'XAU_1m_data_cleaned.csv', index_col=0, parse_dates=True)
print(f'Full: {len(df):,}')

df = df[df.index >= '2025-01-01']
print(f'2025-2026: {len(df):,}')

df.to_csv('data/beta_testing/processed/gold_2025_2026.csv')
print('Saved to data/beta_testing/processed/gold_2025_2026.csv')
