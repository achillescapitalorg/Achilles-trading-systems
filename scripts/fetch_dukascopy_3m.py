"""
Fetch 3 months of XAU/USD tick data from Dukascopy and resample to M1.
Run: source venv/Scripts/activate && python scripts/fetch_dukascopy_3m.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from beta_testing.dukascopy_downloader import DukascopyGoldDownloader
from beta_testing.config import DUKASCOPY_DIR

# Feb 20 - May 22, 2026 = ~3 months
START = datetime(2026, 2, 20, 0, 0)
END = datetime(2026, 5, 22, 0, 0)

OUTPUT_CSV = DUKASCOPY_DIR / "XAUUSD_1m_feb_may_2026.csv"

def main():
    print("=" * 60)
    print("Dukascopy 3-Month Fetch")
    print(f"Range: {START} to {END}")
    print("=" * 60)

    dl = DukascopyGoldDownloader(n_workers=6)

    # Download ticks
    ticks = dl.download_range(START, END, save_ticks=True)
    if ticks.empty:
        print("ERROR: No tick data downloaded!")
        return

    # Resample to M1
    df_1m = dl.resample_to_1m(ticks)
    if df_1m.empty:
        print("ERROR: Resampling failed!")
        return

    # Save
    df_1m.to_csv(OUTPUT_CSV)
    print(f"\nSaved {len(df_1m):,} M1 bars to {OUTPUT_CSV}")
    print(f"Date range: {df_1m.index.min()} to {df_1m.index.max()}")

if __name__ == "__main__":
    main()
