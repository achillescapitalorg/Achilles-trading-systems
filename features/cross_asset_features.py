"""
Cross-asset features for gold prediction.
Captures macro drivers that pure price features miss.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "cross_asset"


def load_cross_asset_data(gold_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load pre-downloaded cross-asset data and align to gold's 1m index."""
    assets = ['DXY', 'TNX', 'VIX', 'SI', 'CL', 'SPX', 'BTC']
    frames = {}

    for name in assets:
        path = DATA_DIR / f"{name}.csv"
        if not path.exists():
            print(f"[CrossAsset] Missing {path}")
            continue
        df = pd.read_csv(path, parse_dates=['Date'])
        # yfinance returns 'Close' column
        close = df.set_index('Date')['Close'].sort_index()
        # Forward-fill to 1m frequency (daily -> 1m)
        close_1m = close.reindex(gold_index, method='ffill')
        frames[name] = close_1m

    return pd.DataFrame(frames)


def compute_cross_asset_features(gold_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-asset features aligned to gold's 1m bars.
    Uses pre-downloaded daily data forward-filled to 1m.
    """
    f = pd.DataFrame(index=gold_df.index)
    close = gold_df['close']

    # Load cross-asset data aligned to gold index
    cross = load_cross_asset_data(gold_df.index)
    if cross.empty:
        print("[CrossAsset] No cross-asset data available")
        return f

    # ─── DXY (inverse correlation with gold) ───
    if 'DXY' in cross.columns:
        dxy = cross['DXY']
        f['dxy_slope_5'] = dxy.ewm(span=5).mean().diff(5) / (dxy + 1e-10)
        f['dxy_slope_20'] = dxy.ewm(span=20).mean().diff(20) / (dxy + 1e-10)
        f['dxy_vol'] = dxy.pct_change().rolling(20).std()
        # Gold/DXY ratio deviation
        gd_ratio = close / dxy
        f['gold_dxy_ratio'] = gd_ratio / gd_ratio.rolling(50).mean()

    # ─── 10Y Treasury Yield (real rates proxy) ───
    if 'TNX' in cross.columns:
        tnx = cross['TNX']
        f['yield_slope_5'] = tnx.ewm(span=5).mean().diff(5) / (tnx + 1e-10)
        f['yield_slope_20'] = tnx.ewm(span=20).mean().diff(20) / (tnx + 1e-10)
        f['yield_gold_ratio'] = tnx / (close * 0.01 + 1e-10)

    # ─── VIX (risk-off proxy) ───
    if 'VIX' in cross.columns:
        vix = cross['VIX']
        f['vix_level'] = vix / 20.0  # Normalize
        f['vix_slope_5'] = vix.ewm(span=5).mean().diff(5) / (vix + 1e-10)
        f['vix_change_1d'] = vix.pct_change(1440)  # 1 day in 1m bars

    # ─── Silver (gold/silver ratio) ───
    if 'SI' in cross.columns:
        si = cross['SI']
        ratio = close / si
        f['gold_silver_ratio'] = ratio / ratio.rolling(50).mean()
        f['silver_slope_5'] = si.ewm(span=5).mean().diff(5) / (si + 1e-10)

    # ─── Oil (inflation expectations) ───
    if 'CL' in cross.columns:
        oil = cross['CL']
        f['oil_slope_5'] = oil.ewm(span=5).mean().diff(5) / (oil + 1e-10)
        f['oil_slope_20'] = oil.ewm(span=20).mean().diff(20) / (oil + 1e-10)

    # ─── S&P 500 (risk appetite) ───
    if 'SPX' in cross.columns:
        spx = cross['SPX']
        f['spx_slope_5'] = spx.ewm(span=5).mean().diff(5) / (spx + 1e-10)
        f['spx_slope_20'] = spx.ewm(span=20).mean().diff(20) / (spx + 1e-10)
        # Gold/SPX correlation
        gold_ret = close.pct_change()
        spx_ret = spx.pct_change()
        f['gold_spx_corr_50'] = gold_ret.rolling(50).corr(spx_ret)
        f['gold_spx_corr_200'] = gold_ret.rolling(200).corr(spx_ret)

    # ─── Bitcoin (alternative safe haven / risk proxy) ───
    if 'BTC' in cross.columns:
        btc = cross['BTC']
        f['btc_slope_5'] = btc.ewm(span=5).mean().diff(5) / (btc + 1e-10)
        f['btc_slope_20'] = btc.ewm(span=20).mean().diff(20) / (btc + 1e-10)
        # Gold/BTC correlation
        btc_ret = btc.pct_change()
        f['gold_btc_corr_50'] = gold_ret.rolling(50).corr(btc_ret)

    # Clean
    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.ffill().fillna(0)
    return f
