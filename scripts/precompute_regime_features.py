#!/usr/bin/env python3
"""
PRECOMPUTE REGIME FEATURES - Run this ONCE manually, not by the dashboard.

Loads full CSV (343K rows), computes regime features, trains HMM,
trains RegimePredictor, and saves everything to disk for fast real-time use.
"""
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path so imports work
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from regime.regime_features import RegimeFeatureEngineer
from regime.hmm_detector import HMMRegimeDetector
from regime.regime_predictor import RegimePredictor

warnings.filterwarnings('ignore')

# --- CONFIG ---
DEFAULT_CSV = PROJECT_ROOT / "data" / "beta_testing" / "processed" / "gold_2025_2026.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "beta_testing" / "processed" / "regime_cache"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "beta_testing" / "processed" / "models"


def load_gold_data(csv_path: Path) -> pd.DataFrame:
    """Load and validate 1m gold CSV."""
    print(f"[1/6] Loading data from {csv_path} ...")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.set_index("date").sort_index()

    # Drop extra columns
    df = df.drop(columns=[c for c in ["is_original", "minutes_since_last_bar"] if c in df.columns], errors="ignore")

    print(f"       Loaded {len(df):,} rows | {df.index.min()} -> {df.index.max()}")
    return df


def compute_and_save_features(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Compute regime features on full dataset and save to parquet."""
    print(f"\n[2/6] Computing regime features on {len(df):,} rows ...")
    print("       This takes a few minutes. Grab coffee.")
    t0 = time.time()

    engineer = RegimeFeatureEngineer()
    features = engineer.compute_all_features(df)

    elapsed = time.time() - t0
    print(f"       Done in {elapsed:.1f}s | Shape: {features.shape}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_path = output_dir / "regime_features_precomputed.parquet"
    features.to_parquet(feat_path, compression='zstd')
    print(f"       Saved -> {feat_path} ({feat_path.stat().st_size / 1e6:.1f} MB)")

    return features


def train_and_save_hmm(features: pd.DataFrame, model_dir: Path) -> HMMRegimeDetector:
    """Train HMM on full feature set and save."""
    print(f"\n[3/6] Training HMM regime detector (6 states) ...")
    t0 = time.time()

    hmm = HMMRegimeDetector(n_regimes=6, n_iter=200, random_state=42)
    hmm.fit(features, verbose=True)

    elapsed = time.time() - t0
    print(f"       Done in {elapsed:.1f}s")

    # Save
    model_dir.mkdir(parents=True, exist_ok=True)
    hmm_path = model_dir / "regime_models_hmm.pkl"
    hmm.save(str(hmm_path))
    print(f"       Saved -> {hmm_path}")

    return hmm


def train_and_save_predictor(
    features: pd.DataFrame,
    regimes: pd.Series,
    model_dir: Path
) -> RegimePredictor:
    """Train regime transition predictor and save."""
    print(f"\n[4/6] Training regime transition predictor (20-bar horizon) ...")
    t0 = time.time()

    predictor = RegimePredictor(forecast_horizon=20)
    predictor.fit(features, regimes)

    elapsed = time.time() - t0
    print(f"       Done in {elapsed:.1f}s")

    pred_path = model_dir / "regime_models_predictor.pkl"
    predictor.save(str(pred_path))
    print(f"       Saved -> {pred_path}")

    return predictor


def save_regime_labels(
    features: pd.DataFrame,
    hmm: HMMRegimeDetector,
    output_dir: Path
):
    """Save regime labels for every bar (useful for analysis)."""
    print(f"\n[5/6] Generating regime labels for all bars ...")

    regime_df = hmm.predict_regime(features)

    # Save
    labels_path = output_dir / "regime_labels_all_bars.parquet"
    regime_df.to_parquet(labels_path, compression='zstd')

    # Print distribution
    print(f"       Regime distribution:")
    for regime, count in regime_df['regime'].value_counts().items():
        pct = 100 * count / len(regime_df)
        print(f"         {regime}: {count:,} bars ({pct:.1f}%)")

    print(f"       Saved -> {labels_path}")


def validate_fast_load(output_dir: Path, model_dir: Path):
    """Verify that fast-loading works correctly."""
    print(f"\n[6/6] Validating fast-load path (simulating callback) ...")
    t0 = time.time()

    # Simulate what the callback does
    hist = pd.read_parquet(output_dir / "regime_features_precomputed.parquet")
    tail = hist.tail(1000)

    hmm = HMMRegimeDetector()
    hmm.load(str(model_dir / "regime_models_hmm.pkl"))

    recent = tail.tail(500)
    result = hmm.predict_regime(recent)
    current = result.iloc[-1]

    elapsed = time.time() - t0
    print(f"       Fast-load test: {elapsed*1000:.1f}ms")
    print(f"       Current regime: {current['regime']} (confidence: {current['regime_confidence']:.1%})")

    if elapsed > 0.5:
        print("       WARNING: Fast-load took >500ms. Check your setup.")
    else:
        print("       Fast-load path is healthy.")


def main():
    print("=" * 60)
    print("REGIME PRECOMPUTATION PIPELINE")
    print("=" * 60)

    # 1. Load data
    df = load_gold_data(DEFAULT_CSV)

    # 2. Compute features
    features = compute_and_save_features(df, DEFAULT_OUTPUT_DIR)

    # 3. Train HMM
    hmm = train_and_save_hmm(features, DEFAULT_MODEL_DIR)

    # 4. Get regime labels from HMM for predictor training
    regime_df = hmm.predict_regime(features)
    regimes = regime_df['regime']

    # 5. Train predictor
    predictor = train_and_save_predictor(features, regimes, DEFAULT_MODEL_DIR)

    # 6. Save full labels
    save_regime_labels(features, hmm, DEFAULT_OUTPUT_DIR)

    # 7. Validate
    validate_fast_load(DEFAULT_OUTPUT_DIR, DEFAULT_MODEL_DIR)

    print("\n" + "=" * 60)
    print("ALL DONE. You can now run the dashboard.")
    print("The callback will load from cache, not recompute on 343K rows.")
    print("=" * 60)


if __name__ == "__main__":
    main()
