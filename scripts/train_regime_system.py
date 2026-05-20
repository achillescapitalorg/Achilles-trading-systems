"""
Train Regime Detection System
==============================
Trains HMM detector + regime predictor on local gold data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from regime import RegimeAwareTradingSystem

DATA_PATH = Path("data/beta_testing/processed/gold_2025_2026.csv")
SAVE_PREFIX = "data/beta_testing/processed/models/regime_system"


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df = df.drop(columns=[c for c in ["is_original", "minutes_since_last_bar"] if c in df.columns], errors="ignore")
    print(f"Data shape: {df.shape}")

    system = RegimeAwareTradingSystem()
    system.fit(df)
    system.save(SAVE_PREFIX)

    # Test prediction on latest data
    print("\n--- Testing prediction on latest 500 bars ---")
    result = system.predict(df.tail(500))
    print(f"Current Regime: {result['current_regime']} (conf: {result['regime_confidence']:.1%})")
    print(f"Predicted Regime: {result['predicted_regime']} (conf: {result['prediction_confidence']:.1%})")
    print(f"Transition Warning: {result['regime_transition_warning']}")
    print(f"Change Point Warning: {result['change_point_warning']}")
    print("\nRegime Probabilities:")
    for name, prob in sorted(result['regime_probs'].items(), key=lambda x: -x[1]):
        print(f"  {name}: {prob:.2%}")

    print(f"\n✅ Regime system trained and saved to {SAVE_PREFIX}_*.pkl")


if __name__ == "__main__":
    main()
