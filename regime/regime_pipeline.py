"""
Regime-Aware Trading Pipeline
==============================
Integrates detection, prediction, and strategy adaptation.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from .regime_features import RegimeFeatureEngineer
from .hmm_detector import HMMRegimeDetector
from .change_point_detector import ChangePointDetector
from .regime_predictor import RegimePredictor
from .regime_strategy import RegimeStrategy


class RegimeAwareTradingSystem:
    """Complete regime-aware trading pipeline."""

    def __init__(self):
        self.feature_engineer = RegimeFeatureEngineer()
        self.hmm_detector = HMMRegimeDetector()
        self.change_detector = ChangePointDetector()
        self.regime_predictor = RegimePredictor(forecast_horizon=20)
        self.strategy = RegimeStrategy()
        self.regime_history = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> 'RegimeAwareTradingSystem':
        """Train all regime detection and prediction models."""
        print("=" * 60)
        print("TRAINING REGIME-AWARE TRADING SYSTEM")
        print("=" * 60)

        # Step 1: Compute regime features
        print("\n[1/4] Computing regime features...")
        features = self.feature_engineer.compute_all_features(df)
        print(f"  Features shape: {features.shape}")

        # Step 2: Fit HMM regime detector
        print("\n[2/4] Training HMM regime detector...")
        self.hmm_detector.fit(features)
        regime_df = self.hmm_detector.predict_regime(features)
        current_regimes = regime_df['regime']
        print(f"  Regime distribution:")
        for r, c in current_regimes.value_counts().items():
            print(f"    {r}: {c:,} bars ({100*c/len(current_regimes):.1f}%)")

        # Step 3: Detect change points
        print("\n[3/4] Detecting historical change points...")
        changes = self.change_detector.detect_all(df['close'])
        n_changes = changes['is_change_point'].sum()
        print(f"  Found {n_changes:,} change points")

        # Step 4: Fit regime predictor
        print("\n[4/4] Training regime transition predictor...")
        self.regime_predictor.fit(features, current_regimes)
        self.is_fitted = True

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        return self

    def predict(self, df: pd.DataFrame) -> dict:
        """Full regime analysis for current market conditions."""
        if not self.is_fitted:
            raise RuntimeError("System not fitted. Call fit() first.")

        features = self.feature_engineer.compute_all_features(df)

        # Detect current regime
        regime_df = self.hmm_detector.predict_regime(features)
        current_regime = regime_df['regime'].iloc[-1]
        regime_confidence = regime_df['regime_confidence'].iloc[-1]

        # Predict future regime
        future_regime_df = self.regime_predictor.predict(features)
        predicted_regime = future_regime_df['most_likely_regime'].iloc[-1]
        prediction_confidence = future_regime_df['prediction_confidence'].iloc[-1]

        # Check for impending change point
        recent_data = df['close'].iloc[-200:]
        changes = self.change_detector.detect_all(recent_data)
        recent_changes = changes['is_change_point'].iloc[-50:].sum()
        change_warning = recent_changes >= 2

        # Get trading config
        config = self.strategy.get_config(current_regime)

        # Regime transition warning
        regime_transition_warning = (
            current_regime != predicted_regime and
            prediction_confidence > 0.4
        )

        # Build regime probabilities dict
        regime_probs = {}
        for col in regime_df.columns:
            if col.startswith('prob_'):
                regime_probs[col.replace('prob_', '')] = float(regime_df[col].iloc[-1])

        return {
            'current_regime': current_regime,
            'regime_confidence': float(regime_confidence),
            'predicted_regime': predicted_regime,
            'prediction_confidence': float(prediction_confidence),
            'regime_transition_warning': regime_transition_warning,
            'change_point_warning': change_warning,
            'trading_config': config,
            'regime_probs': regime_probs,
            'recommendation': self._generate_recommendation(
                current_regime, predicted_regime,
                regime_confidence, prediction_confidence,
                regime_transition_warning
            ),
        }

    def _generate_recommendation(self, current: str, predicted: str,
                                 conf_current: float, conf_predicted: float,
                                 transition_warning: bool) -> str:
        """Generate human-readable trading recommendation."""
        lines = [
            f"CURRENT REGIME: {current} (confidence: {conf_current:.1%})",
            f"PREDICTED REGIME ({self.regime_predictor.forecast_horizon}m): {predicted} (confidence: {conf_predicted:.1%})",
        ]
        if transition_warning:
            lines.append("⚠️ REGIME TRANSITION WARNING")
            lines.append("ACTION: Reduce position size by 50%")

        config = self.strategy.get_config(current)
        if not config.allow_new_entries:
            lines.append(f"🚫 NO NEW TRADES: {current} regime")
        else:
            lines.append(f"✅ TRADING ACTIVE: {current} regime")
            lines.append(f"  Stop: {config.stop_atr_multiple}x ATR")
            lines.append(f"  Target: {config.takeprofit_atr_multiple}x ATR")
        return "\n".join(lines)

    def evaluate_trade(self, df: pd.DataFrame, signal_confidence: float,
                       current_drawdown: float = 0.0) -> dict:
        """Evaluate a trade signal under current regime conditions."""
        prediction = self.predict(df)
        regime = prediction['current_regime']
        evaluation = self.strategy.evaluate_signal(
            regime, signal_confidence, current_drawdown
        )
        return {**prediction, **evaluation}

    def save(self, path_prefix: str):
        """Save all models."""
        Path(path_prefix).parent.mkdir(parents=True, exist_ok=True)
        self.hmm_detector.save(f"{path_prefix}_hmm.pkl")
        self.regime_predictor.save(f"{path_prefix}_predictor.pkl")
        print(f"Models saved to {path_prefix}_*.pkl")

    def load(self, path_prefix: str):
        """Load all models."""
        self.hmm_detector.load(f"{path_prefix}_hmm.pkl")
        self.regime_predictor.load(f"{path_prefix}_predictor.pkl")
        self.is_fitted = True
