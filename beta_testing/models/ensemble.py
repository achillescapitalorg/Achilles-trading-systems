"""
Ensemble Trader for 1-Minute Gold Prediction
"""
import numpy as np
import pandas as pd
from typing import Dict


class Gold1mEnsembleTrader:
    """
    Ensemble of tree models + regime-aware trading logic.
    """

    def __init__(self, models: Dict, weights=None):
        self.models = models
        self.weights = weights or {'lgb': 0.5, 'xgb': 0.3, 'rf': 0.2}
        self.min_confidence = 0.55

    def predict_ensemble(self, X):
        """Weighted average of model probabilities."""
        preds = {}
        for name, model in self.models.items():
            preds[name] = model.predict(X)
        ensemble = sum(self.weights[n] * preds[n] for n in self.models)
        return ensemble, preds

    def generate_signal(self, X, current_price, atr_14):
        """
        Generate complete trading signal.
        Returns dict with action, confidence, stops, targets.
        """
        ensemble_prob, individual = self.predict_ensemble(X)
        prob = ensemble_prob[-1]

        buy_votes = sum(1 for p in individual.values() if p[-1] > 0.5)
        sell_votes = len(self.models) - buy_votes
        agreement = max(buy_votes, sell_votes) / len(self.models)

        is_buy = prob > 0.5
        confidence = prob if is_buy else (1 - prob)
        confidence *= (0.5 + 0.5 * agreement)

        if confidence < self.min_confidence:
            return {
                'action': 'HOLD',
                'confidence': confidence,
                'reason': 'below_threshold',
                'stop_loss': None,
                'take_profit': None,
                'position_size': 0,
                'models_agreement': f'{buy_votes}/{len(self.models)} buy' if is_buy else f'{sell_votes}/{len(self.models)} sell',
                'individual_probs': {n: float(p[-1]) for n, p in individual.items()},
            }

        action = 'BUY' if is_buy else 'SELL'
        position_size = min(1.0, (confidence - 0.5) * 4)
        atr = atr_14.iloc[-1] if hasattr(atr_14, 'iloc') else atr_14

        if action == 'BUY':
            stop = current_price - 2.0 * atr
            target = current_price + 3.0 * atr
        else:
            stop = current_price + 2.0 * atr
            target = current_price - 3.0 * atr

        return {
            'action': action,
            'confidence': confidence,
            'stop_loss': stop,
            'take_profit': target,
            'position_size': position_size,
            'models_agreement': f'{buy_votes}/{len(self.models)} buy' if is_buy else f'{sell_votes}/{len(self.models)} sell',
            'individual_probs': {n: float(p[-1]) for n, p in individual.items()},
        }
