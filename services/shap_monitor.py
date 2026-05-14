"""
SHAP-Based Feature Importance Monitoring.

Problem: You don't know which features your model is actually using.
If it overfits to a spurious feature, you won't detect it until losses occur.

Solution: Integrate SHAP importance tracking into the training pipeline
with automated alerts for feature drift.

Verified: SHAP (Lundberg & Lee 2017) is the gold standard for model
interpretability in tree-based models. Feature drift detection catches
overfitting before it costs money.

Note: Falls back to permutation importance if SHAP is not installed.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPFeatureMonitor:
    """
    Monitor feature importance using SHAP values.
    Alerts on feature drift and overfitting indicators.
    """

    def __init__(self, feature_names: List[str], alert_threshold: float = 0.2):
        self.feature_names = feature_names
        self.alert_threshold = alert_threshold
        self.baseline_importance: Optional[pd.Series] = None
        self.importance_history: List[pd.Series] = []

    def compute_importance(self, model, X_sample: np.ndarray) -> pd.Series:
        """Compute SHAP importance for a model sample."""
        if not SHAP_AVAILABLE:
            warnings.warn("SHAP not installed — using fallback permutation importance")
            return self._permutation_importance(model, X_sample)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                # Multi-class — use mean absolute across classes
                importance = np.mean(
                    [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
                )
            else:
                importance = np.abs(shap_values).mean(axis=0)
            return pd.Series(importance, index=self.feature_names)
        except Exception as e:
            warnings.warn(f"TreeExplainer failed ({e}), falling back to permutation importance")
            return self._permutation_importance(model, X_sample)

    def _permutation_importance(self, model, X_sample: np.ndarray) -> pd.Series:
        """Fallback: permutation importance."""
        baseline_pred = model.predict(X_sample)
        baseline_mae = np.mean(np.abs(baseline_pred - baseline_pred.mean()))
        importances = []
        for i in range(X_sample.shape[1]):
            X_permuted = X_sample.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_pred = model.predict(X_permuted)
            permuted_mae = np.mean(np.abs(permuted_pred - baseline_pred))
            importances.append(permuted_mae - baseline_mae)
        return pd.Series(importances, index=self.feature_names)

    def set_baseline(self, model, X_sample: np.ndarray):
        """Set baseline importance from initial validation."""
        self.baseline_importance = self.compute_importance(model, X_sample)
        total = self.baseline_importance.sum()
        if total > 0:
            self.baseline_importance = self.baseline_importance / total

    def check_drift(self, model, X_sample: np.ndarray) -> Dict:
        """
        Check for feature importance drift.
        Returns alert if top features change significantly.
        """
        if self.baseline_importance is None:
            return {"status": "no_baseline", "messages": ["Baseline not set"]}

        current = self.compute_importance(model, X_sample)
        total = current.sum()
        if total > 0:
            current = current / total
        self.importance_history.append(current)

        # Check if top-5 features have changed
        baseline_top5 = set(self.baseline_importance.nlargest(5).index)
        current_top5 = set(current.nlargest(5).index)
        overlap = len(baseline_top5 & current_top5)

        # Check importance shifts for individual features
        shifts = (current - self.baseline_importance).abs()
        max_shift_feature = shifts.idxmax()
        max_shift_value = shifts.max()

        alert = False
        messages = []
        if overlap < 3:
            alert = True
            messages.append(f"Feature drift: only {overlap}/5 top features stable")
        if max_shift_value > self.alert_threshold:
            alert = True
            messages.append(
                f"Major shift in {max_shift_feature}: {max_shift_value:.3f}"
            )

        return {
            "status": "alert" if alert else "ok",
            "messages": messages,
            "top_feature_overlap": overlap,
            "max_shift_feature": max_shift_feature,
            "max_shift_value": max_shift_value,
            "current_top5": list(current_top5),
        }

    def get_importance_report(self) -> pd.DataFrame:
        """Get feature importance history as DataFrame."""
        if not self.importance_history:
            return pd.DataFrame()
        df = pd.DataFrame(self.importance_history).T
        df.columns = [f"period_{i}" for i in range(len(self.importance_history))]
        df["mean"] = df.mean(axis=1)
        df["std"] = df.std(axis=1)
        df["drift"] = df["std"] / (df["mean"] + 1e-10)
        return df.sort_values("mean", ascending=False)
