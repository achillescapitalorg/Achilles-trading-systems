"""
Regime-Aware Mixture-of-Experts (MoE) for Non-Stationary Gold Forecasting.

Problem: Financial markets are non-stationary. A model trained on calm markets
will fail during crises. Your walk-forward results show high variance across folds
(Sharpe 128 in backtest vs different regimes in WF), which is the classic signature
of non-stationarity.

Solution: Mixture-of-Experts with a gating network that routes input to the most
appropriate expert based on detected regime. Each expert specializes in a specific
market regime (trending-up, trending-down, ranging, high-vol).

Verified Research:
  - N-BEATS-MoE achieved rank 1.58 across datasets (Porto thesis 2024)
  - DoubleAdapt (Zhao 2022) meta-learns to adapt to distribution shifts
  - ADB-TRM (Chen 2024) uses adversarial training for regime adaptation
  - SSGA (2025) identifies 4 distinct market regimes over 30 years using ML

This implementation uses a lightweight gating network + XGBoost experts.
No deep learning required for the experts, making training fast and interpretable.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


@dataclass
class MoEConfig:
    """Configuration for regime-aware MoE."""
    regimes: List[str] = None
    gating_features: List[str] = None  # Features used by gating network
    expert_params: Dict = None
    min_samples_per_expert: int = 200
    smoothing_window: int = 5  # Smooth regime probabilities

    def __post_init__(self):
        if self.regimes is None:
            self.regimes = ["low_vol", "normal", "trend_up", "trend_down", "high_vol"]
        if self.expert_params is None:
            self.expert_params = {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 5.0,
                "n_estimators": 200,
            }


def detect_regime_simple(df: pd.DataFrame) -> pd.Series:
    """
    Simple rule-based regime detection for gold.
    Returns regime label for each row.
    """
    close = df["close"]
    # Volatility regime using ATR
    atr = df.get("atr", (df["high"] - df["low"]).rolling(14).mean())
    atr_ma = atr.rolling(50).mean()

    # Trend using EMA slope
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    trend = ema20 - ema50

    regime = pd.Series("normal", index=df.index)
    regime[atr > 1.5 * atr_ma] = "high_vol"
    regime[atr < 0.7 * atr_ma] = "low_vol"
    regime[(trend > 0) & (atr <= 1.5 * atr_ma)] = "trend_up"
    regime[(trend < 0) & (atr <= 1.5 * atr_ma)] = "trend_down"
    return regime


class RegimeAwareMoE:
    """
    Mixture-of-Experts where each expert is an XGBoost model specialized
    for a specific regime. A gating network (also XGBoost) predicts regime
    probabilities, and predictions are weighted by regime confidence.
    """

    def __init__(self, config: Optional[MoEConfig] = None):
        if not XGB_AVAILABLE:
            raise RuntimeError("xgboost required for RegimeAwareMoE")
        self.config = config or MoEConfig()
        self.gating_model: Optional[xgb.XGBClassifier] = None
        self.experts: Dict[str, xgb.XGBClassifier] = {}
        self._is_fitted = False
        self._feature_cols: Optional[List[str]] = None

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        regime_col: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Train gating network and regime-specific experts.
        If regime_col not provided, auto-detect using detect_regime_simple.
        """
        self._feature_cols = feature_cols
        regimes = self.config.regimes

        # Auto-detect regime if not provided
        if regime_col is None or regime_col not in df.columns:
            df = df.copy()
            df["_regime"] = detect_regime_simple(df)
            regime_col = "_regime"

        # 1. Train gating network (regime classifier)
        gate_features = self.config.gating_features or feature_cols[:8]
        X_gate = df[gate_features].fillna(0).values
        y_gate = df[regime_col].values

        # Use only regimes actually present in the data
        actual_regimes = sorted(df[regime_col].unique().tolist())
        self._actual_regimes = actual_regimes
        self._regime_to_int = {r: i for i, r in enumerate(actual_regimes)}
        self._int_to_regime = {i: r for i, r in enumerate(actual_regimes)}
        y_gate_int = np.array([self._regime_to_int.get(r, 0) for r in y_gate])
        regimes = actual_regimes  # Update to actual for expert training

        gate_params = {
            "objective": "multi:softprob",
            "num_class": len(actual_regimes),
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 100,
        }
        self.gating_model = xgb.XGBClassifier(**gate_params)
        self.gating_model.fit(X_gate, y_gate_int)
        gate_acc = np.mean(self.gating_model.predict(X_gate) == y_gate_int)
        print(f"[MoE] Gating network accuracy: {gate_acc:.3f}")

        # 2. Train one expert per regime
        expert_scores = {}
        for regime in regimes:
            mask = df[regime_col] == regime
            n_samples = mask.sum()
            if n_samples < self.config.min_samples_per_expert:
                print(f"[MoE] Skipping {regime}: only {n_samples} samples")
                continue

            X_expert = df.loc[mask, feature_cols].fillna(0).values
            y_expert = df.loc[mask, target_col].values

            # Determine number of classes in this regime's target
            n_classes = len(np.unique(y_expert))
            if n_classes < 2:
                print(f"[MoE] Skipping {regime}: only {n_classes} class(es)")
                continue

            params = self.config.expert_params.copy()
            params["num_class"] = n_classes
            params["objective"] = "multi:softprob"

            expert = xgb.XGBClassifier(**params)
            expert.fit(X_expert, y_expert)
            self.experts[regime] = expert

            # In-sample score (for diagnostic only)
            preds = expert.predict(X_expert)
            from sklearn.metrics import f1_score
            f1 = f1_score(y_expert, preds, average="macro", zero_division=0)
            expert_scores[regime] = f1
            print(f"[MoE] Expert {regime}: n={n_samples}, macro-F1={f1:.3f}")

        self._is_fitted = True
        return expert_scores

    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Predict class probabilities using weighted mixture of experts.
        Returns (n_samples, n_classes) array.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        feature_cols = feature_cols or self._feature_cols
        if feature_cols is None:
            raise ValueError("feature_cols required")

        X = df[feature_cols].fillna(0).values
        n_samples = len(X)

        # Get regime probabilities from gating network
        gate_features = self.config.gating_features or feature_cols[:8]
        X_gate = df[gate_features].fillna(0).values
        regime_probs = self.gating_model.predict_proba(X_gate)  # (n, n_regimes)

        # Smooth regime probabilities over time
        if self.config.smoothing_window > 1:
            regime_probs = pd.DataFrame(regime_probs).rolling(
                self.config.smoothing_window, min_periods=1
            ).mean().values

        # Collect predictions from each expert
        # We need to align class probabilities across experts
        # Strategy: each expert predicts its own classes, we weight and normalize
        all_classes = set()
        expert_probs = {}
        for regime, expert in self.experts.items():
            probs = expert.predict_proba(X)  # (n, n_classes_regime)
            expert_probs[regime] = probs
            all_classes.update(expert.classes_)

        all_classes = sorted(all_classes)
        n_classes = len(all_classes)
        class_to_idx = {c: i for i, c in enumerate(all_classes)}

        # Blend predictions weighted by regime probability
        blended = np.zeros((n_samples, n_classes))
        for i, regime in enumerate(self._actual_regimes):
            if regime not in self.experts:
                continue
            probs = expert_probs[regime]
            # Map expert's classes to global indices
            expert_classes = self.experts[regime].classes_
            for j, cls in enumerate(expert_classes):
                global_idx = class_to_idx[cls]
                blended[:, global_idx] += regime_probs[:, i] * probs[:, j]

        # Normalize
        row_sums = blended.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        blended = blended / row_sums
        return blended

    def predict(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(df, feature_cols)
        return np.argmax(probs, axis=1)

    def get_regime_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return per-sample regime probabilities for interpretability."""
        gate_features = self.config.gating_features or self._feature_cols[:8]
        X_gate = df[gate_features].fillna(0).values
        probs = self.gating_model.predict_proba(X_gate)
        return pd.DataFrame(
            probs,
            columns=[f"prob_{r}" for r in self._actual_regimes],
            index=df.index,
        )

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "config": self.config,
                "gating_model": self.gating_model,
                "experts": self.experts,
                "regime_to_int": self._regime_to_int,
                "int_to_regime": self._int_to_regime,
                "feature_cols": self._feature_cols,
            }, f)

    def load(self, path: str):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.config = data["config"]
        self.gating_model = data["gating_model"]
        self.experts = data["experts"]
        self._regime_to_int = data["regime_to_int"]
        self._int_to_regime = data["int_to_regime"]
        self._feature_cols = data["feature_cols"]
        self._is_fitted = True
