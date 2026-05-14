"""
Temperature Scaling + Online Recalibration for Trading Confidence Scores.

Replaces isotonic regression (which overfits and cannot update live) with:
1. Temperature Scaling (Guo et al. 2017) — single-parameter calibration
2. OnlineCalibrator — rolling-window recalibration using recent trade outcomes

Verified: Guo et al. 2017 (5000+ citations) shows TS outperforms isotonic
regression on neural networks. It preserves ranking and is robust to
overfitting because it uses only one parameter.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from collections import deque
from typing import List, Tuple, Optional


class TemperatureScaler:
    """
    Temperature scaling for confidence calibration.
    Single parameter T that softens softmax outputs.
    Preserves ranking — only calibrates probabilities.

    Reference: Guo et al. 2017, 'On Calibration of Modern Neural Networks'
    """

    def __init__(self, initial_temperature: float = 1.5):
        self.temperature = initial_temperature
        self._calibrated = False

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Find optimal temperature on validation set.
        logits: (N, C) array of raw model outputs
        labels: (N,) array of true class indices (or binary 0/1)
        """
        # Handle binary case: logits may be (N,) or (N, 1)
        if logits.ndim == 1:
            logits = np.column_stack([-logits, logits])
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.ravel()

        def nll_loss(T: float) -> float:
            scaled_logits = logits / T
            # Numerical stability: subtract max
            max_logits = np.max(scaled_logits, axis=1, keepdims=True)
            exp_logits = np.exp(scaled_logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            # Negative log-likelihood
            nll = -np.mean(
                np.log(probs[np.arange(len(labels)), labels.astype(int)] + 1e-10)
            )
            return nll

        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method="bounded")
        self.temperature = result.x
        self._calibrated = True
        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits."""
        if logits.ndim == 1:
            logits_2d = np.column_stack([-logits, logits])
        else:
            logits_2d = logits
        scaled_logits = logits_2d / self.temperature
        max_logits = np.max(scaled_logits, axis=1, keepdims=True)
        exp_logits = np.exp(scaled_logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated


class OnlineCalibrator:
    """
    Rolling-window calibration that updates continuously in production.
    Uses recency-weighted temperature scaling.

    Critical: This is what makes confidence scores actually useful live.
    """

    def __init__(
        self,
        window_size: int = 500,
        update_every: int = 50,
        min_buffer: int = 100,
    ):
        self.window_size = window_size
        self.update_every = update_every
        self.min_buffer = min_buffer
        self.temperature = 1.5
        self._buffer: deque = deque(maxlen=window_size)
        self._trade_count = 0

    def add_observation(self, logits: np.ndarray, outcome: int):
        """
        Add a trade result for online calibration.
        logits: raw model output at trade time (array-like)
        outcome: 0=loss, 1=win
        """
        self._buffer.append((np.asarray(logits).copy(), int(outcome)))
        self._trade_count += 1
        if self._trade_count % self.update_every == 0 and len(self._buffer) >= self.min_buffer:
            self._recalibrate()

    def _recalibrate(self):
        """Recalibrate temperature using recent trades."""
        logits_list = []
        labels_list = []
        # Recency weighting — recent trades matter more
        n = len(self._buffer)
        for i, (logits, outcome) in enumerate(self._buffer):
            weight = (i + 1) / n  # linear weighting 1/n -> 1
            repeats = max(1, int(weight * 3))
            for _ in range(repeats):
                logits_list.append(logits)
            labels_list.append(outcome)

        logits_array = np.array(logits_list)
        labels_array = np.array(labels_list)
        scaler = TemperatureScaler(initial_temperature=self.temperature)
        self.temperature = scaler.fit(logits_array, labels_array)
        print(
            f"[CALIBRATION] Updated temperature: {self.temperature:.3f} "
            f"(from {len(self._buffer)} recent trades)"
        )

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply current temperature scaling."""
        scaler = TemperatureScaler(initial_temperature=self.temperature)
        return scaler.transform(logits)

    def calibrate_binary_probability(self, logits: np.ndarray) -> float:
        """
        Convenience: return probability of class 1 for binary models.
        logits: scalar or (1,) or (2,) array
        """
        probs = self.calibrate(np.atleast_1d(logits))
        if probs.shape[-1] == 2:
            return float(probs[..., 1])
        return float(probs.ravel()[0])

    @property
    def is_calibrated(self) -> bool:
        return len(self._buffer) >= self.min_buffer

    def get_status(self) -> dict:
        return {
            "temperature": self.temperature,
            "buffer_size": len(self._buffer),
            "trade_count": self._trade_count,
            "is_calibrated": self.is_calibrated,
        }
