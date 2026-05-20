"""
Change Point Detection
======================
CUSUM + Mood test + trend breaks for detecting regime shifts.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple


class ChangePointDetector:
    """Detect regime change points in price series."""

    def __init__(self, window_size: int = 50, significance: float = 0.01):
        self.window_size = window_size
        self.significance = significance

    def detect_cusum(self, series: pd.Series) -> List[int]:
        """
        CUSUM (Cumulative Sum) change point detection.
        Detects shifts in the mean of a time series.
        """
        values = series.dropna().values
        n = len(values)
        mean_val = np.mean(values)
        std_val = np.std(values) + 1e-10
        normalized = (values - mean_val) / std_val

        s_pos = np.zeros(n)
        s_neg = np.zeros(n)
        threshold = 4.0
        change_points = []

        for t in range(1, n):
            s_pos[t] = max(0, s_pos[t - 1] + normalized[t] - 0.5)
            s_neg[t] = max(0, s_neg[t - 1] - normalized[t] - 0.5)
            if s_pos[t] > threshold or s_neg[t] > threshold:
                change_points.append(t)
                s_pos[t] = 0
                s_neg[t] = 0
        return change_points

    def detect_mood_test(self, returns: pd.Series) -> List[Tuple[int, str]]:
        """
        Mood test for variance change points.
        Detects when volatility regime changes.
        """
        values = returns.dropna().values
        n = len(values)
        change_points = []
        step = max(1, self.window_size // 5)

        for i in range(self.window_size, n - self.window_size, step):
            window_before = values[i - self.window_size:i]
            window_after = values[i:i + self.window_size]
            var_before = np.var(window_before, ddof=1)
            var_after = np.var(window_after, ddof=1)
            if var_before < 1e-10 or var_after < 1e-10:
                continue

            f_stat = max(var_after / var_before, var_before / var_after)
            df1 = df2 = self.window_size - 1
            p_value = 2 * min(
                stats.f.cdf(1 / f_stat, df1, df2),
                1 - stats.f.cdf(f_stat, df1, df2)
            )
            if p_value < self.significance:
                direction = 'VOL_INCREASE' if var_after > var_before else 'VOL_DECREASE'
                change_points.append((i, direction))
        return change_points

    def detect_all(self, close: pd.Series) -> pd.DataFrame:
        """Run all change point detection methods and combine results."""
        returns = close.pct_change().dropna()
        # Mean shifts (CUSUM)
        mean_changes = self.detect_cusum(returns)
        # Variance shifts (Mood test)
        var_changes = self.detect_mood_test(returns)
        # Trend changes
        trend_changes = self._detect_trend_changes(close)

        result = pd.DataFrame(index=close.index)
        result['is_change_point'] = False
        result['change_type'] = None
        result['confidence'] = 0.0

        for idx in mean_changes:
            if idx < len(result):
                result.iloc[idx, result.columns.get_loc('is_change_point')] = True
                result.iloc[idx, result.columns.get_loc('change_type')] = 'MEAN_SHIFT'
                result.iloc[idx, result.columns.get_loc('confidence')] = 0.7

        for idx, direction in var_changes:
            if idx < len(result):
                result.iloc[idx, result.columns.get_loc('is_change_point')] = True
                result.iloc[idx, result.columns.get_loc('change_type')] = direction
                result.iloc[idx, result.columns.get_loc('confidence')] = 0.9

        for idx, direction in trend_changes:
            if idx < len(result):
                result.iloc[idx, result.columns.get_loc('is_change_point')] = True
                result.iloc[idx, result.columns.get_loc('change_type')] = direction
                result.iloc[idx, result.columns.get_loc('confidence')] = 0.8

        return result

    def _detect_trend_changes(self, close: pd.Series) -> List[Tuple[int, str]]:
        """Detect trend direction changes using regression slope breaks."""
        values = close.values
        n = len(values)
        changes = []
        window = self.window_size

        for i in range(window * 2, n - window, max(1, window // 3)):
            x_before = np.arange(window)
            y_before = values[i - window:i]
            slope_before, _, _, _, _ = stats.linregress(x_before, y_before)

            x_after = np.arange(window)
            y_after = values[i:i + window]
            slope_after, _, _, _, _ = stats.linregress(x_after, y_after)

            slope_change = abs(slope_after - slope_before)
            avg_price = np.mean(y_before)
            normalized_change = slope_change / (avg_price + 1e-10)
            if normalized_change > 0.001:
                direction = 'TREND_UP' if slope_after > slope_before else 'TREND_DOWN'
                changes.append((i, direction))
        return changes
