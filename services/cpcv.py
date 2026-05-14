"""
Combinatorial Purged Cross-Validation (CPCV) + Probability of Backtest Overfitting (PBO).

Lopez de Prado (2018) demonstrated that walk-forward validation produces only a
single OOS path, making overfitting impossible to detect. CPCV generates C(N,k)
train/test paths with purging and embargo, producing a DISTRIBUTION of Sharpe
ratios and a PBO score.

Verified: Institutional standard. Implemented in multiple open-source quant
libraries (How-To-Backtest-Correctly, mlfinlab, llm-quant).
"""
import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Tuple, Callable, Optional
from scipy import stats


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (Lopez de Prado, 2018).
    Generates C(N, k) train/test paths, each with:
      - Purging: remove overlapping label windows
      - Embargo: gap between train/test to prevent leakage
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        embargo_pct: float = 0.02,
        bar_times: Optional[pd.Series] = None,
        label_times: Optional[pd.Series] = None,
    ):
        """
        n_splits: Total groups to divide data into (N)
        n_test_splits: Groups per test set (k)
        embargo_pct: Fraction of data to embargo between train/test
        bar_times: Series of bar timestamps
        label_times: Series of label end-times (for purging)
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct
        self.bar_times = bar_times
        self.label_times = label_times

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Generate all C(N, k) train/test splits.
        Returns list of (train_indices, [test_indices_list]).
        """
        n_samples = len(X)
        group_size = n_samples // self.n_splits
        # Create groups
        groups = []
        for i in range(self.n_splits):
            start = i * group_size
            end = start + group_size if i < self.n_splits - 1 else n_samples
            groups.append(np.arange(start, end))

        splits = []
        for test_combo in combinations(range(self.n_splits), self.n_test_splits):
            test_indices = np.concatenate([groups[i] for i in test_combo])
            train_groups = [i for i in range(self.n_splits) if i not in test_combo]
            train_indices = np.concatenate([groups[i] for i in train_groups])
            # Apply purging and embargo
            train_indices = self._purge_and_embargo(
                train_indices, test_indices, n_samples
            )
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, [test_indices]))
        return splits

    def _purge_and_embargo(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Remove overlapping train samples and add embargo gap."""
        if self.label_times is None or self.bar_times is None:
            # Simple embargo without label purging
            test_start = test_indices[0]
            test_end = test_indices[-1]
            embargo_start = int(test_end + self.embargo_pct * n_samples)
            valid_train = [
                idx for idx in train_indices
                if idx < test_start or idx > embargo_start
            ]
            return np.array(valid_train)

        test_start = test_indices[0]
        test_end = test_indices[-1]
        # Purge: remove train samples whose labels overlap with test period
        label_test_start = self.label_times.iloc[test_start]
        label_test_end = self.label_times.iloc[test_end]
        valid_train = []
        for idx in train_indices:
            label_end = self.label_times.iloc[idx]
            # Keep only if label ends before test period starts
            if label_end < label_test_start:
                valid_train.append(idx)
        # Embargo: add gap after test period
        embargo_start = int(test_end + self.embargo_pct * n_samples)
        valid_train = [
            idx for idx in valid_train
            if idx > embargo_start or idx < test_start
        ]
        return np.array(valid_train)


def compute_probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame,
    n_splits: int = 8,
    metric_func: Optional[Callable] = None,
) -> dict:
    """
    Compute PBO (Probability of Backtest Overfitting) using CSCV.
    Returns dict with:
      - pbo: Probability that best IS strategy ranks below median OOS
      - n_splits: Number of splits evaluated
      - below_median: Count of below-median occurrences
      - oos_ranks: Distribution of OOS ranks
      - is_overfit: bool
    """
    if n_splits % 2 != 0:
        n_splits += 1
    n_test = n_splits // 2

    if metric_func is None:
        def metric_func(r):
            if r.std(ddof=1) > 0:
                return r.mean() / r.std(ddof=1)
            return 0.0

    # Create neutral label times
    dates = returns_matrix.index
    t1 = pd.Series(dates, index=dates)
    cpcv = CombinatorialPurgedCV(
        n_splits=n_splits,
        n_test_splits=n_test,
        embargo_pct=0.0,
        bar_times=dates,
        label_times=t1,
    )

    n_strategies = returns_matrix.shape[1]
    below_median = 0
    oos_ranks = []

    for train_idx, test_idx_list in cpcv.split(returns_matrix):
        test_idx = test_idx_list[0]
        # Score all strategies IS
        is_scores = {}
        for col in returns_matrix.columns:
            is_ret = returns_matrix[col].iloc[train_idx]
            is_scores[col] = metric_func(is_ret)
        best_is = max(is_scores, key=is_scores.get)

        # Score best IS strategy OOS
        oos_ret = returns_matrix[best_is].iloc[test_idx]
        oos_score = metric_func(oos_ret)

        # Rank among all strategies OOS
        oos_scores = {}
        for col in returns_matrix.columns:
            oos_scores[col] = metric_func(returns_matrix[col].iloc[test_idx])
        rank = pd.Series(oos_scores).rank(ascending=False)[best_is]
        norm_rank = (rank - 1) / (n_strategies - 1)
        oos_ranks.append(norm_rank)

        if norm_rank > 0.5:
            below_median += 1

    n_eval = len(oos_ranks)
    pbo = below_median / n_eval if n_eval > 0 else 0.5
    return {
        "pbo": pbo,
        "n_splits": n_eval,
        "below_median": below_median,
        "oos_ranks": oos_ranks,
        "oos_rank_mean": np.mean(oos_ranks) if oos_ranks else 0.5,
        "oos_rank_std": np.std(oos_ranks) if oos_ranks else 0.0,
        "is_overfit": pbo > 0.5,
    }


def combinatorial_backtest(
    model_factory: Callable,
    feature_df: pd.DataFrame,
    labels: pd.Series,
    n_splits: int = 6,
    n_test_splits: int = 2,
    embargo_pct: float = 0.02,
    metric_func: Optional[Callable] = None,
) -> dict:
    """
    Run CPCV backtest for a single model configuration.
    Returns distribution of OOS performance metrics.
    """
    if metric_func is None:
        def metric_func(y_true, y_pred):
            from sklearn.metrics import accuracy_score
            return accuracy_score(y_true, y_pred)

    cpcv = CombinatorialPurgedCV(
        n_splits=n_splits,
        n_test_splits=n_test_splits,
        embargo_pct=embargo_pct,
    )
    scores = []
    for train_idx, test_idx_list in cpcv.split(feature_df):
        test_idx = test_idx_list[0]
        X_train, X_test = feature_df.iloc[train_idx], feature_df.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(metric_func(y_test, y_pred))

    return {
        "mean": np.mean(scores),
        "std": np.std(scores),
        "min": np.min(scores),
        "max": np.max(scores),
        "median": np.median(scores),
        "scores": scores,
    }
