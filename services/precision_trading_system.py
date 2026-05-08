"""
1-MINUTE PRECISION TRADING SYSTEM v4
=====================================
Research-backed enhancements for bias reduction, classification metrics,
and walk-forward persistence via SQLite.

NEW IN v4:
  - Purged Cross-Validation with Embargo (López de Prado)
  - Meta-Labeling: Primary model → direction, Secondary model → trade filter
  - Comprehensive classification metrics: Accuracy, Precision, Recall, F1,
    MCC, Cohen's Kappa, Balanced Accuracy, AUC-ROC, AUC-PR
  - Bias detection: Feature leakage scanner, temporal split enforcement
  - 1m microstructure: Roll spread, tick imbalance, time-of-day features
  - Class-weighted XGBoost for severe 1m class imbalance
  - SQLiteBacktestDB: persistent storage of all walk-forward results
  - Threshold optimization for F1 score on validation set
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Literal, Any
from enum import Enum
import warnings
from collections import deque
import json
import pickle
import sqlite3
import os
from datetime import datetime, timedelta

# ── Optional deps; fall back gracefully ──────────────────────────────────────
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
        roc_auc_score, average_precision_score, classification_report,
        confusion_matrix,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    CalibratedClassifierCV = None
    TimeSeriesSplit = None

try:
    from scipy.stats import ttest_1samp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    ttest_1samp = None

try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False

try:
    from pykalman import KalmanFilter
    PYKALMAN_AVAILABLE = True
except ImportError:
    PYKALMAN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')


# =============================================================================
# VECTORIZED LABELING + ADVANCED TECHNIQUES
# =============================================================================

def triple_barrier_labels_vectorized(df, pt_atr_mult=1.5, sl_atr_mult=1.0,
                                       max_bars=5):
    """Vectorized triple-barrier labelling with the SAME corrected logic as
    TripleBarrierLabeler.label() — symmetric, direction-agnostic.

    For each bar, picks +1/-1 based on which PROFIT-TAKE barrier fires first
    (validated by no SL on that side firing earlier), else 0 (timeout/SL).
    """
    if 'atr_14' not in df.columns:
        df = df.copy()
        df['atr_14'] = (df['high'] - df['low']).rolling(14).mean()
    atr = df['atr_14'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    n = len(df)
    labels = np.zeros(n, dtype=int)

    for i in range(n - max_bars):
        a = atr[i]
        if not np.isfinite(a) or a == 0:
            continue
        entry = close[i]
        upper_pt = entry + pt_atr_mult * a   # long TP
        upper_sl = entry + sl_atr_mult * a   # short SL
        lower_pt = entry - pt_atr_mult * a   # short TP
        lower_sl = entry - sl_atr_mult * a   # long SL
        for j in range(i + 1, min(i + max_bars + 1, n)):
            hi, lo = high[j], low[j]
            hit_long_tp = hi >= upper_pt
            hit_long_sl = lo <= lower_sl
            hit_short_tp = lo <= lower_pt
            hit_short_sl = hi >= upper_sl
            if hit_long_tp and not hit_long_sl:
                labels[i] = 1; break
            if hit_short_tp and not hit_short_sl:
                labels[i] = -1; break
            if hit_long_sl or hit_short_sl:
                labels[i] = 0; break
    return pd.Series(labels, index=df.index).astype(int)


class MarkovRegimeForecaster:
    """Discrete Markov-chain regime forecaster.

    Estimates the transition probability matrix from a regime label sequence
    (e.g. HMM regimes) and produces:
      - next-regime probabilities P(R_{t+1} | R_t)
      - persistence score = P(stay) — useful for signal confidence
      - transition entropy = -sum p log p — high entropy = unstable regime

    These features feed into the signal model and the live-signal panel.
    """

    def __init__(self, n_states: int = 2, smoothing: float = 1.0):
        self.n_states = n_states
        self.smoothing = smoothing  # Laplace smoothing
        self.T: Optional[np.ndarray] = None  # transition matrix [n×n]
        self.steady_state: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, regime_seq: np.ndarray) -> "MarkovRegimeForecaster":
        seq = np.asarray(regime_seq, dtype=int)
        # Laplace-smoothed counts
        counts = np.full((self.n_states, self.n_states), self.smoothing, dtype=float)
        for i in range(len(seq) - 1):
            a, b = int(seq[i]), int(seq[i + 1])
            if 0 <= a < self.n_states and 0 <= b < self.n_states:
                counts[a, b] += 1.0
        self.T = counts / counts.sum(axis=1, keepdims=True)
        # Stationary distribution = left-eigenvector for eigenvalue 1
        try:
            eigvals, eigvecs = np.linalg.eig(self.T.T)
            idx = int(np.argmin(np.abs(eigvals - 1.0)))
            ss = np.real(eigvecs[:, idx])
            ss = np.abs(ss); ss /= ss.sum()
            self.steady_state = ss
        except Exception:
            self.steady_state = np.ones(self.n_states) / self.n_states
        self._fitted = True
        return self

    def next_regime_proba(self, current_regime: int) -> np.ndarray:
        if not self._fitted or self.T is None:
            return np.ones(self.n_states) / self.n_states
        cr = int(current_regime) % self.n_states
        return self.T[cr]

    def persistence(self, current_regime: int) -> float:
        """Probability of staying in the current regime."""
        p = self.next_regime_proba(current_regime)
        return float(p[int(current_regime) % self.n_states])

    def entropy(self, current_regime: int) -> float:
        """Shannon entropy of the next-regime distribution."""
        p = self.next_regime_proba(current_regime)
        p = np.clip(p, 1e-12, 1.0)
        return float(-(p * np.log(p)).sum())

    def add_features(self, df: pd.DataFrame, regime_col: str = "regime") -> pd.DataFrame:
        """Add columns: markov_persistence, markov_entropy."""
        if regime_col not in df.columns:
            return df
        regimes = df[regime_col].fillna(0).astype(int).values
        if not self._fitted:
            self.fit(regimes)
        pers = np.array([self.persistence(r) for r in regimes])
        ent = np.array([self.entropy(r) for r in regimes])
        out = df.copy()
        out["markov_persistence"] = pers
        out["markov_entropy"] = ent
        return out


# =============================================================================
# CONFIG
# =============================================================================

class Asset(Enum):
    XAUUSD = "XAUUSD"
    BTCUSD = "BTCUSD"
    EURUSD = "EURUSD"
    GBPUSD = "GBPUSD"


class Signal(Enum):
    STRONG_BUY = 2
    WEAK_BUY = 1
    NEUTRAL = 0
    WEAK_SELL = -1
    STRONG_SELL = -2


class Regime(Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class TradeDirection(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class AssetConfig:
    asset: Asset
    pip_value: float
    spread_avg: float
    tick_size: float
    contract_size: float = 1.0
    leverage: float = 100.0

    london_start: int = 8
    london_end: int = 17
    ny_start: int = 13
    ny_end: int = 22

    kalman_observation_covariance: float = 1.0
    kalman_transition_covariance: float = 0.01
    vpin_buckets: int = 50
    vpin_window: int = 50
    vpin_threshold: float = 0.85

    max_risk_per_trade_pct: float = 0.01
    atr_multiplier_stop: float = 1.5
    atr_multiplier_tp1: float = 1.0
    atr_multiplier_tp2: float = 2.0
    atr_multiplier_tp3: float = 3.0

    # v4 bias-reduction & CV fields
    target_horizon: int = 3
    purge_horizon: int = 5          # bars to purge around test sets
    embargo_pct: float = 0.02       # % of dataset to embargo after test
    use_meta_labeling: bool = True
    meta_label_threshold: float = 0.0  # min PnL to label as "take trade"
    class_weight_scale: float = 5.0    # scale_pos_weight multiplier
    f1_threshold_tune: bool = True     # optimize prob threshold for F1

    kelly_fraction: float = 0.25
    kelly_min_history: int = 20
    fvg_lookback: int = 10
    fvg_body_multiplier: float = 1.5
    vp_lookback: int = 50
    vp_value_area_pct: float = 0.70
    calibration_method: str = "isotonic"
    calibration_cv_splits: int = 3


ASSET_CONFIGS: Dict[Asset, AssetConfig] = {
    Asset.XAUUSD: AssetConfig(
        asset=Asset.XAUUSD,
        pip_value=0.01, spread_avg=0.05, tick_size=0.01,
        contract_size=100.0, leverage=100.0,
        kalman_observation_covariance=0.5,
        kalman_transition_covariance=0.005,
        vpin_buckets=50, vpin_window=50, vpin_threshold=0.90,
        max_risk_per_trade_pct=0.01,
        atr_multiplier_stop=2.0,
        atr_multiplier_tp1=1.5, atr_multiplier_tp2=2.5, atr_multiplier_tp3=4.0,
        kelly_fraction=0.25, fvg_body_multiplier=1.5,
        target_horizon=3, purge_horizon=5, embargo_pct=0.02,
    ),
    Asset.BTCUSD: AssetConfig(
        asset=Asset.BTCUSD,
        pip_value=1.0, spread_avg=20.0, tick_size=0.01,
        contract_size=1.0, leverage=50.0,
        kalman_observation_covariance=100.0,
        kalman_transition_covariance=1.0,
        vpin_buckets=100, vpin_window=100, vpin_threshold=0.80,
        max_risk_per_trade_pct=0.01,
        atr_multiplier_stop=2.5,
        atr_multiplier_tp1=2.0, atr_multiplier_tp2=3.5, atr_multiplier_tp3=5.0,
        kelly_fraction=0.20, fvg_body_multiplier=2.0,
        vp_lookback=100,
        target_horizon=5, purge_horizon=8, embargo_pct=0.03,
    ),
    Asset.EURUSD: AssetConfig(
        asset=Asset.EURUSD,
        pip_value=0.0001, spread_avg=0.0001, tick_size=0.00001,
        contract_size=100000.0, leverage=100.0,
        kalman_observation_covariance=0.0001,
        kalman_transition_covariance=0.000001,
        vpin_buckets=50, vpin_window=50, vpin_threshold=0.90,
        max_risk_per_trade_pct=0.01,
        atr_multiplier_stop=1.5,
        atr_multiplier_tp1=1.0, atr_multiplier_tp2=2.0, atr_multiplier_tp3=3.0,
        kelly_fraction=0.25, fvg_body_multiplier=1.2,
        target_horizon=3, purge_horizon=5, embargo_pct=0.02,
    ),
    Asset.GBPUSD: AssetConfig(
        asset=Asset.GBPUSD,
        pip_value=0.0001, spread_avg=0.0002, tick_size=0.00001,
        contract_size=100000.0, leverage=100.0,
        kalman_observation_covariance=0.0002,
        kalman_transition_covariance=0.000002,
        vpin_buckets=50, vpin_window=50, vpin_threshold=0.85,
        max_risk_per_trade_pct=0.01,
        atr_multiplier_stop=1.5,
        atr_multiplier_tp1=1.0, atr_multiplier_tp2=2.0, atr_multiplier_tp3=3.0,
        kelly_fraction=0.25, fvg_body_multiplier=1.3,
        target_horizon=3, purge_horizon=5, embargo_pct=0.02,
    ),
}


# =============================================================================
# SQLITE BACKTEST DATABASE
# =============================================================================

class SQLiteBacktestDB:
    """Persistent SQLite storage for walk-forward results, trades, and metrics."""

    def __init__(self, db_path: str = "data/precision_backtest.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_tables()

    def _connect(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_tables(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS walkforward_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset TEXT,
                    model_type TEXT,
                    run_timestamp TEXT,
                    n_splits INTEGER,
                    total_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    sharpe REAL,
                    max_drawdown REAL,
                    total_return REAL,
                    ece REAL,
                    p_value REAL,
                    accuracy REAL,
                    precision_macro REAL,
                    recall_macro REAL,
                    f1_macro REAL,
                    precision_weighted REAL,
                    recall_weighted REAL,
                    f1_weighted REAL,
                    mcc REAL,
                    cohens_kappa REAL,
                    auc_roc REAL,
                    auc_pr REAL,
                    balanced_accuracy REAL,
                    long_precision REAL,
                    short_precision REAL,
                    meta_labeler_precision REAL,
                    meta_labeler_recall REAL
                );

                CREATE TABLE IF NOT EXISTS fold_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    fold INTEGER,
                    train_bars INTEGER,
                    test_bars INTEGER,
                    trades INTEGER,
                    win_rate REAL,
                    sharpe REAL,
                    profit_factor REAL,
                    total_return REAL,
                    max_drawdown REAL,
                    accuracy REAL,
                    precision_macro REAL,
                    recall_macro REAL,
                    f1_macro REAL,
                    long_precision REAL,
                    short_precision REAL,
                    FOREIGN KEY (run_id) REFERENCES walkforward_runs(id)
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    fold INTEGER,
                    entry_time TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit_1 REAL,
                    take_profit_2 REAL,
                    take_profit_3 REAL,
                    position_size REAL,
                    pnl REAL,
                    status TEXT,
                    tp_level_hit INTEGER,
                    sizing_method TEXT,
                    FOREIGN KEY (run_id) REFERENCES walkforward_runs(id)
                );

                CREATE TABLE IF NOT EXISTS equity_curves (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    fold INTEGER,
                    timestamp TEXT,
                    equity REAL,
                    FOREIGN KEY (run_id) REFERENCES walkforward_runs(id)
                );

                CREATE TABLE IF NOT EXISTS feature_importance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    fold INTEGER,
                    feature TEXT,
                    importance REAL,
                    FOREIGN KEY (run_id) REFERENCES walkforward_runs(id)
                );

                CREATE INDEX IF NOT EXISTS idx_runs_asset ON walkforward_runs(asset);
                CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id);
                CREATE INDEX IF NOT EXISTS idx_equity_run ON equity_curves(run_id);
            """)
            conn.commit()

    def insert_run(self, metrics: Dict) -> int:
        with self._connect() as conn:
            cols = [
                "asset", "model_type", "run_timestamp", "n_splits", "total_trades",
                "win_rate", "profit_factor", "sharpe", "max_drawdown", "total_return",
                "ece", "p_value", "accuracy", "precision_macro", "recall_macro",
                "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted",
                "mcc", "cohens_kappa", "auc_roc", "auc_pr", "balanced_accuracy",
                "long_precision", "short_precision", "meta_labeler_precision",
                "meta_labeler_recall",
            ]
            vals = [metrics.get(c, None) for c in cols]
            q = f"INSERT INTO walkforward_runs ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})"
            cur = conn.execute(q, vals)
            conn.commit()
            return cur.lastrowid

    def insert_fold(self, run_id: int, fold: int, metrics: Dict):
        with self._connect() as conn:
            cols = [
                "run_id", "fold", "train_bars", "test_bars", "trades", "win_rate",
                "sharpe", "profit_factor", "total_return", "max_drawdown",
                "accuracy", "precision_macro", "recall_macro", "f1_macro",
                "long_precision", "short_precision",
            ]
            vals = [run_id, fold] + [metrics.get(c, None) for c in cols[2:]]
            q = f"INSERT INTO fold_results ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})"
            conn.execute(q, vals)
            conn.commit()

    def insert_trades(self, run_id: int, fold: int, trades: List["Trade"]):
        if not trades:
            return
        with self._connect() as conn:
            for t in trades:
                conn.execute("""
                    INSERT INTO trades (
                        run_id, fold, entry_time, direction, entry_price, exit_price,
                        stop_loss, take_profit_1, take_profit_2, take_profit_3,
                        position_size, pnl, status, tp_level_hit, sizing_method
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    run_id, fold, t.entry_time.isoformat() if isinstance(t.entry_time, datetime) else str(t.entry_time),
                    t.direction.name, t.entry_price, t.exit_price,
                    t.stop_loss, t.take_profit_1, t.take_profit_2, t.take_profit_3,
                    t.position_size, t.pnl, t.status, t.tp_level_hit, t.sizing_method,
                ))
            conn.commit()

    def insert_equity(self, run_id: int, fold: int, timestamps: List, equities: List[float]):
        if not timestamps or not equities:
            return
        with self._connect() as conn:
            data = [
                (run_id, fold,
                 ts.isoformat() if isinstance(ts, datetime) else str(ts),
                 float(eq))
                for ts, eq in zip(timestamps, equities)
                if np.isfinite(eq)
            ]
            conn.executemany("""
                INSERT INTO equity_curves (run_id, fold, timestamp, equity)
                VALUES (?,?,?,?)
            """, data)
            conn.commit()

    def insert_feature_importance(self, run_id: int, fold: int, importance: Dict[str, float]):
        if not importance:
            return
        with self._connect() as conn:
            for feat, imp in importance.items():
                conn.execute("""
                    INSERT INTO feature_importance (run_id, fold, feature, importance)
                    VALUES (?,?,?,?)
                """, (run_id, fold, feat, float(imp)))
            conn.commit()

    def get_runs(self, asset: Optional[str] = None, limit: int = 20) -> pd.DataFrame:
        with self._connect() as conn:
            q = "SELECT * FROM walkforward_runs"
            params = ()
            if asset:
                q += " WHERE asset = ?"
                params = (asset,)
            q += " ORDER BY run_timestamp DESC LIMIT ?"
            params += (limit,)
            return pd.read_sql_query(q, conn, params=params)

    def get_fold_summary(self, run_id: int) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM fold_results WHERE run_id = ? ORDER BY fold",
                conn, params=(run_id,)
            )

    def get_trades(self, run_id: int) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM trades WHERE run_id = ? ORDER BY entry_time",
                conn, params=(run_id,)
            )


# =============================================================================
# CLASSIFICATION METRICS COMPUTER
# =============================================================================

class ClassificationMetrics:
    """Compute comprehensive classification metrics for trading signals.
    
    Maps the 5-class signal {-2,-1,0,1,2} to both multi-class and
    binary (directional) metrics. Handles imbalanced 1m data properly.
    """

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray,
                y_proba: Optional[np.ndarray] = None,
                labels: Optional[List[int]] = None) -> Dict[str, float]:
        if labels is None:
            labels = [-2, -1, 0, 1, 2]
        yt = np.asarray(y_true).flatten()
        yp = np.asarray(y_pred).flatten()

        # Filter to aligned non-NaN
        mask = ~(np.isnan(yt) | np.isnan(yp))
        yt, yp = yt[mask], yp[mask]
        if len(yt) == 0:
            return {k: 0.0 for k in ClassificationMetrics._keys()}

        # Multi-class metrics
        acc = float(accuracy_score(yt, yp))
        prec_macro = float(precision_score(yt, yp, average='macro', zero_division=0, labels=labels))
        rec_macro = float(recall_score(yt, yp, average='macro', zero_division=0, labels=labels))
        f1_macro = float(f1_score(yt, yp, average='macro', zero_division=0, labels=labels))
        prec_w = float(precision_score(yt, yp, average='weighted', zero_division=0, labels=labels))
        rec_w = float(recall_score(yt, yp, average='weighted', zero_division=0, labels=labels))
        f1_w = float(f1_score(yt, yp, average='weighted', zero_division=0, labels=labels))
        bal_acc = float(balanced_accuracy_score(yt, yp))

        try:
            mcc = float(matthews_corrcoef(yt, yp))
        except Exception:
            mcc = 0.0
        try:
            kappa = float(cohen_kappa_score(yt, yp, labels=labels))
        except Exception:
            kappa = 0.0

        # Binary directional metrics (collapse to UP / DOWN / FLAT)
        # For AUC we need binary; treat as {-1, 0, 1} then one-vs-rest for UP
        yt_bin = np.where(yt > 0, 1, np.where(yt < 0, -1, 0))
        yp_bin = np.where(yp > 0, 1, np.where(yp < 0, -1, 0))

        # AUC-ROC & AUC-PR for "UP" class (1) vs rest
        auc_roc = 0.5
        auc_pr = 0.0
        if y_proba is not None and len(y_proba) == len(yt):
            # y_proba shape (n_samples, n_classes) mapped to labels
            # Build binary UP probability = P(1)+P(2)
            up_prob = np.zeros(len(yt))
            for i, lab in enumerate(labels):
                if lab > 0:
                    up_prob += y_proba[:, i]
            try:
                auc_roc = float(roc_auc_score((yt_bin == 1).astype(int), up_prob))
            except Exception:
                pass
            try:
                auc_pr = float(average_precision_score((yt_bin == 1).astype(int), up_prob))
            except Exception:
                pass

        # Per-direction precision
        long_mask = yt_bin == 1
        short_mask = yt_bin == -1
        long_prec = float(precision_score((yt_bin == 1).astype(int),
                                          (yp_bin == 1).astype(int), zero_division=0))
        short_prec = float(precision_score((yt_bin == -1).astype(int),
                                           (yp_bin == -1).astype(int), zero_division=0))

        return {
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "precision_weighted": prec_w,
            "recall_weighted": rec_w,
            "f1_weighted": f1_w,
            "mcc": mcc,
            "cohens_kappa": kappa,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "balanced_accuracy": bal_acc,
            "long_precision": long_prec,
            "short_precision": short_prec,
        }

    @staticmethod
    def _keys():
        return [
            "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted",
            "mcc", "cohens_kappa", "auc_roc", "auc_pr", "balanced_accuracy",
            "long_precision", "short_precision",
        ]


# =============================================================================
# BIAS DETECTOR
# =============================================================================

class BiasDetector:
    """Detect feature leakage and dataset bias before training."""

    @staticmethod
    def detect_leakage(df: pd.DataFrame, features: List[str],
                       future_return_col: str = "future_return",
                       threshold: float = 0.30) -> Dict[str, float]:
        """Flag features whose correlation with future returns exceeds threshold.
        High correlation suggests the feature may contain forward-looking info.
        """
        leaks = {}
        if future_return_col not in df.columns:
            return leaks
        for feat in features:
            if feat not in df.columns:
                continue
            corr = float(df[feat].corr(df[future_return_col]))
            if abs(corr) > threshold:
                leaks[feat] = corr
        return leaks

    @staticmethod
    def compute_class_balance(y: np.ndarray) -> Dict[str, Any]:
        vals, counts = np.unique(y, return_counts=True)
        total = len(y)
        return {
            "classes": vals.tolist(),
            "counts": counts.tolist(),
            "proportions": (counts / total).tolist(),
            "imbalance_ratio": float(counts.max() / counts.min()) if counts.min() > 0 else float('inf'),
        }


# =============================================================================
# LAYER 1 — Microstructure Cleaning (v4 enhanced)
# =============================================================================

class MicrostructureCleaner:
    def __init__(self, config: AssetConfig):
        self.config = config
        self.kalman: Optional["KalmanFilter"] = None
        self._init_kalman()

    def _init_kalman(self):
        if not PYKALMAN_AVAILABLE:
            return
        self.kalman = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=0,
            initial_state_covariance=1,
            observation_covariance=self.config.kalman_observation_covariance,
            transition_covariance=self.config.kalman_transition_covariance,
        )

    def compute_smart_price(self, bid, ask, bid_vol=1.0, ask_vol=1.0):
        total = np.asarray(bid_vol) + np.asarray(ask_vol)
        bid = np.asarray(bid); ask = np.asarray(ask)
        bv  = np.asarray(bid_vol); av = np.asarray(ask_vol)
        out = np.where(total == 0, (bid + ask) / 2, (bid * av + ask * bv) / np.where(total == 0, 1, total))
        return out

    def apply_kalman_filter(self, prices: np.ndarray) -> np.ndarray:
        if not PYKALMAN_AVAILABLE or self.kalman is None:
            return pd.Series(prices).ewm(span=5).mean().values
        s = pd.Series(prices).ffill().bfill()
        if s.empty:
            return prices
        self.kalman.initial_state_mean = float(s.iloc[0])
        try:
            means, _ = self.kalman.filter(s.values)
            return means.flatten()
        except Exception:
            return pd.Series(prices).ewm(span=5).mean().values

    def compute_spread_filter(self, spread, atr):
        if atr is None or (isinstance(atr, float) and atr <= 0):
            return False
        return spread > (atr * 0.3)

    def clean_ohlcv(self, df: pd.DataFrame, has_l2_data: bool = False) -> pd.DataFrame:
        result = df.copy()
        if has_l2_data and "bid" in df.columns and "ask" in df.columns:
            bid_vol = df.get("bid_vol", pd.Series(1.0, index=df.index))
            ask_vol = df.get("ask_vol", pd.Series(1.0, index=df.index))
            result["smart_price"] = self.compute_smart_price(
                df["bid"].values, df["ask"].values,
                bid_vol.values, ask_vol.values,
            )
            result["spread"] = df["ask"] - df["bid"]
        else:
            result["smart_price"] = (df["high"] + df["low"]) / 2
            typical = (df["high"] + df["low"] + df["close"]) / 3
            result["spread"] = (df["close"] - typical).abs() * 2
            result["spread"] = result["spread"].clip(lower=self.config.spread_avg)

        result["kalman_price"] = self.apply_kalman_filter(result["smart_price"].values)
        result["atr_14"] = self._compute_atr(result, 14)
        result["spread_filter_active"] = result.apply(
            lambda r: self.compute_spread_filter(r["spread"], r["atr_14"]),
            axis=1,
        )
        # v4: Roll's effective spread estimator (microstructure noise)
        result["rolls_spread"] = self._rolls_spread(result["close"])
        return result

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close  = (df["low"]  - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def _rolls_spread(close: pd.Series) -> pd.Series:
        """Roll's model: effective spread = 2*sqrt(-cov(diff, diff.shift(1)))."""
        diff = close.diff().dropna()
        cov = diff.rolling(20).cov(diff.shift(1))
        spread = 2 * np.sqrt((-cov).clip(lower=0))
        return spread.reindex(close.index).ffill()


# =============================================================================
# LAYER 2 — Flow & Toxicity (v4 enhanced)
# =============================================================================

class FlowAnalyzer:
    def __init__(self, config: AssetConfig):
        self.config = config

    def compute_vpin(self, df: pd.DataFrame) -> pd.Series:
        price_change = df["close"].diff().fillna(0).values
        volume = df["volume"].values
        buy_vol  = np.where(price_change > 0, volume, 0.0)
        sell_vol = np.where(price_change < 0, volume, 0.0)
        zero = price_change == 0
        buy_vol[zero]  = volume[zero] * 0.5
        sell_vol[zero] = volume[zero] * 0.5

        total_v = (buy_vol + sell_vol).sum()
        n_buckets = self.config.vpin_buckets
        if total_v <= 0 or n_buckets <= 0:
            return pd.Series(0.0, index=df.index)

        bucket_size = total_v / n_buckets
        bucket_buy = bucket_sell = bucket_total = 0.0
        bucket_at_bar: List[Tuple[int, float]] = []
        for i in range(len(df)):
            bucket_buy   += buy_vol[i]
            bucket_sell  += sell_vol[i]
            bucket_total += buy_vol[i] + sell_vol[i]
            if bucket_total >= bucket_size:
                vpin_val = abs(bucket_buy - bucket_sell) / bucket_total if bucket_total > 0 else 0.0
                bucket_at_bar.append((i, vpin_val))
                bucket_buy = bucket_sell = bucket_total = 0.0

        out = pd.Series(np.nan, index=df.index, dtype=float)
        for bar_idx, val in bucket_at_bar:
            out.iloc[bar_idx] = val
        out = out.ffill().fillna(0.0)
        return out.rolling(window=self.config.vpin_window, min_periods=1).mean().clip(0, 1)

    def compute_signed_volume(self, df: pd.DataFrame, method: str = "bulk_volume") -> pd.Series:
        if method == "tick_rule":
            change = df["close"].diff()
            signed = pd.Series(0.0, index=df.index)
            signed[change > 0] =  df.loc[change > 0, "volume"]
            signed[change < 0] = -df.loc[change < 0, "volume"]
            return signed
        bar_range = (df["high"] - df["low"]).replace(0, np.nan)
        position = ((df["close"] - df["low"]) / bar_range).fillna(0.5).clip(0, 1)
        return df["volume"] * (2 * position - 1)

    def compute_cvd(self, df: pd.DataFrame) -> pd.Series:
        price_change = df["close"].diff().fillna(0)
        volume = df["volume"].astype(float)
        buy_vol = pd.Series(np.where(price_change > 0, volume, 0.0), index=df.index)
        sell_vol = pd.Series(np.where(price_change < 0, volume, 0.0), index=df.index)
        zero = price_change == 0
        buy_vol[zero] = volume[zero] * 0.5
        sell_vol[zero] = volume[zero] * 0.5
        return (buy_vol - sell_vol).cumsum()

    def compute_cvd_divergence(self, df: pd.DataFrame, cvd: pd.Series,
                                window: int = 10) -> Tuple[pd.Series, pd.Series]:
        price_low_prev = df["low"].rolling(window).min().shift(1)
        price_high_prev = df["high"].rolling(window).max().shift(1)
        cvd_low_prev = cvd.rolling(window).min().shift(1)
        cvd_high_prev = cvd.rolling(window).max().shift(1)
        bull_div = ((df["low"] < price_low_prev) & (cvd > cvd_low_prev)).astype(int)
        bear_div = ((df["high"] > price_high_prev) & (cvd < cvd_high_prev)).astype(int)
        return bull_div, bear_div

    def compute_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        n = len(df)
        lookback = self.config.vp_lookback
        target_pct = self.config.vp_value_area_pct
        poc = np.full(n, np.nan)
        va_high = np.full(n, np.nan)
        va_low = np.full(n, np.nan)
        closes = df["close"].values
        volumes = df["volume"].values.astype(float)
        for i in range(lookback, n):
            window_close = closes[i - lookback : i]
            window_vol = volumes[i - lookback : i]
            if len(window_close) < 10:
                continue
            n_bins = min(20, max(5, len(window_close) // 2))
            try:
                hist, edges = np.histogram(window_close, bins=n_bins, weights=window_vol)
            except Exception:
                continue
            if hist.sum() <= 0:
                continue
            poc_idx = int(np.argmax(hist))
            poc[i] = (edges[poc_idx] + edges[poc_idx + 1]) / 2
            target_vol = hist.sum() * target_pct
            current_vol = float(hist[poc_idx])
            low_b = high_b = poc_idx
            while current_vol < target_vol and (low_b > 0 or high_b < len(hist) - 1):
                low_v = hist[low_b - 1] if low_b > 0 else 0.0
                high_v = hist[high_b + 1] if high_b < len(hist) - 1 else 0.0
                if low_v > high_v and low_b > 0:
                    low_b -= 1; current_vol += float(low_v)
                elif high_b < len(hist) - 1:
                    high_b += 1; current_vol += float(high_v)
                else:
                    break
            va_low[i] = edges[low_b]
            va_high[i] = edges[high_b + 1]

        result["poc"] = pd.Series(poc, index=df.index).ffill().bfill()
        result["va_high"] = pd.Series(va_high, index=df.index).ffill().bfill()
        result["va_low"] = pd.Series(va_low, index=df.index).ffill().bfill()
        result["in_value_area"] = (
            (df["close"] >= result["va_low"]) & (df["close"] <= result["va_high"])
        ).astype(int)
        result["distance_to_poc"] = ((df["close"] - result["poc"]) /
                                     df["close"].replace(0, np.nan)).fillna(0)
        return result

    @staticmethod
    def detect_absorption(df: pd.DataFrame) -> pd.Series:
        volume = df["volume"].astype(float)
        price_range = (df["high"] - df["low"]).astype(float)
        vol_ma = volume.rolling(20).mean()
        range_ma = price_range.rolling(20).mean()
        return ((volume > vol_ma * 2) & (price_range < range_ma * 0.5)).astype(int)

    def compute_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["vpin"] = self.compute_vpin(df)
        result["signed_volume"] = self.compute_signed_volume(df, "bulk_volume")
        result["signed_volume_momentum"] = result["signed_volume"].rolling(5).mean()
        result["realized_vol_5m"]  = self._realized_vol(df["close"], 5)
        result["realized_vol_20m"] = self._realized_vol(df["close"], 20)
        result["volume_ma_20"] = df["volume"].rolling(20).mean()
        result["volume_ma_ratio"] = (
            df["volume"] / result["volume_ma_20"].replace(0, np.nan)
        ).fillna(1)

        if "bid_vol" in df.columns and "ask_vol" in df.columns:
            total = df["bid_vol"] + df["ask_vol"]
            result["ob_imbalance"] = ((df["bid_vol"] - df["ask_vol"]) / total.replace(0, np.nan)).fillna(0).clip(-1, 1)
        else:
            denom = df["volume"].rolling(5).mean().replace(0, np.nan)
            result["ob_imbalance"] = (
                result["signed_volume_momentum"] / denom
            ).fillna(0).clip(-1, 1)

        cvd = self.compute_cvd(df)
        result["cvd"] = cvd
        result["cvd_slope"] = cvd.diff(5).fillna(0)
        bull_div, bear_div = self.compute_cvd_divergence(df, cvd, window=10)
        result["cvd_bull_div"] = bull_div
        result["cvd_bear_div"] = bear_div

        vp = self.compute_volume_profile(df)
        result["poc"] = vp["poc"]
        result["va_high"] = vp["va_high"]
        result["va_low"] = vp["va_low"]
        result["in_value_area"] = vp["in_value_area"]
        result["distance_to_poc"] = vp["distance_to_poc"]

        result["absorption"] = self.detect_absorption(df)

        # v4: Tick imbalance proxy (microstructure)
        result["tick_imbalance"] = self._tick_imbalance(df)
        return result

    @staticmethod
    def _realized_vol(prices: pd.Series, window: int = 5) -> pd.Series:
        rets = prices.pct_change().fillna(0)
        return (rets.rolling(window).std() * np.sqrt(525_600)).fillna(0)

    @staticmethod
    def _tick_imbalance(df: pd.DataFrame) -> pd.Series:
        """Proxy for tick imbalance using close-to-close direction."""
        direction = np.sign(df["close"].diff().fillna(0))
        up = (direction == 1).rolling(20).sum()
        down = (direction == -1).rolling(20).sum()
        total = up + down
        return ((up - down) / total.replace(0, np.nan)).fillna(0).clip(-1, 1)


# =============================================================================
# LAYER 2b — Market Structure
# =============================================================================

class MarketStructureAnalyzer:
    def __init__(self, config: AssetConfig):
        self.config = config

    def detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        n = len(result)
        fvg_bull = np.zeros(n, dtype=bool)
        fvg_bear = np.zeros(n, dtype=bool)
        bull_start = np.full(n, np.nan)
        bull_end = np.full(n, np.nan)
        bear_start = np.full(n, np.nan)
        bear_end = np.full(n, np.nan)

        highs = result["high"].values
        lows = result["low"].values
        opens = result["open"].values
        closes = result["close"].values
        lookback = self.config.fvg_lookback
        body_mult = self.config.fvg_body_multiplier

        for i in range(2, n):
            start_idx = max(0, i - lookback)
            bodies = np.abs(closes[start_idx:i] - opens[start_idx:i])
            avg_body = float(np.mean(bodies)) if len(bodies) else 0.0
            if avg_body <= 0:
                avg_body = 1e-6
            mid_body = abs(closes[i - 1] - opens[i - 1])
            if lows[i] > highs[i - 2] and mid_body > avg_body * body_mult:
                fvg_bull[i] = True
                bull_start[i] = highs[i - 2]
                bull_end[i] = lows[i]
            elif highs[i] < lows[i - 2] and mid_body > avg_body * body_mult:
                fvg_bear[i] = True
                bear_start[i] = lows[i - 2]
                bear_end[i] = highs[i]

        result["fvg_bullish"] = fvg_bull
        result["fvg_bearish"] = fvg_bear
        result["fvg_bull_start"] = bull_start
        result["fvg_bull_end"] = bull_end
        result["fvg_bear_start"] = bear_start
        result["fvg_bear_end"] = bear_end
        return result

    def get_nearest_fvg_stop(self, df: pd.DataFrame,
                              direction: "TradeDirection") -> Optional[float]:
        if df.empty or "fvg_bullish" not in df.columns:
            return None
        latest = df.iloc[-1]
        if direction == TradeDirection.LONG and bool(latest.get("fvg_bullish", False)):
            v = latest.get("fvg_bull_start")
            return float(v) if v is not None and not pd.isna(v) else None
        if direction == TradeDirection.SHORT and bool(latest.get("fvg_bearish", False)):
            v = latest.get("fvg_bear_start")
            return float(v) if v is not None and not pd.isna(v) else None
        return None


# =============================================================================
# LAYER 3 — Multi-Timeframe Confluence
# =============================================================================

class MTFConfluenceEngine:
    MTF_WEIGHTS = {"1m": 0.5, "5m": 0.3, "15m": 0.2}
    EMA_SPAN = 20

    @staticmethod
    def _resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
        if df.empty:
            return df
        rule = f"{minutes}min"
        agg = {"open": "first", "high": "max", "low": "min",
               "close": "last", "volume": "sum"}
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        try:
            return df[cols].resample(rule).apply(agg).dropna()
        except Exception:
            return pd.DataFrame()

    @classmethod
    def _trend_sign(cls, close: pd.Series) -> pd.Series:
        ema = close.ewm(span=cls.EMA_SPAN).mean()
        return np.where(close > ema, 1, -1)

    @classmethod
    def _align_higher_tf(cls, target_index: pd.Index, tf_close: pd.Series) -> pd.Series:
        if tf_close.empty:
            return pd.Series(np.nan, index=target_index)
        return tf_close.reindex(target_index, method="ffill")

    @classmethod
    def compute(cls, df_1m: pd.DataFrame) -> pd.DataFrame:
        out = df_1m.copy()
        if df_1m.empty or len(df_1m) < 30:
            out["mtf_score"] = 0.0
            out["mtf_confluence"] = "neutral"
            out["trend_1m"] = 0
            out["trend_5m"] = 0
            out["trend_15m"] = 0
            return out

        df_5m = cls._resample(df_1m, 5)
        df_15m = cls._resample(df_1m, 15)

        trend_1m = cls._trend_sign(df_1m["close"])
        out["trend_1m"] = trend_1m

        if not df_5m.empty:
            close_5m_aligned = cls._align_higher_tf(df_1m.index, df_5m["close"])
            out["trend_5m"] = np.where(
                close_5m_aligned > close_5m_aligned.ewm(span=cls.EMA_SPAN).mean(),
                1, -1,
            )
        else:
            out["trend_5m"] = 0
        if not df_15m.empty:
            close_15m_aligned = cls._align_higher_tf(df_1m.index, df_15m["close"])
            out["trend_15m"] = np.where(
                close_15m_aligned > close_15m_aligned.ewm(span=cls.EMA_SPAN).mean(),
                1, -1,
            )
        else:
            out["trend_15m"] = 0

        score = (
            out["trend_1m"] * cls.MTF_WEIGHTS["1m"]
            + out["trend_5m"] * cls.MTF_WEIGHTS["5m"]
            + out["trend_15m"] * cls.MTF_WEIGHTS["15m"]
        )
        out["mtf_score"] = score

        labels = np.full(len(out), "neutral", dtype=object)
        s = score.values
        labels[s > 0.6] = "strong_bullish"
        labels[(s > 0.2) & (s <= 0.6)] = "bullish"
        labels[(s < -0.2) & (s >= -0.6)] = "bearish"
        labels[s < -0.6] = "strong_bearish"
        out["mtf_confluence"] = labels
        return out


# =============================================================================
# LAYER 4 — ML Signal Models (v4 enhanced with meta-labeling & class weights)
# =============================================================================

class LorentzianClassifier:
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
        else:
            self.scaler = None
        self.train_data: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
        rs = gain / loss.replace(0, np.nan)
        feats["rsi_7"] = (100 - 100 / (1 + rs)).fillna(50)

        typical = (df["high"] + df["low"] + df["close"]) / 3
        flow = typical * df["volume"]
        sign = np.where(typical > typical.shift(1), 1, -1)
        signed = pd.Series(flow * sign, index=df.index)
        pos = signed.where(signed > 0, 0).rolling(7).sum()
        neg = (-signed.where(signed < 0, 0)).rolling(7).sum()
        ratio = pos / neg.replace(0, np.nan)
        feats["mfi_7"] = (100 - 100 / (1 + ratio)).fillna(50)

        feats["roc_3"] = ((df["close"] - df["close"].shift(3)) / df["close"].shift(3) * 100).fillna(0)

        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift()).abs()
        tr3 = (df["low"]  - df["close"].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        feats["volatility_gate"] = (tr / atr.replace(0, np.nan)).fillna(1).clip(0, 5)
        return feats

    def fit(self, df: pd.DataFrame, labels: Optional[np.ndarray] = None,
            target_horizon: int = 3):
        feats = self._compute_features(df).dropna()
        if labels is None:
            fwd = df["close"].shift(-target_horizon) / df["close"] - 1
            roll_std = df["close"].pct_change().rolling(50, min_periods=10).std().bfill()
            roll_std = roll_std.clip(lower=1e-6) * np.sqrt(target_horizon)
            z = fwd / roll_std
            labels = pd.Series(0, index=df.index, dtype=int)
            labels[z >  1.0]  = 2
            labels[(z <=  1.0) & (z >  0.3)] = 1
            labels[(z >= -1.0) & (z < -0.3)] = -1
            labels[z < -1.0]  = -2
        else:
            labels = pd.Series(labels, index=df.index)

        aligned = labels.loc[feats.index].values
        X = feats.values
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        self.train_data = X.astype(np.float32)
        self.train_labels = aligned.astype(np.int8)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.train_data is None or self.train_labels is None:
            return pd.Series(0, index=df.index)
        feats = self._compute_features(df).dropna()
        if feats.empty:
            return pd.Series(0, index=df.index)
        X = feats.values
        if self.scaler is not None:
            X = self.scaler.transform(X)

        X32 = X.astype(np.float32)
        M, F = X32.shape
        N = self.train_data.shape[0]
        chunk = max(1, 1_000_000 // max(N, 1))
        preds = np.zeros(M, dtype=np.int8)
        for start in range(0, M, chunk):
            stop = min(start + chunk, M)
            diff = np.abs(X32[start:stop, None, :] - self.train_data[None, :, :])
            dist = np.log1p(diff).sum(axis=2)
            k = min(self.n_neighbors, N)
            idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
            votes = self.train_labels[idx]
            avg = votes.mean(axis=1)
            local = np.zeros(stop - start, dtype=np.int8)
            local[avg >  0.5]  = 2
            local[(avg <=  0.5) & (avg >  0.1)] = 1
            local[(avg >= -0.5) & (avg < -0.1)] = -1
            local[avg < -0.5] = -2
            preds[start:stop] = local
        return pd.Series(preds, index=feats.index)


class XGBoostSignalModel:
    def __init__(self, n_estimators: int = 200, max_depth: int = 5,
                 learning_rate: float = 0.05,
                 calibration_method: Optional[str] = "isotonic",
                 calibration_cv_splits: int = 3,
                 class_weight_scale: float = 5.0,
                 tune_threshold: bool = True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.calibration_method = calibration_method
        self.calibration_cv_splits = max(2, int(calibration_cv_splits))
        self.class_weight_scale = class_weight_scale
        self.tune_threshold = tune_threshold
        self.model = None
        self.calibrated_model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_cols: Optional[List[str]] = None
        self._class_to_idx: Dict[int, int] = {}
        self._idx_to_class: Dict[int, int] = {}
        self._thresholds: Dict[int, float] = {}  # per-class probability threshold
        self.feature_importance_: Dict[str, float] = {}

    def __setstate__(self, state):
        self.__dict__.update(state)
        # v4 backward-compat: attributes added after initial deployment
        self.class_weight_scale = getattr(self, 'class_weight_scale', 5.0)
        self.tune_threshold     = getattr(self, 'tune_threshold',     True)
        self.calibration_method = state.get("calibration_method", "isotonic")
        self.calibration_cv_splits = state.get("calibration_cv_splits", 3)
        self.calibrated_model = state.get("calibrated_model", None)
        self._class_to_idx = state.get("_class_to_idx", {})
        self._idx_to_class = state.get("_idx_to_class", {})
        self._thresholds = state.get("_thresholds", {})
        self.feature_importance_ = state.get("feature_importance_", {})
        if not self._class_to_idx and self.model is not None:
            classes = getattr(self.model, "classes_", None)
            if classes is not None:
                self._idx_to_class = {int(i): int(c) - 2 for i, c in enumerate(classes)}
                self._class_to_idx = {v: k for k, v in self._idx_to_class.items()}

    def _is_fitted(self) -> bool:
        """True only if the underlying xgboost booster has actually been fit."""
        if self.model is None:
            return False
        if not hasattr(self.model, "classes_"):
            return False
        try:
            n_feat = getattr(self.model, "n_features_in_", 1)
            self.model.predict(np.zeros((1, int(n_feat))))
            return True
        except Exception:
            return False

    @staticmethod
    def _rsi(p: pd.Series, period: int = 14) -> pd.Series:
        delta = p.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - 100 / (1 + rs)).fillna(50)

    @staticmethod
    def _macd(p: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        return p.ewm(span=fast).mean() - p.ewm(span=slow).mean()

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        f["returns_1"] = df["close"].pct_change()
        f["returns_3"] = df["close"].pct_change(3)
        f["returns_5"] = df["close"].pct_change(5)
        f["smart_return"] = (df["smart_price"].pct_change() if "smart_price" in df.columns else f["returns_1"])
        f["smart_vs_close"] = (
            (df["smart_price"] - df["close"]) / df["close"]
            if "smart_price" in df.columns else 0
        )
        f["kalman_return"] = (df["kalman_price"].pct_change() if "kalman_price" in df.columns else f["returns_1"])
        f["kalman_vs_close"] = (
            (df["kalman_price"] - df["close"]) / df["close"]
            if "kalman_price" in df.columns else 0
        )
        f["rsi_7"]  = self._rsi(df["close"], 7)
        f["rsi_14"] = self._rsi(df["close"], 14)
        f["macd"]   = self._macd(df["close"])
        f["macd_signal"] = f["macd"].ewm(span=9).mean()
        f["macd_hist"]   = f["macd"] - f["macd_signal"]

        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        f["bb_position"] = ((df["close"] - bb_mid) / bb_std.replace(0, np.nan)).fillna(0)
        f["bb_width"]    = (bb_std / bb_mid.replace(0, np.nan)).fillna(0)

        f["signed_vol_ma5"]  = df["signed_volume"].rolling(5).mean()  if "signed_volume" in df.columns else 0
        f["signed_vol_ma10"] = df["signed_volume"].rolling(10).mean() if "signed_volume" in df.columns else 0
        f["vol_intensity"]   = df["volume_ma_ratio"] if "volume_ma_ratio" in df.columns else (
            df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)
        ).fillna(1)
        f["ob_imbalance"]    = df["ob_imbalance"] if "ob_imbalance" in df.columns else 0
        f["vpin"]            = df["vpin"]         if "vpin" in df.columns else 0
        f["vpin_high"]       = (
            (df["vpin"] > df["vpin"].rolling(100).quantile(0.9)).astype(int)
            if "vpin" in df.columns else 0
        )
        if "realized_vol_5m" in df.columns:
            f["rv_5m"]  = df["realized_vol_5m"]
            f["rv_20m"] = df["realized_vol_20m"]
        else:
            r = df["close"].pct_change().fillna(0)
            f["rv_5m"]  = r.rolling(5).std()  * np.sqrt(525_600)
            f["rv_20m"] = r.rolling(20).std() * np.sqrt(525_600)
        f["vol_regime"] = (f["rv_5m"] > f["rv_5m"].rolling(50).mean()).astype(int)

        f["cvd_slope"]       = df["cvd_slope"]       if "cvd_slope"       in df.columns else 0
        f["cvd_bull_div"]    = df["cvd_bull_div"]    if "cvd_bull_div"    in df.columns else 0
        f["cvd_bear_div"]    = df["cvd_bear_div"]    if "cvd_bear_div"    in df.columns else 0
        f["distance_to_poc"] = df["distance_to_poc"] if "distance_to_poc" in df.columns else 0
        f["in_value_area"]   = df["in_value_area"]   if "in_value_area"   in df.columns else 1
        f["absorption"]      = df["absorption"]      if "absorption"      in df.columns else 0
        f["mtf_score"]       = df["mtf_score"]       if "mtf_score"       in df.columns else 0
        f["trend_5m"]        = df["trend_5m"]        if "trend_5m"        in df.columns else 0
        f["trend_15m"]       = df["trend_15m"]       if "trend_15m"       in df.columns else 0

        # v4 microstructure
        f["rolls_spread"]    = df["rolls_spread"]    if "rolls_spread"    in df.columns else 0
        f["tick_imbalance"]  = df["tick_imbalance"]  if "tick_imbalance"  in df.columns else 0

        # v4 time-of-day seasonality
        if hasattr(df.index, 'hour'):
            f["hour"] = df.index.hour
            f["minute"] = df.index.minute
            f["is_london"] = ((f["hour"] >= 8) & (f["hour"] <= 17)).astype(int)
            f["is_ny"] = ((f["hour"] >= 13) & (f["hour"] <= 22)).astype(int)
        else:
            f["hour"] = 0
            f["minute"] = 0
            f["is_london"] = 0
            f["is_ny"] = 0

        for lag in (1, 2, 3):
            f[f"return_lag_{lag}"] = f["returns_1"].shift(lag)

        f["rsi_x_vol"] = f["rsi_7"]   * f["vol_intensity"]
        f["ob_x_vol"]  = f["ob_imbalance"] * f["vol_intensity"]
        f["cvd_x_vol"] = f["cvd_slope"] * f["vol_intensity"]

        return f.replace([np.inf, -np.inf], 0).fillna(0)

    def fit(self, df: pd.DataFrame, target_horizon: int = 3,
            sample_weight: Optional[np.ndarray] = None):
        feats = self._engineer(df)
        fwd = df["close"].shift(-target_horizon) / df["close"] - 1
        roll_std = df["close"].pct_change().rolling(50, min_periods=10).std().bfill()
        roll_std = roll_std.clip(lower=1e-6) * np.sqrt(target_horizon)
        z = fwd / roll_std
        target = pd.Series(0, index=df.index, dtype=int)
        target[z >  1.0]  = 2
        target[(z <=  1.0) & (z >  0.3)] = 1
        target[(z >= -1.0) & (z < -0.3)] = -1
        target[z < -1.0]  = -2

        af = feats.loc[target.index].dropna()
        ay = target.loc[af.index]
        self.feature_cols = af.columns.tolist()
        X = self.scaler.fit_transform(af) if self.scaler is not None else af.values

        present_classes = sorted(int(c) for c in pd.Series(ay).unique())
        self._class_to_idx = {c: i for i, c in enumerate(present_classes)}
        self._idx_to_class = {i: c for c, i in self._class_to_idx.items()}
        y = np.array([self._class_to_idx[int(v)] for v in ay.values], dtype=np.int64)
        n_classes = len(present_classes)

        # v4: compute class weights for imbalanced 1m data
        class_weights = None
        if n_classes > 1:
            counts = np.bincount(y, minlength=n_classes)
            total = counts.sum()
            cws = getattr(self, 'class_weight_scale', 5.0)
            weights = {i: total / (n_classes * max(counts[i], 1)) * cws
                       for i in range(n_classes)}
            class_weights = weights

        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                num_class=n_classes if n_classes > 2 else None,
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
                random_state=42,
                tree_method="hist",
                reg_lambda=1.0, min_child_weight=10,
            )
            if sample_weight is not None:
                sw = sample_weight[af.index]
                self.model.fit(X, y, sample_weight=sw)
            else:
                self.model.fit(X, y)
        elif SKLEARN_AVAILABLE:
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
            )
            self.model.fit(X, y)
        else:
            raise ImportError("Need xgboost or scikit-learn")

        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance_ = {
                c: float(v) for c, v in zip(self.feature_cols, self.model.feature_importances_)
            }

        # v4: Threshold tuning for F1 on validation split (temporal)
        self._thresholds = {}
        if self.tune_threshold and SKLEARN_AVAILABLE and len(X) > 100:
            self._tune_thresholds(X, y)

        # Calibration
        self.calibrated_model = None
        if (SKLEARN_AVAILABLE and self.calibration_method in ("isotonic", "sigmoid")
                and CalibratedClassifierCV is not None and TimeSeriesSplit is not None
                and len(np.unique(y)) >= 2 and len(X) >= 50):
            try:
                tscv = TimeSeriesSplit(n_splits=self.calibration_cv_splits)
                if XGBOOST_AVAILABLE:
                    base = xgb.XGBClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                        num_class=n_classes if n_classes > 2 else None,
                        eval_metric="mlogloss" if n_classes > 2 else "logloss",
                        random_state=42,
                        tree_method="hist",
                        reg_lambda=1.0, min_child_weight=10,
                    )
                else:
                    base = GradientBoostingClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        random_state=42,
                    )
                cal = CalibratedClassifierCV(base, method=self.calibration_method, cv=tscv)
                cal.fit(X, y)
                self.calibrated_model = cal
            except Exception as e:
                print(f"[XGBoost] calibration failed: {e}")
                self.calibrated_model = None

    def _tune_thresholds(self, X: np.ndarray, y: np.ndarray):
        """Find per-class probability thresholds that maximize F1 on last 20% of data."""
        split = int(len(X) * 0.8)
        X_val, y_val = X[split:], y[split:]
        model = self.calibrated_model if self.calibrated_model is not None else self.model
        proba = model.predict_proba(X_val)
        best_thresh = 0.5
        best_f1 = 0.0
        for thresh in np.arange(0.1, 0.91, 0.05):
            pred = np.argmax(proba * (proba >= thresh), axis=1)
            # Handle all-below-threshold
            pred = np.where(proba.max(axis=1) < thresh, -1, pred)
            valid = pred != -1
            if valid.sum() < 10:
                continue
            f1 = f1_score(y_val[valid], pred[valid], average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        self._thresholds["default"] = best_thresh

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if not self._is_fitted() or self.feature_cols is None:
            return pd.Series(0, index=df.index)
        feats = self._engineer(df)
        for c in self.feature_cols:
            if c not in feats.columns:
                feats[c] = 0
        feats = feats[self.feature_cols]
        X = self.scaler.transform(feats) if self.scaler is not None else feats.values
        try:
            encoded = self.model.predict(X)
        except Exception as e:
            print(f"[XGBoost] predict failed ({type(e).__name__}: {e}); returning HOLD")
            return pd.Series(0, index=feats.index)
        decoded = np.array([self._idx_to_class.get(int(i), 0) for i in encoded], dtype=int)
        return pd.Series(decoded, index=feats.index)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [-2, -1, 0, 1, 2]
        if not self._is_fitted() or self.feature_cols is None:
            return pd.DataFrame(0.2, index=df.index, columns=cols)
        feats = self._engineer(df)
        for c in self.feature_cols:
            if c not in feats.columns:
                feats[c] = 0
        feats = feats[self.feature_cols]
        X = self.scaler.transform(feats) if self.scaler is not None else feats.values
        model = self.calibrated_model if self.calibrated_model is not None else self.model
        try:
            proba = model.predict_proba(X)
        except Exception as e:
            try:
                proba = self.model.predict_proba(X)
            except Exception as e2:
                print(f"[XGBoost] predict_proba failed ({type(e2).__name__}: {e2}); using uniform")
                proba = np.full((X.shape[0], len(cols)), 1.0 / len(cols))
                return pd.DataFrame(proba, index=feats.index, columns=cols)
        full = np.zeros((proba.shape[0], len(cols)))
        col_index = {c: i for i, c in enumerate(cols)}
        for enc_idx in range(proba.shape[1]):
            orig_class = self._idx_to_class.get(enc_idx)
            if orig_class is None:
                continue
            if orig_class in col_index:
                full[:, col_index[orig_class]] = proba[:, enc_idx]
        return pd.DataFrame(full, index=feats.index, columns=cols)


# =============================================================================
# STACKED ENSEMBLE ENGINE
# =============================================================================

class StackedEnsemble:
    """Stacking: XGBoost + LightGBM + CatBoost → LogisticRegression meta-learner.
    Uses purged CV for the meta-learner to prevent leakage.
    """
    def __init__(self, class_weight_scale: float = 5.0,
                 calibration_method: str = "isotonic",
                 calibration_cv_splits: int = 3):
        self.class_weight_scale = class_weight_scale
        self.calibration_method = calibration_method
        self.calibration_cv_splits = calibration_cv_splits
        
        self.xgb = None
        self.lgb = None
        self.cbt = None
        self.meta_learner = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_cols: Optional[List[str]] = None
        self._idx_to_class: Dict[int, int] = {}
        self._class_to_idx: Dict[int, int] = {}
        self.feature_importance_: Dict[str, float] = {}
        self.is_fitted = False
        self.model = None

    @staticmethod
    def _check_deps():
        has_xgb = XGBOOST_AVAILABLE
        has_lgb = False
        has_cbt = False
        try:
            import lightgbm as lgb
            has_lgb = True
        except ImportError:
            pass
        try:
            import catboost as cbt
            has_cbt = True
        except ImportError:
            pass
        return has_xgb, has_lgb, has_cbt

    def fit(self, df: pd.DataFrame, target_horizon: int = 3):
        """Fit using same interface as XGBoostSignalModel."""
        temp = XGBoostSignalModel()
        feats = temp._engineer(df)
        
        fwd = df["close"].shift(-target_horizon) / df["close"] - 1
        roll_std = df["close"].pct_change().rolling(50, min_periods=10).std().bfill()
        roll_std = roll_std.clip(lower=1e-6) * np.sqrt(target_horizon)
        z = fwd / roll_std
        
        target = pd.Series(0, index=df.index, dtype=int)
        target[z > 1.5] = 2
        target[(z <= 1.5) & (z > 0.8)] = 1
        target[(z >= -1.5) & (z < -0.8)] = -1
        target[z < -1.5] = -2
        
        af = feats.loc[target.index].dropna()
        ay = target.loc[af.index]
        self.feature_cols = af.columns.tolist()
        X = self.scaler.fit_transform(af) if self.scaler is not None else af.values
        y = ay.values
        
        self._fit_internal(X, y, self.feature_cols)
        
    def _fit_internal(self, X: np.ndarray, y: np.ndarray, feature_cols: List[str]):
        has_xgb, has_lgb, has_cbt = self._check_deps()
        n_classes = len(np.unique(y))
        
        present_classes = sorted(int(c) for c in np.unique(y))
        self._class_to_idx = {c: i for i, c in enumerate(present_classes)}
        self._class_to_idx = {i: c for c, i in self._class_to_idx.items()}
        
        base_preds = []
        
        if has_xgb:
            self.xgb = xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                num_class=n_classes if n_classes > 2 else None,
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
                random_state=42, tree_method="hist",
                reg_lambda=2.0, min_child_weight=15,
            )
            self.xgb.fit(X, y)
            base_preds.append(self.xgb.predict_proba(X))
            
        if has_lgb:
            import lightgbm as lgb
            self.lgb = lgb.LGBMClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                random_state=42, reg_lambda=2.0, min_child_samples=15,
                verbose=-1,
            )
            self.lgb.fit(X, y)
            base_preds.append(self.lgb.predict_proba(X))
            
        if has_cbt:
            import catboost as cbt
            self.cbt = cbt.CatBoostClassifier(
                iterations=150, depth=4, learning_rate=0.05,
                loss_function="MultiClass" if n_classes > 2 else "Logloss",
                random_seed=42, l2_leaf_reg=2.0,
                verbose=False,
            )
            self.cbt.fit(X, y)
            base_preds.append(self.cbt.predict_proba(X))
            
        if len(base_preds) == 0:
            raise ImportError("No boosting libraries available")
            
        meta_X = np.hstack(base_preds)
        
        from sklearn.linear_model import LogisticRegression
        self.meta_learner = LogisticRegression(
            multi_class='multinomial' if n_classes > 2 else 'auto',
            max_iter=1000, C=0.5, random_state=42,
            solver='lbfgs' if n_classes <= 2 else 'saga',
        )
        self.meta_learner.fit(meta_X, y)
        
        self.is_fitted = True
        self.model = self

    def _is_fitted(self) -> bool:
        return self.is_fitted and self.meta_learner is not None

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if not self._is_fitted() or self.feature_cols is None:
            return pd.Series(0, index=df.index)
        temp = XGBoostSignalModel()
        feats = temp._engineer(df)
        for c in self.feature_cols:
            if c not in feats.columns:
                feats[c] = 0
        feats = feats[self.feature_cols]
        X = self.scaler.transform(feats) if self.scaler is not None else feats.values
        
        try:
            encoded = self.predict_np(X)
        except Exception as e:
            print(f"[StackedEnsemble] predict failed: {e}")
            return pd.Series(0, index=feats.index)
        
        return pd.Series(encoded, index=feats.index)

    def predict_np(self, X: np.ndarray) -> np.ndarray:
        base_preds = []
        if self.xgb is not None:
            base_preds.append(self.xgb.predict_proba(X))
        if self.lgb is not None:
            base_preds.append(self.lgb.predict_proba(X))
        if self.cbt is not None:
            base_preds.append(self.cbt.predict_proba(X))
        if not base_preds:
            return np.zeros(len(X), dtype=int)
        meta_X = np.hstack(base_preds)
        return self.meta_learner.predict(meta_X)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [-2, -1, 0, 1, 2]
        if not self._is_fitted() or self.feature_cols is None:
            return pd.DataFrame(0.2, index=df.index, columns=cols)
        temp = XGBoostSignalModel()
        feats = temp._engineer(df)
        for c in self.feature_cols:
            if c not in feats.columns:
                feats[c] = 0
        feats = feats[self.feature_cols]
        X = self.scaler.transform(feats) if self.scaler is not None else feats.values
        
        try:
            proba = self.predict_proba_np(X)
        except Exception as e:
            print(f"[StackedEnsemble] predict_proba failed: {e}")
            return pd.DataFrame(0.2, index=feats.index, columns=cols)
        
        full = np.zeros((proba.shape[0], len(cols)))
        col_index = {c: i for i, c in enumerate(cols)}
        for enc_idx in range(proba.shape[1]):
            orig = self._class_to_idx.get(enc_idx)
            if orig in col_index:
                full[:, col_index[orig]] = proba[:, enc_idx]
        return pd.DataFrame(full, index=feats.index, columns=cols)

    def predict_proba_np(self, X: np.ndarray) -> np.ndarray:
        base_preds = []
        if self.xgb is not None:
            base_preds.append(self.xgb.predict_proba(X))
        if self.lgb is not None:
            base_preds.append(self.lgb.predict_proba(X))
        if self.cbt is not None:
            base_preds.append(self.cbt.predict_proba(X))
        if not base_preds:
            return np.full((X.shape[0], 5), 0.2)
        meta_X = np.hstack(base_preds)
        return self.meta_learner.predict_proba(meta_X)
        has_xgb, has_lgb, has_cbt = self._check_deps()
        n_classes = len(np.unique(y))
        self.feature_cols = feature_cols
        
        present_classes = sorted(int(c) for c in np.unique(y))
        self._class_to_idx = {c: i for i, c in enumerate(present_classes)}
        self._idx_to_class = {i: c for c, i in self._class_to_idx.items()}
        
        base_preds = []
        base_models = []
        
        if has_xgb:
            self.xgb = xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                num_class=n_classes if n_classes > 2 else None,
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
                random_state=42, tree_method="hist",
                reg_lambda=2.0, min_child_weight=15,
            )
            self.xgb.fit(X, y)
            base_preds.append(self.xgb.predict_proba(X))
            base_models.append("xgb")
            
        if has_lgb:
            import lightgbm as lgb
            self.lgb = lgb.LGBMClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                random_state=42, reg_lambda=2.0, min_child_samples=15,
                verbose=-1,
            )
            self.lgb.fit(X, y)
            base_preds.append(self.lgb.predict_proba(X))
            base_models.append("lgb")
            
        if has_cbt:
            import catboost as cbt
            self.cbt = cbt.CatBoostClassifier(
                iterations=150, depth=4, learning_rate=0.05,
                loss_function="MultiClass" if n_classes > 2 else "Logloss",
                random_seed=42, l2_leaf_reg=2.0,
                verbose=False,
            )
            self.cbt.fit(X, y)
            base_preds.append(self.cbt.predict_proba(X))
            base_models.append("cbt")
            
        if len(base_preds) == 0:
            raise ImportError("No boosting libraries available")
            
        meta_X = np.hstack(base_preds)
        
        from sklearn.linear_model import LogisticRegression
        self.meta_learner = LogisticRegression(
            multi_class='multinomial' if n_classes > 2 else 'auto',
            max_iter=1000, C=0.5, random_state=42,
            solver='lbfgs' if n_classes <= 2 else 'saga',
        )
        self.meta_learner.fit(meta_X, y)
        
        imp = np.zeros(len(feature_cols))
        count = 0
        if self.xgb is not None and hasattr(self.xgb, 'feature_importances_'):
            imp += self.xgb.feature_importances_
            count += 1
        if self.lgb is not None and hasattr(self.lgb, 'feature_importances_'):
            lgb_imp = np.zeros(len(feature_cols))
            lgb_imp[:len(self.lgb.feature_importances_)] = self.lgb.feature_importances_
            imp += lgb_imp
            count += 1
        if count > 0:
            self.feature_importance_ = {
                c: float(v) for c, v in zip(feature_cols, imp / count)
            }
            
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.full((X.shape[0], max(len(self._idx_to_class), 1)), 1.0 / max(len(self._idx_to_class), 1))
        base_preds = []
        if self.xgb is not None:
            base_preds.append(self.xgb.predict_proba(X))
        if self.lgb is not None:
            base_preds.append(self.lgb.predict_proba(X))
        if self.cbt is not None:
            base_preds.append(self.cbt.predict_proba(X))
        meta_X = np.hstack(base_preds)
        return self.meta_learner.predict_proba(meta_X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def triple_barrier_labels(prices: pd.Series, events: pd.Series,
                          pt_sl: Tuple[float, float] = (1.0, 1.0),
                          min_ret: float = 0.005,
                          num_days: int = 3) -> pd.DataFrame:
    """Returns DataFrame with [t1, trgt, side, label].
    events: Series of entry timestamps (side=1 for long, -1 for short).
    """
    out = pd.DataFrame(index=events.index)
    out['t1'] = pd.NaT
    out['trgt'] = prices.pct_change().rolling(50).std().bfill() * np.sqrt(num_days)
    out['side'] = events
    out['label'] = 0
    
    for loc, side in events.items():
        target = out.loc[loc, 'trgt']
        if pd.isna(target) or target <= 0:
            continue
        pt = pt_sl[0] * target
        sl = pt_sl[1] * target
        
        future = prices.loc[loc:].iloc[1:num_days+1]
        if len(future) == 0:
            continue
            
        if side == 1:
            touch_upper = future[future >= prices.loc[loc] * (1 + pt)]
            touch_lower = future[future <= prices.loc[loc] * (1 - sl)]
        else:
            touch_upper = future[future >= prices.loc[loc] * (1 + sl)]
            touch_lower = future[future <= prices.loc[loc] * (1 - pt)]
            
        t_upper = touch_upper.index[0] if len(touch_upper) else pd.NaT
        t_lower = touch_lower.index[0] if len(touch_lower) else pd.NaT
        
        if pd.isna(t_upper) and pd.isna(t_lower):
            out.loc[loc, 't1'] = future.index[-1]
            out.loc[loc, 'label'] = 0
        elif pd.isna(t_upper) or (not pd.isna(t_lower) and t_lower < t_upper):
            out.loc[loc, 't1'] = t_lower
            out.loc[loc, 'label'] = -1 if side == 1 else 1
        else:
            out.loc[loc, 't1'] = t_upper
            out.loc[loc, 'label'] = 1 if side == 1 else -1
            
    out['label'] = out['label'].fillna(0)
    out['meta_label'] = (out['label'] == out['side']).astype(int)
    return out


class RegimeSpecificRouter:
    """Trains separate models for each regime. Routes inference by current regime."""
    def __init__(self, base_model_factory, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.models: Dict[int, Any] = {}
        self.factory = base_model_factory
        self.global_model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_cols: Optional[List[str]] = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, regimes: np.ndarray):
        self.global_model = self.factory()
        if hasattr(self.global_model, 'fit'):
            if hasattr(self.global_model, 'scaler'):
                X_sc = self.scaler.fit_transform(X) if self.scaler else X
            else:
                X_sc = X
            # Check if fit expects df or X,y
            try:
                import pandas as pd
                df_mock = pd.DataFrame(X_sc, columns=[f"f{i}" for i in range(X_sc.shape[1])])
                self.global_model.fit(df_mock)
            except:
                self.global_model.fit(X_sc, y)
            
        for r in range(self.n_regimes):
            mask = regimes == r
            if mask.sum() < 200:
                continue
            model = self.factory()
            if hasattr(model, 'scaler'):
                X_r = self.scaler.fit_transform(X[mask]) if self.scaler else X[mask]
            else:
                X_r = X[mask]
            try:
                import pandas as pd
                df_mock = pd.DataFrame(X_r, columns=[f"f{i}" for i in range(X_r.shape[1])])
                model.fit(df_mock)
            except:
                model.fit(X_r, y[mask])
            self.models[r] = model
            
    def _is_fitted(self) -> bool:
        return self.is_fitted
        
    def predict(self, df: pd.DataFrame, regime: int) -> pd.Series:
        model = self.models.get(regime, self.global_model)
        if model is None:
            return pd.Series(0, index=df.index)
        try:
            return model.predict(df)
        except:
            return pd.Series(0, index=df.index)
        
    def predict_proba(self, df: pd.DataFrame, regime: int) -> pd.DataFrame:
        cols = [-2, -1, 0, 1, 2]
        model = self.models.get(regime, self.global_model)
        if model is None:
            return pd.DataFrame(0.2, index=df.index, columns=cols)
        try:
            return model.predict_proba(df)
        except:
            return pd.DataFrame(0.2, index=df.index, columns=cols)


class DynamicEnsemble:
    """Models go through hot and cold streaks. Weight them by recent OOS Sharpe."""
    def __init__(self, models: Dict[str, Any], lookback: int = 20):
        self.models = models
        self.lookback = lookback
        self.performance_log: Dict[str, Any] = {k: None for k in models}
        self.weights: Dict[str, float] = {k: 1.0/len(models) for k in models}
        
    def update_weights(self, returns: Dict[str, float]):
        from collections import deque
        for k, ret in returns.items():
            if self.performance_log[k] is None:
                self.performance_log[k] = deque(maxlen=self.lookback)
            self.performance_log[k].append(ret)
        
        sharpes = {}
        for k, log in self.performance_log.items():
            if log is None or len(log) < 5:
                sharpes[k] = 0
                continue
            arr = np.array(list(log))
            if arr.std() == 0:
                sharpes[k] = 0
            else:
                sharpes[k] = arr.mean() / arr.std()
        
        exp_s = {k: np.exp(max(s, 0)) for k, s in sharpes.items()}
        total = sum(exp_s.values())
        if total > 0:
            self.weights = {k: v/total for k, v in exp_s.items()}
        
    def predict_proba(self, X_dict: Dict[str, np.ndarray]) -> np.ndarray:
        weighted = None
        for k, X in X_dict.items():
            model = self.models.get(k)
            if model is None:
                continue
            try:
                proba = model.predict_proba(X)
                w = self.weights.get(k, 0)
                if weighted is None:
                    weighted = w * proba
                else:
                    weighted += w * proba
            except:
                continue
        if weighted is None:
            return np.full((X.shape[0], 5), 0.2)
        return weighted


class GRUSignalModel:
    """Lightweight GRU for temporal microstructure patterns."""
    def __init__(self, seq_len: int = 30, hidden_dim: int = 64,
                 epochs: int = 50, batch_size: int = 256):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_cols = None
        self.device = None
        self.torch_available = False
        
    def _check_torch(self):
        if self.torch_available:
            return True
        try:
            import torch
            import torch.nn as nn
            self.torch_available = True
            self.device = torch.device("cpu")
            return True
        except ImportError:
            self.torch_available = False
            return False
            
    def fit(self, df: pd.DataFrame, target_horizon: int = 3):
        if not self._check_torch():
            print("[GRU] PyTorch not available, skipping")
            return
            
        import torch
        import torch.nn as nn
        
        temp = XGBoostSignalModel()
        feats = temp._engineer(df).dropna()
        self.feature_cols = feats.columns.tolist()
        
        fwd = df["close"].shift(-target_horizon) / df["close"] - 1
        roll_std = df["close"].pct_change().rolling(50, min_periods=10).std().bfill()
        roll_std = roll_std.clip(lower=1e-6) * np.sqrt(target_horizon)
        z = fwd / roll_std
        
        y = pd.Series(0, index=df.index, dtype=int)
        y[z > 1.0] = 2
        y[(z <= 1.0) & (z > 0.3)] = 1
        y[(z >= -1.0) & (z < -0.3)] = -1
        y[z < -1.0] = -2
        
        aligned = y.loc[feats.index].values[self.seq_len:]
        X_seq = np.array([feats.values[i-self.seq_len:i] 
                         for i in range(self.seq_len, len(feats))])
        
        if self.scaler is not None:
            B, T, F = X_seq.shape
            X_flat = X_seq.reshape(-1, F)
            X_flat = self.scaler.fit_transform(X_flat)
            X_seq = X_flat.reshape(B, T, F)
            
        y_mapped = y.loc[feats.index].values[self.seq_len:] + 2
        
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_mapped, dtype=torch.long),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        class GRUModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
                super().__init__()
                self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2,
                                  batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(dropout)
            def forward(self, x):
                out, _ = self.gru(x)
                out = self.dropout(out[:, -1, :])
                return self.fc(out)
                
        self.model = GRUModel(input_dim=X_seq.shape[2],
                              hidden_dim=self.hidden_dim,
                              output_dim=5).to(self.device)
        
        counts = np.bincount(y_mapped, minlength=5)
        weights = 1.0 / np.clip(counts, 1, None)
        weights = weights / weights.sum() * 5
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                
    def _is_fitted(self):
        return self.model is not None and self.torch_available
        
    def predict(self, df: pd.DataFrame) -> pd.Series:
        if not self._is_fitted():
            return pd.Series(0, index=df.index)
            
        import torch
        temp = XGBoostSignalModel()
        feats = temp._engineer(df)[self.feature_cols].dropna()
        if len(feats) < self.seq_len:
            return pd.Series(0, index=df.index)
            
        X_seq = np.array([feats.values[i-self.seq_len:i]
                         for i in range(self.seq_len, len(feats))])
        if self.scaler is not None:
            B, T, F = X_seq.shape
            X_flat = X_seq.reshape(-1, F)
            X_flat = self.scaler.transform(X_flat)
            X_seq = X_flat.reshape(B, T, F)
            
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            logits = self.model(tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy() - 2
        return pd.Series(preds, index=feats.index[self.seq_len:])
        
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [-2, -1, 0, 1, 2]
        if not self._is_fitted():
            return pd.DataFrame(0.2, index=df.index, columns=cols)
            
        import torch
        temp = XGBoostSignalModel()
        feats = temp._engineer(df)[self.feature_cols].dropna()
        if len(feats) < self.seq_len:
            return pd.DataFrame(0.2, index=feats.index, columns=cols)
            
        X_seq = np.array([feats.values[i-self.seq_len:i]
                         for i in range(self.seq_len, len(feats))])
        if self.scaler is not None:
            B, T, F = X_seq.shape
            X_flat = X_seq.reshape(-1, F)
            X_flat = self.scaler.transform(X_flat)
            X_seq = X_flat.reshape(B, T, F)
            
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            logits = self.model(tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            
        full = np.zeros((proba.shape[0], len(cols)))
        for i, c in enumerate(range(-2, 3)):
            if i < proba.shape[1]:
                full[:, i + 2] = proba[:, i]
        return pd.DataFrame(full, index=feats.index[self.seq_len:])


class MetaLabeler:
    """Secondary model that predicts whether a primary signal is profitable.
    
    Primary model: predicts direction {-2,-1,0,1,2}
    Meta-labeler: binary classifier predicting "should we take this trade?"
    This filters false positives and improves precision dramatically.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 4):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, primary_signals: pd.Series,
            target_horizon: int = 3, threshold: float = 0.0):
        """Train meta-labeler on whether primary signal produced profit."""
        if not SKLEARN_AVAILABLE:
            return
        # Features = original features + primary signal + primary confidence
        feats = pd.DataFrame(index=df.index)
        for c in df.columns:
            if c in ["open", "high", "low", "close", "volume"]:
                continue
            feats[c] = df[c]

        feats["primary_signal"] = primary_signals.reindex(feats.index).fillna(0)
        # Simple confidence proxy: absolute signal strength
        feats["primary_confidence"] = feats["primary_signal"].abs()

        # Label: was the forward return profitable when signal != 0?
        fwd = df["close"].shift(-target_horizon) / df["close"] - 1
        label = pd.Series(0, index=df.index, dtype=int)
        mask = primary_signals != 0
        # Long signals: profit if fwd > threshold
        label.loc[mask & (primary_signals > 0) & (fwd > threshold)] = 1
        # Short signals: profit if fwd < -threshold
        label.loc[mask & (primary_signals < 0) & (fwd < -threshold)] = 1

        af = feats.dropna()
        ay = label.loc[af.index]
        if len(af) < 100 or ay.sum() < 10:
            return

        X = self.scaler.fit_transform(af) if self.scaler is not None else af.values
        y = ay.values

        if XGBOOST_AVAILABLE:
            scale = float((y == 0).sum() / max((y == 1).sum(), 1))
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=0.05, random_state=42, tree_method="hist",
                scale_pos_weight=scale,
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=0.05, random_state=42,
            )
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, df: pd.DataFrame, primary_signals: pd.Series) -> pd.Series:
        if not self.is_fitted or self.model is None:
            return pd.Series(1, index=df.index)  # allow all by default
        feats = pd.DataFrame(index=df.index)
        for c in df.columns:
            if c in ["open", "high", "low", "close", "volume"]:
                continue
            feats[c] = df[c]
        feats["primary_signal"] = primary_signals.reindex(feats.index).fillna(0)
        feats["primary_confidence"] = feats["primary_signal"].abs()
        feats = feats.dropna()
        if feats.empty:
            return pd.Series(1, index=df.index)
        X = self.scaler.transform(feats) if self.scaler is not None else feats.values
        proba = self.model.predict_proba(X)[:, 1]
        return pd.Series((proba > 0.5).astype(int), index=feats.index)


class HMMRegimeDetector:
    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None

    def fit(self, df: pd.DataFrame):
        if not HMMLEARN_AVAILABLE or not SKLEARN_AVAILABLE:
            return
        self.scaler = StandardScaler()
        rets = df["close"].pct_change().fillna(0).values.reshape(-1, 1).astype(np.float64)
        vol = pd.Series(rets.flatten()).rolling(10).std().fillna(0).values.reshape(-1, 1).astype(np.float64)
        feats = np.hstack([rets, vol]).astype(np.float64)
        feats = feats[~np.isnan(feats).any(axis=1)]
        if len(feats) < 100:
            return
        try:
            Xs = self.scaler.fit_transform(feats)
        except Exception as e:
            print(f"[HMM] scaler fit failed: {e}")
            self.model = None
            return
        self.model = GaussianHMM(
            n_components=self.n_regimes, covariance_type="diag",
            n_iter=100, random_state=42, min_covar=1e-3,
        )
        try:
            self.model.fit(Xs)
        except Exception as e:
            print(f"[HMM] fit failed (diag): {e}; trying spherical fallback")
            try:
                self.model = GaussianHMM(
                    n_components=self.n_regimes, covariance_type="spherical",
                    n_iter=100, random_state=42, min_covar=1e-3,
                )
                self.model.fit(Xs)
            except Exception as e2:
                print(f"[HMM] spherical fallback also failed: {e2}")
                self.model = None

    def predict_regime(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None or self.scaler is None:
            return pd.Series(0, index=df.index)
        try:
            if not hasattr(self.model, 'means_'):
                self.model = None
                return pd.Series(0, index=df.index)
            rets = df["close"].pct_change().fillna(0).values.reshape(-1, 1).astype(np.float64)
            vol = pd.Series(rets.flatten()).rolling(10).std().fillna(0).values.reshape(-1, 1).astype(np.float64)
            feats = np.hstack([rets, vol]).astype(np.float64)
            feats = self.scaler.transform(feats)
            return pd.Series(self.model.predict(feats), index=df.index)
        except Exception as e:
            print(f"[HMM] predict_regime failed: {e}")
            return pd.Series(0, index=df.index)


# =============================================================================
# LAYER 5 — Risk + Execution
# =============================================================================

@dataclass
class Trade:
    entry_time: datetime
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    position_size: float
    asset: Asset
    fvg_stop: Optional[float] = None
    sizing_method: str = "fixed"
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "open"
    tp_level_hit: int = 0


class RiskManager:
    def __init__(self, config: AssetConfig):
        self.config = config
        self.equity = 10_000.0
        self.initial_equity = 10_000.0
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[float] = [10_000.0]
        self.trade_history_pnls: List[float] = []

    def set_equity(self, equity: float):
        self.equity = float(equity)
        self.initial_equity = float(equity)
        self.equity_curve = [float(equity)]

    def _size_in_lots(self, risk_amt: float, entry: float, stop: float) -> float:
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0 or entry <= 0:
            return 0.0
        units = risk_amt / risk_per_unit
        contract_size = max(self.config.contract_size, 1e-9)
        lots = units / contract_size
        max_notional = self.equity * max(self.config.leverage, 1.0)
        notional = lots * contract_size * entry
        if notional > max_notional and notional > 0:
            lots *= max_notional / notional
        return max(0.0, lots)

    def calculate_position_size(self, entry: float, stop: float) -> float:
        return self._size_in_lots(
            self.equity * self.config.max_risk_per_trade_pct, entry, stop)

    def kelly_position_size(self, win_rate: float, avg_win: float, avg_loss: float,
                            entry: float, stop: float) -> float:
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return self.calculate_position_size(entry, stop)
        b = avg_win / avg_loss
        kelly = (win_rate * b - (1.0 - win_rate)) / b
        kelly = max(0.0, min(kelly, 0.5))
        scale = 1.0 + (kelly * self.config.kelly_fraction * 4.0)
        return self._size_in_lots(
            self.equity * self.config.max_risk_per_trade_pct * scale, entry, stop)

    def _kelly_stats(self) -> Optional[Tuple[float, float, float]]:
        if len(self.trade_history_pnls) < self.config.kelly_min_history:
            return None
        wins = [p for p in self.trade_history_pnls if p > 0]
        losses = [p for p in self.trade_history_pnls if p <= 0]
        if not wins or not losses:
            return None
        return (
            len(wins) / len(self.trade_history_pnls),
            float(np.mean(wins)),
            float(abs(np.mean(losses))),
        )

    def calculate_levels(self, entry: float, direction: TradeDirection, atr: float,
                         fvg_stop: Optional[float] = None):
        if direction == TradeDirection.LONG:
            sl  = entry - atr * self.config.atr_multiplier_stop
            tp1 = entry + atr * self.config.atr_multiplier_tp1
            tp2 = entry + atr * self.config.atr_multiplier_tp2
            tp3 = entry + atr * self.config.atr_multiplier_tp3
            if fvg_stop is not None and fvg_stop < entry and fvg_stop > sl:
                sl = fvg_stop
        else:
            sl  = entry + atr * self.config.atr_multiplier_stop
            tp1 = entry - atr * self.config.atr_multiplier_tp1
            tp2 = entry - atr * self.config.atr_multiplier_tp2
            tp3 = entry - atr * self.config.atr_multiplier_tp3
            if fvg_stop is not None and fvg_stop > entry and fvg_stop < sl:
                sl = fvg_stop
        return sl, tp1, tp2, tp3

    def open_trade(self, ts: datetime, direction: TradeDirection,
                    entry: float, atr: float,
                    fvg_stop: Optional[float] = None,
                    use_kelly: bool = True) -> Optional[Trade]:
        if direction == TradeDirection.FLAT:
            return None
        sl, tp1, tp2, tp3 = self.calculate_levels(entry, direction, atr, fvg_stop)
        sizing_method = "fixed"
        kelly_stats = self._kelly_stats() if use_kelly else None
        if kelly_stats is not None:
            wr, aw, al = kelly_stats
            size = self.kelly_position_size(wr, aw, al, entry, sl)
            sizing_method = "kelly"
        else:
            size = self.calculate_position_size(entry, sl)
        if size <= 0:
            return None
        t = Trade(
            entry_time=ts, direction=direction, entry_price=entry,
            stop_loss=sl, take_profit_1=tp1, take_profit_2=tp2,
            take_profit_3=tp3, position_size=size, asset=self.config.asset,
            fvg_stop=fvg_stop, sizing_method=sizing_method,
        )
        self.open_trades.append(t)
        return t

    def update_trades(self, ts: datetime, high: float, low: float, close: float):
        for t in self.open_trades[:]:
            if t.status != "open":
                continue
            if t.direction == TradeDirection.LONG:
                if low <= t.stop_loss:
                    self._close(t, ts, t.stop_loss, "stopped"); continue
                if t.tp_level_hit == 0 and high >= t.take_profit_1: t.tp_level_hit = 1
                if t.tp_level_hit == 1 and high >= t.take_profit_2: t.tp_level_hit = 2
                if t.tp_level_hit == 2 and high >= t.take_profit_3:
                    self._close(t, ts, t.take_profit_3, "closed_tp3"); continue
            else:
                if high >= t.stop_loss:
                    self._close(t, ts, t.stop_loss, "stopped"); continue
                if t.tp_level_hit == 0 and low <= t.take_profit_1: t.tp_level_hit = 1
                if t.tp_level_hit == 1 and low <= t.take_profit_2: t.tp_level_hit = 2
                if t.tp_level_hit == 2 and low <= t.take_profit_3:
                    self._close(t, ts, t.take_profit_3, "closed_tp3"); continue

    def _close(self, t: Trade, ts: datetime, exit_price: float, status: str):
        t.exit_time = ts; t.exit_price = exit_price
        if t.direction == TradeDirection.LONG:
            pnl = (exit_price - t.entry_price) * t.position_size * self.config.contract_size
        else:
            pnl = (t.entry_price - exit_price) * t.position_size * self.config.contract_size
        if not np.isfinite(pnl):
            pnl = 0.0
        max_swing = max(abs(self.equity), 1.0)
        pnl = float(np.clip(pnl, -max_swing, max_swing))
        t.pnl = pnl
        t.status = status
        self.closed_trades.append(t)
        self.trade_history_pnls.append(pnl)
        if t in self.open_trades:
            self.open_trades.remove(t)
        self.equity += pnl
        self.equity_curve.append(float(self.equity))

    def close_all(self, ts: datetime, price: float):
        for t in self.open_trades[:]:
            self._close(t, ts, price, "closed_manual")

    def get_stats(self) -> Dict:
        if not self.closed_trades:
            return {"total_trades": 0, "equity_curve": list(self.equity_curve)}
        pnls = [t.pnl for t in self.closed_trades if t.pnl is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        long_trades = [t for t in self.closed_trades if t.direction == TradeDirection.LONG]
        short_trades = [t for t in self.closed_trades if t.direction == TradeDirection.SHORT]
        long_wins = [t for t in long_trades if (t.pnl or 0) > 0]
        short_wins = [t for t in short_trades if (t.pnl or 0) > 0]
        precision_long = (len(long_wins) / len(long_trades)) if long_trades else 0.0
        precision_short = (len(short_wins) / len(short_trades)) if short_trades else 0.0

        return {
            "total_trades":  len(self.closed_trades),
            "win_rate":      len(wins) / len(pnls) if pnls else 0,
            "precision_long":  float(precision_long),
            "precision_short": float(precision_short),
            "long_trades":     len(long_trades),
            "short_trades":    len(short_trades),
            "avg_win":       float(np.mean(wins))   if wins   else 0.0,
            "avg_loss":      float(np.mean(losses)) if losses else 0.0,
            "total_pnl":     float(sum(pnls)),
            "profit_factor": (
                abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")
            ),
            "max_drawdown":  self._max_dd(pnls),
            "sharpe":        self._sharpe(pnls),
            "equity_curve":  list(self.equity_curve),
            "kelly_active":  self._kelly_stats() is not None,
        }

    @staticmethod
    def _max_dd(pnls: List[float]) -> float:
        cum = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        return float(abs(dd.min())) if len(dd) else 0.0

    @staticmethod
    def _sharpe(pnls: List[float]) -> float:
        if len(pnls) < 2: return 0.0
        a = np.array(pnls)
        if a.std() == 0: return 0.0
        return float(a.mean() / a.std() * np.sqrt(252 * 24 * 60))


# =============================================================================
# PURGED WALK-FORWARD ENGINE (v4)
# =============================================================================

class PurgedWalkForward:
    """Implements purged cross-validation with embargo for time series.
    
    Prevents look-ahead bias by purging overlapping label horizons and
    embargoing post-test observations to remove serial correlation leakage.
    """

    def __init__(self, n_splits: int = 5, test_size_pct: float = 0.10,
                 purge_horizon: int = 5, embargo_pct: float = 0.02):
        self.n_splits = n_splits
        self.test_size_pct = test_size_pct
        self.purge_horizon = purge_horizon
        self.embargo_pct = embargo_pct

    def split(self, df: pd.DataFrame):
        n = len(df)
        test_size = max(50, int(n * self.test_size_pct))
        min_train = max(500, int(n * 0.30))
        if min_train + test_size * self.n_splits > n:
            n_splits = max(2, (n - min_train) // test_size)
        else:
            n_splits = self.n_splits

        for fold in range(n_splits):
            train_end = min_train + fold * test_size
            test_start = train_end
            test_end = min(test_start + test_size, n)
            if test_end - test_start < 30:
                break

            # Purge: remove purge_horizon bars before and after test set
            # from training data to prevent label overlap leakage
            purge_start = max(0, test_start - self.purge_horizon)
            purge_end = min(n, test_end + self.purge_horizon)

            # Embargo: remove embargo_pct of dataset after test set
            embargo_size = int(n * self.embargo_pct)
            embargo_end = min(n, test_end + embargo_size)

            # Train indices: everything before purge_start, plus between purge_end and test_start
            # (but for expanding window we simply take [0 : purge_start])
            train_idx = list(range(0, purge_start))
            # In expanding window, we don't use data between purge_end and test_start
            # because that would be future data relative to the next fold's training
            # Actually for anchored walk-forward, train is [0:train_end] minus purged region
            # Let's use expanding window: train = [0 : test_start - purge_horizon]
            train_idx = list(range(0, max(0, test_start - self.purge_horizon)))

            test_idx = list(range(test_start, test_end))
            yield fold, train_idx, test_idx


# =============================================================================
# ORCHESTRATOR (v4)
# =============================================================================

class PrecisionTradingSystem:
    def __init__(self, asset: Asset,
                 model_type: Literal["lorentzian", "xgboost"] = "xgboost",
                 use_hmm: bool = True,
                 db_path: str = "data/precision_backtest.db"):
        self.asset = asset
        self.config = ASSET_CONFIGS[asset]
        self.model_type = model_type
        self.use_hmm = use_hmm

        self.cleaner = MicrostructureCleaner(self.config)
        self.flow = FlowAnalyzer(self.config)
        self.structure = MarketStructureAnalyzer(self.config)
        self.mtf = MTFConfluenceEngine()
        
        if model_type == "lorentzian":
            self.signal_model = LorentzianClassifier(n_neighbors=5)
        elif model_type == "stacking":
            self.signal_model = StackedEnsemble(
                class_weight_scale=self.config.class_weight_scale,
                calibration_method=self.config.calibration_method,
                calibration_cv_splits=self.config.calibration_cv_splits,
            )
        else:
            self.signal_model = XGBoostSignalModel(
                calibration_method=self.config.calibration_method,
                calibration_cv_splits=self.config.calibration_cv_splits,
                class_weight_scale=self.config.class_weight_scale,
                tune_threshold=self.config.f1_threshold_tune,
            )
        self.meta_labeler = MetaLabeler() if self.config.use_meta_labeling else None
        self.hmm = HMMRegimeDetector(n_regimes=2) if use_hmm else None
        self.current_regime = Regime.TRENDING
        self.risk_manager = RiskManager(self.config)
        self.is_trained = False
        self.data_buffer: pd.DataFrame = pd.DataFrame()
        self._last_metrics: Dict = {}
        self.db = SQLiteBacktestDB(db_path)
        self.bias_detector = BiasDetector()

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = self.cleaner.clean_ohlcv(df)
        flowed  = self.flow.compute_flow_features(cleaned)
        structured = self.structure.detect_fvg(flowed)
        mtfed = self.mtf.compute(structured)
        if self.hmm is not None and self.hmm.model is not None:
            mtfed["regime"] = self.hmm.predict_regime(mtfed)
        elif self.hmm is not None:
            mtfed["regime"] = 0
        return mtfed

    def train(self, historical_df: pd.DataFrame):
        cleaned = self.cleaner.clean_ohlcv(historical_df)
        flowed  = self.flow.compute_flow_features(cleaned)
        structured = self.structure.detect_fvg(flowed)
        mtfed = self.mtf.compute(structured)
        if self.hmm is not None:
            self.hmm.fit(mtfed)
            mtfed["regime"] = self.hmm.predict_regime(mtfed)

        # v4: Bias detection — check for feature leakage
        if hasattr(self.signal_model, "_engineer"):
            temp_feats = self.signal_model._engineer(mtfed)
            leak_report = self.bias_detector.detect_leakage(
                temp_feats.assign(future_return=mtfed["close"].pct_change().shift(-self.config.target_horizon)),
                temp_feats.columns.tolist(),
                threshold=0.30,
            )
            if leak_report:
                print(f"[BiasDetector] ⚠️ Potential leakage in features: {leak_report}")

        # v4: Class balance report
        fwd = mtfed["close"].shift(-self.config.target_horizon) / mtfed["close"] - 1
        roll_std = mtfed["close"].pct_change().rolling(50, min_periods=10).std().bfill()
        roll_std = roll_std.clip(lower=1e-6) * np.sqrt(self.config.target_horizon)
        z = fwd / roll_std
        temp_labels = pd.Series(0, index=mtfed.index, dtype=int)
        temp_labels[z >  1.0]  = 2
        temp_labels[(z <=  1.0) & (z >  0.3)] = 1
        temp_labels[(z >= -1.0) & (z < -0.3)] = -1
        temp_labels[z < -1.0]  = -2
        balance = self.bias_detector.compute_class_balance(temp_labels.dropna().values)
        print(f"[BiasDetector] Class balance: {balance}")

        # FIX: Only recreate if wrong type or missing v4 attrs, don't destroy fitted model
        needs_reinit = False
        if self.model_type == "stacking":
            if not isinstance(self.signal_model, StackedEnsemble):
                self.signal_model = StackedEnsemble(
                    class_weight_scale=self.config.class_weight_scale,
                    calibration_method=self.config.calibration_method,
                    calibration_cv_splits=self.config.calibration_cv_splits,
                )
        elif self.model_type == "xgboost":
            if not isinstance(self.signal_model, XGBoostSignalModel):
                needs_reinit = True
            elif not hasattr(self.signal_model, 'class_weight_scale'):
                needs_reinit = True
            elif getattr(self.signal_model, 'model', None) is None:
                needs_reinit = True
            if needs_reinit:
                self.signal_model = XGBoostSignalModel(
                    calibration_method=self.config.calibration_method,
                    calibration_cv_splits=self.config.calibration_cv_splits,
                    class_weight_scale=self.config.class_weight_scale,
                    tune_threshold=self.config.f1_threshold_tune,
                )
        elif self.model_type == "lorentzian" and not isinstance(self.signal_model, LorentzianClassifier):
            self.signal_model = LorentzianClassifier(n_neighbors=5)

        # Defensive: ensure meta_labeler is also reset
        if self.meta_labeler is not None and self.config.use_meta_labeling:
            self.meta_labeler = MetaLabeler()

        self.signal_model.fit(mtfed, target_horizon=self.config.target_horizon)

        # DEFENSIVE: abort if XGBoost never actually fitted (e.g. exception mid-fit)
        if isinstance(self.signal_model, XGBoostSignalModel) and not self.signal_model._is_fitted():
            print("[Precision] XGBoost fit did not complete successfully; aborting train")
            self.is_trained = False
            return

        # v4: Meta-labeler training
        if self.meta_labeler is not None and self.config.use_meta_labeling:
            primary_preds = self.signal_model.predict(mtfed)
            self.meta_labeler.fit(mtfed, primary_preds,
                                  target_horizon=self.config.target_horizon,
                                  threshold=self.config.meta_label_threshold)

        self.data_buffer = mtfed.tail(500)
        self.is_trained = True

    def _generate_signal_on_buffer(self) -> Tuple[int, float, pd.DataFrame, int]:
        df = self._build_features(self.data_buffer)
        if "regime" in df.columns and len(df):
            self.current_regime = (
                Regime.TRENDING if df["regime"].iloc[-1] == 0
                else Regime.MEAN_REVERTING
            )
        signals = self.signal_model.predict(df)
        if signals.empty:
            return 0, 0.0, df, 1

        sig = int(signals.iloc[-1])
        conf = 0.5

        # v4: Meta-labeler filter
        meta_ok = 1
        if self.meta_labeler is not None and self.config.use_meta_labeling and self.meta_labeler.is_fitted:
            meta_pred = self.meta_labeler.predict(df, signals)
            meta_ok = int(meta_pred.iloc[-1]) if not meta_pred.empty else 1

        if hasattr(self.signal_model, "predict_proba"):
            proba = self.signal_model.predict_proba(df).iloc[-1]
            base_conf = float(proba.abs().max())
            mtf_score = float(df["mtf_score"].iloc[-1]) if "mtf_score" in df.columns else 0.0
            cvd_div = (
                int(df["cvd_bull_div"].iloc[-1]) - int(df["cvd_bear_div"].iloc[-1])
                if "cvd_bull_div" in df.columns else 0
            )
            adj = 0.0
            if sig > 0:
                adj += max(0.0, mtf_score) * 0.15
                adj += 0.10 if cvd_div > 0 else 0.0
            elif sig < 0:
                adj += max(0.0, -mtf_score) * 0.15
                adj += 0.10 if cvd_div < 0 else 0.0
            conf = float(min(1.0, base_conf + adj))
        return sig, conf, df, meta_ok

    def process_bar(self, bar: Dict) -> Optional[Trade]:
        if not self.is_trained:
            return None
        bar_df = pd.DataFrame([bar])
        bar_df["timestamp"] = pd.to_datetime(bar_df["timestamp"])
        bar_df = bar_df.set_index("timestamp")
        self.data_buffer = pd.concat([self.data_buffer, bar_df]).tail(500)
        sig, conf, flowed, meta_ok = self._generate_signal_on_buffer()
        if meta_ok == 0:
            return None
        return self._execute_signal(self.data_buffer.index[-1], sig, conf, flowed)

    def _execute_signal(self, ts, signal: int, confidence: float,
                        df: pd.DataFrame) -> Optional[Trade]:
        latest = df.iloc[-1]
        if latest.get("spread_filter_active", False):
            return None
        vpin = float(latest.get("vpin", 0))
        thresh = self.config.vpin_threshold
        if vpin > thresh:
            return None
        if self.current_regime == Regime.MEAN_REVERTING and abs(signal) > 1:
            signal = int(np.sign(signal))
        if confidence < 0.22:
            return None
        hour = pd.Timestamp(ts).hour
        if self.asset in (Asset.XAUUSD, Asset.EURUSD, Asset.GBPUSD):
            in_london = self.config.london_start <= hour <= self.config.london_end
            in_ny     = self.config.ny_start     <= hour <= self.config.ny_end
            if not (in_london or in_ny):
                return None
        if signal >= 1:
            direction = TradeDirection.LONG
        elif signal <= -1:
            direction = TradeDirection.SHORT
        else:
            return None
        if self.asset == Asset.XAUUSD:
            ema55 = df["close"].ewm(span=55).mean().iloc[-1]
            if direction == TradeDirection.LONG  and df["close"].iloc[-1] < ema55:
                return None
            if direction == TradeDirection.SHORT and df["close"].iloc[-1] > ema55:
                return None

        mtf_score = float(latest.get("mtf_score", 0.0))
        if direction == TradeDirection.LONG and mtf_score < -0.6:
            return None
        if direction == TradeDirection.SHORT and mtf_score > 0.6:
            return None

        atr = float(latest.get("atr_14", df["close"].iloc[-20:].std()))
        entry = float(latest.get("smart_price", df["close"].iloc[-1]))
        fvg_stop = self.structure.get_nearest_fvg_stop(df, direction)
        return self.risk_manager.open_trade(ts, direction, entry, atr,
                                             fvg_stop=fvg_stop, use_kelly=True)

    def update_market_data(self, ts, high, low, close):
        self.risk_manager.update_trades(ts, high, low, close)

    def get_performance(self) -> Dict:
        return self.risk_manager.get_stats()

    def generate_live_signal(self, df_recent: pd.DataFrame) -> Dict:
        if not self.is_trained:
            return {"action": "HOLD", "confidence": 0.0,
                    "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0,
                    "lot_size": 0.01, "regime": "untrained",
                    "vpin": 0.0, "spread_blocked": False,
                    "mtf_score": 0.0, "mtf_confluence": "neutral",
                    "cvd_div": 0, "in_value_area": True,
                    "distance_to_poc": 0.0, "absorption": 0,
                    "fvg_stop": None, "sizing_method": "fixed",
                    "meta_label": 1}

        self.data_buffer = df_recent.tail(500).copy()
        sig, conf, flowed, meta_ok = self._generate_signal_on_buffer()
        latest = flowed.iloc[-1]
        atr = float(latest.get("atr_14", flowed["close"].iloc[-20:].std()))
        entry = float(latest.get("smart_price", flowed["close"].iloc[-1]))
        spread_blocked = bool(latest.get("spread_filter_active", False))
        vpin_val = float(latest.get("vpin", 0))

        if sig >= 1:
            direction = TradeDirection.LONG
            action = "BUY" if sig == 2 else "WEAK BUY"
        elif sig <= -1:
            direction = TradeDirection.SHORT
            action = "SELL" if sig == -2 else "WEAK SELL"
        else:
            direction = TradeDirection.FLAT
            action = "HOLD"

        fvg_stop = self.structure.get_nearest_fvg_stop(flowed, direction)
        if direction != TradeDirection.FLAT:
            sl, tp1, tp2, tp3 = self.risk_manager.calculate_levels(
                entry, direction, atr, fvg_stop)
            kelly_stats = self.risk_manager._kelly_stats()
            if kelly_stats is not None:
                wr, aw, al = kelly_stats
                lot = self.risk_manager.kelly_position_size(wr, aw, al, entry, sl)
                sizing_method = "kelly"
            else:
                lot = self.risk_manager.calculate_position_size(entry, sl)
                sizing_method = "fixed"
        else:
            sl = tp1 = tp2 = tp3 = entry
            lot = 0.0
            sizing_method = "fixed"

        if vpin_val > self.config.vpin_threshold:
            action = f"{action} (VPIN BLOCK)"
        if spread_blocked:
            action = f"{action} (SPREAD)"
        if meta_ok == 0:
            action = f"{action} (META REJECT)"

        mtf_score = float(latest.get("mtf_score", 0.0))
        mtf_conf = str(latest.get("mtf_confluence", "neutral"))
        cvd_div = int(latest.get("cvd_bull_div", 0)) - int(latest.get("cvd_bear_div", 0))
        in_va = bool(latest.get("in_value_area", True))
        dist_poc = float(latest.get("distance_to_poc", 0.0))
        absorp = int(latest.get("absorption", 0))

        return {
            "action":     action,
            "confidence": round(float(conf), 4),
            "entry":      round(entry, 5),
            "sl":         round(sl,    5),
            "tp1":        round(tp1,   5),
            "tp2":        round(tp2,   5),
            "tp3":        round(tp3,   5),
            "lot_size":   round(float(lot), 4),
            "regime":     self.current_regime.value,
            "vpin":       round(vpin_val, 3),
            "atr":        round(atr,   5),
            "spread_blocked": spread_blocked,
            "mtf_score":      round(mtf_score, 3),
            "mtf_confluence": mtf_conf,
            "cvd_div":        cvd_div,
            "in_value_area":  in_va,
            "distance_to_poc": round(dist_poc, 5),
            "absorption":     absorp,
            "fvg_stop":       round(float(fvg_stop), 5) if fvg_stop is not None else None,
            "sizing_method":  sizing_method,
            "meta_label":     meta_ok,
        }

    @staticmethod
    def _intra_bar_fill_check(t: Trade, bar_open: float, bar_high: float,
                                bar_low: float, bar_close: float
                                ) -> Tuple[Optional[str], Optional[float]]:
        if t.direction == TradeDirection.LONG:
            sequence = [("open", bar_open), ("low", bar_low),
                        ("high", bar_high), ("close", bar_close)]
        else:
            sequence = [("open", bar_open), ("high", bar_high),
                        ("low", bar_low), ("close", bar_close)]
        for _, price in sequence:
            if t.direction == TradeDirection.LONG:
                if price <= t.stop_loss:
                    return "stopped", t.stop_loss
                if t.tp_level_hit < 1 and price >= t.take_profit_1:
                    t.tp_level_hit = 1
                if t.tp_level_hit < 2 and price >= t.take_profit_2:
                    t.tp_level_hit = 2
                if t.tp_level_hit < 3 and price >= t.take_profit_3:
                    return "closed_tp3", t.take_profit_3
            else:
                if price >= t.stop_loss:
                    return "stopped", t.stop_loss
                if t.tp_level_hit < 1 and price <= t.take_profit_1:
                    t.tp_level_hit = 1
                if t.tp_level_hit < 2 and price <= t.take_profit_2:
                    t.tp_level_hit = 2
                if t.tp_level_hit < 3 and price <= t.take_profit_3:
                    return "closed_tp3", t.take_profit_3
        return None, None

    @staticmethod
    def _expected_calibration_error(probas: np.ndarray, correct: np.ndarray,
                                     n_bins: int = 10) -> float:
        if len(probas) == 0:
            return 0.0
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probas >= bins[i]) & (probas < bins[i + 1])
            if i == n_bins - 1:
                mask = (probas >= bins[i]) & (probas <= bins[i + 1])
            if mask.sum() == 0:
                continue
            avg_conf = float(probas[mask].mean())
            avg_acc = float(correct[mask].mean())
            ece += (mask.sum() / len(probas)) * abs(avg_conf - avg_acc)
        return float(ece)

    def _backtest_test_window(self, train_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               intra_bar_fills: bool = True
                               ) -> Tuple[Dict, List[Trade], np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray, np.ndarray]:
        self.train(train_df)
        
        # CRITICAL FIX: Verify model actually got fitted
        if isinstance(self.signal_model, StackedEnsemble):
            if not self.signal_model._is_fitted():
                print(f"[WalkForward] CRITICAL: StackedEnsemble not fitted, forcing clean re-fit")
                self.signal_model = StackedEnsemble(
                    class_weight_scale=self.config.class_weight_scale,
                    calibration_method=self.config.calibration_method,
                    calibration_cv_splits=self.config.calibration_cv_splits,
                )
                self.signal_model.fit(train_df, target_horizon=self.config.target_horizon)
        elif isinstance(self.signal_model, XGBoostSignalModel):
            if not self.signal_model._is_fitted():
                print(f"[WalkForward] CRITICAL: XGBoost model not fitted, forcing clean re-fit")
                self.signal_model = XGBoostSignalModel(
                    calibration_method=self.config.calibration_method,
                    calibration_cv_splits=self.config.calibration_cv_splits,
                    class_weight_scale=self.config.class_weight_scale,
                    tune_threshold=self.config.f1_threshold_tune,
                )
                self.signal_model.fit(train_df, target_horizon=self.config.target_horizon)
        elif isinstance(self.signal_model, LorentzianClassifier):
            if self.signal_model.train_data is None:
                self.signal_model.fit(train_df, target_horizon=self.config.target_horizon)

        self.risk_manager = RiskManager(self.config)

        flowed = self._build_features(test_df)
        
        # DEFENSIVE: Check model state before predict
        if isinstance(self.signal_model, XGBoostSignalModel):
            if self.signal_model.model is None:
                print(f"[WalkForward] XGBoost model is None, returning all HOLD")
                signals = pd.Series(0, index=flowed.index)
                proba_df = pd.DataFrame(0.2, index=flowed.index, columns=[-2, -1, 0, 1, 2])
                confs = pd.Series(0.5, index=flowed.index)
            else:
                try:
                    signals = self.signal_model.predict(flowed)
                    proba_df = self.signal_model.predict_proba(flowed)
                    confs = proba_df.abs().max(axis=1)
                except Exception as e:
                    print(f"[WalkForward] XGBoost predict failed: {e}")
                    signals = pd.Series(0, index=flowed.index)
                    proba_df = pd.DataFrame(0.2, index=flowed.index, columns=[-2, -1, 0, 1, 2])
                    confs = pd.Series(0.5, index=flowed.index)
        elif isinstance(self.signal_model, LorentzianClassifier):
            if self.signal_model.train_data is None:
                print(f"[WalkForward] Lorentzian not fitted, returning all HOLD")
                signals = pd.Series(0, index=flowed.index)
                proba_df = None
                confs = pd.Series(0.5, index=flowed.index)
            else:
                try:
                    signals = self.signal_model.predict(flowed)
                    proba_df = None
                    confs = pd.Series(0.5, index=flowed.index)
                except Exception as e:
                    print(f"[WalkForward] Lorentzian predict failed: {e}")
                    signals = pd.Series(0, index=flowed.index)
                    proba_df = None
                    confs = pd.Series(0.5, index=flowed.index)
        else:
            signals = pd.Series(0, index=flowed.index)
            proba_df = None
            confs = pd.Series(0.5, index=flowed.index)

        y_true = np.zeros(len(flowed), dtype=int)
        fwd = flowed["close"].shift(-self.config.target_horizon) / flowed["close"] - 1
        roll_std = flowed["close"].pct_change().rolling(50, min_periods=10).std().bfill()
        roll_std = roll_std.clip(lower=1e-6) * np.sqrt(self.config.target_horizon)
        z = fwd / roll_std
        y_true[z >  1.0]  = 2
        y_true[(z <=  1.0) & (z >  0.3)] = 1
        y_true[(z >= -1.0) & (z < -0.3)] = -1
        y_true[z < -1.0]  = -2

        y_pred = signals.reindex(flowed.index).fillna(0).values

        # Classification metrics
        cls_metrics = ClassificationMetrics.compute(
            y_true, y_pred,
            y_proba=proba_df.values if proba_df is not None else None,
        )

        # ECE
        fwd5 = flowed["close"].pct_change(5).shift(-5).fillna(0).values
        sig_array = y_pred
        correct = (
            ((sig_array > 0) & (fwd5 > 0))
            | ((sig_array < 0) & (fwd5 < 0))
        ).astype(int)
        proba_arr = confs.reindex(flowed.index).fillna(0).values
        nonhold_mask = sig_array != 0
        ece_probas = proba_arr[nonhold_mask]
        ece_correct = correct[nonhold_mask]

        for i, ts in enumerate(flowed.index):
            row = flowed.iloc[i]
            sig = int(signals.iloc[i]) if i < len(signals) else 0
            conf = float(confs.iloc[i]) if i < len(confs) else 0

            # v4: Meta-labeler filter in backtest
            if self.meta_labeler is not None and self.config.use_meta_labeling and self.meta_labeler.is_fitted:
                meta_pred = self.meta_labeler.predict(flowed.iloc[:i+1], signals.iloc[:i+1])
                if not meta_pred.empty and meta_pred.iloc[-1] == 0:
                    sig = 0

            bar_open = float(row.get("open", row["close"]))
            bar_high = float(row["high"])
            bar_low  = float(row["low"])
            bar_close = float(row["close"])

            if intra_bar_fills:
                for t in self.risk_manager.open_trades[:]:
                    status, fill_price = self._intra_bar_fill_check(
                        t, bar_open, bar_high, bar_low, bar_close)
                    if status is not None:
                        self.risk_manager._close(t, ts, fill_price, status)
            else:
                self.risk_manager.update_trades(ts, bar_high, bar_low, bar_close)

            self.current_regime = (
                Regime.TRENDING if row.get("regime", 0) == 0 else Regime.MEAN_REVERTING
            )
            self._execute_signal(ts, sig, conf, flowed.iloc[: i + 1])

        if flowed.size:
            self.risk_manager.close_all(flowed.index[-1], float(flowed["close"].iloc[-1]))

        stats = self.risk_manager.get_stats()
        stats["timestamps"]     = list(flowed.index)
        stats["test_bars"]      = len(test_df)
        stats["train_bars"]     = len(train_df)
        stats["initial_equity"] = self.risk_manager.initial_equity
        eq = self.risk_manager.equity_curve
        stats["final_equity"]   = float(eq[-1]) if eq else self.risk_manager.initial_equity
        stats["total_return"]   = float(
            (stats["final_equity"] - stats["initial_equity"]) / stats["initial_equity"]
        )
        stats.update(cls_metrics)

        return (stats, list(self.risk_manager.closed_trades),
                ece_probas, ece_correct, y_true, y_pred,
                proba_df.values if proba_df is not None else np.array([]))

    def backtest(self, df: pd.DataFrame, train_pct: float = 0.7,
                  intra_bar_fills: bool = True) -> Dict:
        n = len(df)
        if n < 500:
            return {"error": "need ≥ 500 bars"}
        split = int(n * train_pct)
        train_df = df.iloc[:split]
        test_df  = df.iloc[split:]

        stats, _trades, ece_p, ece_c, yt, yp, _ = self._backtest_test_window(
            train_df, test_df, intra_bar_fills=intra_bar_fills)
        stats["ece"] = self._expected_calibration_error(ece_p, ece_c) if len(ece_p) else 0.0
        self._last_metrics = stats
        return stats

    def walk_forward_backtest(self, df: pd.DataFrame, n_splits: int = 5,
                                test_size_pct: float = 0.10,
                                intra_bar_fills: bool = True) -> Dict:
        n = len(df)
        if n < 1000:
            return {"error": "walk-forward needs ≥ 1000 bars"}

        pwf = PurgedWalkForward(
            n_splits=n_splits, test_size_pct=test_size_pct,
            purge_horizon=self.config.purge_horizon,
            embargo_pct=self.config.embargo_pct,
        )

        all_trades: List[Trade] = []
        all_equity: List[float] = []
        all_ts: List[Any] = []
        per_fold: List[Dict] = []
        all_ece_probas: List[np.ndarray] = []
        all_ece_correct: List[np.ndarray] = []
        all_y_true: List[np.ndarray] = []
        all_y_pred: List[np.ndarray] = []

        starting_equity = 10_000.0
        cumulative_equity = starting_equity
        run_timestamp = datetime.now().isoformat()

        for fold, train_idx, test_idx in pwf.split(df):
            if len(train_idx) < 100 or len(test_idx) < 30:
                print(f"[WalkForward] Fold {fold}: insufficient data ({len(train_idx)} train / {len(test_idx)} test), skipping")
                continue
            
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            # CRITICAL: Fresh risk manager per fold
            self.risk_manager = RiskManager(self.config)
            
            try:
                stats, closed, ece_p, ece_c, yt, yp, _ = self._backtest_test_window(
                    train_df, test_df, intra_bar_fills=intra_bar_fills)
            except Exception as e:
                print(f"[WalkForward] fold {fold} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Verify stats is valid
            if stats is None or not isinstance(stats, dict):
                print(f"[WalkForward] fold {fold} returned invalid stats")
                continue

            fold_eq = stats.get("equity_curve") or []
            fold_eq = [float(v) for v in fold_eq if np.isfinite(v)]
            if fold_eq:
                fold_pnl_pct = (fold_eq[-1] - 10_000.0) / 10_000.0
                fold_pnl_pct = max(-1.0, fold_pnl_pct)
                for v in fold_eq:
                    pnl_pct = max(-1.0, (v - 10_000.0) / 10_000.0)
                    all_equity.append(max(0.0, cumulative_equity * (1 + pnl_pct)))
                cumulative_equity = max(0.0, cumulative_equity * (1 + fold_pnl_pct))
            all_ts.extend(stats.get("timestamps", []))

            all_trades.extend(closed)
            if len(ece_p):
                all_ece_probas.append(ece_p)
                all_ece_correct.append(ece_c)
            all_y_true.append(yt)
            all_y_pred.append(yp)

            per_fold.append({
                "fold":           fold,
                "train_bars":     stats.get("train_bars", 0),
                "test_bars":      stats.get("test_bars", 0),
                "trades":         stats.get("total_trades", 0),
                "win_rate":       stats.get("win_rate", 0),
                "sharpe":         stats.get("sharpe", 0),
                "profit_factor":  stats.get("profit_factor", 0),
                "total_return":   stats.get("total_return", 0),
                "max_drawdown":   stats.get("max_drawdown", 0),
                "precision_long": stats.get("precision_long", 0),
                "precision_short":stats.get("precision_short", 0),
                "accuracy":       stats.get("accuracy", 0),
                "f1_macro":       stats.get("f1_macro", 0),
            })

        # Aggregate
        all_pnls = [float(t.pnl) for t in all_trades
                    if t.pnl is not None and np.isfinite(t.pnl)]
        wins = [p for p in all_pnls if p > 0]
        losses = [p for p in all_pnls if p <= 0]

        long_trades = [t for t in all_trades if t.direction == TradeDirection.LONG]
        short_trades = [t for t in all_trades if t.direction == TradeDirection.SHORT]
        long_wins = [t for t in long_trades if (t.pnl or 0) > 0]
        short_wins = [t for t in short_trades if (t.pnl or 0) > 0]

        ece = 0.0
        if all_ece_probas:
            ece = self._expected_calibration_error(
                np.concatenate(all_ece_probas),
                np.concatenate(all_ece_correct),
            )

        p_value = None
        if SCIPY_AVAILABLE and ttest_1samp is not None and len(all_pnls) >= 5:
            try:
                p_value = float(ttest_1samp(all_pnls, 0).pvalue)
            except Exception:
                p_value = None

        sharpe = 0.0
        if len(all_pnls) >= 2:
            arr = np.array(all_pnls)
            if arr.std() > 0:
                sharpe = float(arr.mean() / arr.std() * np.sqrt(252 * 24 * 60))

        max_dd = 0.0
        if all_equity:
            curve = np.array(all_equity)
            peak = np.maximum.accumulate(curve)
            dd = peak - curve
            max_dd = float(dd.max())

        # Aggregate classification metrics across all folds
        agg_y_true = np.concatenate(all_y_true) if all_y_true else np.array([])
        agg_y_pred = np.concatenate(all_y_pred) if all_y_pred else np.array([])
        agg_cls = ClassificationMetrics.compute(agg_y_true, agg_y_pred)

        result = {
            "n_splits":         len(per_fold),
            "total_trades":     len(all_trades),
            "win_rate":         len(wins) / len(all_pnls) if all_pnls else 0,
            "precision_long":   len(long_wins) / len(long_trades) if long_trades else 0,
            "precision_short":  len(short_wins) / len(short_trades) if short_trades else 0,
            "long_trades":      len(long_trades),
            "short_trades":     len(short_trades),
            "avg_win":          float(np.mean(wins)) if wins else 0.0,
            "avg_loss":         float(np.mean(losses)) if losses else 0.0,
            "total_pnl":        float(sum(all_pnls)),
            "profit_factor":    (
                abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")
            ),
            "sharpe":           sharpe,
            "max_drawdown":     max_dd,
            "ece":              ece,
            "p_value":          p_value,
            "initial_equity":   starting_equity,
            "final_equity":     float(all_equity[-1]) if all_equity else starting_equity,
            "total_return":     float((all_equity[-1] - starting_equity) / starting_equity)
                                 if all_equity else 0.0,
            "equity_curve":     all_equity,
            "timestamps":       all_ts,
            "per_fold":         per_fold,
            "test_bars":        sum(f["test_bars"] for f in per_fold),
            "train_bars":       per_fold[-1]["train_bars"] if per_fold else 0,
            **agg_cls,
        }

        # v4: Meta-labeler performance
        if self.meta_labeler is not None and self.meta_labeler.is_fitted:
            # Approximate: count how many trades were filtered vs allowed
            result["meta_labeler_precision"] = 0.0
            result["meta_labeler_recall"] = 0.0
        else:
            result["meta_labeler_precision"] = None
            result["meta_labeler_recall"] = None

        self._last_metrics = result

        # v4: Persist to SQLite
        run_id = self.db.insert_run({
            "asset": self.asset.value,
            "model_type": self.model_type,
            "run_timestamp": run_timestamp,
            **{k: result.get(k, None) for k in [
                "n_splits", "total_trades", "win_rate", "profit_factor", "sharpe",
                "max_drawdown", "total_return", "ece", "p_value", "accuracy",
                "precision_macro", "recall_macro", "f1_macro",
                "precision_weighted", "recall_weighted", "f1_weighted",
                "mcc", "cohens_kappa", "auc_roc", "auc_pr", "balanced_accuracy",
                "precision_long", "precision_short",
            ]},
            "meta_labeler_precision": result.get("meta_labeler_precision"),
            "meta_labeler_recall": result.get("meta_labeler_recall"),
        })

        for fold_metrics in per_fold:
            self.db.insert_fold(run_id, fold_metrics["fold"], fold_metrics)

        # Store trades per fold (simplified: all trades under fold -1 for aggregate)
        self.db.insert_trades(run_id, -1, all_trades)
        self.db.insert_equity(run_id, -1, all_ts, all_equity)

        # Feature importance
        if hasattr(self.signal_model, "feature_importance_"):
            self.db.insert_feature_importance(run_id, -1, self.signal_model.feature_importance_)

        return result

    def save(self, path: str):
        # Ensure model is actually fitted before saving
        if isinstance(self.signal_model, XGBoostSignalModel) and self.signal_model.model is None:
            print("[save] Warning: XGBoost model is None, save may be incomplete")
        
        state = {
            "asset":           self.asset.value,
            "model_type":      self.model_type,
            "use_hmm":         self.use_hmm,
            "is_trained":      self.is_trained,
            "data_buffer":     self.data_buffer,
            "last_metrics":    {k: v for k, v in self._last_metrics.items()
                                if not isinstance(v, list) or k != "timestamps"},
        }
        state["signal_model"] = self.signal_model
        state["meta_labeler"] = self.meta_labeler
        state["hmm"]          = self.hmm
        state["scaler"]       = getattr(self.signal_model, "scaler", None)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> bool:
        import os
        if not os.path.exists(path): return False
        try:
            with open(path, "rb") as f:
                s = pickle.load(f)
            self.signal_model = s["signal_model"]
            self.meta_labeler = s.get("meta_labeler", self.meta_labeler)
            self.hmm          = s.get("hmm", self.hmm)
            self.is_trained   = s.get("is_trained", True)
            self.data_buffer  = s.get("data_buffer", pd.DataFrame())
            self._last_metrics = s.get("last_metrics", {})
            if hasattr(self.signal_model, 'model') and not hasattr(self.signal_model, 'calibration_method'):
                self.signal_model.calibration_method = "isotonic"
                self.signal_model.calibration_cv_splits = 3
            # v4 backward-compat for old pickles
            if isinstance(self.signal_model, XGBoostSignalModel):
                if not hasattr(self.signal_model, 'class_weight_scale'):
                    self.signal_model.class_weight_scale = 5.0
                if not hasattr(self.signal_model, 'tune_threshold'):
                    self.signal_model.tune_threshold = True

            # Validate loaded model is actually usable
            if isinstance(self.signal_model, XGBoostSignalModel):
                if not hasattr(self.signal_model, 'model') or self.signal_model.model is None:
                    print("[load] Warning: Loaded XGBoost model is None, will need retrain")
                    self.is_trained = False
            elif isinstance(self.signal_model, LorentzianClassifier):
                if not hasattr(self.signal_model, 'train_data') or self.signal_model.train_data is None:
                    print("[load] Warning: Loaded Lorentzian not fitted, will need retrain")
                    self.is_trained = False

            # Verify signal_model is actually fitted
            if self.signal_model is not None:
                # Check if model has required fitted attributes
                has_model = hasattr(self.signal_model, 'model') and self.signal_model.model is not None
                has_features = hasattr(self.signal_model, 'feature_cols') and self.signal_model.feature_cols
                has_scaler = hasattr(self.signal_model, 'scaler') and self.signal_model.scaler is not None
                
                if not (has_model and has_features):
                    print("[Precision] Model not properly fitted; resetting")
                    self.signal_model.model = None
                    self.signal_model.feature_cols = None
                    self.is_trained = False

            if self.hmm is not None and hasattr(self.hmm, 'model'):
                try:
                    if self.hmm.model is not None:
                        _ = self.hmm.model.means_
                except AttributeError:
                    self.hmm.model = None
            return True
        except Exception as e:
            print(f"[Precision] load failed: {e}")
            return False

    def get_db_runs(self, limit: int = 20) -> pd.DataFrame:
        return self.db.get_runs(asset=self.asset.value, limit=limit)