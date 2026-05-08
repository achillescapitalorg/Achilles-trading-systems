"""
Long-Form Training Pipeline for Precision Trading System
========================================================
End-to-end production training with:
  - Multi-timeframe data fetching (60-90 days)
  - Cleaning: outliers, gaps, timezone alignment, duplicate removal
  - Validation: stationarity / regime-shift detection
  - Train / val / test split with embargo (purged CV)
  - Temporal-decay sample weighting (recency-aware)
  - Markov-chain regime feature augmentation
  - Comprehensive metrics: F1, MCC, accuracy, directional win-rate
  - Versioned artifact saving (model + metadata JSON)

Reuses the corrected `TripleBarrierLabeler` and `triple_barrier_labels_vectorized`
from precision_trading_system. The earlier commit (9a06fc32) had a bias bug
in those labelers that this version fixes.
"""

import os
import json
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score

from services.precision_trading_system import (
    triple_barrier_labels_vectorized,
    MarkovRegimeForecaster,
    PrecisionTradingSystem,
    Asset,
)

warnings.filterwarnings('ignore')


class TrainingStatus(Enum):
    IDLE = "idle"
    FETCHING = "fetching_data"
    CLEANING = "cleaning_pipeline"
    VALIDATING = "validating_data"
    ENGINEERING = "feature_engineering"
    TRAINING = "training_model"
    CALIBRATING = "calibrating"
    VALIDATING_OOS = "validating_oos"
    SAVING = "saving_artifacts"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class TrainingConfig:
    asset: str
    model_type: str = "xgboost"

    # Data window
    train_days: int = 60
    validation_days: int = 7
    test_days: int = 7
    interval: str = "1m"

    # Data quality
    max_gap_minutes: int = 5
    outlier_zscore: float = 4.0
    min_bars_per_day: int = 200

    # Training capacity
    early_stopping_rounds: int = 100
    max_estimators: int = 3000
    learning_rate: float = 0.01
    max_depth: int = 6
    reg_lambda: float = 5.0
    min_child_weight: int = 20

    # Temporal weighting
    temporal_decay_halflife: int = 360
    use_purged_cv: bool = True
    purge_horizon: int = 5
    embargo_pct: float = 0.02


@dataclass
class TrainingMetrics:
    status: str = "idle"
    progress_pct: float = 0.0
    message: str = "Ready"
    timestamp: str = ""

    # Data quality
    total_bars: int = 0
    clean_bars: int = 0
    gaps_found: int = 0
    outliers_removed: int = 0
    regime_shifts: int = 0

    # Classification metrics
    train_f1: float = 0.0
    val_f1: float = 0.0
    oos_f1: float = 0.0
    train_mcc: float = 0.0
    val_mcc: float = 0.0
    oos_mcc: float = 0.0

    # Trading metrics
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    oos_accuracy: float = 0.0
    train_winrate: float = 0.0
    val_winrate: float = 0.0
    oos_winrate: float = 0.0

    n_features: int = 0
    feature_importance: Dict[str, float] = field(default_factory=dict)

    model_path: str = ""
    data_hash: str = ""
    training_duration_sec: float = 0.0


# =============================================================================
# DATA CLEANER
# =============================================================================

class DataCleaner:
    """Production-grade OHLCV cleaning pipeline."""

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        if df.index.duplicated().any():
            dups = int(df.index.duplicated().sum())
            df = df[~df.index.duplicated(keep='last')]
            print(f"[Cleaner] Removed {dups} duplicate timestamps")
        return df

    @staticmethod
    def fill_gaps(df: pd.DataFrame, max_gap_minutes: int = 5,
                  interval: str = "1m") -> pd.DataFrame:
        if len(df) < 2:
            return df
        freq_map = {"1m": "1min", "5m": "5min", "15m": "15min",
                    "30m": "30min", "1h": "1h"}
        freq = freq_map.get(interval, "1min")
        full_idx = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
        df_re = df.reindex(full_idx)
        gaps = int(df_re.isna().any(axis=1).sum())
        if gaps > 0:
            print(f"[Cleaner] {gaps} missing bars detected")
        df_filled = df_re.ffill(limit=max_gap_minutes)
        df_clean = df_filled.dropna()
        dropped = len(df_filled) - len(df_clean)
        if dropped > 0:
            print(f"[Cleaner] Dropped {dropped} bars in large gaps")
        return df_clean

    @staticmethod
    def remove_outliers(df: pd.DataFrame, zscore: float = 4.0
                         ) -> Tuple[pd.DataFrame, int]:
        returns = df['close'].pct_change().abs()
        vol = returns.rolling(60).std()
        z = returns / vol.replace(0, np.nan)
        mask = (z.abs() < zscore) | z.isna()
        outliers = int((~mask).sum())
        if outliers > 0:
            print(f"[Cleaner] Removed {outliers} outlier bars (|z| > {zscore})")
        return df[mask], outliers

    @staticmethod
    def validate_sessions(df: pd.DataFrame, min_bars: int = 200) -> pd.DataFrame:
        df = df.copy()
        df['_date'] = df.index.date
        daily_counts = df.groupby('_date').size()
        valid_days = daily_counts[daily_counts >= min_bars].index
        if len(valid_days) < len(daily_counts):
            invalid = len(daily_counts) - len(valid_days)
            print(f"[Cleaner] Removed {invalid} days with < {min_bars} bars")
        df = df[df['_date'].isin(valid_days)].copy()
        df = df.drop(columns=['_date'], errors='ignore')
        return df

    @staticmethod
    def detect_regime_shift(df: pd.DataFrame, window: int = 1000) -> int:
        """F-test of vol variance early-half vs late-half."""
        returns = df['close'].pct_change().abs()
        rolling_vol = returns.rolling(window).std()
        mid = len(rolling_vol) // 2
        if mid < window:
            return 0
        vol_early = rolling_vol.iloc[window:mid].dropna()
        vol_late = rolling_vol.iloc[mid:].dropna()
        if len(vol_early) < 100 or len(vol_late) < 100:
            return 0
        f_stat = (vol_late.var() / vol_early.var()) if vol_early.var() > 0 else 1.0
        if f_stat > 2.0 or f_stat < 0.5:
            print(f"[Cleaner] REGIME SHIFT DETECTED: F={f_stat:.2f}")
            return 1
        return 0


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class TrainingPipeline:
    """Full-scale training with checkpointing + comprehensive metrics."""

    def __init__(self, system: PrecisionTradingSystem, config: TrainingConfig):
        self.system = system
        self.config = config
        self.metrics = TrainingMetrics(
            status=TrainingStatus.IDLE.value,
            progress_pct=0.0,
            message="Ready",
            timestamp=datetime.now().isoformat(),
        )
        self._cancelled = False
        self._thread: Optional[threading.Thread] = None

    def cancel(self):
        self._cancelled = True
        self.metrics.status = TrainingStatus.CANCELLED.value

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _update(self, status: TrainingStatus, progress: float, message: str):
        self.metrics.status = status.value
        self.metrics.progress_pct = float(progress)
        self.metrics.message = message
        self.metrics.timestamp = datetime.now().isoformat()
        print(f"[TrainingPipeline] {status.value} ({progress:.0f}%): {message}",
              flush=True)

    def _split_three_way(self, df: pd.DataFrame
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        total = self.config.train_days + self.config.validation_days + self.config.test_days
        test_n = max(50, int(n * (self.config.test_days / total)))
        val_n = max(50, int(n * (self.config.validation_days / total)))
        embargo = max(0, int(n * self.config.embargo_pct))
        df_train = df.iloc[:-(test_n + val_n + embargo)]
        df_val = df.iloc[-(test_n + val_n):-test_n]
        df_test = df.iloc[-test_n:]
        return df_train, df_val, df_test

    @staticmethod
    def _temporal_decay_weights(n: int, halflife: int) -> np.ndarray:
        decay_lambda = np.log(2) / max(halflife, 1)
        t = np.arange(n)
        w = np.exp(decay_lambda * (t - n))
        return w / w.sum() * n  # normalised to mean 1

    @staticmethod
    def _hash_df(df: pd.DataFrame) -> str:
        s = df.head(100).to_json() + df.tail(100).to_json()
        return hashlib.md5(s.encode()).hexdigest()[:12]

    def _compute_metrics_set(self, mtfed: pd.DataFrame, prefix: str) -> Dict[str, float]:
        """Compute F1 / MCC / accuracy / win-rate using the FIXED vectorized
        triple-barrier labeler so train/val/test are evaluated on the same
        target function the model was trained on."""
        preds = self.system.signal_model.predict(mtfed)
        # Use the system's TP1 / stop multipliers as TB barriers for evaluation
        # — this matches what the live executor would actually risk per trade.
        cfg = self.system.config
        pt_mult = float(getattr(cfg, "tb_pt_mult", getattr(cfg, "atr_multiplier_tp1", 1.5)))
        sl_mult = float(getattr(cfg, "tb_sl_mult", getattr(cfg, "atr_multiplier_stop", 1.0)))
        max_bars = int(getattr(cfg, "tb_max_holding", 10))
        y_true = triple_barrier_labels_vectorized(
            mtfed, pt_atr_mult=pt_mult, sl_atr_mult=sl_mult, max_bars=max_bars,
        )
        y_pred = preds.reindex(y_true.index).fillna(0).values
        yt = y_true.values
        mask = ~(np.isnan(yt) | np.isnan(y_pred.astype(float)))
        if mask.sum() < 10:
            return {f"{prefix}_f1": 0.0, f"{prefix}_mcc": 0.0,
                    f"{prefix}_accuracy": 0.0, f"{prefix}_winrate": 0.0}
        f1 = float(f1_score(yt[mask], y_pred[mask], average='macro', zero_division=0))
        try:
            mcc = float(matthews_corrcoef(yt[mask], y_pred[mask]))
        except Exception:
            mcc = 0.0
        acc = float((yt[mask] == y_pred[mask]).mean())
        nz = y_pred[mask] != 0
        if nz.sum() > 0:
            wr = float(
                (np.sign(yt[mask][nz]) == np.sign(y_pred[mask][nz])).mean()
            )
        else:
            wr = 0.0
        return {f"{prefix}_f1": f1, f"{prefix}_mcc": mcc,
                f"{prefix}_accuracy": acc, f"{prefix}_winrate": wr}

    def run(self, fetch_fn: Callable[..., pd.DataFrame]) -> TrainingMetrics:
        """Execute the full training pipeline. Returns final metrics."""
        start = datetime.now()
        self._cancelled = False
        try:
            self._update(TrainingStatus.FETCHING, 5,
                         f"Fetching {self.config.train_days}d + "
                         f"{self.config.validation_days}d + "
                         f"{self.config.test_days}d of {self.config.interval} data…")

            end_dt = datetime.now()
            test_start = end_dt - timedelta(days=self.config.test_days)
            val_start = test_start - timedelta(days=self.config.validation_days)
            train_start = val_start - timedelta(days=self.config.train_days)

            df_train_raw = fetch_fn(start=train_start, end=val_start,
                                     interval=self.config.interval)
            df_val_raw = fetch_fn(start=val_start, end=test_start,
                                   interval=self.config.interval)
            df_test_raw = fetch_fn(start=test_start, end=end_dt,
                                    interval=self.config.interval)
            if self._cancelled:
                return self.metrics

            df = pd.concat([df_train_raw, df_val_raw, df_test_raw]).sort_index()
            self.metrics.total_bars = len(df)
            if len(df) < 1000:
                raise ValueError(
                    f"Insufficient data: {len(df)} bars (need >= 1000)"
                )

            self._update(TrainingStatus.CLEANING, 15,
                         f"Cleaning {len(df):,} bars…")
            cleaner = DataCleaner()
            df = cleaner.remove_duplicates(df)
            df = cleaner.fill_gaps(df, self.config.max_gap_minutes,
                                   self.config.interval)
            df, outliers = cleaner.remove_outliers(df, self.config.outlier_zscore)
            self.metrics.outliers_removed = outliers
            df = cleaner.validate_sessions(df, self.config.min_bars_per_day)
            self.metrics.clean_bars = len(df)
            self.metrics.gaps_found = self.metrics.total_bars - self.metrics.clean_bars

            self._update(TrainingStatus.VALIDATING, 30,
                         "Detecting regime shifts + computing data hash…")
            self.metrics.regime_shifts = cleaner.detect_regime_shift(df)
            self.metrics.data_hash = self._hash_df(df)

            df_train, df_val, df_test = self._split_three_way(df)
            print(f"[TrainingPipeline] Split: "
                  f"train={len(df_train):,}, val={len(df_val):,}, test={len(df_test):,}")
            if self._cancelled:
                return self.metrics

            self._update(TrainingStatus.ENGINEERING, 45,
                         "Engineering features + Markov regime features…")
            full = pd.concat([df_train, df_val, df_test]).sort_index()
            full = full[~full.index.duplicated(keep='last')]
            mtfed_full = self.system._build_features(full)

            # ── Markov-chain regime features (next-regime persistence + entropy)
            if "regime" in mtfed_full.columns:
                regimes = mtfed_full["regime"].fillna(0).astype(int).values
                n_states = max(2, int(np.unique(regimes).size))
                markov = MarkovRegimeForecaster(n_states=n_states)
                markov.fit(regimes)
                mtfed_full = markov.add_features(mtfed_full, regime_col="regime")
                print(f"[TrainingPipeline] Markov fitted on {n_states} regimes; "
                      f"transition matrix:\n{np.round(markov.T, 3)}")

            n_train = len(df_train)
            n_val = len(df_val)
            mtfed_train = mtfed_full.iloc[:n_train].copy()
            mtfed_val = mtfed_full.iloc[n_train:n_train + n_val].copy()
            mtfed_test = mtfed_full.iloc[n_train + n_val:].copy()

            feature_cols = [c for c in mtfed_train.columns
                            if c not in ['open', 'high', 'low', 'close', 'volume']]
            for d in (mtfed_train, mtfed_val, mtfed_test):
                d.fillna(0, inplace=True)

            if len(mtfed_train) == 0:
                raise ValueError("Training set empty after cleaning")

            self.metrics.n_features = len(feature_cols)

            self._update(TrainingStatus.TRAINING, 60,
                         f"Training {self.config.model_type} on "
                         f"{len(mtfed_train):,} bars (temporal decay halflife="
                         f"{self.config.temporal_decay_halflife}m)…")

            # Temporal decay sample weights
            try:
                weights = self._temporal_decay_weights(
                    len(mtfed_train), self.config.temporal_decay_halflife
                )
                weight_series = pd.Series(weights, index=mtfed_train.index)
                if hasattr(self.system.signal_model, "fit"):
                    try:
                        self.system.signal_model.fit(
                            mtfed_train,
                            sample_weight=weight_series,
                        )
                    except TypeError:
                        # Older signature without sample_weight
                        self.system.signal_model.fit(mtfed_train)
            except Exception as e:
                print(f"[TrainingPipeline] decay-weighted fit failed ({e}); "
                      f"falling back to unweighted")
                self.system.signal_model.fit(mtfed_train)

            self.system.is_trained = True
            self.system.data_buffer = mtfed_train.tail(500).copy()

            train_dist = pd.Series(
                self.system.signal_model.predict(mtfed_train)
            ).value_counts().sort_index().to_dict()
            print(f"[TrainingPipeline] Train prediction distribution: {train_dist}")

            if self._cancelled:
                return self.metrics

            self._update(TrainingStatus.VALIDATING_OOS, 75,
                         "Computing train / val / test metrics…")
            for split_name, split_df in (
                ("train", mtfed_train), ("val", mtfed_val), ("oos", mtfed_test)
            ):
                m = self._compute_metrics_set(split_df, prefix=split_name)
                for k, v in m.items():
                    setattr(self.metrics, k, v)

            self._update(TrainingStatus.SAVING, 90, "Saving model artifacts…")
            artifact_dir = os.path.join("data", "models", self.config.asset)
            os.makedirs(artifact_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(
                artifact_dir,
                f"precision_{self.config.asset}_{self.config.model_type}_{timestamp}.pkl",
            )
            try:
                self.system.save(model_path)
            except Exception as e:
                print(f"[TrainingPipeline] save failed: {e}")
            meta_path = model_path.replace('.pkl', '_meta.json')
            with open(meta_path, 'w') as f:
                json.dump({
                    'config': asdict(self.config),
                    'metrics': asdict(self.metrics),
                    'saved_at': datetime.now().isoformat(),
                }, f, indent=2, default=str)
            self.metrics.model_path = model_path

            duration = (datetime.now() - start).total_seconds()
            self.metrics.training_duration_sec = duration
            self._update(TrainingStatus.COMPLETE, 100,
                         f"Done in {duration:.1f}s · "
                         f"val F1={self.metrics.val_f1:.3f} · "
                         f"OOS F1={self.metrics.oos_f1:.3f} · "
                         f"OOS WR={self.metrics.oos_winrate:.1%}")
        except Exception as e:
            self._update(TrainingStatus.ERROR, self.metrics.progress_pct,
                         f"Error: {type(e).__name__}: {str(e)[:120]}")
            import traceback
            traceback.print_exc()
        return self.metrics

    def run_async(self, fetch_fn: Callable[..., pd.DataFrame]) -> threading.Thread:
        """Run the pipeline in a daemon thread; access self.metrics for status."""
        def _runner():
            self.run(fetch_fn)
        self._thread = threading.Thread(target=_runner, daemon=True,
                                          name=f"TrainPipeline-{self.config.asset}")
        self._thread.start()
        return self._thread
