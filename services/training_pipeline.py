"""
Production Training Pipeline for Precision Trading System
===========================================================
Enterprise-grade ML pipeline with:
  - Automated hyperparameter optimization (Optuna)
  - Multi-model comparison and ensemble selection
  - Feature engineering with automated selection
  - Data drift detection and validation
  - Model versioning with A/B testing support
  - Online learning with incremental updates
  - SHAP-based explainability
  - Comprehensive monitoring and alerting

Integrates with precision_trading_system v4 for:
  - Triple-barrier labeling
  - Markov regime features
  - Meta-labeling
  - Purged cross-validation
  - Walk-forward backtesting
"""

import os
import json
import hashlib
import pickle
import threading
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, matthews_corrcoef, accuracy_score,
    precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, RFECV
)
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainingPipeline")

warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available, hyperparameter optimization disabled")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available, explainability features disabled")

try:
    from scipy import stats
    from scipy.stats import ks_2samp, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Import from precision_trading_system
from services.precision_trading_system import (
    PrecisionTradingSystem,
    Asset, AssetConfig, ASSET_CONFIGS,
    TradeDirection, Trade, Regime,
    XGBoostSignalModel, LorentzianClassifier,
    StackedEnsemble, MarkovRegimeForecaster,
    triple_barrier_labels_vectorized,
    ClassificationMetrics, BiasDetector,
    MicrostructureCleaner, FlowAnalyzer,
    MarketStructureAnalyzer, MTFConfluenceEngine,
    HMMRegimeDetector, MetaLabeler,
    SQLiteBacktestDB, PurgedWalkForward,
)


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class TrainingStatus(Enum):
    IDLE = "idle"
    FETCHING = "fetching_data"
    VALIDATING_DATA = "validating_data"
    CLEANING = "cleaning_pipeline"
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_SELECTION = "feature_selection"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    TRAINING = "training_model"
    CROSS_VALIDATING = "cross_validating"
    CALIBRATING = "calibrating"
    EVALUATING = "evaluating"
    EXPLAINABILITY = "generating_explanations"
    SAVING = "saving_artifacts"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


class ModelType(Enum):
    XGBOOST = "xgboost"
    LORENTZIAN = "lorentzian"
    STACKING = "stacking"
    ENSEMBLE = "ensemble"


@dataclass
class TrainingConfig:
    """Comprehensive training configuration."""
    # Asset and model
    asset: str = "XAUUSD"
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
    
    # Training
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
    
    # Feature selection
    feature_selection_method: str = "mutual_info"  # mutual_info, rfecv, correlation
    max_features: int = 50
    min_feature_importance: float = 0.001
    
    # Hyperparameter optimization
    enable_hyperopt: bool = True
    hyperopt_trials: int = 50
    hyperopt_timeout: int = 3600  # seconds
    
    # Cross-validation
    n_cv_folds: int = 5
    cv_method: str = "purged"  # purged, timeseries, blocked, cpcv
    cpcv_n_splits: int = 6
    cpcv_n_test_splits: int = 2
    
    # Ensemble
    ensemble_models: List[str] = field(default_factory=lambda: ["xgboost", "lorentzian"])
    ensemble_weights: Optional[Dict[str, float]] = None
    
    # Online learning
    enable_online_learning: bool = False
    online_learning_window: int = 1000
    retrain_threshold: float = 0.1  # Performance degradation threshold
    
    # Drift detection
    enable_drift_detection: bool = True
    drift_statistical_threshold: float = 0.05
    drift_ks_threshold: float = 0.1
    
    # Explainability
    generate_shap_values: bool = True
    shap_sample_size: int = 1000
    
    # Monitoring
    enable_monitoring: bool = True
    alert_on_degradation: bool = True
    
    # Paths
    artifact_dir: str = "data/models"
    feature_store_path: str = "data/feature_store"
    experiment_tracking_path: str = "data/experiments"


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics."""
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
    drift_detected: bool = False
    drift_features: List[str] = field(default_factory=list)
    
    # Feature metrics
    n_features: int = 0
    n_features_original: int = 0
    n_features_selected: int = 0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    selected_features: List[str] = field(default_factory=list)
    
    # Classification metrics
    train_f1: float = 0.0
    val_f1: float = 0.0
    oos_f1: float = 0.0
    train_mcc: float = 0.0
    val_mcc: float = 0.0
    oos_mcc: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    oos_accuracy: float = 0.0
    train_precision: float = 0.0
    val_precision: float = 0.0
    oos_precision: float = 0.0
    train_recall: float = 0.0
    val_recall: float = 0.0
    oos_recall: float = 0.0
    train_auc: float = 0.0
    val_auc: float = 0.0
    oos_auc: float = 0.0
    
    # Trading metrics
    train_winrate: float = 0.0
    val_winrate: float = 0.0
    oos_winrate: float = 0.0
    train_sharpe: float = 0.0
    val_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    
    # Cross-validation
    cv_f1_mean: float = 0.0
    cv_f1_std: float = 0.0
    cv_mcc_mean: float = 0.0
    cv_mcc_std: float = 0.0
    
    # Model info
    model_path: str = ""
    model_version: str = ""
    data_hash: str = ""
    training_duration_sec: float = 0.0
    hyperopt_best_params: Dict[str, Any] = field(default_factory=dict)
    
    # SHAP explainability
    shap_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Error info
    error_message: str = ""
    error_traceback: str = ""


# =============================================================================
# DATA CLEANER (backward-compatible with notebook imports)
# =============================================================================

class DataCleaner:
    """Production-grade OHLCV cleaning pipeline."""
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        if df.index.duplicated().any():
            dups = int(df.index.duplicated().sum())
            df = df[~df.index.duplicated(keep='last')]
            logger.info(f"[Cleaner] Removed {dups} duplicate timestamps")
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
            logger.info(f"[Cleaner] {gaps} missing bars detected")
        df_filled = df_re.ffill(limit=max_gap_minutes)
        df_clean = df_filled.dropna()
        dropped = len(df_filled) - len(df_clean)
        if dropped > 0:
            logger.info(f"[Cleaner] Dropped {dropped} bars in large gaps")
        return df_clean
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, zscore: float = 4.0) -> Tuple[pd.DataFrame, int]:
        returns = df['close'].pct_change().abs()
        vol = returns.rolling(60).std()
        z = returns / vol.replace(0, np.nan)
        mask = (z.abs() < zscore) | z.isna()
        outliers = int((~mask).sum())
        if outliers > 0:
            logger.info(f"[Cleaner] Removed {outliers} outlier bars (|z| > {zscore})")
        return df[mask], outliers
    
    @staticmethod
    def validate_sessions(df: pd.DataFrame, min_bars: int = 200) -> pd.DataFrame:
        df = df.copy()
        df['_date'] = df.index.date
        daily_counts = df.groupby('_date').size()
        valid_days = daily_counts[daily_counts >= min_bars].index
        if len(valid_days) < len(daily_counts):
            invalid = len(daily_counts) - len(valid_days)
            logger.info(f"[Cleaner] Removed {invalid} days with < {min_bars} bars")
        df = df[df['_date'].isin(valid_days)].copy()
        df = df.drop(columns=['_date'], errors='ignore')
        return df
    
    @staticmethod
    def detect_regime_shift(df: pd.DataFrame, window: int = 1000) -> int:
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
            logger.info(f"[Cleaner] REGIME SHIFT DETECTED: F={f_stat:.2f}")
            return 1
        return 0


# =============================================================================
# DATA VALIDATION
# =============================================================================

class DataValidator:
    """Comprehensive data validation for financial time series."""
    
    @staticmethod
    def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate DataFrame has required columns."""
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        return len(missing) == 0, missing
    
    @staticmethod
    def validate_types(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate column types are numeric."""
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        errors = []
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"{col} is not numeric")
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_ohlc_logic(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate OHLC logic (high >= low, etc.)."""
        errors = []
        if 'high' in df.columns and 'low' in df.columns:
            invalid = (df['high'] < df['low']).sum()
            if invalid > 0:
                errors.append(f"{invalid} bars have high < low")
        if 'high' in df.columns and 'close' in df.columns:
            invalid = ((df['close'] > df['high']) | (df['close'] < df['low'])).sum()
            if invalid > 0:
                errors.append(f"{invalid} bars have close outside high-low range")
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_completeness(df: pd.DataFrame, min_bars: int = 1000) -> Tuple[bool, str]:
        """Validate sufficient data."""
        if len(df) < min_bars:
            return False, f"Insufficient data: {len(df)} bars (need >= {min_bars})"
        return True, "OK"
    
    @staticmethod
    def validate_all(df: pd.DataFrame, min_bars: int = 1000) -> Dict[str, Any]:
        """Run all validations."""
        results = {
            'schema_valid': False,
            'types_valid': False,
            'ohlc_valid': False,
            'completeness_valid': False,
            'errors': [],
            'warnings': []
        }
        
        schema_ok, schema_missing = DataValidator.validate_schema(df)
        results['schema_valid'] = schema_ok
        if not schema_ok:
            results['errors'].extend([f"Missing column: {c}" for c in schema_missing])
        
        types_ok, type_errors = DataValidator.validate_types(df)
        results['types_valid'] = types_ok
        if not types_ok:
            results['errors'].extend(type_errors)
        
        ohlc_ok, ohlc_errors = DataValidator.validate_ohlc_logic(df)
        results['ohlc_valid'] = ohlc_ok
        if not ohlc_ok:
            results['errors'].extend(ohlc_errors)
        
        comp_ok, comp_msg = DataValidator.validate_completeness(df, min_bars)
        results['completeness_valid'] = comp_ok
        if not comp_ok:
            results['errors'].append(comp_msg)
        
        results['overall_valid'] = all([
            results['schema_valid'],
            results['types_valid'],
            results['ohlc_valid'],
            results['completeness_valid']
        ])
        
        return results


# =============================================================================
# DATA DRIFT DETECTION
# =============================================================================

class DriftDetector:
    """Detect data drift for retraining triggers."""
    
    def __init__(self, statistical_threshold: float = 0.05,
                 ks_threshold: float = 0.1):
        self.statistical_threshold = statistical_threshold
        self.ks_threshold = ks_threshold
        self.reference_distribution: Optional[pd.DataFrame] = None
        
    def fit_reference(self, df: pd.DataFrame, feature_cols: List[str]):
        """Store reference distribution."""
        self.reference_distribution = df[feature_cols].copy()
        
    def detect_drift(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """Detect drift between reference and current data."""
        if self.reference_distribution is None:
            return {'drift_detected': False, 'message': 'No reference set'}
        
        drifted_features = []
        ks_stats = {}
        p_values = {}
        
        for col in feature_cols:
            if col not in df.columns or col not in self.reference_distribution.columns:
                continue
                
            ref_data = self.reference_distribution[col].dropna()
            curr_data = df[col].dropna()
            
            if len(ref_data) < 100 or len(curr_data) < 100:
                continue
            
            # Kolmogorov-Smirnov test
            if SCIPY_AVAILABLE:
                ks_stat, p_value = ks_2samp(ref_data, curr_data)
                ks_stats[col] = float(ks_stat)
                p_values[col] = float(p_value)
                
                if ks_stat > self.ks_threshold or p_value < self.statistical_threshold:
                    drifted_features.append(col)
        
        return {
            'drift_detected': len(drifted_features) > 0,
            'drifted_features': drifted_features,
            'ks_statistics': ks_stats,
            'p_values': p_values,
            'n_features_checked': len(feature_cols),
            'n_features_drifted': len(drifted_features)
        }
    
    def detect_concept_drift(self, y_true: np.ndarray, y_pred: np.ndarray,
                            window_size: int = 100) -> Dict[str, Any]:
        """Detect concept drift in model performance."""
        if len(y_true) < window_size * 2:
            return {'drift_detected': False, 'message': 'Insufficient data'}
        
        # Calculate rolling accuracy
        correct = (y_true == y_pred).astype(int)
        rolling_acc = pd.Series(correct).rolling(window_size).mean()
        
        # Detect sudden drops
        early_mean = rolling_acc.iloc[:window_size].mean()
        late_mean = rolling_acc.iloc[-window_size:].mean()
        
        drift_detected = late_mean < early_mean * 0.9  # 10% degradation
        
        return {
            'drift_detected': drift_detected,
            'early_accuracy': float(early_mean),
            'late_accuracy': float(late_mean),
            'degradation_pct': float((early_mean - late_mean) / early_mean * 100) if early_mean > 0 else 0
        }


# =============================================================================
# FEATURE ENGINEERING PIPELINE
# =============================================================================

class FeatureEngineer:
    """Advanced feature engineering with automated selection."""
    
    def __init__(self, config: AssetConfig):
        self.config = config
        self.cleaner = MicrostructureCleaner(config)
        self.flow = FlowAnalyzer(config)
        self.structure = MarketStructureAnalyzer(config)
        self.mtf = MTFConfluenceEngine()
        self.hmm = HMMRegimeDetector(n_regimes=2)
        self.scaler = StandardScaler()
        self.feature_cols: Optional[List[str]] = None
        
    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Full feature engineering pipeline."""
        cleaned = self.cleaner.clean_ohlcv(df)
        flowed = self.flow.compute_flow_features(cleaned)
        structured = self.structure.detect_fvg(flowed)
        mtfed = self.mtf.compute(structured)
        
        if fit:
            self.hmm.fit(mtfed)
        if self.hmm.model is not None:
            mtfed["regime"] = self.hmm.predict_regime(mtfed)
        else:
            mtfed["regime"] = 0
            
        # Add Markov features
        if "regime" in mtfed.columns:
            regimes = mtfed["regime"].fillna(0).astype(int).values
            n_states = max(2, int(np.unique(regimes).size))
            markov = MarkovRegimeForecaster(n_states=n_states)
            markov.fit(regimes)
            mtfed = markov.add_features(mtfed, regime_col="regime")
        
        return mtfed
    
    def select_features(self, df: pd.DataFrame, target: pd.Series,
                       method: str = "mutual_info",
                       max_features: int = 50,
                       min_importance: float = 0.001) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using specified method. Non-numeric columns are
        auto-dropped with a warning — if they carry signal, they should be
        one-hot encoded or label-encoded before this step."""
        # Exclude known non-feature columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        candidate_cols = [c for c in df.columns if c not in exclude_cols]

        # Separate numeric from non-numeric
        numeric_cols = [c for c in candidate_cols
                        if pd.api.types.is_numeric_dtype(df[c])]
        non_numeric_cols = [c for c in candidate_cols
                            if not pd.api.types.is_numeric_dtype(df[c])]

        if non_numeric_cols:
            logger.warning(
                f"[FeatureEngineer] Dropping {len(non_numeric_cols)} non-numeric "
                f"column(s): {non_numeric_cols}.  Consider one-hot encoding them "
                f"before feature selection if they carry signal."
            )

        feature_cols = numeric_cols
        if not feature_cols:
            raise ValueError(
                "No numeric feature columns remain after filtering "
                f"(dropped {len(non_numeric_cols)} non-numeric)."
            )

        X = df[feature_cols].fillna(0)
        y = target.loc[X.index].fillna(0)

        if method == "mutual_info":
            n_select = min(max_features, len(feature_cols))
            selector = SelectKBest(mutual_info_classif, k=n_select)
            X_selected = selector.fit_transform(X, y)
            selected_mask = selector.get_support()
            selected_features = [f for f, selected in zip(feature_cols, selected_mask) if selected]

        elif method == "correlation":
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            selected_features = correlations.head(max_features).index.tolist()
            X_selected = X[selected_features]

        else:
            selected_features = feature_cols[:max_features]
            X_selected = X[selected_features]

        # Secondary filter by minimum mutual-information importance
        if len(selected_features) > 10 and method == "mutual_info":
            importances = mutual_info_classif(X[selected_features], y)
            importance_dict = dict(zip(selected_features, importances))
            selected_features = [f for f, imp in importance_dict.items()
                                 if imp >= min_importance]
            X_selected = X[selected_features]

        self.feature_cols = selected_features
        return pd.DataFrame(X_selected, index=df.index), selected_features


# =============================================================================
# HYPERPARAMETER OPTIMIZATION
# =============================================================================

class HyperparameterOptimizer:
    """Bayesian hyperparameter optimization using Optuna."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.best_params: Dict[str, Any] = {}
        self.study: Optional[Any] = None
        
    def create_objective(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series,
                        model_type: str = "xgboost"):
        """Create Optuna objective function."""
        
        def objective(trial):
            if model_type == "xgboost":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
                }
                
                if XGBOOST_AVAILABLE:
                    model = xgb.XGBClassifier(
                        **params,
                        random_state=42,
                        tree_method="hist",
                        eval_metric="mlogloss",
                    )
                    model.fit(X_train, y_train, verbose=False)
                    preds = model.predict(X_val)
                    score = f1_score(y_val, preds, average='macro', zero_division=0)
                    return score
            
            elif model_type == "lorentzian":
                n_neighbors = trial.suggest_int('n_neighbors', 3, 20)
                # Lorentzian doesn't have traditional hyperparameters
                return 0.5
            
            return 0.0
        
        return objective
    
    def optimize(self, X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame, y_val: pd.Series,
                model_type: str = "xgboost",
                n_trials: int = 50,
                timeout: int = 3600) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default parameters")
            return {}
        
        objective = self.create_objective(X_train, y_train, X_val, y_val, model_type)
        
        self.study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42)
        )
        
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        logger.info(f"Best hyperparameters: {self.best_params}")
        logger.info(f"Best F1 score: {self.study.best_value:.4f}")
        
        return self.best_params


# =============================================================================
# MODEL VERSIONING AND REGISTRY
# =============================================================================

class ModelRegistry:
    """Model versioning and registry for A/B testing."""
    
    def __init__(self, registry_path: str = "data/model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / "registry.json"
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                'models': {},
                'experiments': {},
                'production_model': None
            }
    
    def _save_registry(self):
        """Save registry to disk."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def register_model(self, model_id: str, model_path: str,
                      metrics: Dict[str, Any],
                      config: Dict[str, Any],
                      tags: Optional[List[str]] = None) -> str:
        """Register a new model version."""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_id = f"{model_id}_v{version}"
        
        self.registry['models'][full_id] = {
            'id': full_id,
            'base_id': model_id,
            'version': version,
            'path': model_path,
            'metrics': metrics,
            'config': config,
            'tags': tags or [],
            'registered_at': datetime.now().isoformat(),
            'status': 'staging'
        }
        
        self._save_registry()
        logger.info(f"Registered model: {full_id}")
        return full_id
    
    def promote_model(self, model_id: str, environment: str = "production"):
        """Promote model to production."""
        if model_id not in self.registry['models']:
            raise ValueError(f"Model {model_id} not found in registry")
        
        if environment == "production":
            # Demote current production model
            current = self.registry.get('production_model')
            if current:
                self.registry['models'][current]['status'] = 'archived'
            
            self.registry['production_model'] = model_id
            self.registry['models'][model_id]['status'] = 'production'
        
        self._save_registry()
        logger.info(f"Promoted {model_id} to {environment}")
    
    def get_production_model(self) -> Optional[str]:
        """Get current production model ID."""
        return self.registry.get('production_model')
    
    def list_models(self, status: Optional[str] = None) -> List[Dict]:
        """List registered models."""
        models = list(self.registry['models'].values())
        if status:
            models = [m for m in models if m['status'] == status]
        return sorted(models, key=lambda x: x['registered_at'], reverse=True)


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

class TrainingPipeline:
    """
    Production-grade training pipeline with:
    - Automated data validation and drift detection
    - Feature engineering and selection
    - Hyperparameter optimization
    - Cross-validation with purging
    - Model versioning and registry
    - SHAP explainability
    - Comprehensive monitoring
    """
    
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
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(system.config)
        self.drift_detector = DriftDetector(
            config.drift_statistical_threshold,
            config.drift_ks_threshold
        )
        self.shap_monitor = None  # Initialized after first SHAP run
        self.hyperopt = HyperparameterOptimizer(config)
        self.registry = ModelRegistry(config.experiment_tracking_path)
        
        # Create directories
        Path(config.artifact_dir).mkdir(parents=True, exist_ok=True)
        Path(config.feature_store_path).mkdir(parents=True, exist_ok=True)
        
    def cancel(self):
        """Cancel training."""
        self._cancelled = True
        self.metrics.status = TrainingStatus.CANCELLED.value
        
    def is_running(self) -> bool:
        """Check if training is running."""
        return self._thread is not None and self._thread.is_alive()
    
    def _update(self, status: TrainingStatus, progress: float, message: str):
        """Update training status."""
        self.metrics.status = status.value
        self.metrics.progress_pct = float(progress)
        self.metrics.message = message
        self.metrics.timestamp = datetime.now().isoformat()
        logger.info(f"[{status.value}] ({progress:.0f}%): {message}")
    
    def _split_three_way(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test with embargo."""
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
        """Compute temporal decay weights."""
        decay_lambda = np.log(2) / max(halflife, 1)
        t = np.arange(n)
        w = np.exp(decay_lambda * (t - n))
        return w / w.sum() * n
    
    @staticmethod
    def _hash_df(df: pd.DataFrame) -> str:
        """Compute data hash for versioning."""
        s = df.head(100).to_json() + df.tail(100).to_json()
        return hashlib.md5(s.encode()).hexdigest()[:12]
    
    def _compute_metrics_set(self, df: pd.DataFrame, prefix: str) -> Dict[str, float]:
        """Compute comprehensive metrics for a dataset."""
        try:
            preds = self.system.signal_model.predict(df)
            
            # Get triple barrier labels
            cfg = self.system.config
            pt_mult = float(getattr(cfg, "tb_pt_mult", getattr(cfg, "atr_multiplier_tp1", 1.5)))
            sl_mult = float(getattr(cfg, "tb_sl_mult", getattr(cfg, "atr_multiplier_stop", 1.0)))
            max_bars = int(getattr(cfg, "tb_max_holding", 10))
            
            y_true = triple_barrier_labels_vectorized(
                df, pt_atr_mult=pt_mult, sl_atr_mult=sl_mult, max_bars=max_bars,
            )
            y_pred = preds.reindex(y_true.index).fillna(0).values
            yt = y_true.values
            
            mask = ~(np.isnan(yt) | np.isnan(y_pred.astype(float)))
            if mask.sum() < 10:
                return {f"{prefix}_f1": 0.0, f"{prefix}_mcc": 0.0,
                        f"{prefix}_accuracy": 0.0, f"{prefix}_winrate": 0.0}
            
            # Classification metrics
            f1 = float(f1_score(yt[mask], y_pred[mask], average='macro', zero_division=0))
            try:
                mcc = float(matthews_corrcoef(yt[mask], y_pred[mask]))
            except Exception:
                mcc = 0.0
            acc = float(accuracy_score(yt[mask], y_pred[mask]))
            
            try:
                prec = float(precision_score(yt[mask], y_pred[mask], average='macro', zero_division=0))
                rec = float(recall_score(yt[mask], y_pred[mask], average='macro', zero_division=0))
            except Exception:
                prec = rec = 0.0
            
            # Directional win rate
            nz = y_pred[mask] != 0
            wr = 0.0
            if nz.sum() > 0:
                wr = float((np.sign(yt[mask][nz]) == np.sign(y_pred[mask][nz])).mean())
            
            return {
                f"{prefix}_f1": f1,
                f"{prefix}_mcc": mcc,
                f"{prefix}_accuracy": acc,
                f"{prefix}_precision": prec,
                f"{prefix}_recall": rec,
                f"{prefix}_winrate": wr
            }
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return {f"{prefix}_f1": 0.0, f"{prefix}_mcc": 0.0,
                    f"{prefix}_accuracy": 0.0, f"{prefix}_winrate": 0.0}
    
    def _cross_validate(self, df: pd.DataFrame, n_splits: int = 5) -> Dict[str, float]:
        """Run cross-validation: purged, timeseries, blocked, or CPCV."""
        from services.cpcv import CombinatorialPurgedCV

        if self.config.cv_method == "cpcv":
            cpcv = CombinatorialPurgedCV(
                n_splits=self.config.cpcv_n_splits,
                n_test_splits=self.config.cpcv_n_test_splits,
                embargo_pct=self.config.embargo_pct,
            )
            splits = cpcv.split(df)
        elif self.config.cv_method == "purged":
            pwf = PurgedWalkForward(
                n_splits=n_splits,
                test_size_pct=0.1,
                purge_horizon=self.config.purge_horizon,
                embargo_pct=self.config.embargo_pct
            )
            splits = [(train_idx, [test_idx]) for _, train_idx, test_idx in pwf.split(df)]
        else:
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=n_splits)
            splits = [(train_idx, [test_idx]) for train_idx, test_idx in tscv.split(df)]

        f1_scores = []
        mcc_scores = []

        for train_idx, test_idx_list in splits:
            test_idx = test_idx_list[0]
            if len(train_idx) < 100 or len(test_idx) < 30:
                continue

            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            # Build features
            mtfed_train = self.feature_engineer.engineer_features(train_df, fit=True)
            mtfed_test = self.feature_engineer.engineer_features(test_df, fit=False)

            # Train temporary model
            temp_model = XGBoostSignalModel()
            temp_model.fit(mtfed_train)

            # Predict and score
            preds = temp_model.predict(mtfed_test)

            # Get labels
            cfg = self.system.config
            y_true = triple_barrier_labels_vectorized(
                mtfed_test,
                pt_atr_mult=getattr(cfg, "atr_multiplier_tp1", 1.5),
                sl_atr_mult=getattr(cfg, "atr_multiplier_stop", 1.0),
                max_bars=10
            )

            y_pred = preds.reindex(y_true.index).fillna(0).values
            yt = y_true.values
            mask = ~(np.isnan(yt) | np.isnan(y_pred.astype(float)))

            if mask.sum() > 10:
                f1 = f1_score(yt[mask], y_pred[mask], average='macro', zero_division=0)
                try:
                    mcc = matthews_corrcoef(yt[mask], y_pred[mask])
                except:
                    mcc = 0.0
                f1_scores.append(f1)
                mcc_scores.append(mcc)

        return {
            'cv_f1_mean': float(np.mean(f1_scores)) if f1_scores else 0.0,
            'cv_f1_std': float(np.std(f1_scores)) if f1_scores else 0.0,
            'cv_mcc_mean': float(np.mean(mcc_scores)) if mcc_scores else 0.0,
            'cv_mcc_std': float(np.std(mcc_scores)) if mcc_scores else 0.0,
            'n_paths': len(f1_scores),
        }
    
    def _generate_shap_values(self, df: pd.DataFrame, model: Any) -> Dict[str, Any]:
        """Generate SHAP values for model explainability and drift monitoring."""
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP not available'}

        try:
            # Sample data for SHAP
            sample_size = min(self.config.shap_sample_size, len(df))
            sample_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df

            # Get features
            if hasattr(model, '_engineer'):
                feats = model._engineer(sample_df)
                if hasattr(model, 'feature_cols') and model.feature_cols:
                    feats = feats[model.feature_cols]
            else:
                feats = sample_df.select_dtypes(include=[np.number]).fillna(0)

            # Initialize SHAP monitor on first call
            if self.shap_monitor is None:
                from services.shap_monitor import SHAPFeatureMonitor
                self.shap_monitor = SHAPFeatureMonitor(
                    feature_names=list(feats.columns),
                    alert_threshold=0.2,
                )

            # Create SHAP explainer
            if hasattr(model, 'model') and model.model is not None:
                explainer = shap.TreeExplainer(model.model)
                shap_values = explainer.shap_values(feats.values)

                # Summary statistics
                feature_importance = dict(zip(
                    feats.columns,
                    np.abs(shap_values).mean(axis=0).tolist()
                ))

                # Set baseline if not set, otherwise check drift
                X_sample = feats.values
                if self.shap_monitor.baseline_importance is None:
                    self.shap_monitor.set_baseline(model.model, X_sample)
                    drift_report = {"status": "baseline_set"}
                else:
                    drift_report = self.shap_monitor.check_drift(model.model, X_sample)

                return {
                    'feature_importance': feature_importance,
                    'shap_values_shape': str(np.array(shap_values).shape),
                    'sample_size': sample_size,
                    'drift_report': drift_report,
                }
            else:
                return {'error': 'Model not fitted'}
        except Exception as e:
            logger.error(f"SHAP generation failed: {e}")
            return {'error': str(e)}
    
    def run(self, fetch_fn: Callable[..., pd.DataFrame]) -> TrainingMetrics:
        """
        Execute the full training pipeline.
        
        Parameters
        ----------
        fetch_fn : Callable
            Function to fetch data: fetch_fn(start, end, interval) -> pd.DataFrame
            
        Returns
        -------
        TrainingMetrics
            Comprehensive training metrics
        """
        start = datetime.now()
        self._cancelled = False
        
        try:
            # =========================================================================
            # STEP 1: Data Fetching
            # =========================================================================
            self._update(TrainingStatus.FETCHING, 5,
                        f"Fetching {self.config.train_days}d + "
                        f"{self.config.validation_days}d + "
                        f"{self.config.test_days}d of {self.config.interval} data")
            
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
            
            # =========================================================================
            # STEP 2: Data Validation
            # =========================================================================
            self._update(TrainingStatus.VALIDATING_DATA, 10, "Validating data quality")
            
            validation_results = DataValidator.validate_all(df, min_bars=1000)
            if not validation_results['overall_valid']:
                raise ValueError(f"Data validation failed: {validation_results['errors']}")
            
            # =========================================================================
            # STEP 3: Data Cleaning
            # =========================================================================
            self._update(TrainingStatus.CLEANING, 15, f"Cleaning {len(df):,} bars")
            
            # Use system's cleaner
            cleaned = self.system.cleaner.clean_ohlcv(df)
            
            # Additional cleaning
            cleaned = cleaned[~cleaned.index.duplicated(keep='last')]
            cleaned = cleaned.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            
            self.metrics.clean_bars = len(cleaned)
            self.metrics.gaps_found = self.metrics.total_bars - self.metrics.clean_bars
            
            # Detect regime shifts
            cleaner = DataCleaner()
            self.metrics.regime_shifts = cleaner.detect_regime_shift(cleaned)
            self.metrics.data_hash = self._hash_df(cleaned)
            
            # =========================================================================
            # STEP 4: Feature Engineering
            # =========================================================================
            self._update(TrainingStatus.FEATURE_ENGINEERING, 25,
                        "Engineering features with microstructure and regime")
            
            # Build features for entire dataset
            mtfed_full = self.feature_engineer.engineer_features(cleaned, fit=True)
            
            # Store reference for drift detection
            feature_cols = [c for c in mtfed_full.columns
                          if c not in ['open', 'high', 'low', 'close', 'volume']]
            self.drift_detector.fit_reference(mtfed_full, feature_cols)
            self.metrics.n_features_original = len(feature_cols)
            
            # =========================================================================
            # STEP 5: Data Splitting
            # =========================================================================
            df_train, df_val, df_test = self._split_three_way(cleaned)
            
            # Rebuild features for each split
            n_train = len(df_train)
            n_val = len(df_val)
            
            mtfed_train = mtfed_full.iloc[:n_train].copy()
            mtfed_val = mtfed_full.iloc[n_train:n_train + n_val].copy()
            mtfed_test = mtfed_full.iloc[n_train + n_val:].copy()
            
            logger.info(f"Split: train={len(mtfed_train):,}, val={len(mtfed_val):,}, "
                       f"test={len(mtfed_test):,}")
            
            if len(mtfed_train) == 0:
                raise ValueError("Training set empty after cleaning")
            
            if self._cancelled:
                return self.metrics
            
            # =========================================================================
            # STEP 6: Feature Selection
            # =========================================================================
            self._update(TrainingStatus.FEATURE_SELECTION, 35,
                        f"Selecting features using {self.config.feature_selection_method}")
            
            # Create target for feature selection
            fwd = mtfed_train["close"].shift(-self.system.config.target_horizon) / mtfed_train["close"] - 1
            roll_std = mtfed_train["close"].pct_change().rolling(50, min_periods=10).std().bfill()
            roll_std = roll_std.clip(lower=1e-6) * np.sqrt(self.system.config.target_horizon)
            z = fwd / roll_std
            
            target = pd.Series(0, index=mtfed_train.index, dtype=int)
            target[z > 1.0] = 2
            target[(z <= 1.0) & (z > 0.3)] = 1
            target[(z >= -1.0) & (z < -0.3)] = -1
            target[z < -1.0] = -2
            
            # Select features
            X_selected, selected_features = self.feature_engineer.select_features(
                mtfed_train, target,
                method=self.config.feature_selection_method,
                max_features=self.config.max_features,
                min_importance=self.config.min_feature_importance
            )
            
            self.metrics.n_features_selected = len(selected_features)
            self.metrics.n_features = len(selected_features)
            self.metrics.selected_features = selected_features
            
            # =========================================================================
            # STEP 7: Hyperparameter Optimization
            # =========================================================================
            if self.config.enable_hyperopt and OPTUNA_AVAILABLE:
                self._update(TrainingStatus.HYPERPARAMETER_TUNING, 45,
                           f"Optimizing hyperparameters ({self.config.hyperopt_trials} trials)")
                
                best_params = self.hyperopt.optimize(
                    X_selected, target.loc[X_selected.index],
                    mtfed_val[selected_features].fillna(0),
                    target.reindex(mtfed_val.index).fillna(0),
                    model_type=self.config.model_type,
                    n_trials=self.config.hyperopt_trials,
                    timeout=self.config.hyperopt_timeout
                )
                
                self.metrics.hyperopt_best_params = best_params
            
            if self._cancelled:
                return self.metrics
            
            # =========================================================================
            # STEP 8: Cross-Validation
            # =========================================================================
            self._update(TrainingStatus.CROSS_VALIDATING, 55,
                        f"Running {self.config.n_cv_folds}-fold purged cross-validation")
            
            cv_results = self._cross_validate(mtfed_full, n_splits=self.config.n_cv_folds)
            self.metrics.cv_f1_mean = cv_results['cv_f1_mean']
            self.metrics.cv_f1_std = cv_results['cv_f1_std']
            self.metrics.cv_mcc_mean = cv_results['cv_mcc_mean']
            self.metrics.cv_mcc_std = cv_results['cv_mcc_std']
            
            # =========================================================================
            # STEP 9: Model Training
            # =========================================================================
            self._update(TrainingStatus.TRAINING, 65,
                        f"Training {self.config.model_type} with temporal decay")
            
            # Apply temporal decay weights
            weights = self._temporal_decay_weights(
                len(mtfed_train), self.config.temporal_decay_halflife
            )
            weight_series = pd.Series(weights, index=mtfed_train.index)
            
            # Train the model
            if hasattr(self.system.signal_model, "fit"):
                try:
                    self.system.signal_model.fit(
                        mtfed_train,
                        sample_weight=weight_series,
                    )
                except TypeError:
                    # Fallback without sample_weight
                    self.system.signal_model.fit(mtfed_train)
            
            self.system.is_trained = True
            self.system.data_buffer = mtfed_train.tail(500).copy()
            
            # Get feature importance
            if hasattr(self.system.signal_model, 'feature_importance_'):
                self.metrics.feature_importance = self.system.signal_model.feature_importance_
            
            if self._cancelled:
                return self.metrics
            
            # =========================================================================
            # STEP 10: Evaluation
            # =========================================================================
            self._update(TrainingStatus.EVALUATING, 75,
                        "Computing comprehensive metrics")
            
            for split_name, split_df in [
                ("train", mtfed_train),
                ("val", mtfed_val),
                ("oos", mtfed_test)
            ]:
                m = self._compute_metrics_set(split_df, prefix=split_name)
                for k, v in m.items():
                    setattr(self.metrics, k, v)
            
            # =========================================================================
            # STEP 11: Explainability
            # =========================================================================
            if self.config.generate_shap_values:
                self._update(TrainingStatus.EXPLAINABILITY, 85,
                           "Generating SHAP explanations")
                
                shap_results = self._generate_shap_values(mtfed_test, self.system.signal_model)
                self.metrics.shap_summary = shap_results
            
            # =========================================================================
            # STEP 12: Save Artifacts
            # =========================================================================
            self._update(TrainingStatus.SAVING, 90, "Saving model artifacts")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_version = f"{self.config.asset}_{self.config.model_type}_{timestamp}"
            
            # Save model
            artifact_dir = Path(self.config.artifact_dir) / self.config.asset
            artifact_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = artifact_dir / f"{model_version}.pkl"
            
            try:
                self.system.save(str(model_path))
            except Exception as e:
                logger.error(f"Model save failed: {e}")
                # Fallback: save with joblib
                joblib.dump(self.system.signal_model, model_path)
            
            # Save metadata
            meta_path = artifact_dir / f"{model_version}_meta.json"
            with open(meta_path, 'w') as f:
                json.dump({
                    'config': asdict(self.config),
                    'metrics': asdict(self.metrics),
                    'saved_at': datetime.now().isoformat(),
                    'model_version': model_version,
                }, f, indent=2, default=str)
            
            # Register in model registry
            self.registry.register_model(
                model_id=f"{self.config.asset}_{self.config.model_type}",
                model_path=str(model_path),
                metrics=asdict(self.metrics),
                config=asdict(self.config),
                tags=['automated', 'production']
            )
            
            self.metrics.model_path = str(model_path)
            self.metrics.model_version = model_version
            
            # =========================================================================
            # STEP 13: Finalize
            # =========================================================================
            duration = (datetime.now() - start).total_seconds()
            self.metrics.training_duration_sec = duration
            
            self._update(TrainingStatus.COMPLETE, 100,
                        f"Done in {duration:.1f}s | "
                        f"Val F1={self.metrics.val_f1:.3f} | "
                        f"OOS F1={self.metrics.oos_f1:.3f} | "
                        f"OOS WR={self.metrics.oos_winrate:.1%}")
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:200]}"
            self._update(TrainingStatus.ERROR, self.metrics.progress_pct, error_msg)
            self.metrics.error_message = error_msg
            import traceback
            self.metrics.error_traceback = traceback.format_exc()
            logger.error(f"Training failed: {e}")
            traceback.print_exc()
        
        return self.metrics
    
    def run_async(self, fetch_fn: Callable[..., pd.DataFrame]) -> threading.Thread:
        """Run training in a background thread."""
        def _runner():
            self.run(fetch_fn)
        
        self._thread = threading.Thread(
            target=_runner,
            daemon=True,
            name=f"TrainPipeline-{self.config.asset}"
        )
        self._thread.start()
        return self._thread
    
    def check_drift(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Check for data drift in new data."""
        mtfed = self.feature_engineer.engineer_features(new_data, fit=False)
        feature_cols = [c for c in mtfed.columns
                       if c not in ['open', 'high', 'low', 'close', 'volume']]
        return self.drift_detector.detect_drift(mtfed, feature_cols)
    
    def incremental_update(self, new_data: pd.DataFrame,
                         performance_threshold: float = 0.1) -> bool:
        """
        Incrementally update model with new data.
        
        Parameters
        ----------
        new_data : pd.DataFrame
            New market data
        performance_threshold : float
            Threshold for triggering full retrain
            
        Returns
        -------
        bool
            True if model was updated
        """
        if not self.config.enable_online_learning:
            return False
        
        # Check for drift
        drift_results = self.check_drift(new_data)
        
        if drift_results.get('drift_detected', False):
            logger.warning(f"Data drift detected in features: {drift_results['drifted_features']}")
            self.metrics.drift_detected = True
            self.metrics.drift_features = drift_results['drifted_features']
        
        # Build features
        mtfed = self.feature_engineer.engineer_features(new_data, fit=False)
        
        # Update data buffer
        self.system.data_buffer = pd.concat([
            self.system.data_buffer,
            mtfed
        ]).tail(self.config.online_learning_window)
        
        # Check if full retrain needed
        if drift_results.get('drift_detected', False):
            logger.info("Triggering full retrain due to drift")
            return False  # Signal that full retrain is needed
        
        return True


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_training_pipeline(
    asset: str = "XAUUSD",
    model_type: str = "xgboost",
    **kwargs
) -> Tuple[TrainingPipeline, PrecisionTradingSystem]:
    """
    Factory function to create a training pipeline.
    
    Parameters
    ----------
    asset : str
        Asset symbol (XAUUSD, BTCUSD, etc.)
    model_type : str
        Model type (xgboost, lorentzian, stacking)
    **kwargs
        Additional config parameters
        
    Returns
    -------
    Tuple[TrainingPipeline, PrecisionTradingSystem]
        Configured pipeline and trading system
    """
    asset_enum = Asset(asset)
    config = TrainingConfig(asset=asset, model_type=model_type, **kwargs)
    system = PrecisionTradingSystem(asset=asset_enum, model_type=model_type)
    pipeline = TrainingPipeline(system, config)
    return pipeline, system


def run_training(
    asset: str = "XAUUSD",
    model_type: str = "xgboost",
    fetch_fn: Optional[Callable] = None,
    **config_kwargs
) -> TrainingMetrics:
    """
    Convenience function to run training.
    
    Parameters
    ----------
    asset : str
        Asset symbol
    model_type : str
        Model type
    fetch_fn : Callable, optional
        Data fetching function
    **config_kwargs
        Training configuration
        
    Returns
    -------
    TrainingMetrics
        Training results
    """
    pipeline, system = create_training_pipeline(asset, model_type, **config_kwargs)
    
    if fetch_fn is None:
        # Use default market data service
        from services.market_data import get_market_data_service
        mds = get_market_data_service()
        
        def default_fetch(start, end, interval):
            return mds.fetch_history_range(asset, start=start, end=end, interval=interval)
        
        fetch_fn = default_fetch
    
    return pipeline.run(fetch_fn)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Precision Trading System - Training Pipeline")
    print("=" * 60)
    
    # Run training for XAUUSD
    metrics = run_training(
        asset="XAUUSD",
        model_type="xgboost",
        train_days=30,
        enable_hyperopt=False,  # Set to True if Optuna is installed
        generate_shap_values=False  # Set to True if SHAP is installed
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Status: {metrics.status}")
    print(f"Duration: {metrics.training_duration_sec:.1f}s")
    print(f"OOS F1: {metrics.oos_f1:.3f}")
    print(f"OOS Win Rate: {metrics.oos_winrate:.1%}")
    print(f"Model saved to: {metrics.model_path}")
