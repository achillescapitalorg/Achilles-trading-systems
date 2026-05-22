"""
XGBoost Model for 1-Minute Gold Prediction
"""
import xgboost as xgb
import numpy as np
import pandas as pd
import os


class Gold1mXGBoost:
    """XGBoost with different tree growth for ensemble diversity."""

    BEST_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 4,
        'learning_rate': 0.008,
        'n_estimators': 3000,
        'subsample': 0.7,
        'colsample_bytree': 0.65,
        'colsample_bylevel': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 3.0,
        'min_child_weight': 200,
        'gamma': 0.1,
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1,
    }

    def __init__(self, params=None):
        self.params = self.BEST_PARAMS.copy()
        if params:
            self.params.update(params)
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        # Only auto-compute scale_pos_weight if not already provided in params
        if 'scale_pos_weight' not in self.params or self.params.get('scale_pos_weight') is None:
            scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            self.params['scale_pos_weight'] = scale
        self.model = xgb.XGBClassifier(**self.params)
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            verbose=False
        )
        try:
            best_iter = self.model.best_iteration
            print(f"[XGB-1M] Best iteration: {best_iter}")
        except AttributeError:
            print(f"[XGB-1M] Training complete ({self.model.n_estimators} trees)")
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        """Save in native XGBoost UBJSON format to avoid .pkl confusion."""
        # Ensure .ubj extension (not .pkl) so load_model doesn't warn
        ubj_path = str(path).replace('.pkl', '.ubj')
        self.model.save_model(ubj_path)

    def load(self, path):
        """Load from native XGBoost format."""
        self.model = xgb.XGBClassifier()
        # Try .ubj first, then fall back to .pkl for legacy files
        ubj_path = str(path).replace('.pkl', '.ubj')
        if os.path.exists(ubj_path):
            self.model.load_model(ubj_path)
        else:
            self.model.load_model(path)
