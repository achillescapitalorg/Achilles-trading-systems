"""
LightGBM Model for 1-Minute Gold Prediction
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import joblib


class Gold1mLightGBM:
    """LightGBM optimized specifically for 1-minute gold prediction."""

    BEST_PARAMS = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 5,
        'min_data_in_leaf': 300,
        'learning_rate': 0.01,
        'feature_fraction': 0.65,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 0.5,
        'lambda_l2': 2.0,
        'max_bin': 127,
        'verbose': -1,
        'seed': 42,
        'deterministic': True,
        'is_unbalance': False,
    }

    def __init__(self, params=None):
        self.params = self.BEST_PARAMS.copy()
        if params:
            self.params.update(params)
        self.model = None
        self.feature_names = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None, num_boost_round=3000):
        self.feature_names = list(X_train.columns)
        # Only auto-compute scale_pos_weight if not already provided
        if 'scale_pos_weight' not in self.params or self.params.get('scale_pos_weight') is None:
            n_pos = (y_train == 1).sum()
            n_neg = (y_train == 0).sum()
            scale_pos_weight = n_neg / max(n_pos, 1)
            self.params['scale_pos_weight'] = scale_pos_weight

        train_data = lgb.Dataset(X_train, y_train, weight=sample_weight)
        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            valid_sets.append(lgb.Dataset(X_val, y_val, reference=train_data))

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        best_score = self.model.best_score.get('valid_1', {}).get('auc', 'N/A')
        print(f"[LGB-1M] Best iteration: {self.model.best_iteration}")
        print(f"[LGB-1M] Best AUC: {best_score}")
        return self

    def predict(self, X):
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def get_importance(self):
        imp = self.model.feature_importance(importance_type='gain')
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)

    def save(self, path):
        joblib.dump({
            'model': self.model,
            'params': self.params,
            'features': self.feature_names
        }, path)

    def load(self, path):
        d = joblib.load(path)
        self.model = d['model']
        self.params = d['params']
        self.feature_names = d['features']
