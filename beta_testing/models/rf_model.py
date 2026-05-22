"""
Random Forest Model for 1-Minute Gold Prediction
"""
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import joblib


class Gold1mRandomForest:
    """Random Forest — hard to overfit, perfect baseline."""

    BEST_PARAMS = {
        'n_estimators': 200,
        'max_depth': 6,
        'min_samples_leaf': 100,
        'min_samples_split': 50,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': 42,
    }

    def __init__(self, params=None):
        best = self.BEST_PARAMS.copy()
        if params:
            best.update(params)
        self.model = RandomForestClassifier(**best)

    def fit(self, X_train, y_train, sample_weight=None, **kwargs):
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        if hasattr(self.model, "oob_score_"):
            print(f"[RF-1M] OOB Score: {self.model.oob_score_:.4f}")
        else:
            print(f"[RF-1M] Training complete ({self.model.n_estimators} trees)")
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
