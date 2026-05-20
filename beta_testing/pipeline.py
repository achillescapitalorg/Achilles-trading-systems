"""
Gold 1M Training Pipeline
=========================
Trains LightGBM, XGBoost, and Random Forest on 1-minute gold data.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import warnings

warnings.filterwarnings('ignore')

from .features import compute_1m_features
from .models import Gold1mLightGBM, Gold1mXGBoost, Gold1mRandomForest, Gold1mEnsembleTrader
from .config import PROCESSED_DIR


class Gold1mTrainingPipeline:
    """Complete training pipeline for 1m gold models."""

    def __init__(self, horizon=5):
        self.horizon = horizon
        self.models = {}
        self.results = {}
        self.feature_cols = None

    def prepare_data(self, df: pd.DataFrame):
        """Prepare features and targets from raw 1m data."""
        print("Computing features...")
        features = compute_1m_features(df)

        target_col = f'target_dir_{self.horizon}'
        if target_col not in features.columns:
            raise ValueError(f"Target column {target_col} not found")

        target_cols = [c for c in features.columns if c.startswith('target_')]
        y = features[target_col].values
        X = features.drop(columns=target_cols)

        # Binary target: 1 = up, 0 = down (remove zeros by random assignment)
        y_binary = np.where(y > 0, 1, 0)
        zero_mask = y == 0
        if zero_mask.sum() > 0:
            y_binary[zero_mask] = np.random.RandomState(42).randint(0, 2, size=zero_mask.sum())

        # Temporal split
        split = int(len(X) * 0.85)
        split2 = int(split * 0.85)

        X_train, y_train = X.iloc[:split2], y_binary[:split2]
        X_val, y_val = X.iloc[split2:split], y_binary[split2:split]
        X_test, y_test = X.iloc[split:], y_binary[split:]

        print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
        print(f"Train up-ratio: {y_train.mean():.3f}")

        self.feature_cols = list(X.columns)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def train_all(self, X_train, y_train, X_val, y_val):
        """Train all models."""
        print("\n=== Training LightGBM ===")
        lgb = Gold1mLightGBM()
        lgb.fit(X_train, y_train, X_val, y_val)
        self.models['lgb'] = lgb

        print("\n=== Training XGBoost ===")
        xgb = Gold1mXGBoost()
        xgb.fit(X_train, y_train, X_val, y_val)
        self.models['xgb'] = xgb

        print("\n=== Training Random Forest ===")
        rf = Gold1mRandomForest()
        rf.fit(X_train, y_train)
        self.models['rf'] = rf

        return self.models

    def evaluate(self, X_test, y_test):
        """Evaluate all models."""
        print("\n=== EVALUATION ===")
        results = {}
        for name, model in self.models.items():
            preds = model.predict(X_test)
            pred_labels = (preds > 0.5).astype(int)
            acc = accuracy_score(y_test, pred_labels)
            auc = roc_auc_score(y_test, preds)
            loss = log_loss(y_test, preds.clip(1e-6, 1 - 1e-6))
            results[name] = {
                'accuracy': float(acc),
                'auc': float(auc),
                'logloss': float(loss),
            }
            print(f"\n{name.upper()}:")
            print(f"  Accuracy: {acc:.4f} (baseline: {max(y_test.mean(), 1-y_test.mean()):.4f})")
            print(f"  AUC: {auc:.4f}")
            print(f"  LogLoss: {loss:.4f}")

        # Ensemble
        weights = {'lgb': 0.5, 'xgb': 0.3, 'rf': 0.2}
        ensemble = sum(weights[n] * self.models[n].predict(X_test) for n in self.models)
        ens_labels = (ensemble > 0.5).astype(int)
        ens_acc = accuracy_score(y_test, ens_labels)
        ens_auc = roc_auc_score(y_test, ensemble)
        results['ensemble'] = {
            'accuracy': float(ens_acc),
            'auc': float(ens_auc),
        }
        print(f"\nENSEMBLE (50/30/20):")
        print(f"  Accuracy: {ens_acc:.4f}")
        print(f"  AUC: {ens_auc:.4f}")

        self.results = results
        return results

    def save_all(self, prefix='gold_1m'):
        """Save all trained models."""
        save_dir = PROCESSED_DIR / 'models'
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            path = save_dir / f'{prefix}_{name}.pkl'
            model.save(str(path))
        # Save results
        with open(save_dir / f'{prefix}_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        # Save feature columns
        with open(save_dir / f'{prefix}_features.json', 'w') as f:
            json.dump(self.feature_cols, f)
        print(f"\nSaved {len(self.models)} models to {save_dir}")
