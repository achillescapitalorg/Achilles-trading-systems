"""
Regime Predictor
================
Predict the next market regime using leading indicators.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
import joblib


class RegimePredictor:
    """
    Predict the next market regime using leading indicators.
    Key insight: Regime changes are preceded by statistical precursors.
    """

    def __init__(self, forecast_horizon: int = 20):
        self.forecast_horizon = forecast_horizon
        self.models = {}
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _create_leading_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create features that LEAD regime changes (not lag them)."""
        lf = pd.DataFrame(index=features.index)

        # 1. Volatility compression (often precedes expansion)
        lf['vol_compression_ratio'] = (
            features['realized_vol_20'] /
            features['realized_vol_20'].rolling(100).max().replace(0, np.nan)
        ).fillna(0.5)

        # 2. Autocorrelation breakdown (trend exhaustion signal)
        lf['autocorr_breakdown'] = features['autocorr_1'].rolling(10).std().fillna(0)

        # 3. Volume divergence from price
        lf['volume_price_divergence'] = features['volume_ratio'] * np.sign(features['autocorr_1'])

        # 4. Bollinger Band squeeze (pre-breakout)
        lf['bb_squeeze'] = (
            features['bb_width'] /
            features['bb_width'].rolling(50).max().replace(0, np.nan)
        ).fillna(0.5)

        # 5. ADX divergence (trend weakening)
        lf['adx_divergence'] = features['adx_14'].diff(10) / 100

        # 6. Hurst exponent shift
        lf['hurst_shift'] = features['hurst_50'].diff(20).fillna(0)

        # 7. Skewness shift (tail risk building)
        lf['skew_shift'] = features['return_skew'].diff(10).fillna(0)

        # 8. Trend alignment decay
        lf['trend_decay'] = features['trend_alignment'].diff(20).fillna(0)

        # 9. Multiple timeframe disagreement
        lf['mtf_disagreement'] = (
            (features['ema20_slope'] - features['ema50_slope']).abs() /
            (features['ema20_slope'].abs() + features['ema50_slope'].abs() + 1e-10)
        )

        # 10. Distance to historical extremes
        lf['atr_percentile'] = features['atr_ratio'].rolling(200).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) > 50 else 50,
            raw=False
        ).fillna(50) / 100

        return lf.fillna(0)

    def fit(self, features: pd.DataFrame, regimes: pd.Series):
        """
        Train regime transition predictor.
        For each regime, train a binary classifier:
        "Will we be in regime X in N bars?"
        """
        leading_features = self._create_leading_features(features)
        future_regime = regimes.shift(-self.forecast_horizon)
        valid_idx = future_regime.notna()
        X = leading_features[valid_idx]
        y_future = future_regime[valid_idx]
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0)

        unique_regimes = y_future.unique()
        for regime in unique_regimes:
            y_binary = (y_future == regime).astype(int)
            n_pos = y_binary.sum()
            n_neg = len(y_binary) - n_pos
            if n_pos < 100:
                print(f"  Skipping regime {regime}: only {n_pos} positive examples")
                continue

            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )
            sample_weight = np.where(
                y_binary == 1,
                len(y_binary) / (2 * n_pos),
                len(y_binary) / (2 * n_neg)
            )
            model.fit(X_scaled, y_binary, sample_weight=sample_weight)
            self.models[regime] = model
            acc = model.score(X_scaled, y_binary)
            print(f"  Regime {regime}: accuracy={acc:.3f}, n_train={n_pos}")

        self.is_fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict probability of each regime in N bars."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        leading_features = self._create_leading_features(features)
        X_scaled = self.scaler.transform(leading_features)
        X_scaled = np.nan_to_num(X_scaled, nan=0)

        predictions = pd.DataFrame(index=features.index)
        predictions['most_likely_regime'] = 'UNKNOWN'
        predictions['prediction_confidence'] = 0.0

        all_probs = {}
        for regime, model in self.models.items():
            probs = model.predict_proba(X_scaled)[:, 1]
            predictions[f'prob_{regime}'] = probs
            all_probs[regime] = probs

        regime_cols = [c for c in predictions.columns if c.startswith('prob_')]
        if regime_cols:
            prob_matrix = predictions[regime_cols].values
            max_idx = prob_matrix.argmax(axis=1)
            max_prob = prob_matrix.max(axis=1)
            regime_names = [c.replace('prob_', '') for c in regime_cols]
            predictions['most_likely_regime'] = [regime_names[i] for i in max_idx]
            predictions['prediction_confidence'] = max_prob

        return predictions

    def save(self, path: str):
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'forecast_horizon': self.forecast_horizon,
        }, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.models = data['models']
        self.scaler = data['scaler']
        self.forecast_horizon = data['forecast_horizon']
        self.is_fitted = True
