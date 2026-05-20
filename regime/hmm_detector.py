"""
HMM Regime Detector
===================
Hidden Markov Model for detecting 6 gold market regimes.
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')


class HMMRegimeDetector:
    """
    Hidden Markov Model for regime detection.
    Learns 6 hidden states corresponding to gold market regimes.
    """

    REGIME_NAMES = {
        0: 'CHOPPY_RANGE',
        1: 'STRONG_TREND_UP',
        2: 'STRONG_TREND_DOWN',
        3: 'HIGH_VOL_CHAOS',
        4: 'GRIND_UP',
        5: 'LOW_VOL_DRIFT',
    }

    def __init__(self, n_regimes: int = 6, n_iter: int = 200,
                 random_state: int = 42, cov_type: str = 'full'):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.random_state = random_state
        self.cov_type = cov_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.regime_map = None

    def fit(self, features: pd.DataFrame, verbose: bool = True) -> 'HMMRegimeDetector':
        """
        Fit HMM on regime features.
        Strategy: Train multiple HMMs with different random states
        and select the one with the best BIC.
        """
        self.feature_names = [
            'atr_ratio', 'vol_percentile',
            'adx_14', 'ema20_slope', 'ema50_slope',
            'trend_alignment', 'dist_from_ema200',
            'bb_position', 'bb_width',
            'autocorr_1', 'hurst_50',
            'volume_ratio', 'return_skew',
        ]
        # Filter to available columns
        available = [c for c in self.feature_names if c in features.columns]
        if len(available) < 5:
            raise ValueError(f"Too few features available: {available}")
        self.feature_names = available

        X = features[self.feature_names].values
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=5.0, neginf=-5.0)

        best_bic = np.inf
        best_model = None

        for seed in range(self.random_state, self.random_state + 5):
            model = GaussianHMM(
                n_components=self.n_regimes,
                n_iter=self.n_iter,
                random_state=seed,
                covariance_type=self.cov_type,
                tol=1e-4,
                init_params='kmeans',
            )
            try:
                model.fit(X_scaled)
                log_likelihood = model.score(X_scaled)
                n_params = self._count_hmm_params(model)
                bic = n_params * np.log(len(X_scaled)) - 2 * log_likelihood
                if bic < best_bic:
                    best_bic = bic
                    best_model = model
                if verbose:
                    print(f"  Seed {seed}: BIC={bic:.1f}, LL={log_likelihood:.1f}")
            except Exception as e:
                if verbose:
                    print(f"  Seed {seed}: Failed ({e})")
                continue

        if best_model is None:
            raise RuntimeError("All HMM fits failed")

        self.model = best_model
        self.regime_map = self._map_states_to_regimes(features, X_scaled)
        return self

    def _count_hmm_params(self, model) -> int:
        """Count number of free parameters in HMM."""
        n = model.n_components
        d = model.means_.shape[1]
        return (n * d + n * d * (d + 1) // 2 + n * (n - 1) + n - 1)

    def _map_states_to_regimes(self, features: pd.DataFrame, X_scaled: np.ndarray) -> dict:
        """
        Map HMM states to meaningful regime names based on state characteristics.
        """
        states = self.model.predict(X_scaled)
        state_profiles = {}
        for s in range(self.n_regimes):
            mask = states == s
            if mask.sum() < 100:
                continue
            profile = features[self.feature_names].iloc[mask].mean()
            state_profiles[s] = profile

        mapping = {}
        for s, profile in state_profiles.items():
            vol = profile.get('vol_percentile', 50)
            adx = profile.get('adx_14', 20)
            slope = profile.get('ema20_slope', 0)
            autocorr = profile.get('autocorr_1', 0)

            if vol > 70 and adx > 20:
                regime = 3  # HIGH_VOL_CHAOS
            elif adx > 25 and slope > 0.3:
                regime = 1  # STRONG_TREND_UP
            elif adx > 25 and slope < -0.3:
                regime = 2  # STRONG_TREND_DOWN
            elif vol < 30 and abs(slope) < 0.1:
                regime = 5  # LOW_VOL_DRIFT
            elif slope > 0.1 and vol < 60:
                regime = 4  # GRIND_UP
            else:
                regime = 0  # CHOPPY_RANGE
            mapping[s] = regime
        return mapping

    def predict_regime(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict regime for new data."""
        X = features[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=5.0, neginf=-5.0)

        states = self.model.predict(X_scaled)
        regimes = [self.regime_map.get(s, 0) for s in states]
        state_probs = self.model.predict_proba(X_scaled)

        result = pd.DataFrame(index=features.index)
        result['regime'] = [self.REGIME_NAMES.get(r, 'UNKNOWN') for r in regimes]
        result['regime_confidence'] = state_probs.max(axis=1)

        for i, name in self.REGIME_NAMES.items():
            result[f'prob_{name}'] = 0.0
        for hmm_state, regime_id in self.regime_map.items():
            regime_name = self.REGIME_NAMES.get(regime_id, 'UNKNOWN')
            result[f'prob_{regime_name}'] += state_probs[:, hmm_state]
        return result

    def regime_transition_matrix(self) -> pd.DataFrame:
        """Get empirical regime transition probabilities."""
        return pd.DataFrame(
            self.model.transmat_,
            columns=[f'State_{i}' for i in range(self.n_regimes)],
            index=[f'State_{i}' for i in range(self.n_regimes)]
        )

    def save(self, path: str):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'regime_map': self.regime_map,
        }, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.regime_map = data['regime_map']
