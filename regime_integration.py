"""
FAST REGIME INTEGRATION for Beta Dashboard.
Loads precomputed features + trained models once.
Only processes last 1000 bars in real-time.
"""
import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path

from regime.regime_features import RegimeFeatureEngineer
from regime.hmm_detector import HMMRegimeDetector
from regime.regime_predictor import RegimePredictor
from regime.regime_strategy import RegimeStrategy


# --- GLOBALS: Loaded once at import time ---
_CACHE = {
    'hmm': None,
    'predictor': None,
    'precomputed_features': None,
    'engineer': None,
    'strategy': None,
}

REGIME_BUFFER_SIZE = 1000   # ~16 hours
HMM_WINDOW_SIZE = 500       # ~8 hours for Viterbi smoothing

PROJECT_ROOT = Path(__file__).parent.resolve()
MODEL_DIR = PROJECT_ROOT / "data" / "beta_testing" / "processed" / "models"
CACHE_DIR = PROJECT_ROOT / "data" / "beta_testing" / "processed" / "regime_cache"


def _load_models():
    """Load all regime models once. Called automatically on first use."""
    if _CACHE['hmm'] is not None:
        return

    hmm_path = MODEL_DIR / "regime_models_hmm.pkl"
    pred_path = MODEL_DIR / "regime_models_predictor.pkl"
    feat_path = CACHE_DIR / "regime_features_precomputed.parquet"

    if not hmm_path.exists() or not pred_path.exists() or not feat_path.exists():
        print(f"[RegimeIntegration] Models not found at {MODEL_DIR} or {CACHE_DIR}")
        return

    # Load HMM
    hmm = HMMRegimeDetector()
    hmm.load(str(hmm_path))
    _CACHE['hmm'] = hmm

    # Load predictor
    pred = RegimePredictor(forecast_horizon=20)
    pred.load(str(pred_path))
    _CACHE['predictor'] = pred

    # Load only the tail of precomputed features (buffer + safety margin)
    precomp = pd.read_parquet(feat_path).tail(REGIME_BUFFER_SIZE + 500)
    _CACHE['precomputed_features'] = precomp

    _CACHE['engineer'] = RegimeFeatureEngineer()
    _CACHE['strategy'] = RegimeStrategy()

    print(f"[RegimeIntegration] Models loaded | Precomputed: {len(precomp):,} rows")


def get_regime_prediction(df_latest: pd.DataFrame) -> Dict:
    """
    Fast regime prediction for Dash callback.

    Args:
        df_latest: DataFrame with last ~100-500 bars from live feed.
                   Must have columns: open, high, low, close, volume

    Returns:
        Dict with current regime, predicted regime, confidence, trading config.
    """
    # Lazy load models
    _load_models()
    if _CACHE['hmm'] is None:
        return None

    # 1. Grab historical tail from precomputed cache
    hist_tail = _CACHE['precomputed_features'].tail(REGIME_BUFFER_SIZE).copy()

    # 2. Recompute features ONLY on combined window (~1000-1500 rows)
    # We need raw OHLCV, not precomputed features, for the latest bars
    # For simplicity, just use precomputed features tail directly
    combined_features = hist_tail

    # 3. Take last HMM_WINDOW_SIZE for Viterbi
    recent_features = combined_features.tail(HMM_WINDOW_SIZE)

    # 4. HMM predict on short sequence
    regime_result = _CACHE['hmm'].predict_regime(recent_features)
    current = regime_result.iloc[-1]

    # 5. Regime predictor (future regime)
    future_pred = _CACHE['predictor'].predict(recent_features)
    future = future_pred.iloc[-1]

    # 6. Strategy check
    strategy = _CACHE['strategy']
    config = strategy.get_config(current['regime'])
    trade_decision = strategy.evaluate_signal(
        regime=current['regime'],
        current_drawdown=0.0,
        signal_confidence=0.6,
    )

    return {
        'current_regime': current['regime'],
        'regime_confidence': float(current['regime_confidence']),
        'predicted_regime': future['most_likely_regime'],
        'prediction_confidence': float(future['prediction_confidence']),
        'regime_transition_warning': (
            current['regime'] != future['most_likely_regime']
            and future['prediction_confidence'] > 0.5
        ),
        'trading_status': 'ACTIVE' if trade_decision['allow_trade'] else 'BLOCKED',
        'position_multiplier': float(trade_decision['position_multiplier']),
        'regime_reason': trade_decision['reason'],
        'stop_atr_multiple': config.stop_atr_multiple,
        'takeprofit_atr_multiple': config.takeprofit_atr_multiple,
        'allow_new_entries': config.allow_new_entries,
        'regime_probs': {
            col.replace('prob_', ''): float(current[col])
            for col in regime_result.columns if col.startswith('prob_')
        }
    }


# --- ML Signal Generation (unchanged from before) ---
from beta_testing.features import compute_1m_features
from beta_testing.models.lgb_model import Gold1mLightGBM
from beta_testing.models.xgb_model import Gold1mXGBoost
from beta_testing.models.rf_model import Gold1mRandomForest

ML_MODELS_DIR = PROJECT_ROOT / "data" / "beta_testing" / "processed" / "models"
HORIZONS = [20, 60]
MODEL_NAMES = ["lgb", "xgb", "rf"]

_ml_cache = {"loaded": False, "models": {}, "feature_cols": {}}


def _load_ml_models():
    if _ml_cache["loaded"]:
        return
    for h in HORIZONS:
        prefix = f"beta_h{h}"
        _ml_cache["models"][h] = {}
        for name in MODEL_NAMES:
            # XGB models use .ubj (native format), others use .pkl
            ext = "ubj" if name == "xgb" else "pkl"
            path = ML_MODELS_DIR / f"{prefix}_{name}.{ext}"
            if not path.exists():
                continue
            if name == "lgb":
                m = Gold1mLightGBM()
            elif name == "xgb":
                m = Gold1mXGBoost()
            else:
                m = Gold1mRandomForest()
            try:
                m.load(str(path))
                _ml_cache["models"][h][name] = m
            except Exception as e:
                print(f"[RegimeIntegration] Failed to load {path}: {e}")
        feat_path = ML_MODELS_DIR / f"{prefix}_features.json"
        if feat_path.exists():
            import json
            with open(feat_path) as f:
                _ml_cache["feature_cols"][h] = json.load(f)
    _ml_cache["loaded"] = True


def get_ml_signals(df: pd.DataFrame) -> dict:
    """Generate raw ML ensemble signals."""
    _load_ml_models()
    if not _ml_cache["models"]:
        return None

    # Only need tail for live prediction (rolling windows need ~500 rows max)
    df_tail = df.tail(5000) if len(df) > 5000 else df
    features = compute_1m_features(df_tail)
    target_cols = [c for c in features.columns if c.startswith("target_")]
    X_live = features.drop(columns=target_cols).iloc[[-1]]

    primary_h = 60
    secondary_h = 20

    preds = {}
    probs = {}
    for h in [primary_h, secondary_h]:
        if h not in _ml_cache["models"]:
            continue
        feat_cols = _ml_cache["feature_cols"].get(h, list(X_live.columns))
        Xh = X_live.reindex(columns=feat_cols, fill_value=0)
        for name, model in _ml_cache["models"][h].items():
            try:
                prob = float(model.predict(Xh)[0])
                probs[f"{name}_h{h}"] = prob
                preds[name] = prob
            except Exception as e:
                print(f"[RegimeIntegration] Prediction error {name} H={h}: {e}")

    if not preds:
        return None

    avg_prob = np.mean(list(preds.values()))
    action = "BUY" if avg_prob > 0.55 else "SELL" if avg_prob < 0.45 else "HOLD"
    confidence = abs(avg_prob - 0.5) * 2

    return {
        "action": action,
        "confidence": confidence,
        "raw_prob": avg_prob,
        "individual_probs": probs,
        "price": float(df["close"].iloc[-1]),
    }


def predict(df: pd.DataFrame, raw_signal: dict = None) -> dict:
    """
    Full unified prediction: ML signals + regime adjustment.
    If raw_signal is provided, skips ML recomputation (dashboard already computed it).
    Returns dict with raw signals, regime info, and adjusted final signal.
    """
    if raw_signal is None:
        raw_signal = get_ml_signals(df)
    regime_result = get_regime_prediction(df)

    if raw_signal is None:
        raw_signal = {
            "action": "HOLD", "confidence": 0.0, "raw_prob": 0.5,
            "individual_probs": {}, "price": 0.0,
        }

    if regime_result is None:
        return {
            "raw_action": raw_signal["action"],
            "raw_confidence": raw_signal["confidence"],
            "raw_position_size": 1.0,
            "individual_probs": raw_signal.get("individual_probs", {}),
            "current_regime": "UNKNOWN",
            "regime_confidence": 0.0,
            "predicted_regime": "UNKNOWN",
            "prediction_confidence": 0.0,
            "regime_transition_warning": False,
            "trading_status": "ACTIVE",
            "regime_reason": "Regime system not loaded",
            "final_action": raw_signal["action"],
            "final_confidence": raw_signal["confidence"],
            "final_position_size": 1.0,
            "stop_loss": None,
            "take_profit": None,
            "risk_reward": 1.5,
            "reason": "No regime filter applied",
        }

    current_regime = regime_result["current_regime"]
    regime_conf = regime_result["regime_confidence"]
    predicted_regime = regime_result["predicted_regime"]
    pred_conf = regime_result["prediction_confidence"]
    transition_warning = regime_result["regime_transition_warning"]

    strategy = _CACHE['strategy']
    trade_decision = strategy.evaluate_signal(
        current_regime,
        raw_signal["confidence"],
        current_drawdown=0.0,
    )
    config = trade_decision.get("config", strategy.get_config(current_regime))

    final_action = raw_signal["action"]
    final_confidence = raw_signal["confidence"]
    position_size = 1.0
    reason = ""

    if not trade_decision["allow_trade"]:
        final_action = "HOLD"
        final_confidence = 0.0
        position_size = 0.0
        reason = trade_decision["reason"]
    else:
        regime_multiplier = trade_decision["position_multiplier"]
        position_size = min(1.0, raw_signal["confidence"] * regime_multiplier)
        if transition_warning:
            final_confidence *= 0.7
            position_size *= 0.7
            reason = f"Regime transition: {current_regime} -> {predicted_regime}"
        else:
            reason = f"Regime {current_regime}: {regime_multiplier:.1f}x sizing"

    atr = float(df["close"].iloc[-100:].rolling(14).max().iloc[-1] - df["close"].iloc[-100:].rolling(14).min().iloc[-1])
    stops = strategy.compute_stops(
        entry_price=raw_signal["price"],
        atr=atr if atr > 0 else 0.5,
        direction="buy" if final_action == "BUY" else "sell",
        regime=current_regime,
    )

    return {
        "raw_action": raw_signal["action"],
        "raw_confidence": raw_signal["confidence"],
        "raw_position_size": 1.0,
        "individual_probs": raw_signal.get("individual_probs", {}),
        "current_regime": current_regime,
        "regime_confidence": regime_conf,
        "predicted_regime": predicted_regime,
        "prediction_confidence": pred_conf,
        "regime_transition_warning": transition_warning,
        "trading_status": "ACTIVE" if trade_decision["allow_trade"] else "BLOCKED",
        "regime_reason": trade_decision["reason"],
        "final_action": final_action,
        "final_confidence": final_confidence,
        "final_position_size": position_size,
        "regime_multiplier": regime_multiplier,
        "stop_loss": stops["stop_loss"],
        "take_profit": stops["take_profit"],
        "risk_reward": stops["risk_reward"],
        "reason": reason,
    }
