"""
Market Regime Detection & Prediction System
============================================
6-regime detection for gold (XAU/USD) 1-minute trading.
"""
from .regime_features import RegimeFeatureEngineer
from .hmm_detector import HMMRegimeDetector
from .change_point_detector import ChangePointDetector
from .regime_predictor import RegimePredictor
from .regime_strategy import RegimeStrategy, TradingConfig
from .regime_pipeline import RegimeAwareTradingSystem

__all__ = [
    "RegimeFeatureEngineer",
    "HMMRegimeDetector",
    "ChangePointDetector",
    "RegimePredictor",
    "RegimeStrategy",
    "TradingConfig",
    "RegimeAwareTradingSystem",
]
