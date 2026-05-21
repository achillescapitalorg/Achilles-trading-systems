"""
regime_integration_v2.py
==========================
Updated integration layer for VibeTrading Gold System.

This module bridges:
  - ML Ensemble (LGB + XGB + RF) predictions
  - HMM Regime Detection
  - Microstructure Features (OFI, VPIN, RV, Entropy)
  - Sentiment Analysis (5-layer hardened)
  - Risk Manager (position sizing, stops, daily limits)

Architecture:
  1. 15m Signal Generator provides directional bias (BUY/SELL/HOLD)
  2. 1m Microstructure Filter provides execution timing and quality control
  3. Risk Manager enforces position sizing, stops, and daily limits
  4. Final signal only fires when ALL layers agree

Key Changes from v1:
  - Confidence threshold raised to 0.65
  - Added OFI-proxy directional alignment filter
  - Added VPIN flow toxicity filter (< 0.6)
  - Added sign entropy market structure filter (< 0.7)
  - Added HFT activity anomaly filter
  - Integrated RiskManager for live trade execution
  - Added microstructure-based emergency exits
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Import existing project modules (adjust paths as needed)
try:
    from features.features_microstructure import MicrostructureFeatureEngine, get_microstructure_filter_mask
    from risk_manager import RiskManager, Trade
except ImportError as e:
    print(f"[Integration] Import warning: {e}. Some features may be unavailable.")
    MicrostructureFeatureEngine = None
    RiskManager = None

# Optional imports — gracefully degrade if modules unavailable
try:
    from sentiment.sentiment_hardened import HardenedSentimentAnalyzer
except ImportError:
    HardenedSentimentAnalyzer = None

try:
    import regime_integration as ri
except ImportError:
    ri = None

try:
    from beta_testing.features_15m import resample_to_15m, compute_15m_features
except ImportError:
    resample_to_15m = None
    compute_15m_features = None


@dataclass
class SignalPackage:
    """Complete signal output from the integrated system."""
    timestamp: datetime

    # Layer 1: 15m Directional Signal
    signal_15m_direction: str  # 'BUY', 'SELL', 'HOLD'
    signal_15m_strength: float   # 0.0 - 1.0

    # Layer 2: 1m ML Ensemble
    ensemble_direction: str    # 'BUY', 'SELL', 'NEUTRAL'
    ensemble_confidence: float # 0.0 - 1.0
    model_agreement: float     # 0.0 - 1.0 (how many models agree)

    # Layer 3: Regime
    current_regime: str
    regime_allows_trading: bool
    regime_position_multiplier: float

    # Layer 4: Microstructure
    ofi_proxy: float
    vpin_proxy: float
    sign_entropy: float
    hft_activity_proxy: float
    microstructure_quality: float  # 0-1 composite score

    # Layer 5: Sentiment
    sentiment_score: float     # -1.0 to 1.0
    sentiment_confidence: float
    sentiment_validated: bool

    # Risk Management
    position_size_lots: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float

    # Final Decision
    final_decision: str        # 'OPEN_LONG', 'OPEN_SHORT', 'HOLD', 'BLOCKED'
    decision_reason: str

    # Filter diagnostics (why was signal blocked?)
    filters_passed: Dict[str, bool]


class IntegratedTradingSystem:
    """
    Main trading system integrating all layers.

    Usage:
        system = IntegratedTradingSystem(account_balance=10000)
        signal = system.process_bar(df_1m, df_15m_signal, current_time)
        if signal.final_decision in ['OPEN_LONG', 'OPEN_SHORT']:
            # Execute via broker API
            pass
    """

    # FILTER THRESHOLDS (Research-backed)
    CONFIDENCE_THRESHOLD: float = 0.65
    MODEL_AGREEMENT_THRESHOLD: float = 0.67  # At least 2 of 3 models agree
    VPIN_THRESHOLD: float = 0.60
    ENTROPY_THRESHOLD: float = 0.70
    HFT_ANOMALY_MULTIPLIER: float = 3.0   # Reject if HFT proxy > 3x median

    def __init__(self, account_balance: float = 10000.0):
        # Feature engines
        self.micro_engine = MicrostructureFeatureEngine(tick_size=0.01) if MicrostructureFeatureEngine else None

        # Analysis layers
        self.sentiment_analyzer = HardenedSentimentAnalyzer() if HardenedSentimentAnalyzer else None

        # Risk manager
        self.risk_manager = RiskManager(account_balance=account_balance) if RiskManager else None

        # 15m model cache (lazy load)
        self._15m_model_cache = {
            'lgb': None,
            'xgb': None,
            'rf': None,
            'loaded': False,
        }

        # State
        self.last_15m_signal = None
        self.feature_cache = {}

    # =====================================================================
    # MAIN PROCESSING PIPELINE
    # =====================================================================
    def process_bar(
        self,
        df_1m: pd.DataFrame,
        signal_15m: Optional[Dict],
        current_time: datetime,
        live_price: float,
        precomputed_ensemble: Optional[Dict] = None
    ) -> SignalPackage:
        """
        Process a single 1m bar through the full pipeline.

        Args:
            df_1m: Recent 1m bars (last 200 rows minimum)
            signal_15m: Dict with 'direction' and 'strength' from 15m model
            current_time: Current timestamp
            live_price: Current market price
            precomputed_ensemble: Optional dict with 'direction', 'confidence', 'model_agreement'
                                  from existing dashboard signal generation

        Returns:
            SignalPackage with complete decision and diagnostics
        """
        # Update 15m signal cache
        if signal_15m is not None:
            self.last_15m_signal = signal_15m

        # Default to real 15m prediction if no external signal provided
        if self.last_15m_signal is None:
            self.last_15m_signal = self._predict_15m(df_1m)

        # Step 1: Generate microstructure features
        micro = self._analyze_microstructure(df_1m)

        # Step 2: ML Ensemble prediction (use precomputed if provided)
        if precomputed_ensemble is not None:
            ensemble_dir = precomputed_ensemble.get('direction', 'NEUTRAL')
            ensemble_conf = precomputed_ensemble.get('confidence', 0.0)
            model_agreement = precomputed_ensemble.get('model_agreement', 0.0)
        else:
            ensemble_dir, ensemble_conf, model_agreement = self._ensemble_predict(df_1m)

        # Step 3: Regime detection (delegate to existing regime_integration if available)
        regime, regime_allows, regime_mult = self._detect_regime(df_1m)

        # Step 4: Sentiment analysis
        sentiment_score, sentiment_conf, sentiment_valid = self._analyze_sentiment()

        # Step 5: Apply hardened filters
        filters_passed, decision, reason = self._apply_filters(
            ensemble_dir, ensemble_conf, model_agreement,
            regime, regime_allows, regime_mult,
            micro, sentiment_score, sentiment_valid
        )

        # Step 6: Risk management (position sizing, stops)
        position_size, stop_loss, take_profit, rr = self._calculate_risk_params(
            live_price, ensemble_dir, ensemble_conf, regime, regime_mult,
            micro, df_1m
        )

        # Step 7: Check for open trade exits
        if self.risk_manager is not None and self.risk_manager.open_trade is not None:
            exit_reason = self.risk_manager.check_exit_conditions(
                current_time, live_price, micro['ofi_proxy'], micro['vpin_proxy']
            )
            if exit_reason is not None:
                pnl = self.risk_manager.close_position(current_time, live_price, exit_reason)
                decision = 'HOLD'
                reason = f'Closed previous trade: {exit_reason} (PnL: ${pnl:.2f})'

        # Step 8: Attempt to open position if signal is valid
        if decision in ['OPEN_LONG', 'OPEN_SHORT'] and self.risk_manager is not None:
            direction = 'BUY' if decision == 'OPEN_LONG' else 'SELL'

            allowed, risk_reason = self.risk_manager.can_trade(current_time)
            if not allowed:
                decision = 'BLOCKED'
                reason = f'Risk manager blocked: {risk_reason}'
            else:
                atr_14 = self._get_atr(df_1m)
                trade = self.risk_manager.open_position(
                    current_time=current_time,
                    direction=direction,
                    entry_price=live_price,
                    atr_14=atr_14,
                    regime=regime,
                    confidence=ensemble_conf,
                    ofi_proxy=micro['ofi_proxy'],
                    vpin_proxy=micro['vpin_proxy'],
                    microstructure_quality=micro['quality_score']
                )

                if trade is None:
                    decision = 'BLOCKED'
                    reason = 'Risk manager rejected position sizing'

        # Build signal package
        signal = SignalPackage(
            timestamp=current_time,
            signal_15m_direction=self.last_15m_signal['direction'],
            signal_15m_strength=self.last_15m_signal['strength'],
            ensemble_direction=ensemble_dir,
            ensemble_confidence=ensemble_conf,
            model_agreement=model_agreement,
            current_regime=regime,
            regime_allows_trading=regime_allows,
            regime_position_multiplier=regime_mult,
            ofi_proxy=micro['ofi_proxy'],
            vpin_proxy=micro['vpin_proxy'],
            sign_entropy=micro['sign_entropy'],
            hft_activity_proxy=micro['hft_activity'],
            microstructure_quality=micro['quality_score'],
            sentiment_score=sentiment_score,
            sentiment_confidence=sentiment_conf,
            sentiment_validated=sentiment_valid,
            position_size_lots=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=rr,
            final_decision=decision,
            decision_reason=reason,
            filters_passed=filters_passed
        )

        return signal

    # =====================================================================
    # UTILITY: ATR helper
    # =====================================================================
    def _get_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Compute latest ATR from OHLCV dataframe."""
        try:
            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift()).abs(),
                (df['low'] - df['close'].shift()).abs()
            ], axis=1).max(axis=1)
            return tr.rolling(period, min_periods=1).mean().iloc[-1]
        except Exception:
            return 0.5

    # =====================================================================
    # 15M SIGNAL GENERATION
    # =====================================================================
    def _load_15m_models(self):
        """Lazy-load 15m LGB/XGB/RF models."""
        if self._15m_model_cache['loaded']:
            return

        PROJECT_ROOT = Path(__file__).parent.resolve()
        MODEL_DIR = PROJECT_ROOT / "data" / "beta_testing" / "processed" / "models"

        try:
            from beta_testing.models.lgb_model import Gold1mLightGBM
            from beta_testing.models.xgb_model import Gold1mXGBoost
            from beta_testing.models.rf_model import Gold1mRandomForest

            lgb_path = MODEL_DIR / 'gold_15m_lgb.pkl'
            xgb_path = MODEL_DIR / 'gold_15m_xgb.ubj'
            rf_path = MODEL_DIR / 'gold_15m_rf.pkl'

            if lgb_path.exists():
                m = Gold1mLightGBM()
                m.load(str(lgb_path))
                self._15m_model_cache['lgb'] = m

            if xgb_path.exists():
                m = Gold1mXGBoost()
                m.load(str(xgb_path))
                self._15m_model_cache['xgb'] = m

            if rf_path.exists():
                m = Gold1mRandomForest()
                m.load(str(rf_path))
                self._15m_model_cache['rf'] = m

            self._15m_model_cache['loaded'] = True
            print(f"[15m] Models loaded from {MODEL_DIR}")
        except Exception as e:
            print(f"[15m] Model load error: {e}")

    def _predict_15m(self, df_1m: pd.DataFrame) -> Dict:
        """
        Generate 15m directional bias from latest 1m bars.

        Returns:
            {'direction': 'BUY'|'SELL'|'HOLD', 'strength': float}
        """
        if resample_to_15m is None or compute_15m_features is None:
            return {'direction': 'HOLD', 'strength': 0.0}

        self._load_15m_models()
        if not any(self._15m_model_cache.get(k) for k in ['lgb', 'xgb', 'rf']):
            return {'direction': 'HOLD', 'strength': 0.0}

        try:
            # Need enough 1m bars to form meaningful 15m bars
            if len(df_1m) < 200:
                return {'direction': 'HOLD', 'strength': 0.0}

            # Resample last ~500 bars to 15m
            df_15m = resample_to_15m(df_1m.tail(500))
            if len(df_15m) < 10:
                return {'direction': 'HOLD', 'strength': 0.0}

            features = compute_15m_features(df_15m)

            # Drop target columns if present
            target_cols = [c for c in features.columns if c.startswith('target_')]
            X = features.drop(columns=target_cols, errors='ignore')
            X_latest = X.iloc[[-1]]

            preds = {}
            if self._15m_model_cache['lgb'] is not None:
                preds['lgb'] = float(self._15m_model_cache['lgb'].predict(X_latest)[0])
            if self._15m_model_cache['xgb'] is not None:
                preds['xgb'] = float(self._15m_model_cache['xgb'].predict(X_latest)[0])
            if self._15m_model_cache['rf'] is not None:
                preds['rf'] = float(self._15m_model_cache['rf'].predict(X_latest)[0])

            if not preds:
                return {'direction': 'HOLD', 'strength': 0.0}

            avg_prob = np.mean(list(preds.values()))
            direction = 'BUY' if avg_prob > 0.55 else 'SELL' if avg_prob < 0.45 else 'HOLD'
            strength = abs(avg_prob - 0.5) * 2

            return {'direction': direction, 'strength': strength}

        except Exception as e:
            print(f"[15m] Prediction error: {e}")
            return {'direction': 'HOLD', 'strength': 0.0}

    # =====================================================================
    # STEP 2: ENSEMBLE PREDICTION (fallback if no precomputed)
    # =====================================================================
    def _ensemble_predict(self, df_1m: pd.DataFrame) -> Tuple[str, float, float]:
        """
        Load pre-trained models and generate ensemble prediction.

        Returns:
            (direction, confidence, model_agreement)
        """
        # NOTE: In the VibeTrading dashboard, predictions are pre-computed
        # via _model_cache and passed as precomputed_ensemble. This fallback
        # only runs when no precomputed ensemble is provided.
        try:
            import joblib
            import xgboost as xgb

            PROJECT_ROOT = Path(__file__).parent.resolve()
            MODEL_DIR = PROJECT_ROOT / "data" / "beta_testing" / "processed" / "models"

            # Try loading via custom wrappers first (project-specific)
            try:
                from beta_testing.models.lgb_model import Gold1mLightGBM
                from beta_testing.models.xgb_model import Gold1mXGBoost
                from beta_testing.models.rf_model import Gold1mRandomForest

                lgb_m = Gold1mLightGBM()
                xgb_m = Gold1mXGBoost()
                rf_m = Gold1mRandomForest()

                lgb_m.load(str(MODEL_DIR / 'beta_h20_lgb.pkl'))
                xgb_m.load(str(MODEL_DIR / 'beta_h20_xgb.ubj'))
                rf_m.load(str(MODEL_DIR / 'beta_h20_rf.pkl'))

                # Use the latest row's numeric features only (exclude targets)
                exclude = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
                X = df_1m.select_dtypes(include=[np.number]).iloc[[-1]].drop(
                    columns=[c for c in exclude if c in df_1m.columns], errors='ignore'
                )

                pred_lgb = float(lgb_m.predict(X)[0])
                pred_xgb = float(xgb_m.predict(X)[0])
                pred_rf = float(rf_m.predict(X)[0])
            except Exception:
                # Fallback: try direct sklearn/xgboost loading
                lgb_h20 = joblib.load(MODEL_DIR / 'beta_h20_lgb.pkl')
                xgb_h20 = xgb.Booster()
                xgb_h20.load_model(str(MODEL_DIR / 'beta_h20_xgb.ubj'))
                rf_h20 = joblib.load(MODEL_DIR / 'beta_h20_rf.pkl')

                exclude = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
                X = df_1m.select_dtypes(include=[np.number]).iloc[[-1]].drop(
                    columns=[c for c in exclude if c in df_1m.columns], errors='ignore'
                )

                pred_lgb = lgb_h20.predict_proba(X)[0][1]
                pred_xgb = xgb_h20.predict(xgb.DMatrix(X))[0]
                pred_rf = rf_h20.predict_proba(X)[0][1]

            ensemble_prob = (pred_lgb + pred_xgb + pred_rf) / 3.0

            if ensemble_prob > 0.55:
                direction = 'BUY'
            elif ensemble_prob < 0.45:
                direction = 'SELL'
            else:
                direction = 'NEUTRAL'

            confidence = abs(ensemble_prob - 0.5) * 2

            votes = [
                1 if pred_lgb > 0.5 else -1,
                1 if pred_xgb > 0.5 else -1,
                1 if pred_rf > 0.5 else -1
            ]
            ensemble_vote = 1 if ensemble_prob > 0.5 else -1
            agreement = sum(1 for v in votes if v == ensemble_vote) / 3.0

            return direction, confidence, agreement

        except Exception as e:
            print(f"[Ensemble] Prediction error: {e}")
            return 'NEUTRAL', 0.0, 0.0

    # =====================================================================
    # STEP 3: REGIME DETECTION
    # =====================================================================
    def _detect_regime(self, df_1m: pd.DataFrame) -> Tuple[str, bool, float]:
        """
        Detect current market regime.

        Returns:
            (regime_name, allows_trading, position_multiplier)
        """
        try:
            if ri is not None:
                # Use existing regime_integration bridge
                regime_pred = ri.get_regime_prediction(df_1m)
                if regime_pred is not None:
                    regime = regime_pred.get('current_regime', 'UNKNOWN')
                    allows = regime_pred.get('trading_status', 'BLOCKED') == 'ACTIVE'
                    mult = RiskManager.REGIME_MULTIPLIERS.get(regime, 0.25) if RiskManager else 0.25
                    return regime, allows, mult

            # Fallback: simple volatility-based regime
            close = df_1m['close']
            atr = self._get_atr(df_1m)
            atr_mean = close.rolling(20, min_periods=1).std().iloc[-1]

            if atr > atr_mean * 2:
                regime = 'HIGH_VOL_CHAOS'
            elif atr < atr_mean * 0.3:
                regime = 'LOW_VOL_DRIFT'
            elif close.iloc[-1] > close.rolling(50, min_periods=1).mean().iloc[-1]:
                regime = 'STRONG_TREND_UP'
            else:
                regime = 'CHOPPY'

            blocked_regimes = ['HIGH_VOL_CHAOS', 'LOW_VOL_DRIFT']
            allows = regime not in blocked_regimes
            mult = RiskManager.REGIME_MULTIPLIERS.get(regime, 0.25) if RiskManager else 0.25
            return regime, allows, mult

        except Exception as e:
            print(f"[Regime] Detection error: {e}")
            return 'UNKNOWN', False, 0.0

    # =====================================================================
    # STEP 4: MICROSTRUCTURE ANALYSIS
    # =====================================================================
    def _analyze_microstructure(self, df_1m: pd.DataFrame) -> Dict:
        """Extract microstructure signals from latest bars."""
        if self.micro_engine is None:
            return {
                'ofi_proxy': 0.0,
                'vpin_proxy': 0.5,
                'sign_entropy': 0.5,
                'hft_activity': 1.0,
                'quality_score': 0.5,
                'rv_parkinson': 0,
                'micro_return': 0
            }

        micro = self.micro_engine.generate_all_features(df_1m)
        latest = micro.iloc[-1]

        vpin_score = max(0, 1 - latest['vpin_proxy'] / self.VPIN_THRESHOLD)
        entropy_score = max(0, 1 - latest['sign_entropy'] / self.ENTROPY_THRESHOLD)

        hft_median = micro['hft_activity_proxy'].rolling(100, min_periods=10).median().iloc[-1]
        hft_score = 1.0 if latest['hft_activity_proxy'] < hft_median * self.HFT_ANOMALY_MULTIPLIER else 0.0

        quality = (vpin_score + entropy_score + hft_score) / 3.0

        return {
            'ofi_proxy': latest['ofi_proxy'],
            'vpin_proxy': latest['vpin_proxy'],
            'sign_entropy': latest['sign_entropy'],
            'hft_activity': latest['hft_activity_proxy'],
            'quality_score': quality,
            'rv_parkinson': latest.get('rv_parkinson', 0),
            'micro_return': latest.get('micro_return', 0)
        }

    # =====================================================================
    # STEP 5: SENTIMENT ANALYSIS
    # =====================================================================
    def _analyze_sentiment(self) -> Tuple[float, float, bool]:
        """
        Get hardened sentiment score.

        Returns:
            (score, confidence, validated)
        """
        try:
            if self.sentiment_analyzer is not None:
                # The actual API requires headlines_df and price_df.
                # In standalone mode we don't have these, so return neutral.
                return 0.0, 0.0, False
            else:
                return 0.0, 0.0, False
        except Exception as e:
            print(f"[Sentiment] Analysis error: {e}")
            return 0.0, 0.0, False

    # =====================================================================
    # STEP 6: HARDENED FILTERS
    # =====================================================================
    def _apply_filters(
        self,
        ensemble_dir: str,
        ensemble_conf: float,
        model_agreement: float,
        regime: str,
        regime_allows: bool,
        regime_mult: float,
        micro: Dict,
        sentiment_score: float,
        sentiment_valid: bool
    ) -> Tuple[Dict, str, str]:
        """
        Apply all research-backed filters.

        Returns:
            (filters_dict, decision, reason)
        """
        filters = {
            '15m_signal_active': self.last_15m_signal['direction'] in ['BUY', 'SELL'],
            'ensemble_confident': ensemble_conf >= self.CONFIDENCE_THRESHOLD,
            'models_agree': model_agreement >= self.MODEL_AGREEMENT_THRESHOLD,
            'regime_allows': regime_allows,
            'ofi_aligned': False,
            'vpin_safe': micro['vpin_proxy'] < self.VPIN_THRESHOLD,
            'entropy_structured': micro['sign_entropy'] < self.ENTROPY_THRESHOLD,
            'hft_normal': micro['hft_activity'] < (micro.get('hft_median', 1.0) * self.HFT_ANOMALY_MULTIPLIER),
            'sentiment_valid': sentiment_valid,
            'sentiment_aligned': False
        }

        # OFI alignment: microstructure pressure confirms ensemble direction
        if ensemble_dir == 'BUY' and micro['ofi_proxy'] > 0:
            filters['ofi_aligned'] = True
        elif ensemble_dir == 'SELL' and micro['ofi_proxy'] < 0:
            filters['ofi_aligned'] = True

        # Sentiment alignment
        if sentiment_valid:
            if ensemble_dir == 'BUY' and sentiment_score > -0.3:
                filters['sentiment_aligned'] = True
            elif ensemble_dir == 'SELL' and sentiment_score < 0.3:
                filters['sentiment_aligned'] = True
            elif abs(sentiment_score) < 0.3:
                filters['sentiment_aligned'] = True
        else:
            filters['sentiment_aligned'] = True

        critical_filters = [
            '15m_signal_active',
            'ensemble_confident',
            'models_agree',
            'regime_allows',
            'ofi_aligned',
            'vpin_safe',
            'entropy_structured'
        ]

        failed = [f for f in critical_filters if not filters[f]]

        if failed:
            decision = 'BLOCKED'
            reason = f"Failed filters: {', '.join(failed)}"
        else:
            if ensemble_dir == 'BUY':
                decision = 'OPEN_LONG'
            elif ensemble_dir == 'SELL':
                decision = 'OPEN_SHORT'
            else:
                decision = 'HOLD'
                reason = 'Ensemble direction neutral'

            if decision != 'HOLD':
                reason = (f"All critical filters passed | Conf:{ensemble_conf:.2f} | "
                          f"Regime:{regime} | OFI:{micro['ofi_proxy']:.2f} | "
                          f"VPIN:{micro['vpin_proxy']:.2f} | Entropy:{micro['sign_entropy']:.2f}")

        return filters, decision, reason

    # =====================================================================
    # STEP 7: RISK PARAMETER CALCULATION
    # =====================================================================
    def _calculate_risk_params(
        self,
        live_price: float,
        ensemble_dir: str,
        ensemble_conf: float,
        regime: str,
        regime_mult: float,
        micro: Dict,
        df_1m: pd.DataFrame
    ) -> Tuple[float, float, float, float]:
        """
        Calculate position size, stop loss, take profit, and R:R ratio.

        Returns:
            (size_lots, stop_loss, take_profit, rr_ratio)
        """
        if ensemble_dir == 'NEUTRAL' or self.risk_manager is None:
            return 0.0, 0.0, 0.0, 0.0

        atr_14 = self._get_atr(df_1m)

        direction = 'BUY' if ensemble_dir == 'BUY' else 'SELL'
        stop_loss, take_profit = self.risk_manager.calculate_stops(
            entry_price=live_price,
            direction=direction,
            atr_14=atr_14,
            regime=regime,
            ofi_proxy=micro['ofi_proxy']
        )

        size = self.risk_manager.calculate_position_size(
            entry_price=live_price,
            stop_loss=stop_loss,
            regime=regime,
            confidence=ensemble_conf,
            microstructure_quality=micro['quality_score']
        )

        risk = abs(live_price - stop_loss)
        reward = abs(take_profit - live_price)
        rr = reward / risk if risk > 0 else 0

        return size, stop_loss, take_profit, rr

    # =====================================================================
    # DASHBOARD DATA HELPERS
    # =====================================================================
    def get_dashboard_data(self) -> Dict:
        """Get formatted data for the Dash dashboard."""
        if self.risk_manager is None:
            return {
                'account_balance': 0.0,
                'daily_pnl': 0.0,
                'win_rate': 0.0,
                'max_drawdown_pct': 0.0,
                'consecutive_losses': 0,
                'trading_halted': False,
                'halt_reason': '',
                'total_trades': 0,
                'sharpe_proxy': 0.0
            }
        risk_metrics = self.risk_manager.get_metrics()
        return {
            'account_balance': risk_metrics['account_balance'],
            'daily_pnl': risk_metrics['daily_pnl'],
            'win_rate': risk_metrics['win_rate'],
            'max_drawdown_pct': risk_metrics['max_drawdown_pct'],
            'consecutive_losses': risk_metrics['consecutive_losses'],
            'trading_halted': risk_metrics['trading_halted'],
            'halt_reason': risk_metrics['halt_reason'],
            'total_trades': risk_metrics['total_trades'],
            'sharpe_proxy': risk_metrics['sharpe_proxy']
        }

    def get_trade_history_df(self) -> pd.DataFrame:
        """Get trade history as DataFrame for dashboard."""
        if self.risk_manager is None:
            return pd.DataFrame()
        return self.risk_manager.get_trade_log()


# =====================================================================
# BACKWARD-COMPATIBILITY: predict() wrapper matching regime_integration.py
# =====================================================================

def predict(df: pd.DataFrame, raw_signal: dict = None) -> Optional[Dict]:
    """
    Backward-compatible predict() that enriches regime_integration output
    with microstructure data.

    This allows beta_dashboard.py to call `ri_v2.predict(df, raw_signal=...)`
    and get enhanced results.
    """
    # Start with existing regime_integration result if available
    if ri is not None and hasattr(ri, 'predict'):
        base = ri.predict(df, raw_signal=raw_signal)
    elif ri is not None and hasattr(ri, 'get_regime_prediction'):
        base = ri.get_regime_prediction(df)
    else:
        base = {}

    if base is None:
        base = {}

    # Add microstructure if we can compute it
    try:
        micro_engine = MicrostructureFeatureEngine(tick_size=0.01) if MicrostructureFeatureEngine else None
        if micro_engine is not None:
            micro = micro_engine.generate_all_features(df)
            latest = micro.iloc[-1]
            base['ofi_proxy'] = latest['ofi_proxy']
            base['vpin_proxy'] = latest['vpin_proxy']
            base['sign_entropy'] = latest['sign_entropy']
            base['hft_activity_proxy'] = latest['hft_activity_proxy']
            base['microstructure_quality'] = (
                max(0, 1 - latest['vpin_proxy'] / 0.6) +
                max(0, 1 - latest['sign_entropy'] / 0.7)
            ) / 2.0
    except Exception as e:
        print(f"[regime_integration_v2] Microstructure enrich error: {e}")

    return base
