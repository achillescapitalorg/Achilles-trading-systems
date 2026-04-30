"""
Gold RL Trainer — 1-Minute XAUUSD Dueling Double DQN
=====================================================
Improved version with:
  - StandardScaler feature normalization
  - Supervised pre-training on 5-bar direction labels (massive accuracy boost)
  - Dueling Double DQN with dropout + LayerNorm (resists overfitting)
  - Direct PnL reward (basis-point scaled) — clearer learning signal
  - Held-out test set (not seen during training or validation)
  - Continuous direction prediction (always BUY or SELL with confidence)
"""

import os
import json
import time
import pickle
import threading
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

warnings.filterwarnings("ignore")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from services.gold_rl_env import GoldTradingEnv
from services.gold_rl_dueling import GoldDuelingAgent, TORCH_AVAILABLE as DUEL_TORCH
from services.gold_rl_seq_agent import SequenceAgent, TORCH_AVAILABLE as SEQ_TORCH


# Action mapping: env actions are already 0=HOLD, 1=BUY, 2=SELL
ACTION_HOLD, ACTION_BUY, ACTION_SELL = 0, 1, 2
_ENV_TO_STR = {ACTION_HOLD: "HOLD", ACTION_BUY: "BUY", ACTION_SELL: "SELL"}


class GoldRLTrainer:
    """
    Orchestrates gold 1m DQN training:
      1. Fetch real 1m gold data (chunked)
      2. Engineer features (SMMA, regime, GARCH, momentum z-scores, crossovers)
      3. Fit StandardScaler on training portion only
      4. Generate direction labels for supervised pre-training
      5. Pre-train Dueling DQN as 3-class classifier
      6. RL fine-tune with walk-forward + held-out test
      7. Continuously generate live BUY/SELL signals with confidence
    """

    GOLD_YF_SYMBOL = "GC=F"
    STATE_SIZE  = GoldTradingEnv.STATE_SIZE   # 25
    STATIC_SIZE = GoldTradingEnv.STATIC_SIZE  # 21 (0..20 — scaler applied here)
    ACTION_SIZE = 3

    SAVE_PATH    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data", "gold_rl_model.pt")
    SEQ_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "data", "gold_rl_seq_model.pt")
    SCALER_PATH  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data", "gold_rl_scaler.pkl")
    HISTORY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data", "gold_rl_history.json")

    # Sequence-based config
    SEQ_LEN = 30   # last 30 bars as input

    # Direction-label threshold: classify as BUY/SELL only if 5-bar fwd return
    # exceeds this fraction of rolling ATR. Smaller moves → HOLD label.
    LABEL_THRESHOLD_ATR_FRAC = 0.5

    def __init__(self):
        self.agent: Optional[GoldDuelingAgent] = None
        self.scaler = None  # sklearn StandardScaler
        self.is_training: bool = False
        self.progress: Dict = {"episode": 0, "total": 0, "reward": 0.0, "phase": "idle"}
        self._lock = threading.Lock()
        self.training_history: List[Dict] = []
        self._last_regime_result: Optional[Dict] = None
        self._last_transition_matrix: Optional[np.ndarray] = None

        # Load history if it exists
        if os.path.exists(self.HISTORY_PATH):
            try:
                with open(self.HISTORY_PATH) as f:
                    self.training_history = json.load(f)
            except Exception:
                pass

        # Load scaler if it exists
        self._try_load_scaler()

    # =========================================================================
    # Scaler persistence
    # =========================================================================
    def _try_load_scaler(self) -> bool:
        if not os.path.exists(self.SCALER_PATH):
            return False
        try:
            with open(self.SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
            return True
        except Exception:
            self.scaler = None
            return False

    def _save_scaler(self):
        if self.scaler is None:
            return
        os.makedirs(os.path.dirname(self.SCALER_PATH), exist_ok=True)
        with open(self.SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)

    # =========================================================================
    # Data Fetching
    # =========================================================================
    def fetch_training_data(self, months_back: int = 3) -> pd.DataFrame:
        import yfinance as yf

        chunks = []
        end_dt = datetime.now()
        max_days = min(months_back * 30, 25)

        print(f"[GoldRL] Fetching {max_days} days of 1m gold data…")
        with self._lock:
            self.progress = {"episode": 0, "total": 0, "reward": 0.0, "phase": "fetching_data"}

        for offset in range(0, max_days, 7):
            chunk_end   = end_dt - timedelta(days=offset)
            chunk_start = end_dt - timedelta(days=min(offset + 7, max_days))
            try:
                df_chunk = yf.download(
                    self.GOLD_YF_SYMBOL,
                    start=chunk_start.strftime("%Y-%m-%d"),
                    end=chunk_end.strftime("%Y-%m-%d"),
                    interval="1m",
                    progress=False,
                    auto_adjust=True,
                )
                if not df_chunk.empty:
                    chunks.append(df_chunk)
                    print(f"  chunk {chunk_start.date()} → {chunk_end.date()}: {len(df_chunk)} bars")
            except Exception as e:
                print(f"  [GoldRL] chunk error {chunk_start.date()}: {e}")
            time.sleep(0.4)

        if chunks:
            df = pd.concat(chunks)
            df = df[~df.index.duplicated(keep="last")].sort_index()
        else:
            df = pd.DataFrame()

        if len(df) < 1000:
            print(f"[GoldRL] 1m data sparse ({len(df)} bars) — falling back to 15m")
            try:
                df = yf.download(
                    self.GOLD_YF_SYMBOL,
                    period=f"{months_back * 30}d",
                    interval="15m",
                    progress=False,
                    auto_adjust=True,
                )
            except Exception as e:
                print(f"[GoldRL] 15m fallback failed: {e}")
                return pd.DataFrame()

        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        for want, candidates in [("close", ["close", "adj close"]),
                                  ("open",  ["open"]),
                                  ("high",  ["high"]),
                                  ("low",   ["low"]),
                                  ("volume",["volume"])]:
            for cand in candidates:
                if cand in df.columns and want not in df.columns:
                    df.rename(columns={cand: want}, inplace=True)
                    break

        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df["timestamp"] = df.index
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        print(f"[GoldRL] Training data: {len(df)} bars")
        return df

    # =========================================================================
    # Feature Engineering — extended with predictive features
    # =========================================================================
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        from pages.smma_strategy import (
            _smma, _atr_wilder, _vwap_session, _mfi, _keltner, _delta, _hurst
        )
        from services.markov_model import MarkovRegimeModel

        df = df.copy().reset_index(drop=True)
        closes = df["close"]

        with self._lock:
            self.progress["phase"] = "preparing_features"

        # ── SMMA ──
        df["smma3"]  = _smma(closes, 3)
        df["smma9"]  = _smma(closes, 9)
        df["smma40"] = _smma(closes, 40)
        df["smma75"] = _smma(closes, 75)

        # ── ATR ──
        atr_s = _atr_wilder(df, 14)
        df["atr"] = atr_s
        df["atr_pct"] = atr_s / closes.replace(0, np.nan)
        atr_mean = atr_s.rolling(50, min_periods=10).mean()
        df["atr_regime"] = np.where(
            atr_s > atr_mean * 1.1,  1.0,
            np.where(atr_s < atr_mean * 0.8, -1.0, 0.0)
        )

        # ── VWAP / Keltner ──
        try:    df["vwap"] = _vwap_session(df)
        except: df["vwap"] = closes
        try:
            kc = _keltner(df, ema_p=20, atr_p=10, mult=2.0)
            if kc and "upper" in kc:
                df["kc_upper"] = kc["upper"]; df["kc_lower"] = kc["lower"]
            else:
                df["kc_upper"] = closes * 1.002; df["kc_lower"] = closes * 0.998
        except:
            df["kc_upper"] = closes * 1.002; df["kc_lower"] = closes * 0.998

        # ── MFI / Delta ──
        try:    df["mfi"] = _mfi(df, 14)
        except: df["mfi"] = 50.0
        try:
            ddf = _delta(df); df["cum_delta"] = ddf["cum"]
        except:
            df["cum_delta"] = 0.0

        # ── Hurst (rolling 200, stride 20) ──
        hv = np.full(len(df), 0.5)
        ret_arr = closes.pct_change().fillna(0).values
        for i in range(200, len(df), 20):
            try:
                H, _ = _hurst(ret_arr[i - 200:i], min_lag=5, max_lag=40)
                hv[i] = H
            except: pass
        for i in range(1, len(df)):
            if hv[i] == 0.5 and hv[i - 1] != 0.5:
                hv[i] = hv[i - 1]
        df["hurst"] = hv

        # ── Returns + time of day ──
        df["ret1"]  = closes.pct_change(1)
        df["ret5"]  = closes.pct_change(5)
        df["ret20"] = closes.pct_change(20)
        ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.to_datetime(df.index)
        df["time_of_day"] = ts.dt.hour / 24.0 + ts.dt.minute / 1440.0

        # ── Regime (HMM/fallback) ──
        regime_id = np.ones(len(df))
        next_regime_prob = np.full(len(df), 0.5)
        try:
            rm = MarkovRegimeModel("XAUUSD", n_regimes=3)
            rr = rm.fit(closes.dropna())
            self._last_regime_result = rr
            hs = rr.get("hidden_states"); tm = rr.get("transition_matrix")
            self._last_transition_matrix = tm
            if hs is not None:
                hs = np.array(hs, dtype=float)
                start_idx = len(df) - len(hs)
                if start_idx >= 0:
                    regime_id[start_idx:] = hs
                else:
                    regime_id = hs[-len(df):]
                if tm is not None:
                    hs_int = np.clip(regime_id, 0, tm.shape[0] - 1).astype(int)
                    next_regime_prob = np.array([float(np.max(tm[s])) for s in hs_int])
        except Exception as e:
            print(f"[GoldRL] Regime detection failed: {e}")
        df["regime_id"]        = regime_id
        df["regime_norm"]      = regime_id / 2.0
        df["next_regime_prob"] = next_regime_prob

        # ── GARCH ──
        garch_vol = np.full(len(df), 0.15)
        try:
            from volatility_models import VolatilityModels
            ret_clean = closes.pct_change().dropna().values
            if len(ret_clean) > 200:
                vm = VolatilityModels(ret_clean)
                vm.fit_garch(1, 1)
                if hasattr(vm, "conditional_variance") and vm.conditional_variance is not None:
                    cv = vm.conditional_variance
                    ann = np.clip(np.sqrt(cv * 525_600), 0, 2.0)
                    garch_vol[1:1 + len(ann)] = ann[:len(df) - 1]
                    garch_vol[0] = garch_vol[1]
        except Exception as e:
            print(f"[GoldRL] GARCH failed (using 0.15 default): {e}")
        df["garch_vol"] = garch_vol

        df = df.dropna(subset=["smma75", "mfi", "atr"]).reset_index(drop=True)
        print(f"[GoldRL] Features prepared: {len(df)} bars, {len(df.columns)} columns")
        return df

    # =========================================================================
    # Direction-label generation for supervised pre-training
    # =========================================================================
    def _make_direction_labels(self, df: pd.DataFrame, horizon: int = 5) -> np.ndarray:
        """
        Generate y in {0=HOLD, 1=BUY, 2=SELL} based on the next `horizon`-bar
        return relative to a fraction of the local ATR.
        """
        closes = df["close"].values
        atr    = df["atr"].values if "atr" in df.columns else np.full(len(df), closes[0] * 0.002)
        n = len(closes)
        y = np.zeros(n, dtype=np.int64)
        for i in range(n - horizon):
            fwd = closes[i + horizon] - closes[i]
            thresh = atr[i] * self.LABEL_THRESHOLD_ATR_FRAC
            if fwd > thresh:
                y[i] = ACTION_BUY
            elif fwd < -thresh:
                y[i] = ACTION_SELL
            else:
                y[i] = ACTION_HOLD
        return y

    # =========================================================================
    # Build observation matrix from feature DataFrame
    # =========================================================================
    def _build_obs_matrix(self, df: pd.DataFrame, scaler=None) -> np.ndarray:
        """
        Construct a (n, STATE_SIZE) observation matrix without running the env.
        All position-state features default to flat (0, 0, 0.5, 1.0).
        Used for supervised pre-training and accuracy evaluation.
        """
        env = self._make_env(df, scaler=None)  # no scaler at env level — we'll scale here
        static = env._static_feats  # (n, 21)
        n = len(static)
        dynamic = np.tile(np.array([0.0, 0.0, 0.5, 1.0], dtype=np.float32), (n, 1))
        obs = np.concatenate([static, dynamic], axis=1).astype(np.float32)
        if scaler is not None:
            obs[:, :self.STATIC_SIZE] = np.clip(
                scaler.transform(obs[:, :self.STATIC_SIZE]), -5.0, 5.0
            )
        return obs

    # =========================================================================
    # Env / agent constructors
    # =========================================================================
    def _make_env(self, df: pd.DataFrame, scaler=None) -> GoldTradingEnv:
        regime_hs = df["regime_id"].values if "regime_id" in df.columns else None
        garch_vol = df["garch_vol"].values if "garch_vol" in df.columns else None
        return GoldTradingEnv(
            df,
            regime_hidden_states=regime_hs,
            transition_matrix=self._last_transition_matrix,
            garch_vol_series=garch_vol,
            scaler=scaler,
        )

    def _make_agent(self) -> GoldDuelingAgent:
        return GoldDuelingAgent(
            state_size=self.STATE_SIZE,
            action_size=self.ACTION_SIZE,
            learning_rate=0.0003,
            discount_factor=0.97,
            epsilon=1.0,
            epsilon_min=0.05,
            buffer_size=100_000,
            batch_size=128,
            target_update=500,
            hidden=64,
            dropout=0.3,
        )

    # =========================================================================
    # TRAINING — supervised pre-train + RL fine-tune + held-out test
    # =========================================================================
    def train(self, episodes: int = 1500, pretrain_epochs: int = 80):
        with self._lock:
            self.is_training = True
            self.progress = {"episode": 0, "total": episodes, "reward": 0.0,
                             "phase": "fetching_data"}
        try:
            # 1. Fetch & engineer features
            df_raw = self.fetch_training_data(months_back=3)
            if df_raw.empty or len(df_raw) < 800:
                print("[GoldRL] Insufficient training data — aborting")
                return

            df_feat = self.prepare_features(df_raw)
            if len(df_feat) < 400:
                print("[GoldRL] Not enough bars after feature engineering — aborting")
                return

            # 2. Three-way temporal split: 70% train / 15% val / 15% held-out test
            n = len(df_feat)
            i_train_end = int(n * 0.70)
            i_val_end   = int(n * 0.85)
            train_df = df_feat.iloc[:i_train_end].reset_index(drop=True)
            val_df   = df_feat.iloc[i_train_end:i_val_end].reset_index(drop=True)
            test_df  = df_feat.iloc[i_val_end:].reset_index(drop=True)
            print(f"[GoldRL] Split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

            # 3. Fit scaler on TRAINING DATA ONLY (no leakage)
            if not SKLEARN_AVAILABLE:
                print("[GoldRL] sklearn not available — proceeding without scaling")
                self.scaler = None
            else:
                with self._lock:
                    self.progress["phase"] = "fitting_scaler"
                obs_train_unscaled = self._build_obs_matrix(train_df, scaler=None)
                self.scaler = StandardScaler()
                self.scaler.fit(obs_train_unscaled[:, :self.STATIC_SIZE])
                self._save_scaler()
                print(f"[GoldRL] Fitted StandardScaler on {len(obs_train_unscaled)} training obs")

            # 4. Build agent (always start fresh for cleanest training)
            self.agent = self._make_agent()

            # 5. Supervised pre-training on direction labels
            with self._lock:
                self.progress["phase"] = "supervised_pretrain"
            print(f"\n[GoldRL] ── Phase 1: Supervised pre-training ({pretrain_epochs} epochs) ──")
            X_train = self._build_obs_matrix(train_df, scaler=self.scaler)
            y_train = self._make_direction_labels(train_df, horizon=5)
            # Drop the last `horizon` rows (no forward label)
            X_train = X_train[:-5]
            y_train = y_train[:-5]
            print(f"  Pre-train samples: {len(X_train)}")
            print(f"  Class distribution: HOLD={np.sum(y_train==0)}, "
                  f"BUY={np.sum(y_train==1)}, SELL={np.sum(y_train==2)}")
            self.agent.supervised_pretrain(X_train, y_train,
                                            epochs=pretrain_epochs, batch_size=128, lr=0.001)

            # Evaluate pre-train direction accuracy on validation
            X_val = self._build_obs_matrix(val_df, scaler=self.scaler)[:-5]
            y_val = self._make_direction_labels(val_df, horizon=5)[:-5]
            pre_acc = self._eval_direction(self.agent, X_val, y_val)
            print(f"  ✓ Pre-train val direction accuracy: {pre_acc:.2%}")

            # 6. RL fine-tuning with walk-forward
            with self._lock:
                self.progress["phase"] = "rl_finetune"
            print(f"\n[GoldRL] ── Phase 2: RL fine-tuning ({episodes} episodes) ──")
            best_val_acc = pre_acc
            self.agent.save(self.SAVE_PATH)  # save pre-trained baseline

            n_folds = 5
            fold_size = len(train_df) // n_folds
            ep_global = 0

            for fold in range(n_folds):
                fold_train = train_df.iloc[:fold_size * (fold + 1)].reset_index(drop=True)
                if len(fold_train) < 200:
                    continue

                env_train = self._make_env(fold_train, scaler=self.scaler)
                eps_this_fold = max(20, episodes // n_folds)

                for ep in range(eps_this_fold):
                    ep_global += 1
                    # Slow epsilon decay over first 800 global episodes
                    self.agent.epsilon = max(0.05, 1.0 - ep_global / 800.0 * 0.95)

                    obs, _ = env_train.reset()
                    done = False
                    total_reward = 0.0
                    steps = 0
                    while not done and steps < env_train.MAX_STEPS:
                        action_int = self.agent.get_action(obs, training=True)
                        next_obs, reward, done, _, _ = env_train.step(action_int)
                        self.agent.store_transition(obs, action_int, reward, next_obs, done)
                        self.agent.update()
                        obs = next_obs
                        total_reward += reward
                        steps += 1

                    with self._lock:
                        self.progress = {
                            "episode": ep_global, "total": episodes,
                            "reward":  round(total_reward, 4),
                            "phase":   f"fold {fold+1}/{n_folds}",
                        }
                    if ep_global % 50 == 0:
                        print(f"[GoldRL] ep {ep_global}/{episodes} "
                              f"fold {fold+1} reward={total_reward:.3f} "
                              f"ε={self.agent.epsilon:.3f}")

                # Validate on val_df after each fold
                val_acc = self._eval_direction(self.agent, X_val, y_val)
                print(f"[GoldRL] Fold {fold+1} val direction accuracy: {val_acc:.2%}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.agent.save(self.SAVE_PATH)
                    print(f"[GoldRL]   ↑ New best val acc — saved")

            # 7. Reload best-validation model and evaluate on held-out test
            self.load()
            with self._lock:
                self.progress["phase"] = "evaluating"

            X_test = self._build_obs_matrix(test_df, scaler=self.scaler)[:-5]
            y_test = self._make_direction_labels(test_df, horizon=5)[:-5]
            test_acc = self._eval_direction(self.agent, X_test, y_test)

            test_metrics = self._evaluate(self.agent, test_df)
            test_metrics["direction_accuracy"] = round(test_acc, 4)
            test_metrics["pretrain_val_acc"]   = round(pre_acc, 4)
            test_metrics["best_val_acc"]       = round(best_val_acc, 4)

            self.training_history.append({
                "timestamp":  datetime.now().isoformat(),
                "episodes":   episodes,
                "train_bars": len(train_df),
                "val_bars":   len(val_df),
                "test_bars":  len(test_df),
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in test_metrics.items()},
            })
            self._save_history()

            print("\n[GoldRL] ══════ TRAINING COMPLETE ══════")
            print(f"  Pre-train Val Acc:  {pre_acc:.2%}")
            print(f"  Best Val Acc:       {best_val_acc:.2%}")
            print(f"  Held-out Test Acc:  {test_acc:.2%}  ← unseen data")
            print(f"  Sharpe (test):      {test_metrics.get('sharpe', 0):.3f}")
            print(f"  Win Rate (test):    {test_metrics.get('win_rate', 0):.1%}")
            print(f"  Max DD (test):      {test_metrics.get('max_drawdown', 0):.1%}")
            print(f"  Total Return:       {test_metrics.get('total_return', 0):.2%}")

        except Exception as e:
            import traceback
            print(f"[GoldRL] Training error: {e}")
            traceback.print_exc()
        finally:
            with self._lock:
                self.is_training = False
                self.progress["phase"] = "idle"

    # =========================================================================
    # Evaluation helpers
    # =========================================================================
    def _eval_direction(self, agent, X: np.ndarray, y: np.ndarray) -> float:
        """Direction prediction accuracy: % of bars where greedy action == label."""
        if not TORCH_AVAILABLE or len(X) == 0:
            return 0.0
        agent.q_network.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X).to(agent.device)
            preds = agent.q_network(t).argmax(dim=1).cpu().numpy()
        agent.q_network.train()
        return float(np.mean(preds == y))

    def _evaluate(self, agent, df: pd.DataFrame) -> Dict:
        """Run agent greedily through env and return performance metrics."""
        if len(df) < 50:
            return {}
        env = self._make_env(df, scaler=self.scaler)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action_int = agent.get_action(obs, training=False)
            obs, reward, done, _, _ = env.step(action_int)
            ep_reward += reward

        trades  = env.get_trades()
        capital = env.get_portfolio_value()
        pnls    = [t["pnl"] for t in trades] if trades else [0.0]

        n_wins    = sum(1 for p in pnls if p > 0)
        win_rate  = n_wins / len(pnls) if pnls else 0.0
        gp        = sum(p for p in pnls if p > 0)
        gl        = abs(sum(p for p in pnls if p < 0))
        pf        = gp / gl if gl > 1e-9 else gp
        total_ret = (capital - env._initial_capital) / env._initial_capital
        mu, sig   = np.mean(pnls), np.std(pnls)
        sharpe    = float(mu / sig * np.sqrt(max(len(pnls), 1))) if sig > 1e-9 else 0.0

        equity = [env._initial_capital * (1 + p) for p in np.cumsum(pnls)]
        peak, max_dd = env._initial_capital, 0.0
        for eq in equity:
            peak = max(peak, eq)
            max_dd = max(max_dd, (peak - eq) / peak)

        return {
            "sharpe":        round(sharpe, 4),
            "win_rate":      round(win_rate, 4),
            "profit_factor": round(pf, 4),
            "total_return":  round(total_ret, 4),
            "total_trades":  len(trades),
            "max_drawdown":  round(max_dd, 4),
            "ep_reward":     round(ep_reward, 4),
        }

    # =========================================================================
    # Live signal — continuous BUY/SELL prediction
    # =========================================================================
    def generate_live_signal(self, agent, df_recent: pd.DataFrame,
                              hold_threshold: float = 0.4) -> Dict:
        """
        Continuous prediction: always returns BUY or SELL with confidence.
        Only returns HOLD when both BUY and SELL probabilities are below
        `hold_threshold` (very uncertain). Otherwise picks the higher of BUY/SELL.
        """
        if agent is None:
            return {"action": "HOLD", "confidence": 0.0, "sl": 0.0, "tp": 0.0,
                    "lot_size": 0.01, "q_values": [0, 0, 0]}

        tail = df_recent.tail(400).copy()
        try:
            df_feat = self.prepare_features(tail)
        except Exception as e:
            print(f"[GoldRL] Live feature error: {e}")
            return {"action": "HOLD", "confidence": 0.0, "sl": 0.0, "tp": 0.0,
                    "lot_size": 0.01, "q_values": [0, 0, 0]}

        if len(df_feat) < 80:
            return {"action": "HOLD", "confidence": 0.0, "sl": 0.0, "tp": 0.0,
                    "lot_size": 0.01, "q_values": [0, 0, 0]}

        # Build observation for the LAST bar. SequenceAgent and DuelingAgent
        # have different inputs:
        #   SequenceAgent.get_q_values(seq, dyn)  ← (T,F) sequence + (D,) dyn
        #   GoldDuelingAgent.get_q_values(state)  ← flat (state_size,) vector
        env = self._make_env(df_feat, scaler=self.scaler)
        last_static = env._static_feats[-1]   # already scaled if scaler present
        dynamic = np.array([0.0, 0.0, 0.5, 1.0], dtype=np.float32)
        try:
            if isinstance(agent, SequenceAgent):
                # Use the env's sequence helper if available; otherwise build
                # the trailing window manually from _static_feats.
                seq_len = getattr(agent, "seq_len", self.SEQ_LEN)
                if hasattr(env, "get_sequence_obs"):
                    # _step is at the start; seek to the last bar so that
                    # get_sequence_obs returns the trailing window.
                    env._step = len(env._static_feats) - 1
                    seq = env.get_sequence_obs(seq_len=seq_len)
                else:
                    static = env._static_feats
                    if len(static) >= seq_len:
                        seq = static[-seq_len:].astype(np.float32)
                    else:
                        pad = np.zeros((seq_len - len(static), static.shape[1]),
                                        dtype=np.float32)
                        seq = np.concatenate([pad, static], axis=0).astype(np.float32)
                q_vals = agent.get_q_values(seq, dynamic)
            else:
                # Single-bar agent (DuelingAgent)
                obs = np.concatenate([last_static, dynamic]).astype(np.float32)
                q_vals = agent.get_q_values(obs)
        except Exception as e:
            print(f"[GoldRL] get_q_values failed: {e}")
            return {"action": "HOLD", "confidence": 0.0, "sl": 0.0, "tp": 0.0,
                    "lot_size": 0.01, "q_values": [0, 0, 0]}

        if q_vals is None or len(q_vals) < 3:
            q_vals = np.zeros(3)

        # Softmax over Q-values for probabilities
        q_shift = q_vals - q_vals.max()
        probs   = np.exp(q_shift) / np.exp(q_shift).sum()

        p_hold, p_buy, p_sell = float(probs[0]), float(probs[1]), float(probs[2])

        # Continuous direction logic:
        #   - If BUY prob >= hold_threshold AND > SELL prob → BUY
        #   - If SELL prob >= hold_threshold AND > BUY prob → SELL
        #   - Else → HOLD (very uncertain)
        if p_buy >= hold_threshold and p_buy > p_sell:
            action_str = "BUY"
            confidence = p_buy
        elif p_sell >= hold_threshold and p_sell > p_buy:
            action_str = "SELL"
            confidence = p_sell
        elif p_buy > p_sell:
            action_str = "WEAK BUY"
            confidence = p_buy
        elif p_sell > p_buy:
            action_str = "WEAK SELL"
            confidence = p_sell
        else:
            action_str = "HOLD"
            confidence = p_hold

        # Risk management
        price = float(df_feat["close"].iloc[-1])
        atr   = float(df_feat["atr"].iloc[-1]) if "atr" in df_feat.columns else price * 0.002
        sl_dist = atr * 2.0
        tp_dist = atr * 4.0

        is_buy = action_str.endswith("BUY")
        if is_buy:
            sl, tp = price - sl_dist, price + tp_dist
        elif action_str.endswith("SELL"):
            sl, tp = price + sl_dist, price - tp_dist
        else:
            sl = tp = price

        # 1% risk position sizing
        risk_amount = 10_000.0 * 0.01
        lot_size = round(max(0.01, risk_amount / max(sl_dist * 100, 0.01)), 2)

        return {
            "action":     action_str,
            "confidence": round(float(confidence), 4),
            "sl":         round(sl, 2),
            "tp":         round(tp, 2),
            "lot_size":   lot_size,
            "q_values":   q_vals.tolist(),
            "probs":      {"HOLD": round(p_hold, 4),
                           "BUY":  round(p_buy, 4),
                           "SELL": round(p_sell, 4)},
            "atr":        round(atr, 2),
            "price":      round(price, 2),
        }

    # =========================================================================
    # Persistence
    # =========================================================================
    def _save_history(self):
        os.makedirs(os.path.dirname(self.HISTORY_PATH), exist_ok=True)
        try:
            with open(self.HISTORY_PATH, "w") as f:
                json.dump(self.training_history, f, indent=2)
        except Exception as e:
            print(f"[GoldRL] History save failed: {e}")

    def save(self, path: Optional[str] = None):
        if self.agent is not None:
            self.agent.save(path or self.SAVE_PATH)
        self._save_scaler()
        self._save_history()

    def load(self, path: Optional[str] = None) -> bool:
        path = path or self.SAVE_PATH
        if not os.path.exists(path):
            return False
        try:
            self.agent = self._make_agent()
            ok = self.agent.load(path)
            if ok:
                self._try_load_scaler()
                print(f"[GoldRL] Loaded model + scaler "
                      f"(ε={self.agent.epsilon:.3f}, steps={self.agent.steps})")
            return ok
        except Exception as e:
            print(f"[GoldRL] Load failed: {e}")
            return False

    def get_last_metrics(self) -> Dict:
        return self.training_history[-1] if self.training_history else {}

    # =========================================================================
    # SEQUENCE-BASED TRAINING (CNN+multi-head, the high-accuracy version)
    # =========================================================================
    def _build_sequence_dataset(self, df: pd.DataFrame, seq_len: int = 30):
        """
        Build (seqs, dyns, dir_labels, vol_labels) for supervised pre-training
        of the sequence agent.

        seqs:       (N, T, F) where T=seq_len, F=21
        dyns:       (N, 4)   — flat-position default (0,0,0.5,1)
        dir_labels: (N,)     — 0=HOLD, 1=BUY, 2=SELL based on 5-bar fwd vs ATR
        vol_labels: (N,)     — next 5-bar realized vol (for auxiliary regression)
        """
        env = self._make_env(df, scaler=None)  # raw static features
        static = env._static_feats             # (n, 21)
        if self.scaler is not None:
            static = np.clip(self.scaler.transform(static), -5.0, 5.0).astype(np.float32)

        n = len(static)
        seqs = []; dyns = []; dirs = []; vols = []
        closes = df["close"].values
        atr    = df["atr"].values if "atr" in df.columns else np.full(n, closes[0] * 0.002)

        for i in range(seq_len, n - 5):
            # Skip rows where ATR or label window invalid
            seqs.append(static[i - seq_len:i])
            dyns.append(np.array([0.0, 0.0, 0.5, 1.0], dtype=np.float32))

            fwd = closes[i + 5] - closes[i]
            thresh = atr[i] * self.LABEL_THRESHOLD_ATR_FRAC
            if fwd > thresh:
                dirs.append(1)
            elif fwd < -thresh:
                dirs.append(2)
            else:
                dirs.append(0)

            # Aux vol: next 5-bar realized vol normalized by ATR
            fwd_window = closes[i:i + 5]
            rv = float(np.std(np.diff(fwd_window) / fwd_window[:-1])) if len(fwd_window) > 1 else 0.0
            vols.append(rv * 1000.0)   # scale to similar magnitude as dir loss

        return (np.array(seqs, dtype=np.float32),
                np.array(dyns, dtype=np.float32),
                np.array(dirs, dtype=np.int64),
                np.array(vols, dtype=np.float32))

    def _make_seq_agent(self) -> "SequenceAgent":
        return SequenceAgent(
            n_features=self.STATIC_SIZE,
            dyn_size=4,
            n_actions=self.ACTION_SIZE,
            seq_len=self.SEQ_LEN,
            learning_rate=0.0003,
            discount_factor=0.97,
            buffer_size=80_000,
            batch_size=128,
            target_update=500,
            dropout=0.3,
            input_noise=0.05,
            aux_weight=0.3,
        )

    def train_sequence_model(self, pretrain_epochs: int = 60,
                             rl_episodes: int = 0, months_back: int = 3) -> Dict:
        """
        Two-phase training of the SequenceAgent:
          Phase 1: multi-task supervised pre-training (CE on direction + MSE on vol)
                   with early stopping on validation accuracy.
          Phase 2 (optional): RL fine-tune over `rl_episodes` episodes.

        Returns: dict with all metrics including OOS test direction accuracy.
        """
        from sklearn.preprocessing import StandardScaler

        with self._lock:
            self.is_training = True
            self.progress = {"episode": 0, "total": pretrain_epochs,
                             "reward": 0.0, "phase": "fetching_data"}

        try:
            # 1. Fetch
            df_raw = self.fetch_training_data(months_back=months_back)
            if df_raw.empty or len(df_raw) < 800:
                return {"error": "Insufficient data"}

            # 2. Features
            df_feat = self.prepare_features(df_raw)
            if len(df_feat) < 400:
                return {"error": "Not enough feature bars"}

            # 3. 70/15/15 temporal split
            n = len(df_feat)
            i_tr = int(n * 0.70); i_va = int(n * 0.85)
            train_df = df_feat.iloc[:i_tr].reset_index(drop=True)
            val_df   = df_feat.iloc[i_tr:i_va].reset_index(drop=True)
            test_df  = df_feat.iloc[i_va:].reset_index(drop=True)
            print(f"[GoldRL-Seq] Split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

            # 4. Fit scaler on training only
            with self._lock:
                self.progress["phase"] = "fitting_scaler"
            env_for_scaler = self._make_env(train_df, scaler=None)
            obs_train_static = env_for_scaler._static_feats
            self.scaler = StandardScaler()
            self.scaler.fit(obs_train_static)
            self._save_scaler()

            # 5. Build sequence datasets
            seqs_tr, dyns_tr, ydir_tr, yvol_tr = self._build_sequence_dataset(
                train_df, seq_len=self.SEQ_LEN)
            seqs_va, dyns_va, ydir_va, yvol_va = self._build_sequence_dataset(
                val_df, seq_len=self.SEQ_LEN)
            seqs_te, dyns_te, ydir_te, yvol_te = self._build_sequence_dataset(
                test_df, seq_len=self.SEQ_LEN)
            print(f"[GoldRL-Seq] Datasets: train={len(seqs_tr)}, val={len(seqs_va)}, "
                  f"test={len(seqs_te)} sequences")
            print(f"[GoldRL-Seq] Class dist (train): "
                  f"HOLD={int((ydir_tr==0).sum())}, "
                  f"BUY={int((ydir_tr==1).sum())}, "
                  f"SELL={int((ydir_tr==2).sum())}")

            # 6. Build agent and pre-train
            self.agent = self._make_seq_agent()
            with self._lock:
                self.progress["phase"] = "supervised_pretrain"
            print(f"\n[GoldRL-Seq] Phase 1: Multi-task supervised pre-training...")
            best_val_acc = self.agent.supervised_pretrain(
                seqs_tr, dyns_tr, ydir_tr, yvol_tr,
                epochs=pretrain_epochs, batch_size=128, lr=0.001,
                val_split=0.15, patience=12, verbose=True,
            )

            # 7. Evaluate on held-out test set (UNSEEN data)
            self.agent.q_network.eval()
            with torch.no_grad():
                seq_t = torch.FloatTensor(seqs_te).to(self.agent.device)
                dyn_t = torch.FloatTensor(dyns_te).to(self.agent.device)
                _, logits, _ = self.agent.q_network(seq_t, dyn_t)
                test_preds = logits.argmax(dim=1).cpu().numpy()

            test_acc = float((test_preds == ydir_te).mean())

            # Per-class precision/recall for honesty
            from sklearn.metrics import classification_report, confusion_matrix
            try:
                cls_report = classification_report(
                    ydir_te, test_preds,
                    labels=[0, 1, 2],
                    target_names=["HOLD", "BUY", "SELL"],
                    output_dict=True, zero_division=0,
                )
                cm = confusion_matrix(ydir_te, test_preds, labels=[0, 1, 2])
            except Exception:
                cls_report, cm = {}, np.zeros((3, 3))

            # 8. Save model
            self.agent.save(self.SEQ_SAVE_PATH)

            metrics = {
                "timestamp":          datetime.now().isoformat(),
                "model_type":         "sequence_cnn_multihead",
                "seq_len":            self.SEQ_LEN,
                "pretrain_epochs":    pretrain_epochs,
                "train_bars":         len(train_df),
                "val_bars":           len(val_df),
                "test_bars":          len(test_df),
                "train_sequences":    len(seqs_tr),
                "val_sequences":      len(seqs_va),
                "test_sequences":     len(seqs_te),
                "best_val_acc":       round(best_val_acc, 4),
                "test_direction_acc": round(test_acc, 4),
                "test_confusion":     cm.tolist(),
                "test_class_report":  cls_report,
                "class_dist_train":   {"HOLD": int((ydir_tr==0).sum()),
                                        "BUY":  int((ydir_tr==1).sum()),
                                        "SELL": int((ydir_tr==2).sum())},
            }
            self.training_history.append(metrics)
            self._save_history()

            print(f"\n[GoldRL-Seq] ══════ TRAINING COMPLETE ══════")
            print(f"  Best Val Acc       : {best_val_acc:.2%}")
            print(f"  Held-out Test Acc  : {test_acc:.2%}")
            print(f"  Confusion (rows=true, cols=pred):")
            print(f"           HOLD    BUY    SELL")
            for i, name in enumerate(["HOLD", "BUY ", "SELL"]):
                row = cm[i]
                print(f"  {name}    {row[0]:5d}  {row[1]:5d}  {row[2]:5d}")
            return metrics

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        finally:
            with self._lock:
                self.is_training = False
                self.progress["phase"] = "idle"

    def generate_seq_signals(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate signal array (0=HOLD, 1=BUY, 2=SELL) and confidence array
        for every row in `df`. Used by the backtester.
        """
        if self.agent is None or not isinstance(self.agent, SequenceAgent):
            return np.zeros(len(df), dtype=int), np.zeros(len(df), dtype=float)

        df_feat = self.prepare_features(df.copy())
        env = self._make_env(df_feat, scaler=None)
        static = env._static_feats
        if self.scaler is not None:
            static = np.clip(self.scaler.transform(static), -5.0, 5.0).astype(np.float32)

        n = len(static)
        signals = np.zeros(n, dtype=int)
        confs   = np.zeros(n, dtype=float)
        T = self.SEQ_LEN

        self.agent.q_network.eval()
        # Batch all sequences
        seq_batches = []
        idxs = []
        for i in range(T, n):
            seq_batches.append(static[i - T:i])
            idxs.append(i)
        if not seq_batches:
            return signals, confs

        seq_arr = np.array(seq_batches, dtype=np.float32)
        dyn_arr = np.tile(np.array([0.0, 0.0, 0.5, 1.0], dtype=np.float32),
                           (len(seq_batches), 1))

        with torch.no_grad():
            seq_t = torch.FloatTensor(seq_arr).to(self.agent.device)
            dyn_t = torch.FloatTensor(dyn_arr).to(self.agent.device)
            _, logits, _ = self.agent.q_network(seq_t, dyn_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            top_p = probs.max(axis=1)

        for k, i in enumerate(idxs):
            signals[i] = int(preds[k])
            confs[i] = float(top_p[k])

        # Map back from feature-row indexing to original df indexing — assume
        # they match since prepare_features only drops cold-start rows
        return signals, confs

    def load_sequence_model(self) -> bool:
        """Load the sequence model from disk."""
        if not os.path.exists(self.SEQ_SAVE_PATH):
            return False
        self.agent = self._make_seq_agent()
        ok = self.agent.load(self.SEQ_SAVE_PATH)
        if ok:
            self._try_load_scaler()
        return ok
