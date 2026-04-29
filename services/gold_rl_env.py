"""
Gold 1-Minute RL Trading Environment
=====================================
Custom gymnasium.Env for XAUUSD/GC=F 1-minute timeframe.

State: 25 features (21 precomputed static + 4 dynamic position features)
Actions: 0=HOLD, 1=BUY(long), 2=SELL(short)
Reward: Sharpe-scaled step PnL + regime/alignment shaping terms
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

import gymnasium
from gymnasium import spaces


class GoldTradingEnv(gymnasium.Env):
    """
    Gymnasium environment for gold futures 1-minute RL trading.

    Static features (precomputed per bar):
      [0]  smma3_dev      close/smma3 - 1
      [1]  smma9_dev      close/smma9 - 1
      [2]  smma40_dev     close/smma40 - 1
      [3]  smma75_dev     close/smma75 - 1
      [4]  smma_align     +1 bull stack / -1 bear stack / 0 mixed
      [5]  hurst          Hurst exponent [0,1]
      [6]  mfi_norm       MFI / 100
      [7]  atr_pct        ATR / close
      [8]  atr_regime     -1/0/+1 (contracting/normal/expanding)
      [9]  delta_sign     sign(cum_delta)
      [10] delta_mag      tanh(cum_delta / rolling_std_delta)
      [11] regime_norm    regime_id / 2  (0=LOW_VOL, 0.5=NORMAL, 1=HIGH_VOL)
      [12] garch_vol      annualized GARCH vol clipped [0,2]
      [13] ret1           1-bar return
      [14] ret5           5-bar return
      [15] ret20          20-bar return
      [16] vwap_dev       close/vwap - 1
      [17] kc_upper_dev   close/kc_upper - 1
      [18] kc_lower_dev   close/kc_lower - 1
      [19] next_reg_prob  max(transition_matrix[current_regime])
      [20] time_of_day    hour/24 + minute/1440

    Dynamic features (appended each step):
      [21] position       -1/0/+1
      [22] unrealized_pnl (price-entry)/entry if open else 0
      [23] steps_since    bars since last trade open/close, capped 100, /100
      [24] account_norm   capital / initial_capital
    """

    metadata = {"render_modes": ["human"]}
    STATE_SIZE = 25
    STATIC_SIZE = 21

    # Action constants
    HOLD = 0
    BUY  = 1
    SELL = 2

    TRANSACTION_COST = 0.0005   # 0.05% per leg
    DRAWDOWN_THRESHOLD = 0.05   # 5% drawdown penalty trigger
    MAX_STEPS = 500

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10_000.0,
        regime_hidden_states: Optional[np.ndarray] = None,
        transition_matrix: Optional[np.ndarray] = None,
        garch_vol_series: Optional[np.ndarray] = None,
        scaler=None,
    ):
        super().__init__()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.STATE_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._df = df.reset_index(drop=True)
        self._closes = self._df["close"].values.astype(np.float64)
        self._initial_capital = initial_capital
        self._scaler = scaler   # sklearn StandardScaler or None

        # Build per-bar static feature matrix
        self._static_feats = self._precompute_features(
            regime_hidden_states, transition_matrix, garch_vol_series
        )
        self._n = len(self._static_feats)

        # Rolling 20-bar std of returns for Sharpe scaling
        rets = np.diff(self._closes) / np.maximum(self._closes[:-1], 1e-9)
        self._rolling_std = np.full(len(self._closes), 1e-6)
        for i in range(1, len(self._closes)):
            w = self._closes[max(0, i-20):i]
            if len(w) >= 2:
                r = np.diff(w) / np.maximum(w[:-1], 1e-9)
                self._rolling_std[i] = max(np.std(r), 1e-6)

        # State reset
        self._step = 0
        self._position = 0
        self._entry_price = 0.0
        self._capital = initial_capital
        self._peak_capital = initial_capital
        self._steps_since_trade = 0
        self._trades: list = []

    # ------------------------------------------------------------------
    def _precompute_features(
        self,
        regime_hidden_states: Optional[np.ndarray],
        transition_matrix: Optional[np.ndarray],
        garch_vol_series: Optional[np.ndarray],
    ) -> np.ndarray:
        df = self._df
        n = len(df)
        closes = df["close"].values

        def safe_col(col, default=0.0):
            if col in df.columns:
                vals = df[col].values.astype(np.float64)
                vals = np.where(np.isfinite(vals), vals,
                                default if np.isscalar(default) else np.nanmean(vals))
                return vals
            if not np.isscalar(default):
                return np.array(default, dtype=np.float64)[:n]
            return np.full(n, default, dtype=np.float64)

        smma3  = safe_col("smma3",  closes)
        smma9  = safe_col("smma9",  closes)
        smma40 = safe_col("smma40", closes)
        smma75 = safe_col("smma75", closes)

        # Deviations — clipped to avoid extreme values from cold-start NaN fill
        dev3  = np.clip(closes / np.where(smma3  > 0, smma3,  closes) - 1, -0.1, 0.1)
        dev9  = np.clip(closes / np.where(smma9  > 0, smma9,  closes) - 1, -0.1, 0.1)
        dev40 = np.clip(closes / np.where(smma40 > 0, smma40, closes) - 1, -0.1, 0.1)
        dev75 = np.clip(closes / np.where(smma75 > 0, smma75, closes) - 1, -0.1, 0.1)

        bull = (smma3 > smma9) & (smma9 > smma40) & (smma40 > smma75)
        bear = (smma3 < smma9) & (smma9 < smma40) & (smma40 < smma75)
        align = np.where(bull, 1.0, np.where(bear, -1.0, 0.0))

        hurst   = np.clip(safe_col("hurst", 0.5), 0.1, 0.9)
        mfi     = safe_col("mfi", 50.0) / 100.0
        atr_pct = np.clip(safe_col("atr_pct", 0.002), 0, 0.05)

        atr_reg_raw = safe_col("atr_regime", 0.0)
        atr_reg = np.clip(atr_reg_raw, -1, 1)

        cum_delta = safe_col("cum_delta", 0.0)
        delta_std = np.std(cum_delta) if np.std(cum_delta) > 1e-9 else 1.0
        delta_sign = np.sign(cum_delta)
        delta_mag  = np.tanh(cum_delta / delta_std)

        # Regime
        if regime_hidden_states is not None:
            hs = np.array(regime_hidden_states, dtype=float)
            if len(hs) < n:
                padded = np.full(n, 1.0)  # default NORMAL
                padded[n - len(hs):] = hs
                hs = padded
            elif len(hs) > n:
                hs = hs[-n:]
            regime_norm = hs / 2.0
        else:
            regime_norm = safe_col("regime_norm", 0.5)

        if transition_matrix is not None and regime_hidden_states is not None:
            hs_int = np.clip(regime_norm * 2, 0, transition_matrix.shape[0] - 1).astype(int)
            next_reg_prob = np.array([
                float(np.max(transition_matrix[s])) for s in hs_int
            ])
        else:
            next_reg_prob = safe_col("next_regime_prob", 0.5)

        # GARCH vol
        if garch_vol_series is not None:
            gvol = np.array(garch_vol_series, dtype=float)
            if len(gvol) < n:
                gvol = np.concatenate([np.full(n - len(gvol), 0.15), gvol])
            elif len(gvol) > n:
                gvol = gvol[-n:]
            garch_vol = np.clip(gvol, 0, 2.0)
        else:
            garch_vol = np.clip(safe_col("garch_vol", 0.15), 0, 2.0)

        # Returns
        ret1  = np.clip(safe_col("ret1",  0.0), -0.05, 0.05)
        ret5  = np.clip(safe_col("ret5",  0.0), -0.10, 0.10)
        ret20 = np.clip(safe_col("ret20", 0.0), -0.15, 0.15)

        # VWAP / Keltner deviations
        vwap = safe_col("vwap", closes)
        kc_upper = safe_col("kc_upper", closes * 1.002)
        kc_lower = safe_col("kc_lower", closes * 0.998)
        vwap_dev     = np.clip(closes / np.where(vwap     > 0, vwap,     closes) - 1, -0.05, 0.05)
        kc_upper_dev = np.clip(closes / np.where(kc_upper > 0, kc_upper, closes) - 1, -0.05, 0.05)
        kc_lower_dev = np.clip(closes / np.where(kc_lower > 0, kc_lower, closes) - 1, -0.05, 0.05)

        # Time of day
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
        else:
            ts = pd.to_datetime(df.index)
        time_of_day = (ts.dt.hour / 24.0 + ts.dt.minute / 1440.0).values

        feats = np.column_stack([
            dev3, dev9, dev40, dev75, align,
            hurst, mfi, atr_pct, atr_reg,
            delta_sign, delta_mag,
            regime_norm, garch_vol,
            ret1, ret5, ret20,
            vwap_dev, kc_upper_dev, kc_lower_dev,
            next_reg_prob, time_of_day,
        ]).astype(np.float32)

        # Replace any remaining NaN/Inf with 0
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        static = self._static_feats[self._step]
        unrealized = 0.0
        if self._position != 0 and self._entry_price > 0:
            price = self._closes[self._step]
            unrealized = self._position * (price - self._entry_price) / self._entry_price

        dynamic = np.array([
            float(self._position),
            float(np.clip(unrealized, -0.1, 0.1)),
            float(min(self._steps_since_trade, 100) / 100.0),
            float(self._capital / self._initial_capital),
        ], dtype=np.float32)

        obs = np.concatenate([static, dynamic]).astype(np.float32)

        # Apply scaler ONLY to static features (cols 0..STATIC_SIZE) — dynamic
        # position features are already bounded and meaningful as-is
        if self._scaler is not None:
            try:
                scaled_static = self._scaler.transform(obs[:self.STATIC_SIZE].reshape(1, -1))[0]
                obs[:self.STATIC_SIZE] = np.clip(scaled_static, -5.0, 5.0)
            except Exception:
                pass
        return obs

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        self._position = 0
        self._entry_price = 0.0
        self._capital = self._initial_capital
        self._peak_capital = self._initial_capital
        self._steps_since_trade = 0
        self._trades = []
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert 0 <= action <= 2, f"Invalid action {action}"

        price_prev = self._closes[self._step]
        self._step += 1
        step_capped = min(self._step, self._n - 1)
        price_now = self._closes[step_capped]

        reward = 0.0
        new_trade = False

        # --- Execute action ---
        if action == self.BUY:
            if self._position == 0:
                self._position = 1
                self._entry_price = price_now * (1 + self.TRANSACTION_COST)
                new_trade = True
                self._steps_since_trade = 0
            elif self._position == -1:
                # Close short then go long
                pnl = -1 * (price_now - self._entry_price) / self._entry_price
                self._capital *= (1 + pnl)
                self._trades.append({"type": "short", "pnl": pnl})
                self._position = 1
                self._entry_price = price_now * (1 + self.TRANSACTION_COST)
                new_trade = True
                self._steps_since_trade = 0

        elif action == self.SELL:
            if self._position == 0:
                self._position = -1
                self._entry_price = price_now * (1 - self.TRANSACTION_COST)
                new_trade = True
                self._steps_since_trade = 0
            elif self._position == 1:
                # Close long then go short
                pnl = (price_now - self._entry_price) / self._entry_price
                self._capital *= (1 + pnl)
                self._trades.append({"type": "long", "pnl": pnl})
                self._position = -1
                self._entry_price = price_now * (1 - self.TRANSACTION_COST)
                new_trade = True
                self._steps_since_trade = 0

        # else HOLD — do nothing

        # --- Compute step reward (direct PnL in basis points, 1 bp = 0.01%) ---
        # Using bp scaling gives reward magnitudes ~1-10 instead of 0.0001-0.001,
        # which makes the DQN learning signal far less noisy.
        if self._position != 0 and self._entry_price > 0:
            step_pnl = self._position * (price_now - price_prev) / self._entry_price
            reward = step_pnl * 10_000.0   # convert to basis points

            # Light volatility normalization (avoid divide-by-zero blow-ups)
            vol_norm = max(self._rolling_std[step_capped] * 10_000.0, 1.0)
            reward = np.clip(reward / vol_norm, -5.0, 5.0)

            # Regime shaping (small, doesn't dominate PnL signal)
            regime = int(round(self._static_feats[step_capped][11] * 2))
            unrealized = self._position * (price_now - self._entry_price) / self._entry_price
            if regime == 2:                          # HIGH_VOL — discourage exposure
                reward -= 0.05
            elif regime == 0 and unrealized > 0:     # LOW_VOL profitable — encourage
                reward += 0.02

            # SMMA alignment bonus
            align = self._static_feats[step_capped][4]
            if (self._position == 1 and align > 0.5) or (self._position == -1 and align < -0.5):
                reward += 0.02

            # Drawdown penalty
            self._peak_capital = max(self._peak_capital, self._capital)
            dd = (self._peak_capital - self._capital) / self._peak_capital
            if dd > self.DRAWDOWN_THRESHOLD:
                reward -= 1.0

        elif action == self.HOLD and self._position == 0:
            # Small penalty for sitting flat — encourages the agent to take
            # informed positions rather than always defaulting to HOLD
            reward = -0.02

        # Transaction cost on new trade — significant enough to matter
        if new_trade:
            reward -= 0.5

        self._steps_since_trade += 1

        # Mark-to-market capital for open positions
        if self._position != 0 and self._entry_price > 0:
            pnl_now = self._position * (price_now - self._entry_price) / self._entry_price
            self._capital = self._initial_capital * (1 + pnl_now)

        done = (
            self._capital < self._initial_capital * 0.80
            or self._step >= min(self.MAX_STEPS, self._n - 1)
        )

        info = {
            "capital": self._capital,
            "position": self._position,
            "step": self._step,
            "price": price_now,
            "trades": len(self._trades),
        }
        return self._get_obs(), float(reward), done, False, info

    # ------------------------------------------------------------------
    def get_sequence_obs(self, seq_len: int = 30) -> np.ndarray:
        """
        Returns the last `seq_len` bars of static features as shape (seq_len, STATIC_SIZE).
        Used by the sequence-based CNN agent. Pads with zeros if not enough history.
        """
        end = min(self._step + 1, self._n)
        start = max(0, end - seq_len)
        seq = self._static_feats[start:end]
        if len(seq) < seq_len:
            # Left-pad with zeros
            pad = np.zeros((seq_len - len(seq), self.STATIC_SIZE), dtype=np.float32)
            seq = np.concatenate([pad, seq], axis=0)
        return seq.astype(np.float32)

    def get_dynamic_features(self) -> np.ndarray:
        """The 4 dynamic position features for the current step."""
        unrealized = 0.0
        if self._position != 0 and self._entry_price > 0:
            price = self._closes[min(self._step, self._n - 1)]
            unrealized = self._position * (price - self._entry_price) / self._entry_price
        return np.array([
            float(self._position),
            float(np.clip(unrealized, -0.1, 0.1)),
            float(min(self._steps_since_trade, 100) / 100.0),
            float(self._capital / self._initial_capital),
        ], dtype=np.float32)

    def get_portfolio_value(self) -> float:
        return self._capital

    def get_trades(self) -> list:
        return self._trades

    def get_sharpe(self) -> float:
        if not self._trades:
            return 0.0
        pnls = [t["pnl"] for t in self._trades]
        mu = np.mean(pnls)
        sigma = np.std(pnls)
        return float(mu / sigma * np.sqrt(len(pnls))) if sigma > 1e-9 else 0.0

    def render(self):
        price = self._closes[min(self._step, self._n - 1)]
        print(
            f"Step {self._step:4d} | Price {price:8.2f} | "
            f"Pos {self._position:+d} | Capital {self._capital:,.2f} | "
            f"Trades {len(self._trades)}"
        )
