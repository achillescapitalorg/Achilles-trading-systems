"""
Continuous Action Gold Trading Environment for SAC.

Wraps the existing discrete GoldTradingEnv with a continuous action space.
Action: [position_size, stop_distance_atr, takeprofit_distance_atr]
  - position_size: [-1, 1] -> mapped to [0, 1] fraction of max position
  - stop_distance: [-1, 1] -> mapped to [0.5, 2.5] ATR multiplier
  - takeprofit_distance: [-1, 1] -> mapped to [1.0, 4.0] ATR multiplier

This allows the agent to learn optimal position sizing and risk placement
end-to-end, rather than forcing discrete Buy/Sell/Hold decisions.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import gymnasium
from gymnasium import spaces

# Import the base discrete environment
try:
    from services.gold_rl_env import GoldTradingEnv
except ImportError:
    GoldTradingEnv = None


class GoldTradingEnvContinuous(gymnasium.Env):
    """
    Continuous-action gold trading environment.
    Delegates state observation to GoldTradingEnv but overrides action space.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10_000.0,
        regime_hidden_states: Optional[np.ndarray] = None,
        transition_matrix: Optional[np.ndarray] = None,
        garch_vol_series: Optional[np.ndarray] = None,
        scaler=None,
        max_position_size_oz: float = 10.0,
        commission_per_lot: float = 7.0,
    ):
        super().__init__()
        if GoldTradingEnv is None:
            raise RuntimeError("GoldTradingEnv not available; cannot create continuous wrapper.")

        self._discrete_env = GoldTradingEnv(
            df=df,
            initial_capital=initial_capital,
            regime_hidden_states=regime_hidden_states,
            transition_matrix=transition_matrix,
            garch_vol_series=garch_vol_series,
            scaler=scaler,
        )
        self.observation_space = self._discrete_env.observation_space
        # Continuous action: [position_size_norm, sl_mult, tp_mult]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.max_position_size_oz = max_position_size_oz
        self.commission_per_lot = commission_per_lot

        # State tracking
        self._position = 0.0  # continuous position in oz
        self._entry_price = 0.0
        self._capital = initial_capital
        self._peak_equity = initial_capital
        self._equity_curve = [initial_capital]
        self._trades = []

    def reset(self, seed=None, options=None):
        obs, info = self._discrete_env.reset(seed=seed, options=options)
        self._position = 0.0
        self._entry_price = 0.0
        self._capital = self._discrete_env._initial_capital
        self._peak_equity = self._capital
        self._equity_curve = [self._capital]
        self._trades = []
        return obs, info

    def step(self, action: np.ndarray):
        """
        action: [size_norm, sl_mult, tp_mult] in [-1, 1]
        """
        size_norm = float(np.clip(action[0], -1.0, 1.0))
        sl_mult = 0.5 + (float(np.clip(action[1], -1.0, 1.0)) + 1.0) * 1.0  # [0.5, 2.5]
        tp_mult = 1.0 + (float(np.clip(action[2], -1.0, 1.0)) + 1.0) * 1.5  # [1.0, 4.0]

        # Map to discrete signal for compatibility with base env logic
        # If size_norm > 0.2 -> BUY, < -0.2 -> SELL, else HOLD
        if size_norm > 0.2:
            discrete_action = 1  # BUY
        elif size_norm < -0.2:
            discrete_action = 2  # SELL
        else:
            discrete_action = 0  # HOLD

        # Run discrete env step
        obs, reward, terminated, truncated, info = self._discrete_env.step(discrete_action)

        # Override position sizing info if a trade was opened
        info["continuous_action"] = {
            "size_norm": size_norm,
            "sl_mult": sl_mult,
            "tp_mult": tp_mult,
            "target_size_oz": abs(size_norm) * self.max_position_size_oz,
        }

        # Shape reward: penalize large stops and tiny position sizes
        # This encourages the agent to use meaningful size and reasonable risk
        shaping = 0.0
        if discrete_action != 0:
            if abs(size_norm) < 0.1:
                shaping -= 0.01  # Penalty for miniscule size
            if sl_mult > 2.0:
                shaping -= 0.005  # Penalty for very wide stops
            if tp_mult < sl_mult:
                shaping -= 0.01  # Penalty for R:R < 1

        reward = float(reward) + shaping
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._discrete_env.render()

    def close(self):
        return self._discrete_env.close()
