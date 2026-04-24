"""
Reinforcement Learning Trading Agent
=====================================
Implements Q-Learning and Deep RL for optimal trading decisions.
Uses Bellman equation for value iteration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from collections import deque
import random

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None


class Action(Enum):
    """Trading actions."""
    BUY = 0
    SELL = 1
    HOLD = 2
    CLOSE_LONG = 3
    CLOSE_SHORT = 4


@dataclass
class TradingState:
    """Current trading state."""
    position: int  # -1 (short), 0 (flat), 1 (long)
    price: float
    returns: np.ndarray
    indicators: Dict[str, float]
    account_value: float
    step: int


class QLearningAgent:
    """
    Q-Learning Agent for Trading

    Uses Bellman equation for value updates:
    Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
    """

    def __init__(self,
                 state_size: int,
                 action_size: int = 5,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent.

        Parameters
        ----------
        state_size : int
            Number of state features
        action_size : int
            Number of possible actions
        learning_rate : float
            Learning rate (alpha)
        discount_factor : float
            Discount factor (gamma)
        epsilon : float
            Exploration rate
        epsilon_decay : float
            Epsilon decay rate
        epsilon_min : float
            Minimum epsilon
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: state -> action -> value
        self.q_table: Dict[tuple, np.ndarray] = {}

        # Training history
        self.rewards = []

    def _discretize_state(self, state: TradingState) -> tuple:
        """Discretize continuous state for Q-table lookup."""
        # Discretize returns
        returns_bucket = np.digitize(
            np.mean(state.returns[-5:]),
            bins=np.linspace(-0.05, 0.05, 10)
        )

        # Discretize indicators
        rsi = state.indicators.get('rsi', 50)
        rsi_bucket = int(rsi // 10)

        macd = state.indicators.get('macd', 0)
        macd_bucket = int(np.sign(macd) + 1)

        # Position
        pos = state.position + 1  # 0, 1, 2

        return (returns_bucket, rsi_bucket, macd_bucket, pos)

    def get_action(self, state: TradingState, training: bool = True) -> Action:
        """
        Get action using epsilon-greedy policy.
        """
        if training and random.random() < self.epsilon:
            return random.choice(list(Action))

        state_key = self._discretize_state(state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        action_idx = np.argmax(self.q_table[state_key])
        return Action(action_idx)

    def update(self, state: TradingState, action: Action,
               reward: float, next_state: TradingState, done: bool):
        """
        Update Q-value using Bellman equation.

        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        """
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        # Current Q-value
        current_q = self.q_table[state_key][action.value]

        # Max Q-value for next state
        max_next_q = np.max(self.q_table[next_state_key])

        # Bellman update
        if done:
            target = reward
        else:
            target = reward + self.gamma * max_next_q

        # Update Q-value
        self.q_table[state_key][action.value] += self.lr * (target - current_q)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.rewards.append(reward)

    def save(self, filepath: str):
        """Save agent state (Q-table, epsilon, rewards) to disk via pickle."""
        import pickle, os
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "rewards": self.rewards,
            }, f)

    def load(self, filepath: str):
        """Load agent state from disk. Returns True on success."""
        import pickle, os
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, "rb") as f:
                state = pickle.load(f)
            self.q_table = state.get("q_table", {})
            self.epsilon = state.get("epsilon", self.epsilon_min)
            self.rewards = state.get("rewards", [])
            return True
        except Exception as e:
            print(f"[RLAgent] Could not load from {filepath}: {e}")
            return False


if TORCH_AVAILABLE:
    class DeepQNetwork(nn.Module):
        """
        Deep Q-Network for trading.
        """

        def __init__(self, state_size: int, action_size: int,
                     hidden_size: int = 128):
            super().__init__()

            self.network = nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, action_size)
            )

        def forward(self, x):
            return self.network(x)


    class DeepRLAgent:
        """
        Deep Reinforcement Learning Agent using DQN with Experience Replay.
        """

        def __init__(self,
                     state_size: int,
                     action_size: int = 5,
                     learning_rate: float = 0.001,
                     discount_factor: float = 0.95,
                     epsilon: float = 1.0,
                     epsilon_decay: float = 0.995,
                     epsilon_min: float = 0.01,
                     buffer_size: int = 10000,
                     batch_size: int = 32,
                     target_update: int = 100):
            """
            Initialize Deep RL agent.

            Parameters
            ----------
            state_size : int
                Number of state features
            action_size : int
                Number of actions
            learning_rate : float
                Learning rate
            discount_factor : float
                Discount factor (gamma)
            epsilon : float
                Exploration rate
            epsilon_decay : float
                Epsilon decay
            epsilon_min : float
                Minimum epsilon
            buffer_size : int
                Replay buffer size
            batch_size : int
                Training batch size
            target_update : int
                Target network update frequency
            """
            self.state_size = state_size
            self.action_size = action_size
            self.gamma = discount_factor
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.epsilon_min = epsilon_min
            self.batch_size = batch_size
            self.target_update = target_update

            # Device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Networks
            self.q_network = DeepQNetwork(state_size, action_size).to(self.device)
            self.target_network = DeepQNetwork(state_size, action_size).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())

            # Optimizer
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

            # Replay buffer
            self.buffer = deque(maxlen=buffer_size)

            # Training
            self.steps = 0
            self.losses = []

        def get_action(self, state: np.ndarray, training: bool = True) -> Action:
            """Get action using epsilon-greedy policy."""
            if training and random.random() < self.epsilon:
                return random.choice(list(Action))

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_values = self.q_network(state_tensor)

            action_idx = q_values.argmax().item()
            return Action(action_idx)

        def store_transition(self, state: np.ndarray, action: Action,
                            reward: float, next_state: np.ndarray, done: bool):
            """Store transition in replay buffer."""
            self.buffer.append((state, action.value, reward, next_state, done))

        def update(self):
            """Update Q-network using experience replay."""
            if len(self.buffer) < self.batch_size:
                return

            # Sample batch
            batch = random.sample(self.buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)

            # Current Q-values
            current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

            # Target Q-values (Bellman equation)
            with torch.no_grad():
                next_q = self.target_network(next_states).max(1)[0]

            target = rewards + (1 - dones) * self.gamma * next_q

            # Loss and optimization
            loss = nn.MSELoss()(current_q.squeeze(), target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.losses.append(loss.item())
            self.steps += 1

            # Update target network
            if self.steps % self.target_update == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        def save(self, filepath: str):
            """Save model."""
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps': self.steps
            }, filepath)

        def load(self, filepath: str):
            """Load model."""
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']


class TradingEnvironment:
    """
    Trading Environment for RL Agent.
    """

    def __init__(self, data: pd.DataFrame,
                 initial_capital: float = 10000,
                 transaction_cost: float = 0.001):
        """
        Initialize trading environment.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with indicators
        initial_capital : float
            Starting capital
        transaction_cost : float
            Transaction cost per trade
        """
        self.data = data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.current_step = 0
        self.capital = self.initial_capital
        self.position = 0  # 0: flat, 1: long, -1: short
        self.entry_price = 0
        self.trades = []
        self.portfolio_values = [self.initial_capital]

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        row = self.data.iloc[self.current_step]

        # Price features
        returns = self.data['close'].pct_change().iloc[max(0, self.current_step-20):self.current_step+1].values
        returns = np.nan_to_num(returns, nan=0)

        # Technical indicators (if available)
        rsi = row.get('rsi', 50) / 100
        macd = row.get('macd', 0)
        bb_pos = row.get('bb_position', 0.5)

        # Normalize
        macd = np.tanh(macd * 10)

        # Position
        pos = (self.position + 1) / 2  # 0, 0.5, 1

        # Account value
        account_norm = self.capital / self.initial_capital

        state = np.array([
            np.mean(returns[-5:]),
            np.std(returns[-5:]),
            returns[-1] if len(returns) > 0 else 0,
            rsi,
            macd,
            bb_pos,
            pos,
            account_norm
        ])

        return state

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (state, reward, done, info).
        """
        self.current_step += 1

        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True, {'portfolio_value': self.capital}

        row = self.data.iloc[self.current_step]
        price = row['close']

        reward = 0
        info = {}

        # Execute action
        if action == Action.BUY and self.position == 0:
            self.position = 1
            self.entry_price = price * (1 + self.transaction_cost)

        elif action == Action.SELL and self.position == 0:
            self.position = -1
            self.entry_price = price * (1 - self.transaction_cost)

        elif action == Action.CLOSE_LONG and self.position == 1:
            pnl = (price - self.entry_price) / self.entry_price
            self.capital *= (1 + pnl)
            reward = pnl
            self.position = 0
            self.trades.append({'type': 'long', 'pnl': pnl})

        elif action == Action.CLOSE_SHORT and self.position == -1:
            pnl = (self.entry_price - price) / self.entry_price
            self.capital *= (1 + pnl)
            reward = pnl
            self.position = 0
            self.trades.append({'type': 'short', 'pnl': pnl})

        elif action == Action.HOLD:
            # Mark to market
            if self.position == 1:
                reward = (price - self.entry_price) / self.entry_price * 0.01
            elif self.position == -1:
                reward = (self.entry_price - price) / self.entry_price * 0.01

        # Penalty for being in losing position
        if self.position != 0:
            unrealized_pnl = (price - self.entry_price) / self.entry_price
            if self.position == 1 and unrealized_pnl < -0.02:
                reward -= 0.1
            elif self.position == -1 and unrealized_pnl > 0.02:
                reward -= 0.1

        self.portfolio_values.append(self.capital)

        done = self.capital < self.initial_capital * 0.5  # Stop if down 50%

        info = {
            'portfolio_value': self.capital,
            'position': self.position,
            'step': self.current_step
        }

        return self._get_state(), reward, done, info

    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        return self.capital

    def get_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of trades."""
        if not self.trades:
            return 0

        pnls = [t['pnl'] for t in self.trades]
        return np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls)) if np.std(pnls) > 0 else 0


def train_rl_agent(data: pd.DataFrame,
                   agent_type: str = 'dqn',
                   episodes: int = 100,
                   max_steps: int = None) -> Tuple:
    """
    Train RL trading agent.

    Parameters
    ----------
    data : pd.DataFrame
        Training data with OHLCV and indicators
    agent_type : str
        'qlearning' or 'dqn'
    episodes : int
        Number of training episodes
    max_steps : int
        Maximum steps per episode

    Returns
    -------
    Tuple with (trained agent, environment, training history)
    """
    # Create environment
    env = TradingEnvironment(data)
    state_size = 8  # Number of state features

    # Create agent
    if agent_type == 'qlearning':
        agent = QLearningAgent(state_size=state_size)
    else:
        if TORCH_AVAILABLE:
            agent = DeepRLAgent(state_size=state_size)
        else:
            print("PyTorch not available, falling back to Q-Learning")
            agent = QLearningAgent(state_size=state_size)

    # Training
    history = {'rewards': [], 'portfolio_values': [], 'sharpe_ratios': []}

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        pct = data['close'].pct_change()

        while True:
            # QLearningAgent expects TradingState; DeepRLAgent takes raw ndarray
            if agent_type == 'qlearning':
                cs = env.current_step
                _qstate = TradingState(
                    position=env.position,
                    price=data.iloc[cs]['close'],
                    returns=np.array([pct.iloc[max(0, cs-20):cs+1].mean()]),
                    indicators={'rsi': data.iloc[cs].get('rsi', 50)},
                    account_value=env.capital,
                    step=cs,
                )
                action = agent.get_action(_qstate)
            else:
                action = agent.get_action(state)

            next_state, reward, done, info = env.step(action)

            if agent_type == 'qlearning':
                cs = env.current_step
                ns = min(cs + 1, len(data) - 1)
                pct = data['close'].pct_change()
                agent.update(
                    TradingState(
                        position=env.position,
                        price=data.iloc[cs]['close'],
                        returns=np.array([pct.iloc[max(0, cs-20):cs+1].mean()]),
                        indicators={'rsi': data.iloc[cs].get('rsi', 50)},
                        account_value=info['portfolio_value'],
                        step=cs
                    ),
                    action, reward,
                    TradingState(
                        position=env.position,
                        price=data.iloc[ns]['close'],
                        returns=np.array([pct.iloc[max(0, ns-20):ns+1].mean()]),
                        indicators={'rsi': data.iloc[ns].get('rsi', 50)},
                        account_value=info['portfolio_value'],
                        step=ns
                    ),
                    done
                )
            else:
                agent.store_transition(state, action, reward, next_state, done)
                agent.update()

            state = next_state
            total_reward += reward
            steps += 1

            if done or (max_steps and steps >= max_steps):
                break

        history['rewards'].append(total_reward)
        history['portfolio_values'].append(env.get_portfolio_value())
        history['sharpe_ratios'].append(env.get_sharpe_ratio())

        if episode % 10 == 0:
            print(f"Episode {episode}: Reward={total_reward:.4f}, "
                  f"Portfolio=${env.get_portfolio_value():.2f}, "
                  f"Sharpe={env.get_sharpe_ratio():.2f}")

    return agent, env, history
