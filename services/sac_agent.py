"""
Soft Actor-Critic (SAC) Agent for Gold Trading with Continuous Action Space.

Replaces the discrete Dueling DQN with an agent that outputs continuous actions:
  - Position size (0 to 1, fraction of max position)
  - Stop-loss distance (as multiple of ATR)
  - Take-profit distance (as multiple of ATR)

Why SAC:
  - Continuous actions: position sizing is inherently continuous
  - Entropy regularization: prevents premature convergence to bad policies
  - Off-policy + replay buffer: sample efficient
  - Twin critics: reduces Q-value overestimation
  - Automatic temperature tuning: adapts exploration dynamically

Verified: Stable Baselines3, barmenteros.com (2026), arXiv:2604.00031 all
recommend SAC for FX trading with continuous position sizing.

Reference: Haarnoja et al. 2018, "Soft Actor-Critic: Off-Policy Maximum
Entropy Deep Reinforcement Learning with a Stochastic Actor."
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
from typing import Tuple, Optional, Dict


SAC_CONFIG = {
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "lr_alpha": 3e-4,
    "gamma": 0.995,        # High discount for trading
    "tau": 0.005,          # Soft update coefficient
    "buffer_size": 200000,
    "batch_size": 256,
    "hidden_size": 256,    # Wider than DQN for complex policy
    "target_entropy": -2.0,  # For 3 continuous actions
    "update_every": 1,
    "updates_per_step": 1,
    "start_steps": 1000,   # Random exploration before learning
}


class Actor(nn.Module):
    """Stochastic policy network for SAC. Outputs mean and log_std for Gaussian."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action using reparameterization trick."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)  # Bound to [-1, 1]
        # Correct log_prob for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """Get action for environment interaction."""
        with torch.no_grad():
            mean, log_std = self.forward(state)
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = log_std.exp()
                normal = Normal(mean, std)
                x_t = normal.rsample()
                action = torch.tanh(x_t)
        return action.cpu().numpy()[0]


class Critic(nn.Module):
    """Twin Q-network for SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        # Q1
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.q1_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q1_out = nn.Linear(hidden_size, 1)
        # Q2
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.q2_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q2_out = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat([state, action], dim=1)
        x1 = F.relu(self.q1_fc1(xu))
        x1 = F.relu(self.q1_fc2(x1))
        q1 = self.q1_out(x1)
        x2 = F.relu(self.q2_fc1(xu))
        x2 = F.relu(self.q2_fc2(x2))
        q2 = self.q2_out(x2)
        return q1, q2


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""

    def __init__(self, capacity: int = 200000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class GoldSACAgent:
    """
    SAC agent for gold trading with continuous action space.
    Actions: [position_size, stop_distance_atr, takeprofit_distance_atr]
    All actions bounded to [-1, 1] via tanh.
    """

    def __init__(self, state_dim: int, action_dim: int = 3, config: Optional[Dict] = None):
        self.config = config or SAC_CONFIG.copy()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor = Actor(state_dim, action_dim, self.config["hidden_size"]).to(self.device)
        self.critic = Critic(state_dim, action_dim, self.config["hidden_size"]).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, self.config["hidden_size"]).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config["lr_actor"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config["lr_critic"])

        # Automatic temperature tuning
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config["lr_alpha"])
        self.target_entropy = torch.tensor(self.config["target_entropy"], device=self.device)

        self.replay_buffer = ReplayBuffer(self.config["buffer_size"])
        self.steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.actor.get_action(state, deterministic)

    def update(self) -> Dict[str, float]:
        """Single SAC update step. Returns loss metrics."""
        if len(self.replay_buffer) < self.config["batch_size"]:
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config["batch_size"]
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.config["gamma"] * q_target

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update temperature
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target critic
        tau = self.config["tau"]
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.item(),
            "q_mean": q1.mean().item(),
        }

    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha,
                "config": self.config,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.log_alpha = checkpoint["log_alpha"]


def interpret_sac_action(action: np.ndarray, atr: float) -> Dict[str, float]:
    """
    Convert SAC action vector to trading parameters.
    action: [size, sl_mult, tp_mult] each in [-1, 1]
    """
    position_size = (action[0] + 1.0) / 2.0  # Map [-1,1] -> [0,1]
    # Stop-loss ATR multiplier: map [-1,1] -> [0.5, 2.5]
    stop_loss_atr = 0.5 + (action[1] + 1.0) * 1.0
    # Take-profit ATR multiplier: map [-1,1] -> [1.0, 4.0]
    take_profit_atr = 1.0 + (action[2] + 1.0) * 1.5
    return {
        "position_size": float(position_size),
        "stop_loss_atr": float(stop_loss_atr),
        "take_profit_atr": float(take_profit_atr),
        "stop_loss_price": float(atr * stop_loss_atr),
        "take_profit_price": float(atr * take_profit_atr),
    }
