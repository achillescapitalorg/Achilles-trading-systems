"""
Dueling Double DQN Agent for Gold 1-Minute Trading
====================================================
Replaces the generic DeepRLAgent with an architecture better suited to
financial decision-making and overfitting resistance:

  - Dueling network: separate V(s) and A(s,a) streams
  - Double DQN: target net for value evaluation, online net for action selection
  - Layer normalization (more stable than BatchNorm in RL)
  - Dropout 0.3 for regularization
  - Smaller hidden size (64) than original (128) — fewer params, less overfit
  - Huber loss (more robust to reward outliers than MSE)
"""

import os
import random
import pickle
import numpy as np
from collections import deque
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class DuelingDQN(nn.Module):
        """
        Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
        """

        def __init__(self, state_size: int, action_size: int,
                     hidden: int = 64, dropout: float = 0.3):
            super().__init__()
            self.feature = nn.Sequential(
                nn.Linear(state_size, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.value = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )
            self.advantage = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, action_size),
            )

        def forward(self, x):
            f = self.feature(x)
            v = self.value(f)
            a = self.advantage(f)
            # Q = V + (A - mean(A))   ← centred advantage for identifiability
            return v + a - a.mean(dim=-1, keepdim=True)


    class GoldDuelingAgent:
        """
        Dueling Double DQN agent with Huber loss + experience replay.
        Action space: 0 = HOLD, 1 = BUY, 2 = SELL.
        """

        def __init__(
            self,
            state_size: int,
            action_size: int = 3,
            learning_rate: float = 0.0003,
            discount_factor: float = 0.97,
            epsilon: float = 1.0,
            epsilon_min: float = 0.05,
            buffer_size: int = 100_000,
            batch_size: int = 128,
            target_update: int = 500,
            gradient_clip: float = 1.0,
            hidden: int = 64,
            dropout: float = 0.3,
        ):
            self.state_size = state_size
            self.action_size = action_size
            self.gamma = discount_factor
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.batch_size = batch_size
            self.target_update = target_update
            self.gradient_clip = gradient_clip

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.q_network = DuelingDQN(state_size, action_size,
                                         hidden=hidden, dropout=dropout).to(self.device)
            self.target_network = DuelingDQN(state_size, action_size,
                                              hidden=hidden, dropout=dropout).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()

            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate,
                                         weight_decay=1e-5)

            self.buffer = deque(maxlen=buffer_size)
            self.steps = 0
            self.losses = []

        # ──────────────────────────────────────────────────────────────────
        def get_action(self, state, training: bool = True) -> int:
            """Returns int action (0/1/2) — NOT an Action enum."""
            if training and random.random() < self.epsilon:
                return random.randint(0, self.action_size - 1)

            self.q_network.eval()
            with torch.no_grad():
                t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q = self.q_network(t)
            self.q_network.train()
            return int(q.argmax(dim=-1).item())

        def get_q_values(self, state) -> np.ndarray:
            """Return raw Q-values for confidence scoring."""
            self.q_network.eval()
            with torch.no_grad():
                t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q = self.q_network(t).squeeze(0).cpu().numpy()
            self.q_network.train()
            return q

        # ──────────────────────────────────────────────────────────────────
        def store_transition(self, state, action: int, reward: float,
                             next_state, done: bool):
            self.buffer.append((
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            ))

        def update(self):
            """Perform one DQN update step using Double DQN target."""
            if len(self.buffer) < self.batch_size:
                return None

            batch = random.sample(self.buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states_t      = torch.FloatTensor(np.stack(states)).to(self.device)
            actions_t     = torch.LongTensor(actions).to(self.device)
            rewards_t     = torch.FloatTensor(rewards).to(self.device)
            next_states_t = torch.FloatTensor(np.stack(next_states)).to(self.device)
            dones_t       = torch.FloatTensor(dones).to(self.device)

            # Current Q(s, a)
            current_q = self.q_network(states_t).gather(
                1, actions_t.unsqueeze(1)).squeeze(1)

            # Double DQN: action from online net, value from target net
            with torch.no_grad():
                next_actions = self.q_network(next_states_t).argmax(dim=1)
                next_q = self.target_network(next_states_t).gather(
                    1, next_actions.unsqueeze(1)).squeeze(1)
                target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

            # Huber loss is more robust than MSE for noisy financial rewards
            loss = F.smooth_l1_loss(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
            self.optimizer.step()

            self.steps += 1
            if self.steps % self.target_update == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            self.losses.append(loss.item())
            return loss.item()

        # ──────────────────────────────────────────────────────────────────
        def supervised_pretrain(self, X: np.ndarray, y: np.ndarray,
                                epochs: int = 50, batch_size: int = 128,
                                lr: float = 0.001, verbose: bool = True):
            """
            Pre-train the Q-network as a 3-class classifier on direction labels.

            X: shape (N, state_size) — observations
            y: shape (N,) integer labels in {0=HOLD, 1=BUY, 2=SELL}

            We treat Q-values as logits and minimize cross-entropy.
            This bootstraps the network with direction-prediction prior, so
            DQN fine-tuning starts from a sensible policy instead of random.
            """
            if len(X) < 100:
                return

            X_t = torch.FloatTensor(X).to(self.device)
            y_t = torch.LongTensor(y).to(self.device)

            # Class-balanced loss weights (gold often has slight downward bias)
            counts = np.bincount(y, minlength=self.action_size).astype(float)
            counts = np.where(counts == 0, 1.0, counts)
            class_w = torch.FloatTensor(len(y) / counts).to(self.device)
            class_w /= class_w.sum()
            class_w *= self.action_size

            # Use a separate optimizer with higher LR for pre-training
            opt = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)

            self.q_network.train()
            n = len(X)
            for ep in range(epochs):
                perm = torch.randperm(n)
                ep_loss = 0.0
                ep_correct = 0
                for i in range(0, n, batch_size):
                    idx = perm[i:i + batch_size]
                    xb = X_t[idx]
                    yb = y_t[idx]
                    logits = self.q_network(xb)  # treat Q-values as logits
                    loss = F.cross_entropy(logits, yb, weight=class_w)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
                    opt.step()
                    ep_loss += loss.item() * len(yb)
                    ep_correct += (logits.argmax(dim=1) == yb).sum().item()
                if verbose and ep % 10 == 0:
                    print(f"  [PreTrain] ep {ep}/{epochs} "
                          f"loss={ep_loss/n:.4f} acc={ep_correct/n:.3%}")

            # Sync target net to pre-trained weights
            self.target_network.load_state_dict(self.q_network.state_dict())

        # ──────────────────────────────────────────────────────────────────
        def save(self, path: str):
            torch.save({
                "q_network":      self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer":      self.optimizer.state_dict(),
                "epsilon":        self.epsilon,
                "steps":          self.steps,
                "state_size":     self.state_size,
                "action_size":    self.action_size,
            }, path)

        def load(self, path: str) -> bool:
            if not os.path.exists(path):
                return False
            try:
                ck = torch.load(path, map_location=self.device, weights_only=False)
                self.q_network.load_state_dict(ck["q_network"])
                self.target_network.load_state_dict(ck["target_network"])
                try:
                    self.optimizer.load_state_dict(ck["optimizer"])
                except Exception:
                    pass
                self.epsilon = ck.get("epsilon", self.epsilon_min)
                self.steps = ck.get("steps", 0)
                return True
            except Exception as e:
                print(f"[GoldDuelingAgent] Load failed: {e}")
                return False

else:
    # Fallback when torch is unavailable
    class GoldDuelingAgent:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available — cannot use GoldDuelingAgent")
