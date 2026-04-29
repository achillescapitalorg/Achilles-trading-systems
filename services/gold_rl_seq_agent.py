"""
Sequence-based CNN Multi-Head Dueling DQN for Gold 1m Trading
==============================================================
Research-backed improvements over single-bar DQN:

  1. Sequence input (last 30 bars × 21 features) — captures temporal patterns
     that single-bar input fundamentally cannot see.
  2. 1D CNN feature extractor — well-established for time series, fast on CPU.
  3. Multi-head architecture:
       - Q-value head (dueling V + A)
       - Auxiliary direction-prediction head (3-class)
       - Auxiliary volatility regression head (next-bar realized vol)
     Multi-task learning regularizes representations and dramatically improves
     generalization (Caruana 1997, recent applications: DeepScalper 2022).
  4. Dropout 0.3 + LayerNorm throughout for overfit resistance.
  5. Huber loss + gradient clipping (robust to reward outliers).
  6. Optional Gaussian noise augmentation on inputs during training.
"""

import os
import random
import numpy as np
from collections import deque
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class SequenceMultiHead(nn.Module):
        """
        CNN sequence encoder + dueling Q-head + auxiliary direction & vol heads.

        Inputs:
          seq:    (B, T, F)   — F=21 static features over T=30 bars
          dyn:    (B, D)      — D=4 dynamic position features

        Outputs:
          q:      (B, A)      — Q-values for A=3 actions
          logits: (B, A)      — direction logits (auxiliary, used in pretraining)
          vol:    (B, 1)      — predicted next-bar volatility (auxiliary)
        """
        def __init__(self, n_features: int = 21, dyn_size: int = 4,
                     n_actions: int = 3, seq_len: int = 30,
                     channels=(32, 64, 64), kernel: int = 3,
                     dropout: float = 0.3):
            super().__init__()
            self.seq_len = seq_len
            self.n_features = n_features
            self.dyn_size = dyn_size
            self.n_actions = n_actions

            # 1D CNN feature extractor along time axis. We treat features as
            # channels and time as length: input shape (B, F, T).
            layers = []
            in_ch = n_features
            for out_ch in channels:
                layers += [
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                in_ch = out_ch
            self.cnn = nn.Sequential(*layers)
            cnn_out = channels[-1]

            # Pool over time dimension (global average + global max for richer rep)
            self.pool_dim = cnn_out * 2

            # Combine pooled CNN output with dynamic position features
            combined = self.pool_dim + dyn_size
            self.shared = nn.Sequential(
                nn.Linear(combined, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Dueling Q heads
            self.value     = nn.Linear(64, 1)
            self.advantage = nn.Linear(64, n_actions)

            # Auxiliary heads
            self.direction = nn.Linear(64, n_actions)   # logits for cross-entropy
            self.vol_head  = nn.Linear(64, 1)           # regression

        def encode(self, seq, dyn):
            # seq: (B, T, F) → (B, F, T) for Conv1d
            x = seq.transpose(1, 2)
            x = self.cnn(x)                              # (B, C_out, T)
            mean_pool = x.mean(dim=-1)
            max_pool, _ = x.max(dim=-1)
            pooled = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 2*C_out)
            combined = torch.cat([pooled, dyn], dim=-1)         # (B, 2*C_out + D)
            return self.shared(combined)                        # (B, 64)

        def forward(self, seq, dyn) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            h = self.encode(seq, dyn)
            v = self.value(h)
            a = self.advantage(h)
            q = v + a - a.mean(dim=-1, keepdim=True)
            logits = self.direction(h)
            vol = self.vol_head(h)
            return q, logits, vol

        def q_only(self, seq, dyn):
            h = self.encode(seq, dyn)
            v = self.value(h)
            a = self.advantage(h)
            return v + a - a.mean(dim=-1, keepdim=True)


    class SequenceAgent:
        """
        Sequence-based dueling double DQN agent with multi-task auxiliary losses.

        Action space: 0=HOLD, 1=BUY, 2=SELL.
        Observation: tuple (seq[T,F], dyn[D]).
        """
        def __init__(
            self,
            n_features: int = 21,
            dyn_size: int = 4,
            n_actions: int = 3,
            seq_len: int = 30,
            learning_rate: float = 0.0003,
            discount_factor: float = 0.97,
            epsilon: float = 1.0,
            epsilon_min: float = 0.05,
            buffer_size: int = 100_000,
            batch_size: int = 128,
            target_update: int = 500,
            dropout: float = 0.3,
            input_noise: float = 0.05,
            aux_weight: float = 0.3,
        ):
            self.n_features = n_features
            self.dyn_size = dyn_size
            self.n_actions = n_actions
            self.seq_len = seq_len
            self.gamma = discount_factor
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.batch_size = batch_size
            self.target_update = target_update
            self.input_noise = input_noise
            self.aux_weight = aux_weight

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.q_network = SequenceMultiHead(n_features, dyn_size, n_actions,
                                                seq_len, dropout=dropout).to(self.device)
            self.target_network = SequenceMultiHead(n_features, dyn_size, n_actions,
                                                     seq_len, dropout=dropout).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()

            self.optimizer = optim.Adam(self.q_network.parameters(),
                                         lr=learning_rate, weight_decay=1e-5)

            self.buffer = deque(maxlen=buffer_size)
            self.steps = 0
            self.losses = []

        # ──────────────────────────────────────────────────────────────────
        def _to_tensor(self, seq, dyn):
            seq_t = torch.FloatTensor(np.asarray(seq, dtype=np.float32)).to(self.device)
            dyn_t = torch.FloatTensor(np.asarray(dyn, dtype=np.float32)).to(self.device)
            if seq_t.dim() == 2:
                seq_t = seq_t.unsqueeze(0)
            if dyn_t.dim() == 1:
                dyn_t = dyn_t.unsqueeze(0)
            return seq_t, dyn_t

        def get_action(self, seq, dyn, training: bool = True) -> int:
            if training and random.random() < self.epsilon:
                return random.randint(0, self.n_actions - 1)
            self.q_network.eval()
            with torch.no_grad():
                seq_t, dyn_t = self._to_tensor(seq, dyn)
                q = self.q_network.q_only(seq_t, dyn_t)
            self.q_network.train()
            return int(q.argmax(dim=-1).item())

        def get_q_values(self, seq, dyn) -> np.ndarray:
            self.q_network.eval()
            with torch.no_grad():
                seq_t, dyn_t = self._to_tensor(seq, dyn)
                q = self.q_network.q_only(seq_t, dyn_t).squeeze(0).cpu().numpy()
            self.q_network.train()
            return q

        def store_transition(self, seq, dyn, action, reward, next_seq, next_dyn, done):
            self.buffer.append((
                np.asarray(seq, dtype=np.float32),
                np.asarray(dyn, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_seq, dtype=np.float32),
                np.asarray(next_dyn, dtype=np.float32),
                bool(done),
            ))

        def update(self):
            if len(self.buffer) < self.batch_size:
                return None
            batch = random.sample(self.buffer, self.batch_size)
            seqs, dyns, actions, rewards, next_seqs, next_dyns, dones = zip(*batch)

            seq_t      = torch.FloatTensor(np.stack(seqs)).to(self.device)
            dyn_t      = torch.FloatTensor(np.stack(dyns)).to(self.device)
            act_t      = torch.LongTensor(actions).to(self.device)
            rew_t      = torch.FloatTensor(rewards).to(self.device)
            nseq_t     = torch.FloatTensor(np.stack(next_seqs)).to(self.device)
            ndyn_t     = torch.FloatTensor(np.stack(next_dyns)).to(self.device)
            done_t     = torch.FloatTensor(dones).to(self.device)

            # Optional input noise for regularization
            if self.input_noise > 0:
                seq_t  = seq_t  + torch.randn_like(seq_t)  * self.input_noise
                nseq_t = nseq_t + torch.randn_like(nseq_t) * self.input_noise

            # Current Q
            q_all = self.q_network.q_only(seq_t, dyn_t)
            current_q = q_all.gather(1, act_t.unsqueeze(1)).squeeze(1)

            # Double DQN target
            with torch.no_grad():
                next_actions = self.q_network.q_only(nseq_t, ndyn_t).argmax(dim=1)
                next_q = self.target_network.q_only(nseq_t, ndyn_t).gather(
                    1, next_actions.unsqueeze(1)).squeeze(1)
                target_q = rew_t + self.gamma * next_q * (1 - done_t)

            loss = F.smooth_l1_loss(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()

            self.steps += 1
            if self.steps % self.target_update == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            self.losses.append(loss.item())
            return loss.item()

        # ──────────────────────────────────────────────────────────────────
        def supervised_pretrain(self, seqs: np.ndarray, dyns: np.ndarray,
                                 dir_labels: np.ndarray, vol_labels: np.ndarray,
                                 epochs: int = 60, batch_size: int = 128,
                                 lr: float = 0.001, val_split: float = 0.15,
                                 patience: int = 10, verbose: bool = True):
            """
            Multi-task supervised pre-training:
              loss = direction_CE + aux_weight * vol_MSE
            Includes early stopping on validation accuracy with patience.
            """
            n = len(seqs)
            if n < 200:
                return 0.0

            # Train/val split
            n_val = int(n * val_split)
            idx = np.random.permutation(n)
            val_idx, tr_idx = idx[:n_val], idx[n_val:]

            seqs_t = torch.FloatTensor(seqs).to(self.device)
            dyns_t = torch.FloatTensor(dyns).to(self.device)
            ydir_t = torch.LongTensor(dir_labels).to(self.device)
            yvol_t = torch.FloatTensor(vol_labels).to(self.device)

            # Class weights to handle imbalance
            counts = np.bincount(dir_labels, minlength=self.n_actions).astype(float)
            counts = np.where(counts == 0, 1.0, counts)
            class_w = torch.FloatTensor(len(dir_labels) / counts).to(self.device)
            class_w /= class_w.sum()
            class_w *= self.n_actions

            opt = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

            best_val_acc = 0.0
            best_state = None
            patience_counter = 0

            for ep in range(epochs):
                self.q_network.train()
                perm = np.random.permutation(len(tr_idx))
                ep_loss = 0.0; ep_dir_correct = 0; ep_count = 0
                for i in range(0, len(tr_idx), batch_size):
                    b = tr_idx[perm[i:i + batch_size]]
                    sb = seqs_t[b]; db = dyns_t[b]
                    ydb = ydir_t[b]; yvb = yvol_t[b]

                    _, logits, vol = self.q_network(sb, db)
                    dir_loss = F.cross_entropy(logits, ydb, weight=class_w)
                    vol_loss = F.smooth_l1_loss(vol.squeeze(-1), yvb)
                    loss = dir_loss + self.aux_weight * vol_loss

                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
                    opt.step()

                    ep_loss += loss.item() * len(b)
                    ep_dir_correct += (logits.argmax(dim=1) == ydb).sum().item()
                    ep_count += len(b)
                scheduler.step()

                # Validation
                self.q_network.eval()
                with torch.no_grad():
                    _, val_logits, _ = self.q_network(seqs_t[val_idx], dyns_t[val_idx])
                    val_acc = (val_logits.argmax(dim=1) == ydir_t[val_idx]).float().mean().item()

                tr_acc = ep_dir_correct / max(ep_count, 1)
                if verbose and (ep % 10 == 0 or ep == epochs - 1):
                    print(f"  [PreTrain] ep {ep:3d}/{epochs} "
                          f"loss={ep_loss/max(ep_count,1):.4f} "
                          f"train_acc={tr_acc:.3%} val_acc={val_acc:.3%}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.detach().clone()
                                  for k, v in self.q_network.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"  [PreTrain] Early stopping at ep {ep} "
                                  f"(no val improvement for {patience} epochs)")
                        break

            if best_state is not None:
                self.q_network.load_state_dict(best_state)
            self.target_network.load_state_dict(self.q_network.state_dict())
            return best_val_acc

        # ──────────────────────────────────────────────────────────────────
        def save(self, path: str):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                "q_network":      self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer":      self.optimizer.state_dict(),
                "epsilon":        self.epsilon,
                "steps":          self.steps,
                "n_features":     self.n_features,
                "dyn_size":       self.dyn_size,
                "n_actions":      self.n_actions,
                "seq_len":        self.seq_len,
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
                print(f"[SeqAgent] Load failed: {e}")
                return False
else:
    class SequenceAgent:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available — SequenceAgent unavailable")
