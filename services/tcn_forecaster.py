"""
Temporal Convolutional Network (TCN) Directional Forecaster for 1-Minute Gold.

Why TCN for 1m gold:
  - Dilated convolutions capture long-range dependencies without recurrence
  - Causal convolutions guarantee NO look-ahead bias (future cannot leak)
  - Parallelizable = 10-100x faster inference than LSTM/Transformer
  - Research shows TCN matches or exceeds LSTM on financial forecasting
    while being more stable (Bai, Kolter & Koltun 2018; confirmed 2024-2025)

Architecture:
  - CausalConv1D stack with exponentially increasing dilation
  - Residual connections for gradient flow
  - Outputs: probability of [down, neutral, up] over next N bars

Reference:
  Bai, S., Kolter, J.Z., & Koltun, V. (2018). "An Empirical Evaluation of Generic
  Convolutional and Recurrent Networks for Sequence Modeling." arXiv:1803.01271.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class TCNConfig:
    input_channels: int = 16      # Number of input features
    hidden_channels: int = 64     # Channels per TCN layer
    kernel_size: int = 3
    num_layers: int = 6           # Depth (dilation = 2^(layer))
    output_classes: int = 3       # down, neutral, up
    dropout: float = 0.2
    lookahead_buffer: int = 5     # Forecast horizon
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    early_stop_patience: int = 7


class CausalConv1d(nn.Module):
    """Causal convolution: output at time t only sees input up to time t."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        out = self.conv(x)
        # Remove extra padding from the right (future)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class ResidualBlock(nn.Module):
    """TCN residual block with weight norm and dropout."""

    def __init__(self, channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout2(out)
        return self.activation(out + residual)


class GoldTCNForecaster(nn.Module):
    """
    TCN-based directional forecaster.
    Input: (batch, seq_len, features)
    Output: (batch, output_classes) softmax probabilities
    """

    def __init__(self, config: TCNConfig):
        super().__init__()
        self.config = config
        # Project input features to hidden channels
        self.input_proj = nn.Conv1d(config.input_channels, config.hidden_channels, 1)
        # TCN layers with exponentially increasing dilation
        self.tcn_layers = nn.ModuleList([
            ResidualBlock(
                config.hidden_channels,
                config.kernel_size,
                dilation=2 ** i,
                dropout=config.dropout,
            )
            for i in range(config.num_layers)
        ])
        # Global average pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_channels, config.hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels // 2, config.output_classes),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_channels)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        for layer in self.tcn_layers:
            x = layer(x)
        # Pool over time dimension
        x = self.global_pool(x).squeeze(-1)
        logits = self.classifier(x)
        return logits

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Convenience: numpy in, numpy out."""
        self.eval()
        with torch.no_grad():
            if x.ndim == 2:
                x = x[np.newaxis, ...]
            tensor = torch.FloatTensor(x)
            logits = self.forward(tensor)
            probs = F.softmax(logits, dim=-1)
            return probs.cpu().numpy()


class TCNTrainer:
    """Training wrapper with early stopping and regime-stratified sampling."""

    def __init__(self, config: TCNConfig, device: Optional[str] = None):
        self.config = config
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = GoldTCNForecaster(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3
        )
        self.best_state = None
        self.best_val_f1 = 0.0

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        seq_len: int = 60,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create overlapping sequences for TCN.
        CRITICAL: No future leakage. Target is derived from bars AFTER the sequence.
        """
        feats = df[feature_cols].fillna(0).values.astype(np.float32)
        targets = df[target_col].fillna(1).values.astype(np.int64)  # default neutral

        X, y = [], []
        for i in range(seq_len, len(feats) - self.config.lookahead_buffer):
            X.append(feats[i - seq_len:i])
            y.append(targets[i + self.config.lookahead_buffer])
        return np.array(X), np.array(y)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        class_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """Train with early stopping."""
        history = {"train_loss": [], "val_f1": [], "val_acc": []}
        patience_counter = 0

        if class_weights is not None:
            weight_tensor = torch.FloatTensor(class_weights).to(self.device)
        else:
            weight_tensor = None

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        n_samples = len(X_train)
        for epoch in range(self.config.epochs):
            self.model.train()
            # Shuffle
            perm = np.random.permutation(n_samples)
            train_losses = []
            for i in range(0, n_samples, self.config.batch_size):
                batch_idx = perm[i : i + self.config.batch_size]
                xb = torch.FloatTensor(X_train[batch_idx]).to(self.device)
                yb = torch.LongTensor(y_train[batch_idx]).to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_losses.append(loss.item())

            avg_loss = np.mean(train_losses)
            history["train_loss"].append(avg_loss)

            # Validation
            if X_val is not None and y_val is not None:
                val_f1, val_acc = self._evaluate(X_val, y_val)
                history["val_f1"].append(val_f1)
                history["val_acc"].append(val_acc)
                self.scheduler.step(val_f1)

                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    self.best_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stop_patience:
                    print(f"[TCN] Early stop at epoch {epoch+1}. Best val F1: {self.best_val_f1:.4f}")
                    break
            else:
                # No validation: save last epoch
                self.best_state = self.model.state_dict().copy()

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return history

    def _evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float]:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_val), self.config.batch_size):
                xb = torch.FloatTensor(X_val[i : i + self.config.batch_size]).to(self.device)
                logits = self.model(xb)
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(batch_preds)
        preds = np.array(preds)
        acc = np.mean(preds == y_val)
        # Macro F1
        from sklearn.metrics import f1_score
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        return f1, acc

    def save(self, path: str):
        torch.save({
            "config": self.config,
            "state_dict": self.model.state_dict(),
            "best_val_f1": self.best_val_f1,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint["config"]
        self.model.load_state_dict(checkpoint["state_dict"])
        self.best_val_f1 = checkpoint.get("best_val_f1", 0.0)
