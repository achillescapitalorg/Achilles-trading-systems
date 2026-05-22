"""
paper_trading/logger.py
=======================
Paper trading analytics and performance tracking.

Reads trade_log.jsonl and computes:
- Equity curve
- Win/loss breakdown by regime, direction
- Performance metrics (Sharpe, profit factor, avg win/loss)
- Trade history table for dashboard
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


LOG_FILE = Path("data/paper_trades/trade_log.jsonl")


def load_trades(limit: int = 100) -> pd.DataFrame:
    """Load recent trades from JSONL log."""
    if not LOG_FILE.exists():
        return pd.DataFrame()

    records = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Sort by exit time descending, take most recent
    df = df.sort_values("exit_time", ascending=False).head(limit).reset_index(drop=True)
    return df


def compute_equity_curve(initial_balance: float = 10000.0) -> pd.DataFrame:
    """Compute equity curve from trade history."""
    df = load_trades(limit=1000)
    if df.empty:
        return pd.DataFrame({"time": [], "equity": []})

    df = df.sort_values("exit_time").copy()
    df["cumulative_pnl"] = df["pnl"].cumsum()
    df["equity"] = initial_balance + df["cumulative_pnl"]
    df["time"] = pd.to_datetime(df["exit_time"])
    return df[["time", "equity", "pnl"]].copy()


def get_performance_summary(initial_balance: float = 10000.0) -> Dict:
    """Compute full performance summary."""
    df = load_trades(limit=1000)
    if df.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "sharpe_annual": 0.0,
            "total_pnl": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "by_regime": {},
            "by_direction": {},
        }

    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    total_trades = len(df)
    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0

    gross_profit = wins["pnl"].sum() if not wins.empty else 0.0
    gross_loss = abs(losses["pnl"].sum()) if not losses.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win = wins["pnl"].mean() if not wins.empty else 0.0
    avg_loss = losses["pnl"].mean() if not losses.empty else 0.0

    # Consecutive wins/losses
    streaks = _compute_streaks(df.sort_values("exit_time")["pnl"].tolist())
    max_wins = max([s for s, _ in streaks] + [0])
    max_losses = max([s for _, s in streaks] + [0])

    # Sharpe (simplified, daily returns)
    df_sorted = df.sort_values("exit_time").copy()
    df_sorted["date"] = pd.to_datetime(df_sorted["exit_time"]).dt.date
    daily = df_sorted.groupby("date")["pnl"].sum()
    sharpe = 0.0
    if len(daily) > 1 and daily.std() > 0:
        sharpe = (daily.mean() / daily.std()) * np.sqrt(252)

    # By regime
    by_regime = {}
    for regime, group in df.groupby("regime"):
        by_regime[regime] = {
            "trades": len(group),
            "win_rate": (group["pnl"] > 0).sum() / len(group),
            "avg_pnl": group["pnl"].mean(),
            "total_pnl": group["pnl"].sum(),
        }

    # By direction
    by_direction = {}
    for direction, group in df.groupby("direction"):
        by_direction[direction] = {
            "trades": len(group),
            "win_rate": (group["pnl"] > 0).sum() / len(group),
            "avg_pnl": group["pnl"].mean(),
            "total_pnl": group["pnl"].sum(),
        }

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_consecutive_wins": max_wins,
        "max_consecutive_losses": max_losses,
        "sharpe_annual": sharpe,
        "total_pnl": df["pnl"].sum(),
        "best_trade": df["pnl"].max(),
        "worst_trade": df["pnl"].min(),
        "by_regime": by_regime,
        "by_direction": by_direction,
    }


def _compute_streaks(pnls: List[float]) -> List[Tuple[int, int]]:
    """Return list of (win_streak, loss_streak) transitions."""
    streaks = []
    current_wins = 0
    current_losses = 0
    for pnl in pnls:
        if pnl > 0:
            if current_losses > 0:
                streaks.append((0, current_losses))
                current_losses = 0
            current_wins += 1
        else:
            if current_wins > 0:
                streaks.append((current_wins, 0))
                current_wins = 0
            current_losses += 1
    if current_wins > 0:
        streaks.append((current_wins, 0))
    if current_losses > 0:
        streaks.append((0, current_losses))
    return streaks


def get_recent_trades_table(limit: int = 10) -> pd.DataFrame:
    """Get formatted recent trades for dashboard table."""
    df = load_trades(limit=limit)
    if df.empty:
        return pd.DataFrame()

    df["entry_time"] = pd.to_datetime(df["entry_time"]).dt.strftime("%H:%M")
    df["exit_time"] = pd.to_datetime(df["exit_time"]).dt.strftime("%H:%M")
    df["pnl_fmt"] = df["pnl"].apply(lambda x: f"${x:+.2f}")
    df["hold_fmt"] = df["hold_time_min"].apply(lambda x: f"{x:.1f}m" if pd.notna(x) else "—")
    df["direction_icon"] = df["direction"].apply(lambda x: "📈" if x == "BUY" else "📉")

    return df[[
        "direction_icon", "direction", "entry_time", "exit_time",
        "entry_price", "exit_price", "size_lots", "pnl_fmt",
        "hold_fmt", "exit_reason", "regime"
    ]].copy()
