"""
End-to-end Gold RL training + backtest demonstration.
Runs the new sequence-based CNN multi-head model and reports honest metrics.
"""
import warnings
warnings.filterwarnings("ignore")
import os, sys, time
import numpy as np
import pandas as pd

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

# Clean any stale models so we start fresh
for f in ["data/gold_rl_seq_model.pt", "data/gold_rl_scaler.pkl"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"[Setup] Removed stale {f}")

print("=" * 70)
print("GOLD RL — End-to-End Training + Backtest")
print("=" * 70)

# ── 1. Train the sequence model ──────────────────────────────────────────
from services.gold_rl_trainer import GoldRLTrainer
trainer = GoldRLTrainer()

t0 = time.time()
print(f"\n[1/3] Training sequence CNN model...")
metrics = trainer.train_sequence_model(pretrain_epochs=40, months_back=1)
elapsed = time.time() - t0
print(f"\n  Training time: {elapsed:.1f}s")
print(f"  Best val acc:  {metrics.get('best_val_acc', 0):.2%}")
print(f"  Test acc:      {metrics.get('test_direction_acc', 0):.2%}")

# ── 2. Run backtest on truly unseen data ─────────────────────────────────
print(f"\n[2/3] Running backtest on held-out data...")
import yfinance as yf
df = yf.download("GC=F", period="60d", interval="15m", progress=False, auto_adjust=True)
df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
df = df[["open", "high", "low", "close", "volume"]].dropna().reset_index()
df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)

# Use last 25% as held-out (not seen during training)
held_out_start = int(len(df) * 0.75)
held_out = df.iloc[held_out_start:].reset_index(drop=True)
print(f"  Held-out bars: {len(held_out)}")

# Generate signals on held-out data
signals, confs = trainer.generate_seq_signals(held_out)

# Need feature df for ATR (for SL/TP sizing)
held_out_feat = trainer.prepare_features(held_out.copy())
# Align signals length to feat length
min_len = min(len(held_out_feat), len(signals))
held_out_feat = held_out_feat.iloc[:min_len].reset_index(drop=True)
signals = signals[:min_len]
confs = confs[:min_len]

print(f"  Signal distribution: HOLD={int((signals==0).sum())}, "
      f"BUY={int((signals==1).sum())}, SELL={int((signals==2).sum())}")

# ── 3. Run backtester ────────────────────────────────────────────────────
print(f"\n[3/3] Running rigorous backtest with realistic costs...")
from services.gold_rl_backtest import run_backtest

result = run_backtest(
    held_out_feat, signals,
    initial_capital=10_000.0,
    risk_per_trade=0.01,
    sl_atr_mult=2.0,
    tp_atr_mult=4.0,
    confidence=confs,
    min_confidence=0.40,
    n_trials_for_dsr=10,
)

print(result.summary())

# ── Per-fold breakdown for honest reporting ─────────────────────────────
print("\n[Bonus] Direction accuracy by signal class on held-out test:")
buy_mask = signals == 1
sell_mask = signals == 2
hold_mask = signals == 0

closes = held_out_feat["close"].values
fwd5 = pd.Series(closes).pct_change(5).shift(-5).fillna(0).values

if buy_mask.sum() > 0:
    buy_correct = ((fwd5[buy_mask] > 0).sum() / buy_mask.sum())
    print(f"  BUY  precision : {buy_correct:.2%}  ({int(buy_mask.sum())} signals)")
if sell_mask.sum() > 0:
    sell_correct = ((fwd5[sell_mask] < 0).sum() / sell_mask.sum())
    print(f"  SELL precision : {sell_correct:.2%}  ({int(sell_mask.sum())} signals)")
print(f"  Overall directional accuracy on actionable signals: "
      f"{result.direction_accuracy:.2%}")
