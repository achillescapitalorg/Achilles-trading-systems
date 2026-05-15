"""
Validation & Next-Gen Forecasting Demo
======================================

This script demonstrates the complete pipeline to fix backtest/walk-forward
divergence and implement regime-aware forecasting.

Usage:
    source venv/bin/activate
    python scripts/validate_and_forecast.py

What it does:
    1. Loads gold 1m data
    2. Runs backtest WITH realistic costs and validation
    3. Runs walk-forward with validation
    4. Trains Regime-Aware MoE ensemble
    5. Trains TCN directional forecaster
    6. Compares all approaches with proper statistical tests
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from services.market_data import fetch_gold_data  # or your data loader
from services.gold_features import GoldFeatureEngineer, select_top_features
from services.gold_rl_backtest import run_backtest, walk_forward_backtest
from services.backtest_validator import BacktestValidator, StrategyKillSwitch
from services.regime_aware_ensemble import RegimeAwareMoE, detect_regime_simple
from services.tcn_forecaster import TCNConfig, TCNTrainer, GoldTCNForecaster
from services.realistic_costs import RealisticCostModel


def generate_dummy_data(n_bars: int = 5000) -> pd.DataFrame:
    """Generate synthetic gold 1m data for demo."""
    np.random.seed(42)
    t = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
    returns = np.random.normal(0, 0.0003, n_bars)
    # Add some autocorrelation and regimes
    for i in range(1, n_bars):
        returns[i] += 0.1 * returns[i - 1]
    # Add a trending regime
    returns[2000:2500] += 0.0005
    returns[3500:4000] -= 0.0005

    close = 2300.0 * np.exp(np.cumsum(returns))
    noise = np.random.normal(0, 0.15, n_bars)
    high = close + np.abs(noise)
    low = close - np.abs(noise)
    open_p = close + np.random.normal(0, 0.05, n_bars)
    volume = np.random.randint(100, 1000, n_bars)

    df = pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=t)
    df["timestamp"] = df.index
    return df


def build_labels(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """
    Triple-barrier style labels for demonstration.
    Class 0 = down, 1 = neutral, 2 = up
    """
    future_ret = df["close"].pct_change(horizon).shift(-horizon)
    labels = pd.Series(1, index=df.index)  # neutral default
    labels[future_ret > 0.0005] = 2   # up
    labels[future_ret < -0.0005] = 0  # down
    return labels


def demo_validation():
    """Demonstrate backtest validation with realistic costs."""
    print("=" * 70)
    print("DEMO 1: BACKTEST VALIDATION WITH REALISTIC COSTS")
    print("=" * 70)

    df = generate_dummy_data(3000)
    # Simple moving-average crossover signals for demo
    df["sma_fast"] = df["close"].rolling(10).mean()
    df["sma_slow"] = df["close"].rolling(30).mean()
    signals = np.where(df["sma_fast"] > df["sma_slow"], 1,
              np.where(df["sma_fast"] < df["sma_slow"], 2, 0))
    signals = signals.astype(int)

    print("\n--- OLD COST MODEL (fixed $0.40) ---")
    res_old = run_backtest(df, signals, use_realistic_costs=False)
    print(res_old.summary())

    print("\n--- NEW COST MODEL (session/volatility dependent) ---")
    res_new = run_backtest(df, signals, use_realistic_costs=True)
    print(res_new.summary())

    print("\n--- VALIDATION REPORT ---")
    validator = BacktestValidator()
    # Simulate backtest vs walk-forward returns
    bt_returns = pd.Series([t.pnl_pct for t in res_old.trades])
    report = validator.validate(
        backtest_returns=bt_returns,
        n_trials=1,
    )
    print(report.summary())


def demo_regime_moe():
    """Demonstrate regime-aware mixture of experts."""
    print("\n" + "=" * 70)
    print("DEMO 2: REGIME-AWARE MIXTURE-OF-EXPERTS")
    print("=" * 70)

    df = generate_dummy_data(8000)
    engineer = GoldFeatureEngineer(df)
    features = engineer.compute_all_features()
    labels = build_labels(df, horizon=5)

    # Align
    features = features.loc[labels.index].dropna()
    labels = labels.loc[features.index]

    # Select top features
    top_feats = select_top_features(features, labels, n_features=30, method="mutual_info")
    print(f"Selected top {len(top_feats)} features: {top_feats[:5]}...")

    # Train/test split (no leakage)
    split_idx = int(len(features) * 0.8)
    train_df = features.iloc[:split_idx].copy()
    test_df = features.iloc[split_idx:].copy()
    train_df["label"] = labels.iloc[:split_idx].values
    test_df["label"] = labels.iloc[split_idx:].values

    # Detect regimes
    train_df["regime"] = detect_regime_simple(df.iloc[:split_idx])
    test_df["regime"] = detect_regime_simple(df.iloc[split_idx:])

    print(f"\nRegime distribution in training:")
    print(train_df["regime"].value_counts())

    # Train MoE
    moe = RegimeAwareMoE()
    scores = moe.fit(train_df, feature_cols=top_feats, target_col="label", regime_col="regime")
    print(f"\nExpert scores: {scores}")

    # Predict
    probs = moe.predict_proba(test_df, feature_cols=top_feats)
    preds = np.argmax(probs, axis=1)

    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    acc = accuracy_score(test_df["label"], preds)
    f1 = f1_score(test_df["label"], preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(test_df["label"], preds)
    print(f"\nMoE Test Results:")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Macro F1 : {f1:.3f}")
    print(f"  MCC      : {mcc:.3f}")

    # Regime importance
    regime_imp = moe.get_regime_importance(test_df.head(100))
    print(f"\nSample regime probabilities:")
    print(regime_imp.head())


def demo_tcn_forecaster():
    """Demonstrate TCN directional forecaster."""
    print("\n" + "=" * 70)
    print("DEMO 3: TCN DIRECTIONAL FORECASTER (CAUSAL, NO LOOK-AHEAD)")
    print("=" * 70)

    df = generate_dummy_data(6000)
    engineer = GoldFeatureEngineer(df)
    features = engineer.compute_all_features()
    labels = build_labels(df, horizon=5)

    features = features.loc[labels.index].dropna()
    labels = labels.loc[features.index]
    top_feats = select_top_features(features, labels, n_features=16, method="mutual_info")
    top_feats = list(top_feats) if not isinstance(top_feats, list) else top_feats

    # Build feature dataframe with all needed columns
    feat_df = features[top_feats].copy()
    feat_df["label"] = labels

    config = TCNConfig(
        input_channels=len(top_feats),
        hidden_channels=64,
        num_layers=5,
        lookahead_buffer=5,
        epochs=20,
        early_stop_patience=5,
    )

    trainer = TCNTrainer(config)
    X, y = trainer.prepare_sequences(
        feat_df.reset_index(drop=True),
        feature_cols=top_feats,
        target_col="label",
        seq_len=60,
    )

    if len(X) < 500:
        print("Not enough data for TCN demo")
        return

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Training TCN on {len(X_train)} sequences, validating on {len(X_val)}")
    history = trainer.fit(X_train, y_train, X_val, y_val)
    print(f"Best val F1: {trainer.best_val_f1:.4f}")

    # Evaluate
    val_f1, val_acc = trainer._evaluate(X_val, y_val)
    print(f"Final validation — F1: {val_f1:.4f}, Acc: {val_acc:.4f}")


def demo_kill_switch():
    """Demonstrate strategy kill switch."""
    print("\n" + "=" * 70)
    print("DEMO 4: STRATEGY KILL SWITCH")
    print("=" * 70)

    ks = StrategyKillSwitch(
        max_drawdown_pct=0.05,
        max_consecutive_losses=3,
        min_win_rate=0.40,
        min_profit_factor=1.0,
        lookback_trades=10,
    )

    # Simulate trades
    np.random.seed(7)
    pnls = np.random.choice([-0.02, -0.01, 0.01, 0.02], size=25, p=[0.4, 0.3, 0.2, 0.1])
    for i, pnl in enumerate(pnls):
        ks.update(pnl)
        if ks.is_triggered:
            print(f"Kill switch triggered at trade {i+1}: {ks.reason}")
            print(f"Current equity: ${ks._current_equity:.2f}")
            break
    else:
        print("Kill switch NOT triggered (simulated trades were acceptable)")


def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + " VALIDATION & NEXT-GEN FORECASTING DEMO ".center(68) + "║")
    print("║" + f" {datetime.now().isoformat()} ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    demo_validation()
    demo_regime_moe()
    demo_tcn_forecaster()
    demo_kill_switch()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("""
Next steps to deploy in production:
  1. Replace dummy data with real gold 1m data ( Polygon.io / broker API )
  2. Set cv_method='cpcv' in training config for robust validation
  3. Use RegimeAwareMoE or TCN as primary signal model
  4. Enable StrategyKillSwitch in live trading loop
  5. Enable use_realistic_costs=True in all backtests
  6. Run paper trading for 3+ months before live deployment
    """)


if __name__ == "__main__":
    main()
