#!/usr/bin/env python3
"""
Gold/BTC Trading Bot - Main Entry Point
========================================
Professional algorithmic trading bot for Gold (XAUUSD) and Bitcoin (BTCUSD).

Usage:
    python run_bot.py [--paper] [--live] [--backtest] [--config CONFIG]

Examples:
    python run_bot.py --paper          # Run in paper trading mode
    python run_bot.py --live           # Run with live Exness account
    python run_bot.py --backtest       # Run backtest
    python run_bot.py --config my_config.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from trading_bot import (
    GoldBTCTradingBot,
    BotConfig,
    create_bot_config,
)
from backtester import Backtester, print_backtest_report
from technical_indicators import TechnicalIndicators
from quantitative_signals import QuantitativeSignals


def load_config(config_path: str = None) -> dict:
    """Load configuration from JSON file."""
    default_config = Path(__file__).parent / 'config.json'
    
    if config_path:
        config_file = Path(config_path)
    else:
        config_file = default_config
        
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Config file not found: {config_file}")
        return {}


def run_paper_trading(config: dict):
    """Run bot in paper trading mode."""
    print("\n" + "=" * 70)
    print("GOLD/BTC TRADING BOT - PAPER TRADING MODE")
    print("=" * 70)
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Assets: {config.get('trading', {}).get('assets', ['XAUUSD', 'BTCUSD'])}")
    print(f"Initial Capital: $10,000 (virtual)")
    print("-" * 70)
    
    # Create bot configuration
    bot_config = create_bot_config(
        paper_trading=True,
        assets=config.get('trading', {}).get('assets'),
        risk_per_trade=config.get('risk', {}).get('risk_per_trade', 1.0),
        max_drawdown=config.get('risk', {}).get('max_drawdown', 10.0),
        min_confidence=config.get('risk', {}).get('min_confidence', 0.6),
    )
    
    # Create and run bot
    bot = GoldBTCTradingBot(bot_config)
    
    try:
        # Run a few cycles for demo
        cycle_seconds = config.get('trading', {}).get('cycle_seconds', 60)
        max_cycles = 5  # Run 5 cycles for demo
        
        for i in range(max_cycles):
            print(f"\n[Cycle {i+1}/{max_cycles}]")
            bot.run_cycle()
            
            if i < max_cycles - 1:
                print(f"Waiting {cycle_seconds} seconds for next cycle...")
                # Shorter wait for demo
                time.sleep(min(cycle_seconds, 10))
                
    except KeyboardInterrupt:
        print("\n\nBot stopped by user")
    finally:
        bot.disconnect()
        
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    account = bot.broker.get_account_info()
    stats = bot.risk_manager.get_statistics()
    
    print(f"\n  Initial Balance:   $10,000.00")
    print(f"  Final Balance:     ${account.get('balance', 0):,.2f}")
    print(f"  Total PnL:         ${account.get('balance', 0) - 10000:,.2f}")
    print(f"  Total Trades:      {stats.get('total_trades', 0)}")
    print(f"  Win Rate:          {stats.get('win_rate', 0)*100:.1f}%")
    print(f"  Max Drawdown:      {stats.get('max_drawdown', 0):.2f}%")
    
    pnl_percent = (account.get('balance', 0) - 10000) / 10000 * 100
    print(f"  Return:            {pnl_percent:.2f}%")
    print("\n" + "=" * 70)


def run_live_trading(config: dict):
    """Run bot with live Exness account."""
    print("\n" + "=" * 70)
    print("GOLD/BTC TRADING BOT - LIVE TRADING MODE")
    print("=" * 70)
    print("\n⚠️  WARNING: This will execute REAL trades with REAL money!")
    print("-" * 70)
    
    exness_config = config.get('exness', {})
    
    if not exness_config.get('login'):
        print("\nError: MT5 login credentials not configured!")
        print("Please set your Exness MT5 credentials in config.json")
        return
        
    # Create bot configuration
    bot_config = BotConfig(
        paper_trading=False,
        assets=config.get('trading', {}).get('assets', ['XAUUSD', 'BTCUSD']),
        mt5_login=exness_config.get('login'),
        mt5_password=exness_config.get('password'),
        mt5_server=exness_config.get('server', 'Exness-MT5Trial'),
        risk_per_trade=config.get('risk', {}).get('risk_per_trade', 1.0),
        max_drawdown=config.get('risk', {}).get('max_drawdown', 10.0),
        min_confidence=config.get('risk', {}).get('min_confidence', 0.6),
    )
    
    # Create and run bot
    bot = GoldBTCTradingBot(bot_config)
    
    try:
        bot.start(cycle_seconds=config.get('trading', {}).get('cycle_seconds', 300))
    except KeyboardInterrupt:
        print("\n\nBot stopped by user")
    finally:
        bot.disconnect()


def run_backtest(config: dict, symbol: str = 'XAUUSD'):
    """Run backtest on historical data."""
    print("\n" + "=" * 70)
    print("GOLD/BTC TRADING BOT - BACKTEST MODE")
    print("=" * 70)
    
    # For demo, create synthetic data
    # In production, load real historical data
    np = __import__('numpy')
    pd = __import__('pandas')
    
    # Generate synthetic price data with realistic characteristics
    n_days = 500
    np.random.seed(42)
    
    if symbol == 'XAUUSD':
        base_price = 2000
        daily_vol = 0.012
    else:  # BTCUSD
        base_price = 40000
        daily_vol = 0.03
        
    # Generate returns with volatility clustering
    returns = np.random.normal(0.0001, daily_vol, n_days)
    
    # Add some GARCH-like volatility clustering
    volatility = np.ones(n_days) * daily_vol
    for i in range(1, n_days):
        volatility[i] = 0.9 * volatility[i-1] + 0.1 * abs(returns[i-1])
        returns[i] = np.random.normal(0.0001, volatility[i])
    
    # Generate prices
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV dataframe
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.002, 0.002, n_days)),
        'high': prices * (1 + np.random.uniform(0, 0.015, n_days)),
        'low': prices * (1 - np.random.uniform(0, 0.015, n_days)),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_days) * 100,
    })
    
    print(f"\nSymbol: {symbol}")
    print(f"Data range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"Total bars: {len(df)}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Generate signals using our strategy
    indicators = TechnicalIndicators(df)
    quant = QuantitativeSignals(df)
    
    # Generate combined signals
    signals = pd.Series(0, index=df.index)
    
    for i in range(50, len(df)):
        subset_df = df.iloc[:i].copy()
        
        # Technical signals
        tech_ind = TechnicalIndicators(subset_df)
        tech_signal, tech_conf = tech_ind.get_aggregated_signal()
        
        # Quantitative signals
        quant_sig = QuantitativeSignals(subset_df)
        quant_signal, quant_conf, _ = quant_sig.get_aggregated_signal()
        
        # Combine signals
        signal_map = {'BUY': 1, 'SELL': -1, 'HOLD': 0, 'NEUTRAL': 0}
        
        combined_score = (
            signal_map.get(tech_signal, 0) * tech_conf * 0.5 +
            signal_map.get(quant_signal, 0) * quant_conf * 0.5
        )
        
        if combined_score > 0.3:
            signals.iloc[i] = 1
        elif combined_score < -0.3:
            signals.iloc[i] = -1
        else:
            signals.iloc[i] = 0
            
    # Run backtest
    backtester = Backtester(
        initial_capital=10000,
        commission=0.0001,
        slippage=0.0005,
    )
    
    result = backtester.run(df, signals, symbol)
    
    # Print report
    print_backtest_report(result)
    
    # Monte Carlo simulation
    if result.trades:
        print("\n[MONTE CARLO SIMULATION - 1000 runs]")
        mc_results = backtester.monte_carlo_simulation(result.trades, n_simulations=1000)
        
        print(f"  Median Final Equity:    ${mc_results.get('median_final_equity', 0):,.2f}")
        print(f"  5th Percentile:         ${mc_results.get('final_equity_5th_percentile', 0):,.2f}")
        print(f"  95th Percentile:        ${mc_results.get('final_equity_95th_percentile', 0):,.2f}")
        print(f"  Probability of Profit:  {mc_results.get('prob_profit', 0)*100:.1f}%")
        print(f"  Median Max DD:          {mc_results.get('median_max_dd', 0):.2f}%")
        print(f"  Worst Max DD:           {mc_results.get('worst_max_dd', 0):.2f}%")
    
    print("\n" + "=" * 70)


def run_demo_analysis(config: dict):
    """Run a single analysis cycle with detailed output."""
    print("\n" + "=" * 70)
    print("GOLD/BTC TRADING BOT - DEMO ANALYSIS")
    print("=" * 70)
    
    bot_config = create_bot_config(paper_trading=True)
    bot = GoldBTCTradingBot(bot_config)
    bot.connect()
    
    for symbol in bot_config.assets:
        print(f"\n{'='*50}")
        print(f"ANALYZING: {symbol}")
        print('='*50)
        
        signal = bot.run_analysis(symbol)
        
        print(f"\n[FINAL SIGNAL]")
        print(f"  Action:           {signal.action}")
        print(f"  Confidence:       {signal.confidence:.2%}")
        print(f"  Entry Price:      ${signal.entry_price:.2f}" if signal.entry_price else "  Entry Price:      N/A")
        print(f"  Stop Loss:        ${signal.stop_loss:.2f}" if signal.stop_loss else "  Stop Loss:        N/A")
        print(f"  Take Profit:      ${signal.take_profit:.2f}" if signal.take_profit else "  Take Profit:      N/A")
        print(f"  Position Size:    {signal.position_size:.4f} lots")
        
        print(f"\n[SIGNAL BREAKDOWN]")
        reasoning = signal.reasoning
        print(f"  Technical:        {reasoning['technical']['signal']} ({reasoning['technical']['confidence']:.2f})")
        print(f"  Quantitative:     {reasoning['quantitative']['signal']} ({reasoning['quantitative']['confidence']:.2f})")
        print(f"  Sentiment:        {reasoning['sentiment']['signal']} ({reasoning['sentiment']['confidence']:.2f})")
        print(f"  Total Score:      {reasoning['total_score']:.3f}")
        print(f"  Sources Agree:    {reasoning['sources_agree']}")
        
    bot.disconnect()
    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Gold/BTC Trading Bot - Professional Algorithmic Trading System'
    )
    
    parser.add_argument(
        '--paper',
        action='store_true',
        help='Run in paper trading mode (default)'
    )
    
    parser.add_argument(
        '--live',
        action='store_true',
        help='Run with live Exness account'
    )
    
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run backtest'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo analysis'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='XAUUSD',
        help='Symbol for backtest (XAUUSD or BTCUSD)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine mode
    if args.live:
        run_live_trading(config)
    elif args.backtest:
        run_backtest(config, args.symbol)
    elif args.demo:
        run_demo_analysis(config)
    else:
        # Default to paper trading
        run_paper_trading(config)


if __name__ == '__main__':
    main()
