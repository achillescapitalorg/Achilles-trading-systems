"""
Paper Trading Analytics
=======================
Performance summary, equity curve, and trade log analysis.
"""
from .logger import load_trades, get_performance_summary, compute_equity_curve

__all__ = ["load_trades", "get_performance_summary", "compute_equity_curve"]
