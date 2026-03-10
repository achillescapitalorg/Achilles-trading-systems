"""
Trading Terminal Services
==========================
Core services for market data, advanced models, RL agent, and news scraping.
"""

from .market_data import MarketDataService, get_market_data_service
from .advanced_models import (
    BlackScholes,
    HestonModel,
    HestonParams,
    SABRModel,
    SABRParams,
    LocalVolatilityModel,
    VarianceGammaModel,
    RoughBergomiModel,
    RegimeSwitchingModel,
    calculate_var,
    calculate_expected_shortfall,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
)

# Import RL agent components conditionally (PyTorch may not be available)
try:
    from .rl_agent import (
        Action,
        QLearningAgent,
        DeepRLAgent,
        TradingEnvironment,
        train_rl_agent,
    )
    RL_AVAILABLE = True
except ImportError:
    Action = None
    QLearningAgent = None
    DeepRLAgent = None
    TradingEnvironment = None
    train_rl_agent = None
    RL_AVAILABLE = False

from .news_scraper import (
    NewsSource,
    NewsArticle,
    ForexFactoryScraper,
    FXStreetScraper,
    InvestingComScraper,
    NewsAggregator,
)

__all__ = [
    # Market Data
    'MarketDataService',
    'get_market_data_service',

    # Advanced Models
    'BlackScholes',
    'HestonModel',
    'HestonParams',
    'SABRModel',
    'SABRParams',
    'LocalVolatilityModel',
    'VarianceGammaModel',
    'RoughBergomiModel',
    'RegimeSwitchingModel',
    'calculate_var',
    'calculate_expected_shortfall',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',

    # RL Agent (conditional)
    'Action',
    'QLearningAgent',
    'DeepRLAgent',
    'TradingEnvironment',
    'train_rl_agent',
    'RL_AVAILABLE',

    # News Scraper
    'NewsSource',
    'NewsArticle',
    'ForexFactoryScraper',
    'FXStreetScraper',
    'InvestingComScraper',
    'NewsAggregator',
]
