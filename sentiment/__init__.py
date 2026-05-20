"""
Hardened Sentiment Analysis Module
===================================
5-layer defense against fake news, bias, and black swans.
"""
from .sentiment_hardened import (
    HardenedSentimentConfig,
    FakeNewsGuard,
    CrossSourceValidator,
    PriceImpactValidator,
    BlackSwanDetector,
    HardenedSentimentAnalyzer,
)
from .hardened_pretrade_booster import HardenedPreTradeBooster

__all__ = [
    "HardenedSentimentConfig",
    "FakeNewsGuard",
    "CrossSourceValidator",
    "PriceImpactValidator",
    "BlackSwanDetector",
    "HardenedSentimentAnalyzer",
    "HardenedPreTradeBooster",
]
