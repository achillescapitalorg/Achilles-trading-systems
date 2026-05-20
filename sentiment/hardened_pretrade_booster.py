"""
HARDENED PRE-TRADE BOOSTER
Uses the 5-layer hardened sentiment system.
Only boosts if sentiment passed ALL validation layers.
"""
import pandas as pd
from typing import Dict

from .sentiment_hardened import HardenedSentimentAnalyzer, HardenedSentimentConfig


class HardenedPreTradeBooster:
    """
    Uses the 5-layer hardened sentiment system.
    Only boosts if sentiment passed ALL validation layers.
    """

    def __init__(self):
        self.analyzer = HardenedSentimentAnalyzer()
        self.config = HardenedSentimentConfig()

    def apply(self, trade_signal: Dict, headlines_df: pd.DataFrame,
              price_df: pd.DataFrame) -> Dict:
        """Apply hardened sentiment to trade signal."""
        result = trade_signal.copy()

        # Run full 5-layer sentiment analysis
        sentiment = self.analyzer.compute_aggregate(headlines_df, price_df)

        result['sentiment_raw_score'] = sentiment['score']
        result['sentiment_valid'] = sentiment['valid']
        result['sentiment_reason'] = sentiment['reason']
        result['sentiment_sources'] = sentiment.get('sources', 0)
        result['l1_headlines'] = sentiment.get('l1_headlines', 0)
        result['l2_real_headlines'] = sentiment.get('l2_real_headlines', 0)
        result['trigger_type'] = sentiment.get('trigger_type', '')
        result['cluster_count'] = sentiment.get('cluster_count', 0)

        # If sentiment invalid (any layer failed), pass through unchanged but log
        if not sentiment['valid']:
            result['sentiment_adjusted'] = False
            result['sentiment_blocked'] = False
            return result

        # Valid sentiment — apply boost
        action = trade_signal['action']
        score = sentiment['score']
        direction = sentiment['direction']

        # Agreement check
        if action == 'BUY' and direction == 'negative':
            multiplier = 0.70
            result['sentiment_blocked'] = True if trade_signal['confidence'] * 0.70 < 0.55 else False
        elif action == 'SELL' and direction == 'positive':
            multiplier = 0.70
            result['sentiment_blocked'] = True if trade_signal['confidence'] * 0.70 < 0.55 else False
        elif action == 'BUY' and direction == 'positive':
            multiplier = 1.20 if score > 0.4 else 1.10
            result['sentiment_blocked'] = False
        elif action == 'SELL' and direction == 'negative':
            multiplier = 1.20 if score < -0.4 else 1.10
            result['sentiment_blocked'] = False
        else:
            multiplier = 1.0
            result['sentiment_blocked'] = False

        # Apply
        new_conf = trade_signal['confidence'] * multiplier
        result['confidence'] = min(1.0, new_conf)
        result['sentiment_boost'] = multiplier
        result['sentiment_adjusted'] = True

        if result['sentiment_blocked']:
            result['action'] = 'HOLD'

        return result

    def analyze_headlines_only(self, headlines_df: pd.DataFrame) -> pd.DataFrame:
        """Process headlines through fake news guard and return analyzed DataFrame."""
        records = []
        for _, row in headlines_df.iterrows():
            analyzed = self.analyzer.analyze_headline(
                title=row.get('title', ''),
                summary=row.get('summary', ''),
                source=row.get('source', 'unknown'),
                published=row.get('published', pd.Timestamp.now())
            )
            records.append(analyzed)
        return pd.DataFrame(records)
