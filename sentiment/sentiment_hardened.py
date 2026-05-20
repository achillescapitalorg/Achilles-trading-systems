"""
HARDENED GOLD SENTIMENT SYSTEM
5-layer defense against fake news, bias, and black swans.
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass


# ─── CONFIGURATION ───

@dataclass
class HardenedSentimentConfig:
    """Production-hardened sentiment configuration."""

    # Layer 1: Source Tiers
    SOURCE_TIERS = {
        'reuters': {'tier': 1, 'weight': 1.0, 'rss': 'https://feeds.reuters.com/reuters/commoditiesNews'},
        'bloomberg': {'tier': 1, 'weight': 1.0, 'api': True},
        'ft': {'tier': 1, 'weight': 1.0, 'rss': 'https://www.ft.com/?format=rss'},
        'kitco': {'tier': 2, 'weight': 0.8, 'rss': 'https://www.kitco.com/rss/gold.xml'},
        'bullionvault': {'tier': 2, 'weight': 0.8, 'rss': 'https://www.bullionvault.com/gold-news/rss.xml'},
        'forexlive': {'tier': 2, 'weight': 0.8, 'rss': 'https://www.forexlive.com/feed/gold'},
        'fxempire': {'tier': 2, 'weight': 0.8, 'rss': 'https://www.fxempire.com/feed/?tag=gold'},
        'newsapi': {'tier': 3, 'weight': 0.5, 'api': True},
        'reddit': {'tier': 3, 'weight': 0.4, 'api': True},
        'twitter': {'tier': 3, 'weight': 0.4, 'api': True},
    }

    # Layer 2: Fake News Detection
    SENSATIONALISM_PATTERNS = [
        r'\b(shock|shocking|bombshell|explosive|devastating|collapse|implode|meltdown)\b',
        r'\b(gold to \$\d{3,}|gold will hit|gold set to|inevitable crash|guaranteed)\b',
        r'[!?]{2,}',
    ]

    # Layer 3: Consensus
    MIN_SOURCES_FOR_CONSENSUS = 2
    CONSENSUS_WINDOW_MINUTES = 10
    CONFLICT_ACTION = 'HOLD'

    # Layer 4: Price Impact
    PRICE_IMPACT_WINDOW_BARS = 5
    MIN_PRICE_MOVE_PCT = 0.03
    MAX_PRICE_MOVE_PCT = 0.5

    # Layer 5: Black Swan Cooldown
    BLACK_SWAN_SENTIMENT_THRESHOLD = 0.8
    BLACK_SWAN_HEADLINE_CLUSTER = 3
    COOLDOWN_MINUTES = 15

    # General
    LOOKBACK_MINUTES = 10
    PRE_TRADE_WINDOW = 3
    TIME_DECAY_HALF_LIFE = 3.5


# ─── LAYER 2: FAKE NEWS GUARD ───

class FakeNewsGuard:
    """Detects fake/manipulated/sensationalized news."""

    def __init__(self, config: HardenedSentimentConfig):
        self.config = config

    def analyze(self, headline: Dict) -> Dict:
        """Returns: {'is_fake': bool, 'risk_score': 0-1, 'reasons': [str]}"""
        text = f"{headline.get('title', '')} {headline.get('summary', '')}"
        text_lower = text.lower()
        reasons = []
        risk = 0.0

        # Check 1: Sensationalism score
        sensationalism = 0
        for pattern in self.config.SENSATIONALISM_PATTERNS:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            sensationalism += matches
        if sensationalism >= 2:
            risk += 0.4
            reasons.append(f"High sensationalism ({sensationalism} triggers)")
        elif sensationalism == 1:
            risk += 0.15
            reasons.append("Moderate sensationalism")

        # Check 2: Source credibility
        source = headline.get('source', '').lower()
        tier = self.config.SOURCE_TIERS.get(source, {}).get('tier', 3)
        if tier == 3 and not headline.get('verified', False):
            risk += 0.2
            reasons.append("Low-credibility source (Tier 3)")

        # Check 3: Extreme claims without data
        extreme_claims = [
            r'gold (surges|plunges|crashes|rallies) \d+%',
            r'(\d{2,})% (crash|rally|surge|plunge)',
        ]
        for pattern in extreme_claims:
            if re.search(pattern, text_lower):
                risk += 0.25
                reasons.append("Extreme percentage claim without verification")
                break

        # Check 4: Unicode homoglyph / hidden text
        ascii_only = text.encode('ascii', 'ignore').decode()
        if len(text) > len(ascii_only) + 5:
            risk += 0.5
            reasons.append("WARNING: Hidden Unicode characters detected (possible adversarial attack)")

        # Check 5: Single-source exclusivity
        if headline.get('exclusive', False) or 'exclusive' in text_lower:
            risk += 0.15
            reasons.append("Exclusive/single-source claim")

        is_fake = risk > 0.5
        return {
            'is_fake': is_fake,
            'risk_score': min(1.0, risk),
            'reasons': reasons,
            'tier': tier
        }


# ─── LAYER 3: CROSS-SOURCE VALIDATOR ───

class CrossSourceValidator:
    """Require multiple sources to agree."""

    def __init__(self, config: HardenedSentimentConfig):
        self.config = config

    def validate(self, analyzed_headlines: pd.DataFrame) -> Dict:
        """Returns consensus direction only if >=2 sources agree."""
        if analyzed_headlines.empty:
            return {'valid': False, 'direction': 'neutral', 'confidence': 0, 'sources': 0}

        positive = analyzed_headlines[analyzed_headlines['direction'] == 'positive']
        negative = analyzed_headlines[analyzed_headlines['direction'] == 'negative']

        pos_sources = positive['source'].nunique() if not positive.empty else 0
        neg_sources = negative['source'].nunique() if not negative.empty else 0

        pos_weight = positive['source_weight'].sum() if not positive.empty else 0
        neg_weight = negative['source_weight'].sum() if not negative.empty else 0

        if pos_sources >= self.config.MIN_SOURCES_FOR_CONSENSUS and pos_weight > neg_weight * 1.2:
            return {
                'valid': True,
                'direction': 'positive',
                'confidence': min(1.0, pos_weight / 3.0),
                'sources': pos_sources,
                'opposing': neg_sources
            }
        elif neg_sources >= self.config.MIN_SOURCES_FOR_CONSENSUS and neg_weight > pos_weight * 1.2:
            return {
                'valid': True,
                'direction': 'negative',
                'confidence': min(1.0, neg_weight / 3.0),
                'sources': neg_sources,
                'opposing': pos_sources
            }
        else:
            return {
                'valid': False,
                'direction': 'neutral',
                'confidence': 0,
                'sources': max(pos_sources, neg_sources),
                'opposing': min(pos_sources, neg_sources),
                'reason': f"Insufficient consensus: {pos_sources} pos vs {neg_sources} neg sources"
            }


# ─── LAYER 4: PRICE IMPACT VALIDATOR ───

class PriceImpactValidator:
    """Verify news actually moved gold."""

    def __init__(self, config: HardenedSentimentConfig):
        self.config = config

    def validate(self, news_timestamp: datetime, news_direction: str,
                 price_df: pd.DataFrame) -> Dict:
        """Check price movement in the 5 minutes after news."""
        if price_df.empty or len(price_df) < self.config.PRICE_IMPACT_WINDOW_BARS:
            return {'valid': False, 'reason': 'Insufficient price data'}

        mask = price_df.index >= news_timestamp
        if not mask.any():
            return {'valid': False, 'reason': 'News newer than price data'}

        post_news = price_df[mask].head(self.config.PRICE_IMPACT_WINDOW_BARS)
        if len(post_news) < 2:
            return {'valid': False, 'reason': 'Not enough post-news bars'}

        price_at_news = post_news['close'].iloc[0]
        price_after = post_news['close'].iloc[-1]
        price_change_pct = (price_after - price_at_news) / price_at_news * 100

        # Black swan check
        if abs(price_change_pct) > self.config.MAX_PRICE_MOVE_PCT:
            return {
                'valid': False,
                'reason': f'EXTREME MOVE: {price_change_pct:.2f}% in 5 min — black swan cooldown activated',
                'price_change': price_change_pct,
                'is_black_swan': True
            }

        # Direction validation
        if news_direction == 'positive' and price_change_pct < -self.config.MIN_PRICE_MOVE_PCT:
            return {
                'valid': False,
                'reason': f'News bullish but price dropped {price_change_pct:.2f}% — fake or already priced in',
                'price_change': price_change_pct
            }
        elif news_direction == 'negative' and price_change_pct > self.config.MIN_PRICE_MOVE_PCT:
            return {
                'valid': False,
                'reason': f'News bearish but price rose +{price_change_pct:.2f}% — fake or already priced in',
                'price_change': price_change_pct
            }
        elif abs(price_change_pct) < self.config.MIN_PRICE_MOVE_PCT:
            return {
                'valid': False,
                'reason': f'Price barely moved ({price_change_pct:.3f}%) — news is noise',
                'price_change': price_change_pct
            }

        return {
            'valid': True,
            'price_change': price_change_pct,
            'reason': f'Price validated: {price_change_pct:.3f}% move'
        }


# ─── LAYER 5: BLACK SWAN DETECTOR ───

class BlackSwanDetector:
    """Detect abnormal news clusters suggesting manipulation."""

    def __init__(self, config: HardenedSentimentConfig):
        self.config = config
        self.headline_history = []
        self.cooldown_until = None
        self.last_trigger_type = None  # remember what triggered the cooldown
        self.last_cluster_count = 0

    def check(self, analyzed_headlines: pd.DataFrame, current_time: datetime) -> Dict:
        """Returns: {'is_black_swan': bool, 'cooldown_active': bool, 'reason': str, 'trigger_type': str}"""
        if self.cooldown_until and current_time < self.cooldown_until:
            return {
                'is_black_swan': False,
                'cooldown_active': True,
                'reason': f'Cooldown active until {self.cooldown_until.strftime("%H:%M:%S")}',
                'action': 'HOLD',
                'trigger_type': self.last_trigger_type or 'unknown',
                'cluster_count': self.last_cluster_count,
            }

        if analyzed_headlines.empty:
            return {'is_black_swan': False, 'cooldown_active': False, 'reason': 'No headlines'}

        # Check 1: Extreme sentiment concentration
        max_sentiment = analyzed_headlines['adjusted_score'].abs().max()
        if max_sentiment > self.config.BLACK_SWAN_SENTIMENT_THRESHOLD:
            self._activate_cooldown(current_time)
            self.last_trigger_type = 'extreme_score'
            self.last_cluster_count = 0
            return {
                'is_black_swan': True,
                'cooldown_active': True,
                'reason': f'Extreme sentiment score ({max_sentiment:.2f}) — possible manipulation',
                'action': 'HOLD',
                'trigger_type': 'extreme_score',
                'cluster_count': 0,
            }

        # Check 2: Headline cluster bomb
        recent = analyzed_headlines[analyzed_headlines['age_minutes'] <= 2]
        cluster_count = len(recent)
        if cluster_count >= self.config.BLACK_SWAN_HEADLINE_CLUSTER:
            directions = recent['direction'].value_counts()
            if directions.iloc[0] / cluster_count > 0.8:
                self._activate_cooldown(current_time)
                self.last_trigger_type = 'cluster'
                self.last_cluster_count = cluster_count
                return {
                    'is_black_swan': True,
                    'cooldown_active': True,
                    'reason': f'{cluster_count} headlines in 2 min, {directions.iloc[0]} same direction — coordinated campaign?',
                    'action': 'HOLD',
                    'trigger_type': 'cluster',
                    'cluster_count': cluster_count,
                }

        return {
            'is_black_swan': False,
            'cooldown_active': False,
            'reason': 'Normal news flow'
        }

    def _activate_cooldown(self, current_time: datetime):
        self.cooldown_until = current_time + timedelta(minutes=self.config.COOLDOWN_MINUTES)


# ─── FULL 5-LAYER ANALYZER ───

class HardenedSentimentAnalyzer:
    """Full 5-layer hardened sentiment system."""

    def __init__(self, config: Optional[HardenedSentimentConfig] = None):
        self.config = config or HardenedSentimentConfig()
        self.fake_guard = FakeNewsGuard(self.config)
        self.consensus = CrossSourceValidator(self.config)
        self.price_validator = PriceImpactValidator(self.config)
        self.black_swan = BlackSwanDetector(self.config)

    def analyze_headline(self, title: str, summary: str, source: str,
                         published: datetime) -> Dict:
        """Layer 1 + 2: Source weighting + Fake news check."""
        headline = {
            'title': title,
            'summary': summary,
            'source': source,
            'published': published,
            'age_minutes': (datetime.now() - published).total_seconds() / 60
        }

        source_info = self.config.SOURCE_TIERS.get(source.lower(), {'tier': 3, 'weight': 0.3})
        headline['source_weight'] = source_info['weight']
        headline['tier'] = source_info['tier']

        fake_check = self.fake_guard.analyze(headline)
        headline['fake_risk'] = fake_check['risk_score']
        headline['is_fake'] = fake_check['is_fake']
        headline['fake_reasons'] = fake_check['reasons']

        if fake_check['is_fake']:
            headline['direction'] = 'neutral'
            headline['adjusted_score'] = 0
            headline['impact_multiplier'] = 0
            return headline

        # Basic keyword sentiment
        text = f"{title} {summary}".lower()
        bullish = sum(1 for kw in ['surge', 'rally', 'soar', 'breakout', 'rate cut', 'dovish', 'safe haven', 'hedge'] if kw in text)
        bearish = sum(1 for kw in ['plunge', 'crash', 'tumble', 'sell-off', 'rate hike', 'hawkish', 'strong dollar', 'dump'] if kw in text)

        score = (bullish - bearish) / max(bullish + bearish, 1)
        score = max(-1, min(1, score))

        impact = 1.0
        if any(x in text for x in ['fed', 'federal reserve', 'powell']):
            impact = 2.0
        elif any(x in text for x in ['war', 'conflict', 'geopolitical', 'sanctions']):
            impact = 1.6
        elif any(x in text for x in ['inflation', 'cpi', 'ppi']):
            impact = 1.8

        headline['direction'] = 'positive' if score > 0.1 else 'negative' if score < -0.1 else 'neutral'
        headline['adjusted_score'] = score * impact * source_info['weight']
        headline['impact_multiplier'] = impact

        return headline

    def compute_aggregate(self, headlines_df: pd.DataFrame,
                          price_df: pd.DataFrame) -> Dict:
        """Full 5-layer pipeline. Returns final sentiment only if all layers pass."""
        # Pre-compute L1/L2 counts for all return paths
        l1_headlines = len(headlines_df)
        real_headlines = headlines_df[~headlines_df['is_fake']].copy() if not headlines_df.empty else headlines_df
        l2_real_headlines = len(real_headlines)

        base_meta = {
            'l1_headlines': l1_headlines,
            'l2_real_headlines': l2_real_headlines,
        }

        if headlines_df.empty:
            return {**base_meta, 'score': 0, 'valid': False, 'reason': 'No headlines'}

        if l2_real_headlines < 2:
            return {
                **base_meta,
                'score': 0,
                'valid': False,
                'reason': f'Only {l2_real_headlines} real headlines after fake filter (need >=2)'
            }

        # Layer 3: Cross-source consensus
        consensus = self.consensus.validate(real_headlines)
        if not consensus['valid']:
            return {
                **base_meta,
                'score': 0,
                'valid': False,
                'reason': consensus.get('reason', 'No consensus'),
                'sources': consensus['sources'],
                'opposing': consensus['opposing']
            }

        # Layer 5: Black swan check
        current_time = datetime.now()
        swan = self.black_swan.check(real_headlines, current_time)
        if swan['cooldown_active']:
            return {
                **base_meta,
                'score': 0,
                'valid': False,
                'reason': swan['reason'],
                'action': 'HOLD',
                'black_swan': True,
                'trigger_type': swan.get('trigger_type', 'unknown'),
                'cluster_count': swan.get('cluster_count', 0),
            }

        # Layer 4: Price impact validation
        price_check = self.price_validator.validate(
            news_timestamp=real_headlines['published'].max(),
            news_direction=consensus['direction'],
            price_df=price_df
        )
        if not price_check['valid']:
            return {
                **base_meta,
                'score': 0,
                'valid': False,
                'reason': price_check['reason'],
                'price_change': price_check.get('price_change', 0)
            }

        # ALL LAYERS PASSED
        recent = real_headlines[real_headlines['age_minutes'] <= self.config.LOOKBACK_MINUTES]
        if recent.empty:
            return {**base_meta, 'score': 0, 'valid': False, 'reason': 'No recent real headlines'}

        weights = np.exp(-recent['age_minutes'] / self.config.TIME_DECAY_HALF_LIFE)
        final_score = np.average(recent['adjusted_score'], weights=weights)

        return {
            **base_meta,
            'score': round(final_score, 4),
            'valid': True,
            'direction': consensus['direction'],
            'confidence': round(consensus['confidence'], 4),
            'sources': consensus['sources'],
            'n_headlines': len(recent),
            'price_change': price_check.get('price_change', 0),
            'reason': f'All layers passed | Price validated: {price_check["reason"]}'
        }
