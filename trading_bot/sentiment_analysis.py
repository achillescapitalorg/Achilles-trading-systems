"""
News & Sentiment Analysis Module
=================================
Aggregates news and sentiment data from multiple sources for trading decisions.
Includes crypto-specific and gold/forex sentiment analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import re


class SentimentSource(Enum):
    """News/Sentiment data sources."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS_API = "news_api"
    CRYPTO_NEWS = "crypto_news"
    GOLD_REPORTS = "gold_reports"
    ECONOMIC_CALENDAR = "economic_calendar"
    CENTRAL_BANK = "central_bank"


@dataclass
class NewsItem:
    """Container for news item data."""
    title: str
    source: SentimentSource
    sentiment: float  # -1 to 1
    relevance: float  # 0 to 1
    timestamp: datetime
    url: Optional[str] = None
    content: Optional[str] = None


@dataclass
class SentimentSignal:
    """Container for sentiment-based trading signal."""
    signal: str  # 'BUY', 'SELL', 'NEUTRAL'
    confidence: float
    sentiment_score: float
    news_count: int
    sources: List[str]
    key_topics: List[str]


class SentimentAnalyzer:
    """
    Multi-Source Sentiment Analysis for Trading
    
    Aggregates sentiment from:
    - Social media (Twitter, Reddit)
    - News APIs
    - Crypto-specific sources
    - Gold/precious metals reports
    - Economic calendar events
    - Central bank communications
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize sentiment analyzer.
        
        Parameters
        ----------
        api_keys : Dict[str, str]
            API keys for various services:
            - 'twitter': Twitter API bearer token
            - 'newsapi': NewsAPI.org key
            - 'cryptopanic': CryptoPanic API key
            - 'alpha_vantage': Alpha Vantage key
        """
        self.api_keys = api_keys or {}
        self.news_cache: List[NewsItem] = []
        self.sentiment_history: List[float] = []
        
        # Keyword dictionaries for sentiment scoring
        self.bullish_keywords = {
            'general': ['bullish', 'rally', 'surge', 'gain', 'rise', 'breakout', 
                       'positive', 'upgrade', 'outperform', 'buy'],
            'crypto': ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'blockchain',
                      'adoption', 'institutional', 'etf', 'halving'],
            'gold': ['gold', 'xau', 'precious', 'safe haven', 'inflation', 
                    'hedge', 'central bank', 'reserves'],
            'economic': ['rate cut', 'stimulus', 'dovish', 'weak dollar', 
                        'recession', 'crisis']
        }
        
        self.bearish_keywords = {
            'general': ['bearish', 'crash', 'plunge', 'drop', 'fall', 'breakdown',
                       'negative', 'downgrade', 'underperform', 'sell'],
            'crypto': ['ban', 'crackdown', 'hack', 'scam', 'collapse', 'ftx',
                      'regulation', 'sec lawsuit'],
            'gold': ['gold sell-off', 'real rates', 'strong dollar', 'opportunity cost'],
            'economic': ['rate hike', 'hawkish', 'strong dollar', 'overheating']
        }
        
    def _calculate_sentiment_score(self, text: str, asset_type: str = 'general') -> float:
        """
        Calculate sentiment score from text.
        
        Parameters
        ----------
        text : str
            Text to analyze
        asset_type : str
            'general', 'crypto', 'gold', or 'economic'
            
        Returns
        -------
        float
            Sentiment score from -1 (very bearish) to 1 (very bullish)
        """
        text = text.lower()
        
        bullish_count = 0
        bearish_count = 0
        
        # General keywords
        for word in self.bullish_keywords['general']:
            if word in text:
                bullish_count += 1
        for word in self.bearish_keywords['general']:
            if word in text:
                bearish_count += 1
                
        # Asset-specific keywords
        if asset_type in self.bullish_keywords:
            for word in self.bullish_keywords[asset_type]:
                if word in text:
                    bullish_count += 1.5  # Higher weight for asset-specific
            for word in self.bearish_keywords[asset_type]:
                if word in text:
                    bearish_count += 1.5
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
            
        # Normalize to -1 to 1
        score = (bullish_count - bearish_count) / total
        return score
    
    def fetch_twitter_sentiment(self, query: str, count: int = 100) -> List[NewsItem]:
        """
        Fetch and analyze Twitter sentiment.
        
        Note: Requires Twitter API v2 access.
        For demo, returns simulated data.
        """
        news_items = []
        
        if self.api_keys.get('twitter'):
            # Real implementation would use tweepy
            pass
        
        # Simulated Twitter sentiment for demo
        import random
        for i in range(min(count, 50)):
            sentiment = random.gauss(0.1, 0.5)
            sentiment = np.clip(sentiment, -1, 1)
            
            news_items.append(NewsItem(
                title=f"Twitter sentiment sample {i}",
                source=SentimentSource.TWITTER,
                sentiment=sentiment,
                relevance=0.6,
                timestamp=datetime.now() - timedelta(minutes=i*5)
            ))
            
        return news_items
    
    def fetch_reddit_sentiment(self, subreddits: List[str] = None) -> List[NewsItem]:
        """
        Fetch Reddit sentiment from trading subreddits.
        
        Subreddits: wallstreetbets, cryptocurrencies, gold, investing
        """
        subreddits = subreddits or ['wallstreetbets', 'cryptocurrencies', 'gold']
        news_items = []
        
        # Simulated Reddit sentiment
        import random
        for sub in subreddits:
            for i in range(20):
                sentiment = random.gauss(0.05, 0.4)
                sentiment = np.clip(sentiment, -1, 1)
                
                news_items.append(NewsItem(
                    title=f"r/{sub} post sentiment",
                    source=SentimentSource.REDDIT,
                    sentiment=sentiment,
                    relevance=0.5 if sub == 'gold' else 0.7,
                    timestamp=datetime.now() - timedelta(hours=i)
                ))
                
        return news_items
    
    def fetch_news_api(self, query: str, days: int = 1) -> List[NewsItem]:
        """
        Fetch news from NewsAPI.org.
        
        Parameters
        ----------
        query : str
            Search query (e.g., 'bitcoin', 'gold price')
        days : int
            Number of days of news to fetch
        """
        news_items = []
        
        if self.api_keys.get('newsapi'):
            # Real implementation would use newsapi-python
            pass
            
        # Simulated news for demo
        headlines = [
            ("Market analysis shows positive outlook", 0.6),
            ("Analysts predict continued volatility", 0.0),
            ("Institutional interest growing", 0.7),
            ("Regulatory concerns weigh on sentiment", -0.5),
            ("Technical breakout signals bullish trend", 0.8),
        ]
        
        for title, sentiment in headlines:
            news_items.append(NewsItem(
                title=title,
                source=SentimentSource.NEWS_API,
                sentiment=sentiment,
                relevance=0.8,
                timestamp=datetime.now()
            ))
            
        return news_items
    
    def fetch_crypto_panic(self, currencies: List[str] = None) -> List[NewsItem]:
        """
        Fetch crypto news from CryptoPanic API.
        
        Parameters
        ----------
        currencies : List[str]
            Currencies to track (e.g., ['BTC', 'ETH', 'XAU'])
        """
        currencies = currencies or ['BTC', 'ETH']
        news_items = []
        
        if self.api_keys.get('cryptopanic'):
            # Real implementation would use CryptoPanic API
            pass
            
        # Simulated crypto news
        crypto_headlines = [
            ("Bitcoin breaks key resistance level", 0.7),
            ("Crypto market cap reaches new milestone", 0.5),
            ("Major exchange announces new features", 0.3),
            ("Whale activity detected on-chain", 0.4),
        ]
        
        for title, sentiment in crypto_headlines:
            news_items.append(NewsItem(
                title=title,
                source=SentimentSource.CRYPTO_NEWS,
                sentiment=sentiment,
                relevance=0.9,
                timestamp=datetime.now()
            ))
            
        return news_items
    
    def fetch_gold_reports(self) -> List[NewsItem]:
        """
        Fetch gold-specific reports and analysis.
        
        Sources:
        - World Gold Council
        - LBMA reports
        - Central bank reports
        """
        news_items = []
        
        # Simulated gold reports
        gold_headlines = [
            ("Central banks continue gold accumulation", 0.8),
            ("Gold ETF inflows reach monthly high", 0.6),
            ("Real rates decline supports gold prices", 0.5),
            ("Jewelry demand from Asia strengthens", 0.4),
            ("Mining supply constraints emerge", 0.3),
        ]
        
        for title, sentiment in gold_headlines:
            news_items.append(NewsItem(
                title=title,
                source=SentimentSource.GOLD_REPORTS,
                sentiment=sentiment,
                relevance=0.85,
                timestamp=datetime.now()
            ))
            
        return news_items
    
    def fetch_economic_calendar(self, days: int = 7) -> List[Dict]:
        """
        Fetch economic calendar events.
        
        High-impact events:
        - Central bank decisions
        - CPI/Inflation data
        - Employment reports
        - GDP releases
        """
        # Simulated economic calendar
        events = [
            {
                'event': 'FOMC Rate Decision',
                'impact': 'HIGH',
                'date': datetime.now() + timedelta(days=2),
                'forecast': '5.50%',
                'previous': '5.50%',
                'gold_impact': 'HIGH',
                'crypto_impact': 'MEDIUM'
            },
            {
                'event': 'US CPI (YoY)',
                'impact': 'HIGH',
                'date': datetime.now() + timedelta(days=3),
                'forecast': '3.2%',
                'previous': '3.7%',
                'gold_impact': 'HIGH',
                'crypto_impact': 'MEDIUM'
            },
            {
                'event': 'Non-Farm Payrolls',
                'impact': 'HIGH',
                'date': datetime.now() + timedelta(days=5),
                'forecast': '180K',
                'previous': '199K',
                'gold_impact': 'MEDIUM',
                'crypto_impact': 'LOW'
            },
        ]
        
        return events
    
    def analyze_central_bank_tone(self, text: str) -> Dict[str, float]:
        """
        Analyze central bank communication tone.
        
        Parameters
        ----------
        text : str
            Central bank statement/speech text
            
        Returns
        -------
        Dict with hawkish/dovish scores
        """
        hawkish_words = ['inflation', 'hike', 'tightening', 'hawkish', 
                        'overheating', 'restrictive', 'higher for longer']
        dovish_words = ['growth', 'cut', 'stimulus', 'dovish',
                       'supportive', 'accommodative', 'patient']
        
        text = text.lower()
        hawkish_score = sum(1 for word in hawkish_words if word in text)
        dovish_score = sum(1 for word in dovish_words if word in text)
        
        total = hawkish_score + dovish_score
        if total == 0:
            return {'hawkish': 0.5, 'dovish': 0.5}
            
        return {
            'hawkish': hawkish_score / total,
            'dovish': dovish_score / total
        }
    
    def get_aggregated_sentiment(self, asset: str = 'BTC') -> SentimentSignal:
        """
        Get aggregated sentiment signal for an asset.
        
        Parameters
        ----------
        asset : str
            Asset symbol ('BTC', 'ETH', 'XAU', 'GOLD')
            
        Returns
        -------
        SentimentSignal
            Aggregated sentiment signal
        """
        # Determine asset type
        asset = asset.upper()
        if asset in ['BTC', 'ETH', 'CRYPTO']:
            asset_type = 'crypto'
        elif asset in ['XAU', 'GOLD']:
            asset_type = 'gold'
        else:
            asset_type = 'general'
            
        # Fetch from all sources
        all_news = []
        
        # Always fetch these
        all_news.extend(self.fetch_news_api(asset, days=1))
        all_news.extend(self.fetch_reddit_sentiment())
        
        # Asset-specific sources
        if asset_type == 'crypto':
            all_news.extend(self.fetch_twitter_sentiment(asset))
            all_news.extend(self.fetch_crypto_panic([asset]))
        elif asset_type == 'gold':
            all_news.extend(self.fetch_gold_reports())
            
        if not all_news:
            return SentimentSignal(
                signal='NEUTRAL',
                confidence=0.0,
                sentiment_score=0.0,
                news_count=0,
                sources=[],
                key_topics=[]
            )
            
        # Weight by recency and relevance
        now = datetime.now()
        weighted_sentiment = 0
        total_weight = 0
        source_sentiments = {}
        
        for item in all_news:
            # Recency weight (exponential decay, half-life = 6 hours)
            hours_old = (now - item.timestamp).total_seconds() / 3600
            recency_weight = np.exp(-hours_old * np.log(2) / 6)
            
            # Combined weight
            weight = recency_weight * item.relevance
            weighted_sentiment += item.sentiment * weight
            total_weight += weight
            
            # Track by source
            source = item.source.value
            if source not in source_sentiments:
                source_sentiments[source] = {'sum': 0, 'count': 0}
            source_sentiments[source]['sum'] += item.sentiment * weight
            source_sentiments[source]['count'] += 1
            
        # Calculate average sentiment
        avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
        
        # Generate signal
        if avg_sentiment > 0.3:
            signal = 'BUY'
            confidence = min(avg_sentiment, 1.0)
        elif avg_sentiment < -0.3:
            signal = 'SELL'
            confidence = min(abs(avg_sentiment), 1.0)
        else:
            signal = 'NEUTRAL'
            confidence = abs(avg_sentiment) / 0.3
            
        # Extract key topics (simplified)
        key_topics = []
        if asset_type == 'crypto':
            key_topics = ['bitcoin', 'institutional', 'regulation', 'etf']
        elif asset_type == 'gold':
            key_topics = ['inflation', 'central banks', 'real rates', 'dollar']
            
        return SentimentSignal(
            signal=signal,
            confidence=confidence * 0.5,  # Sentiment gets 50% weight
            sentiment_score=avg_sentiment,
            news_count=len(all_news),
            sources=list(source_sentiments.keys()),
            key_topics=key_topics
        )
    
    def get_economic_event_impact(self, asset: str = 'GOLD') -> Dict[str, any]:
        """
        Get upcoming economic events and their potential impact.
        
        Parameters
        ----------
        asset : str
            Asset to check impact for
            
        Returns
        -------
        Dict with event impact analysis
        """
        events = self.fetch_economic_calendar()
        
        impact_map = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        
        total_bullish_impact = 0
        total_bearish_impact = 0
        
        for event in events:
            if asset.upper() in ['GOLD', 'XAU']:
                impact_key = 'gold_impact'
            else:
                impact_key = 'crypto_impact'
                
            impact_level = impact_map.get(event.get(impact_key, 'LOW'), 1)
            
            # Simplified impact analysis
            # In reality, would compare forecast vs previous
            if 'rate' in event['event'].lower():
                # Rate decisions affect gold inversely
                total_bearish_impact += impact_level * 0.5
            elif 'cpi' in event['event'].lower() or 'inflation' in event['event'].lower():
                # Higher inflation is bullish for gold
                total_bullish_impact += impact_level * 0.5
                
        net_impact = total_bullish_impact - total_bearish_impact
        
        if net_impact > 1:
            signal = 'BULLISH'
        elif net_impact < -1:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'
            
        return {
            'signal': signal,
            'net_impact': net_impact,
            'upcoming_events': events,
            'bullish_factors': total_bullish_impact,
            'bearish_factors': total_bearish_impact
        }


class NewsSentimentSignal:
    """
    Wrapper class for news sentiment trading signals.
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.analyzer = SentimentAnalyzer(api_keys)
        
    def get_signal(self, asset: str = 'BTC') -> Tuple[str, float, Dict]:
        """
        Get sentiment-based trading signal.
        
        Returns
        -------
        Tuple[str, float, Dict]
            (signal, confidence, metadata)
        """
        sentiment = self.analyzer.get_aggregated_sentiment(asset)
        economic = self.analyzer.get_economic_event_impact(asset)
        
        # Combine sentiment and economic impact
        if sentiment.signal == 'BUY' and economic['signal'] != 'BEARISH':
            signal = 'BUY'
            confidence = sentiment.confidence
        elif sentiment.signal == 'SELL' and economic['signal'] != 'BULLISH':
            signal = 'SELL'
            confidence = sentiment.confidence
        else:
            signal = 'NEUTRAL'
            confidence = sentiment.confidence * 0.5
            
        metadata = {
            'sentiment': sentiment,
            'economic_events': economic,
            'news_count': sentiment.news_count,
            'sources': sentiment.sources
        }
        
        return signal, confidence, metadata
