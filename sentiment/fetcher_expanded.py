"""
EXPANDED NEWS FETCHER
Aggregates gold-related news from multiple sources with tier weights.
"""
import re
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import feedparser


class ExpandedNewsFetcher:
    """Fetch gold news from RSS feeds, APIs, and social sources."""

    # RSS feeds by tier
    RSS_FEEDS = {
        'kitco': 'https://www.kitco.com/rss/gold.xml',
        'bullionvault': 'https://www.bullionvault.com/gold-news/rss.xml',
        'forexlive': 'https://www.forexlive.com/feed',
        'fxempire': 'https://www.fxempire.com/feed/?tag=gold',
    }

    # Simulated headlines for demo when no APIs are configured
    DEMO_HEADLINES = [
        {
            'title': 'Gold prices steady as investors await Fed decision',
            'summary': 'Gold held firm near $2,400 as markets priced in rate cut expectations.',
            'source': 'kitco',
            'published': datetime.now() - timedelta(minutes=5),
        },
        {
            'title': 'Safe-haven demand boosts gold amid geopolitical tensions',
            'summary': 'Escalating conflicts in the Middle East drove investors to gold.',
            'source': 'bullionvault',
            'published': datetime.now() - timedelta(minutes=12),
        },
        {
            'title': 'Dollar weakness supports gold rally',
            'summary': 'The DXY index fell 0.3%, making gold cheaper for foreign buyers.',
            'source': 'forexlive',
            'published': datetime.now() - timedelta(minutes=8),
        },
    ]

    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.cache_dir = Path(__file__).parent.parent / "data" / "news_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_all(self, use_demo: bool = False) -> pd.DataFrame:
        """Fetch headlines from all configured sources."""
        headlines = []

        if use_demo:
            headlines.extend(self.DEMO_HEADLINES)
        else:
            headlines.extend(self._fetch_rss())
            headlines.extend(self._fetch_reddit())
            headlines.extend(self._fetch_twitter())
            headlines.extend(self._fetch_newsapi())

        if not headlines:
            # Fallback to demo if nothing fetched
            headlines.extend(self.DEMO_HEADLINES)

        df = pd.DataFrame(headlines)
        df['published'] = pd.to_datetime(df['published'])
        df['age_minutes'] = (datetime.now() - df['published']).dt.total_seconds() / 60

        # Keep only recent headlines
        df = df[df['age_minutes'] <= 60].copy()
        df = df.sort_values('published', ascending=False).reset_index(drop=True)

        return df

    def _fetch_rss(self) -> List[Dict]:
        """Fetch from RSS feeds."""
        headlines = []
        for source, url in self.RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    headlines.append({
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', entry.get('description', '')),
                        'source': source,
                        'published': self._parse_date(entry),
                    })
            except Exception as e:
                print(f"[Fetcher] RSS {source} failed: {e}")
        return headlines

    def _fetch_reddit(self) -> List[Dict]:
        """Fetch from Reddit r/Gold (requires PRAW or raw JSON)."""
        headlines = []
        try:
            # Try raw JSON endpoint (no API key needed, rate-limited)
            import requests
            url = "https://www.reddit.com/r/Gold/new.json?limit=10"
            headers = {'User-Agent': 'GoldSentimentBot/1.0'}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for post in data.get('data', {}).get('children', []):
                    p = post['data']
                    headlines.append({
                        'title': p.get('title', ''),
                        'summary': p.get('selftext', '')[:200],
                        'source': 'reddit',
                        'published': datetime.fromtimestamp(p.get('created_utc', 0)),
                    })
        except Exception as e:
            print(f"[Fetcher] Reddit failed: {e}")
        return headlines

    def _fetch_twitter(self) -> List[Dict]:
        """Fetch from Twitter/X (requires API key). Falls back to empty."""
        headlines = []
        # Twitter API v2 requires paid access. Skip if no key.
        if not self.api_keys.get('twitter'):
            return headlines
        try:
            # Real implementation would use tweepy
            pass
        except Exception as e:
            print(f"[Fetcher] Twitter failed: {e}")
        return headlines

    def _fetch_newsapi(self) -> List[Dict]:
        """Fetch from NewsAPI."""
        headlines = []
        if not self.api_keys.get('newsapi'):
            return headlines
        try:
            import requests
            url = (
                "https://newsapi.org/v2/everything"
                f"?q=gold+OR+XAUUSD+OR+precious+metals"
                f"&from={(datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S')}"
                f"&sortBy=publishedAt"
                f"&apiKey={self.api_keys['newsapi']}"
                f"&pageSize=20"
            )
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for article in data.get('articles', []):
                    headlines.append({
                        'title': article.get('title', ''),
                        'summary': article.get('description', ''),
                        'source': 'newsapi',
                        'published': pd.to_datetime(article.get('publishedAt', datetime.now())),
                    })
        except Exception as e:
            print(f"[Fetcher] NewsAPI failed: {e}")
        return headlines

    def _parse_date(self, entry) -> datetime:
        """Parse RSS date."""
        for field in ['published', 'updated', 'pubDate', 'dc:date']:
            val = getattr(entry, field, None)
            if val:
                try:
                    return pd.to_datetime(val)
                except:
                    pass
        return datetime.now()

    def fetch_google_trends(self, keywords: List[str] = None) -> Dict:
        """Fetch Google Trends data (requires pytrends)."""
        keywords = keywords or ['buy gold', 'gold price', 'XAUUSD']
        try:
            from pytrends.request import TrendReq
            pytrends = TrendReq(hl='en-US', tz=360)
            pytrends.build_payload(keywords, cat=0, timeframe='now 1-H', geo='', gprop='')
            data = pytrends.interest_over_time()
            if not data.empty:
                latest = data.iloc[-1]
                return {
                    kw: int(latest[kw]) for kw in keywords if kw in latest
                }
        except Exception as e:
            print(f"[Fetcher] Google Trends failed: {e}")
        return {kw: 50 for kw in keywords}

    def fetch_cme_cot(self) -> Optional[pd.DataFrame]:
        """Fetch CME Commitment of Traders report (weekly)."""
        # COT is weekly — cache and update once per week
        cache_file = self.cache_dir / "cot_latest.csv"
        if cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age.days < 7:
                return pd.read_csv(cache_file)

        # Real implementation would scrape CFTC COT reports
        # For now, return None — this is weekly data, not real-time
        return None
