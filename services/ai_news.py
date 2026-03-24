"""
Intelligent News Service - Unified News Aggregator
================================================
Features:
- Parallel fetching from multiple sources
- Deduplication by headline similarity
- Batch sentiment analysis with caching
- Fallback chain with graceful degradation
"""
import os
import json
import time
import requests
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from services.deepseek_sentiment import analyze_sentiment, get_aggregate_sentiment, sentiment_analyzer


class UnifiedNewsService:
    """
    Unified news service with parallel fetching and intelligent deduplication.
    """
    
    SOURCE_ICONS = {
        "Reuters": "📰",
        "Bloomberg": "💼",
        "CNBC": "📺",
        "FX Street": "💱",
        "FXStreet": "💱",
        "Forex Factory": "🏭",
        "Forex.com": "💹",
        "Investopedia": "📖",
        "Kitco": "🥇",
        "DailyFX": "📊",
        "Investing.com": "📈",
        "MarketWatch": "⌚",
        "Yahoo Finance": "🟣",
        "CoinDesk": "₿",
        "CoinTelegraph": "📱",
        "WSJ": "📰",
        "Financial Times": "📊",
        "The Wall Street Journal": "📰",
        "Marketaux": "🤖",
        "Google News": "🔍",
        "Web": "🌐",
    }
    
    SOURCE_PRIORITY = {
        "Reuters": 10,
        "Bloomberg": 10,
        "CNBC": 9,
        "FXStreet": 8,
        "Forex.com": 8,
        "Kitco": 7,
        "Investing.com": 7,
        "MarketWatch": 6,
        "Yahoo Finance": 6,
        "CoinDesk": 7,
        "CoinTelegraph": 7,
        "WSJ": 8,
        "Financial Times": 8,
        "DailyFX": 6,
        "Google News": 4,
        "Web": 3,
    }
    
    def __init__(self):
        self.newsapi_key = os.getenv("NEWSAPI_KEY", "")
        self.marketaux_api_key = os.getenv("MARKETAUX_API_KEY", "")
        self._symbol_keywords = {
            "XAUUSD": ["gold price", "gold futures", "precious metals", "XAUUSD"],
            "BTCUSD": ["bitcoin price", "BTC", "cryptocurrency", "BTCUSD"],
            "ETHUSD": ["ethereum price", "ETH", "ether", "ETHUSD"],
            "EURUSD": ["EUR USD", "euro forex", "eurozone", "EURUSD"],
            "GBPUSD": ["GBP USD", "british pound", "sterling forex", "GBPUSD"],
            "USDJPY": ["USD JPY", "yen", "japanese yen", "USDJPY"],
            "SPX500": ["S&P 500", "US stock market", "US equities"],
            "NAS100": ["NASDAQ", "tech stocks", "nasdaq"],
        }
        
        self._financial_domains = "reuters.com,bloomberg.com,cnbc.com,fxstreet.com,kitco.com,investing.com,yahoofinance.com,marketwatch.com,wsj.com,ft.com"
        
        self._sentiment_cache: Dict[str, Dict] = {}
        self._sentiment_cache_lock = threading.Lock()
        self._SENTIMENT_CACHE_TTL = 900
        
        self._max_workers = 4
    
    def get_news(self, symbol: str, max_items: int = 15) -> List[Dict]:
        """
        Fetch news from all sources in parallel, deduplicate, and return top items.
        """
        keywords = self._symbol_keywords.get(symbol, [symbol])
        
        results = self._fetch_all_sources_parallel(symbol, keywords)
        
        if not results:
            return []
        
        deduplicated = self._deduplicate_and_rank(results)
        
        for item in deduplicated[:max_items]:
            item["sentiment_label"] = item.get("sentiment_label") or self._get_cached_sentiment(item["headline"])
        
        return deduplicated[:max_items]
    
    def _fetch_all_sources_parallel(self, symbol: str, keywords: List[str]) -> List[Dict]:
        """
        Fetch from all sources in parallel using ThreadPoolExecutor.
        """
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {}
            
            if self.newsapi_key:
                futures[executor.submit(self._fetch_newsapi, keywords)] = "newsapi"
            
            futures[executor.submit(self._fetch_forexcom_news, keywords)] = "forexcom"
            futures[executor.submit(self._fetch_google_news, keywords, symbol)] = "google"
            
            if self.marketaux_api_key:
                futures[executor.submit(self._fetch_marketaux, symbol)] = "marketaux"
            
            for future in futures:
                source = futures[future]
                try:
                    items = future.result(timeout=8)
                    if items:
                        for item in items:
                            item["_source"] = source
                        all_results.extend(items)
                except Exception as e:
                    print(f"[UnifiedNews] {source} failed: {e}")
        
        return all_results
    
    def _deduplicate_and_rank(self, items: List[Dict]) -> List[Dict]:
        """
        Deduplicate by headline similarity and URL, then rank by source priority.
        """
        if not items:
            return []
        
        seen_headlines = []
        seen_urls = set()
        unique_items = []
        
        for item in items:
            headline = item.get("headline", "")
            url = item.get("url", "")
            
            if not headline or len(headline) < 20:
                continue
            
            if url and url in seen_urls:
                continue
            
            is_duplicate = False
            for seen in seen_headlines:
                similarity = SequenceMatcher(None, headline.lower(), seen.lower()).ratio()
                if similarity > 0.75:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            seen_headlines.append(headline)
            if url:
                seen_urls.add(url)
            
            item["_priority"] = self.SOURCE_PRIORITY.get(item.get("source", ""), 5)
            unique_items.append(item)
        
        unique_items.sort(key=lambda x: (x["_priority"], x.get("sentiment", 0)), reverse=True)
        
        return unique_items
    
    def _get_cached_sentiment(self, headline: str) -> str:
        """Get cached sentiment or compute and cache it."""
        cache_key = hash(headline)
        
        with self._sentiment_cache_lock:
            if cache_key in self._sentiment_cache:
                cached = self._sentiment_cache[cache_key]
                if time.time() - cached["timestamp"] < self._SENTIMENT_CACHE_TTL:
                    return cached["sentiment"]
        
        result = analyze_sentiment(headline)
        sentiment = result.get("sentiment", "neutral")
        
        with self._sentiment_cache_lock:
            self._sentiment_cache[cache_key] = {
                "sentiment": sentiment,
                "score": result.get("score", 0),
                "confidence": result.get("confidence", 0),
                "timestamp": time.time()
            }
        
        return sentiment
    
    def _fetch_newsapi(self, keywords: List[str]) -> List[Dict]:
        """Fetch from NewsAPI.org with recent news only."""
        try:
            from datetime import datetime, timedelta
            
            query = " OR ".join([f'"{k}"' for k in keywords[:3]])
            
            # Get date range: last 5 days
            to_date = datetime.now()
            from_date = to_date - timedelta(days=5)
            
            url = "https://newsapi.org/v2/everything"
            params = {
                "apiKey": self.newsapi_key,
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 15,
                "domains": self._financial_domains,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return []
            
            data = response.json()
            articles = data.get("articles", [])
            
            news_items = []
            for article in articles:
                headline = article.get("title", "")
                if not headline or headline == "[Removed]":
                    continue
                
                sentiment_result = self._analyze_with_cache(headline)
                impact = self._analyze_impact(headline)
                source = article.get("source", {}).get("name", "News")
                
                news_items.append({
                    "headline": headline[:250],
                    "sentiment": sentiment_result["score"],
                    "sentiment_label": sentiment_result["sentiment"],
                    "confidence": sentiment_result["confidence"],
                    "impact": impact,
                    "time_ago": "Live",
                    "source": source,
                    "source_icon": self.SOURCE_ICONS.get(source, "📰"),
                    "url": article.get("url", ""),
                    "impact_timing": "Market hours"
                })
            
            return news_items
            
        except Exception as e:
            print(f"[UnifiedNews] NewsAPI error: {e}")
            return []
    
    def _fetch_forexcom_news(self, keywords: List[str]) -> List[Dict]:
        """Fetch forex.com news via Google News RSS with recent filter."""
        try:
            import xml.etree.ElementTree as ET
            from datetime import datetime, timedelta
            
            query = "+".join(keywords[:2])
            
            # Google News doesn't support date filtering directly, but we filter after
            url = f"https://news.google.com/rss/search?q=site:forex.com+{query}&hl=en-US&gl=US&ceid=US:en"
            
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            if response.status_code != 200:
                return []
            
            root = ET.fromstring(response.text)
            items = root.findall('.//item')
            
            if not items:
                return []
            
            # Filter to last 5 days
            cutoff = timedelta(days=5)
            news_items = []
            
            for item in items[:15]:
                title_elem = item.find('title')
                link_elem = item.find('link')
                pubdate_elem = item.find('pubDate')
                
                title = title_elem.text if title_elem is not None else ""
                link = str(link_elem.text) if link_elem is not None else ""
                pubdate = str(pubdate_elem.text) if pubdate_elem is not None else ""
                
                if not title or len(title) < 20:
                    continue
                
                # Check date
                item_date = self._parse_pubdate_to_dt(pubdate)
                now = datetime.now(item_date.tzinfo) if item_date else None
                if item_date and now and (now - item_date) > cutoff:
                    continue
                
                sentiment_result = self._analyze_with_cache(title)
                impact = self._analyze_impact(title)
                
                news_items.append({
                    "headline": title[:250],
                    "sentiment": sentiment_result["score"],
                    "sentiment_label": sentiment_result["sentiment"],
                    "confidence": sentiment_result["confidence"],
                    "impact": impact,
                    "time_ago": self._parse_pubdate(pubdate),
                    "source": "Forex.com",
                    "source_icon": "💹",
                    "url": self._extract_article_url(link),
                    "impact_timing": "Market hours",
                    "pubdate": pubdate,
                })
            
            return news_items
            
        except Exception as e:
            print(f"[UnifiedNews] Forex.com error: {e}")
            return []
    
    def _fetch_google_news(self, keywords: List[str], symbol: str) -> List[Dict]:
        """Fetch from Google News RSS for financial sources with recent filter."""
        try:
            import xml.etree.ElementTree as ET
            from datetime import datetime, timedelta
            
            query = "+".join(keywords[:2])
            sources = ["fxstreet.com", "investing.com", "kitco.com", "dailyfx.com", "forexfactory.com"]
            
            news_items = []
            cutoff = timedelta(days=5)
            
            for source_domain in sources[:3]:
                try:
                    url = f"https://news.google.com/rss/search?q=site:{source_domain}+{query}&hl=en-US&gl=US&ceid=US:en"
                    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
                    
                    if response.status_code != 200:
                        continue
                    
                    root = ET.fromstring(response.text)
                    items = root.findall('.//item')
                    
                    source_name = source_domain.split('.')[0].capitalize()
                    
                    for item in items[:5]:
                        title_elem = item.find('title')
                        link_elem = item.find('link')
                        pubdate_elem = item.find('pubDate')
                        
                        title = title_elem.text if title_elem is not None else ""
                        link = str(link_elem.text) if link_elem is not None else ""
                        pubdate = str(pubdate_elem.text) if pubdate_elem is not None else ""
                        
                        if not title or len(title) < 20:
                            continue
                        
                        # Check date filter
                        item_date = self._parse_pubdate_to_dt(pubdate)
                        now = datetime.now(item_date.tzinfo) if item_date else None
                        if item_date and now and (now - item_date) > cutoff:
                            continue
                        
                        sentiment_result = self._analyze_with_cache(title)
                        impact = self._analyze_impact(title)
                        
                        news_items.append({
                            "headline": title[:250],
                            "sentiment": sentiment_result["score"],
                            "sentiment_label": sentiment_result["sentiment"],
                            "confidence": sentiment_result["confidence"],
                            "impact": impact,
                            "time_ago": self._parse_pubdate(pubdate),
                            "source": f"{source_name}",
                            "source_icon": self.SOURCE_ICONS.get(source_name, "📰"),
                            "url": self._extract_article_url(link),
                            "impact_timing": "Market hours",
                            "pubdate": pubdate,
                        })
                except:
                    continue
            
            return news_items
            
        except Exception as e:
            print(f"[UnifiedNews] Google News error: {e}")
            return []
    
    def _fetch_marketaux(self, symbol: str) -> List[Dict]:
        """Fetch from Marketaux API."""
        try:
            url = "https://api.marketaux.com/v1/news/all"
            params = {
                "api_key": self.marketaux_api_key,
                "symbols": symbol,
                "limit": 10,
                "sort": "published_at:desc"
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return []
            
            data = response.json()
            if "data" not in data:
                return []
            
            news_items = []
            for item in data.get("data", []):
                title = item.get("title", "")
                if not title or len(title) < 20:
                    continue
                
                sentiment = item.get("sentiment_score", 0)
                impact = "HIGH" if abs(sentiment) > 0.3 else "MEDIUM"
                
                sentiment_label = "neutral"
                if sentiment > 0.2:
                    sentiment_label = "bullish"
                elif sentiment < -0.2:
                    sentiment_label = "bearish"
                
                source = item.get("source", {})
                if isinstance(source, dict):
                    source = source.get("name", "Marketaux")
                
                news_items.append({
                    "headline": title[:250],
                    "sentiment": sentiment,
                    "sentiment_label": sentiment_label,
                    "confidence": 0.7,
                    "impact": impact,
                    "time_ago": self._time_ago(item.get("published_at")),
                    "source": source,
                    "source_icon": self.SOURCE_ICONS.get(source, "🤖"),
                    "url": item.get("url", ""),
                    "impact_timing": "Market hours"
                })
            
            return news_items
            
        except Exception as e:
            print(f"[UnifiedNews] Marketaux error: {e}")
            return []
    
    def _analyze_with_cache(self, headline: str) -> Dict:
        """Analyze sentiment with caching."""
        cache_key = hash(headline)
        
        with self._sentiment_cache_lock:
            if cache_key in self._sentiment_cache:
                cached = self._sentiment_cache[cache_key]
                if time.time() - cached["timestamp"] < self._SENTIMENT_CACHE_TTL:
                    return {
                        "sentiment": cached["sentiment"],
                        "score": cached["score"],
                        "confidence": cached["confidence"]
                    }
        
        result = analyze_sentiment(headline)
        
        with self._sentiment_cache_lock:
            self._sentiment_cache[cache_key] = {
                "sentiment": result.get("sentiment", "neutral"),
                "score": result.get("score", 0),
                "confidence": result.get("confidence", 0),
                "timestamp": time.time()
            }
        
        return result
    
    def _analyze_impact(self, text: str) -> str:
        """Analyze news impact level."""
        if not text:
            return "MEDIUM"
        
        text_lower = text.lower()
        
        high_impact = ['breaking', 'crash', 'surprise', 'fomc', 'rate decision',
                       'bankruptcy', 'recession', 'war', 'fed chair', 'powell',
                       'emergency', 'crisis', 'plunge', 'surges']
        
        low_impact = ['analysis', 'outlook', 'forecast', 'technical', 'review',
                      'weekly', 'monthly', 'seasonal']
        
        if any(word in text_lower for word in high_impact):
            return "HIGH"
        elif any(word in text_lower for word in low_impact):
            return "LOW"
        return "MEDIUM"
    
    def _extract_article_url(self, google_url: str) -> str:
        """Extract original URL from Google News link."""
        try:
            if 'articles/' in google_url:
                import re
                match = re.search(r'url=(.*?)(?:\?|$)', google_url)
                if match:
                    from urllib.parse import unquote
                    return unquote(match.group(1))
            return google_url
        except:
            return google_url
    
    def _parse_pubdate(self, pubdate: str) -> str:
        """Parse pubDate to time ago."""
        try:
            from email.utils import parsedate_to_datetime
            from datetime import datetime as dt_now
            dt = parsedate_to_datetime(pubdate)
            diff = dt_now.now(dt.tzinfo) - dt
            minutes = diff.total_seconds() / 60
            if minutes < 60:
                return f"{int(minutes)}m ago"
            elif minutes < 1440:
                return f"{int(minutes/60)}h ago"
            else:
                return f"{int(minutes/1440)}d ago"
        except:
            return "Live"
    
    def _parse_pubdate_to_dt(self, pubdate: str):
        """Parse pubDate to datetime object for filtering."""
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(pubdate)
        except:
            return None
    
    def _time_ago(self, dt_str: str) -> str:
        """Convert datetime to 'time ago' string."""
        if not dt_str:
            return "Live"
        
        try:
            if 'T' in dt_str:
                dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                diff = datetime.now(dt.tzinfo) - dt
                minutes = diff.total_seconds() / 60
                if minutes < 60:
                    return f"{int(minutes)}m ago"
                elif minutes < 1440:
                    return f"{int(minutes/60)}h ago"
                else:
                    return f"{int(minutes/1440)}d ago"
        except:
            pass
        return "Live"


news_service = UnifiedNewsService()


def get_intelligent_news(symbol: str, max_items: int = 15) -> List[Dict]:
    """Convenience function to get intelligent news."""
    return news_service.get_news(symbol, max_items)
