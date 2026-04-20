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

from services.local_ai_service import analyze_sentiment, get_local_ai_service


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
        "MyFXBook": "📒",
        "ForexFactory": "🏭",
        "Myfxbook": "📒",
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
        "MyFXBook": 7,
        "Myfxbook": 7,
        "ForexFactory": 7,
    }

    # Direct RSS feed configuration
    # url: primary feed URL; icon, priority match SOURCE_ICONS/PRIORITY above
    RSS_FEEDS: Dict[str, Dict] = {
        "Reuters": {
            "url": "https://feeds.reuters.com/reuters/businessNews",
            "icon": "📰",
            "priority": 10,
        },
        "Financial Times": {
            "url": "https://www.ft.com/rss/home/uk",
            "icon": "📊",
            "priority": 8,
        },
        "FXStreet": {
            "url": "https://www.fxstreet.com/rss/news",
            "icon": "💱",
            "priority": 8,
        },
        "CoinDesk": {
            "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "icon": "₿",
            "priority": 7,
        },
        "Investing.com Commodities": {
            "url": "https://www.investing.com/rss/news_301.rss",
            "icon": "📈",
            "priority": 7,
        },
        "Investing.com Forex": {
            "url": "https://www.investing.com/rss/news_1.rss",
            "icon": "📈",
            "priority": 7,
        },
        "Investing.com Crypto": {
            "url": "https://www.investing.com/rss/news_25.rss",
            "icon": "📈",
            "priority": 7,
        },
        "MarketWatch": {
            "url": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
            "icon": "⌚",
            "priority": 6,
        },
        "MyFXBook": {
            "url": "https://www.myfxbook.com/rss/forex-news-rss",
            "icon": "📒",
            "priority": 7,
        },
        "ForexFactory": {
            "url": "https://www.forexfactory.com/ff_calendar.php?week=this&ctype=impact&timezone=NY&format=xml",
            "icon": "🏭",
            "priority": 7,
        },
    }

    # Which RSS sources are relevant for each symbol
    SYMBOL_RSS_MAP: Dict[str, List[str]] = {
        "XAUUSD": ["Reuters", "FXStreet", "Investing.com Commodities", "MyFXBook", "ForexFactory"],
        "BTCUSD": ["CoinDesk", "Investing.com Crypto", "Reuters"],
        "ETHUSD": ["CoinDesk", "Investing.com Crypto", "Reuters"],
        "EURUSD": ["FXStreet", "Reuters", "Investing.com Forex", "MyFXBook", "ForexFactory"],
        "GBPUSD": ["FXStreet", "Reuters", "Investing.com Forex", "MyFXBook"],
        "USDJPY": ["FXStreet", "Reuters", "Investing.com Forex", "MyFXBook"],
        "SPX500": ["MarketWatch", "Reuters", "Financial Times", "Investing.com Commodities"],
        "NAS100": ["MarketWatch", "Reuters", "Financial Times"],
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
        Uses as_completed() so fast sources are not blocked by slow ones.
        Includes direct RSS feeds in addition to existing sources.
        """
        all_results = []
        relevant_rss = self.SYMBOL_RSS_MAP.get(symbol, list(self.RSS_FEEDS.keys())[:4])

        max_workers = self._max_workers + len(relevant_rss)

        # Sources that use Google News RSS are unreliable (often blocked).
        # Only include them if the direct RSS sources are insufficient.
        _google_news_sources = {"forexcom", "google"}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            if self.newsapi_key:
                futures[executor.submit(self._fetch_newsapi, keywords)] = "newsapi"

            # Google News RSS — skip silently if key-less scraping is blocked
            futures[executor.submit(self._fetch_forexcom_news, keywords)] = "forexcom"
            futures[executor.submit(self._fetch_google_news, keywords, symbol)] = "google"

            if self.marketaux_api_key:
                futures[executor.submit(self._fetch_marketaux, symbol)] = "marketaux"

            # Direct RSS feeds for this symbol
            for src_name in relevant_rss:
                feed_cfg = self.RSS_FEEDS.get(src_name, {})
                feed_url = feed_cfg.get("url", "")
                if feed_url:
                    futures[executor.submit(
                        self._fetch_rss_direct, src_name, feed_url, keywords
                    )] = f"rss_{src_name}"

            # Use as_completed so fast sources return immediately; 20s total budget
            try:
                for future in as_completed(futures, timeout=20):
                    source = futures[future]
                    try:
                        items = future.result()   # already done — no extra wait
                        if items:
                            for item in items:
                                item["_source"] = source
                            all_results.extend(items)
                    except Exception as e:
                        exc_type = type(e).__name__
                        # Silently skip unreliable Google-News scraping failures
                        if source in _google_news_sources:
                            pass
                        else:
                            msg = str(e) or exc_type
                            print(f"[UnifiedNews] {source} failed ({exc_type}): {msg}")
            except TimeoutError:
                # Some futures exceeded the 20s wall-clock budget — return what we have
                print(f"[UnifiedNews] Partial timeout — returning {len(all_results)} items collected so far")

        return all_results

    def _fetch_rss_direct(
        self, source_name: str, url: str, keywords: List[str]
    ) -> List[Dict]:
        """
        Fetch and parse an RSS/Atom feed directly using feedparser.

        Requires `feedparser>=6.0.0` (added to requirements.txt).
        Returns [] gracefully if feedparser is not installed or the feed is unreachable.

        Args:
            source_name: Human-readable label shown in the UI
            url:         RSS/Atom feed URL
            keywords:    Filter articles to those containing at least one keyword
                         in title or summary

        Returns:
            List of news item dicts in the standard UnifiedNewsService format.
        """
        try:
            import feedparser
        except ImportError:
            print("[UnifiedNews] feedparser not installed — skipping direct RSS")
            return []

        try:
            from datetime import datetime, timedelta

            feed = feedparser.parse(url)
            if not feed.entries:
                return []

            cutoff = datetime.now() - timedelta(days=3)
            kw_lower = [k.lower() for k in keywords]
            icon = self.RSS_FEEDS.get(source_name, {}).get("icon", "📰")
            news_items = []

            for entry in feed.entries[:20]:
                title = getattr(entry, "title", "") or ""
                if not title or len(title) < 20:
                    continue

                # Keyword filter — match in title or summary
                summary = getattr(entry, "summary", "") or ""
                combined = (title + " " + summary).lower()
                if kw_lower and not any(kw in combined for kw in kw_lower):
                    continue

                # Recency filter
                published = getattr(entry, "published_parsed", None)
                if published:
                    try:
                        import time as _time
                        pub_dt = datetime.fromtimestamp(_time.mktime(published))
                        if pub_dt < cutoff:
                            continue
                        time_ago_str = self._parse_pubdate(
                            entry.get("published", "")
                        )
                    except Exception:
                        time_ago_str = "Live"
                else:
                    time_ago_str = "Live"

                link = getattr(entry, "link", "") or ""
                pubdate_raw = getattr(entry, "published", "") or ""

                sentiment_result = self._analyze_with_cache(title)
                impact = self._analyze_impact(title)

                news_items.append({
                    "headline": title[:250],
                    "sentiment": sentiment_result["score"],
                    "sentiment_label": sentiment_result["sentiment"],
                    "confidence": sentiment_result["confidence"],
                    "impact": impact,
                    "time_ago": time_ago_str,
                    "source": source_name,
                    "source_icon": icon,
                    "url": link,
                    "impact_timing": "Market hours",
                    "pubdate": pubdate_raw,
                })

            return news_items

        except Exception as exc:
            print(f"[UnifiedNews] RSS {source_name} error: {exc}")
            return []
    
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
                err = response.json().get("message", response.text[:100]) if response.content else ""
                print(f"[UnifiedNews] NewsAPI HTTP {response.status_code}: {err}")
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

            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=6)
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
                    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
                    
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
        """Fetch from Marketaux API (tries both api_token and api_key auth styles)."""
        try:
            url = "https://api.marketaux.com/v1/news/all"
            base_params = {
                "symbols": symbol,
                "limit": 10,
                "sort": "published_at:desc",
                "language": "en",
            }

            # Try newer 'api_token' style first (current plans)
            params = {**base_params, "api_token": self.marketaux_api_key}
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 401:
                # Fall back to legacy 'api_key' style
                params = {**base_params, "api_key": self.marketaux_api_key}
                response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                print(f"[UnifiedNews] Marketaux returned {response.status_code} — skipping")
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
