"""
Intelligent News Service - Unified News Aggregator
================================================
Features:
- Parallel fetching from multiple sources (RSS, yfinance, Alpha Vantage, NewsAPI, Marketaux)
- Jaccard token deduplication (faster than SequenceMatcher at scale)
- Recency bonus + relevance scoring for smarter ranking
- Batch sentiment analysis with caching
- Fallback chain with graceful degradation
"""
import os
import json
import time
import requests
import threading
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from services.local_ai_service import analyze_sentiment, get_local_ai_service


def _jaccard(a: str, b: str) -> float:
    """Token-based Jaccard similarity — faster than SequenceMatcher for dedup."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _relevance(text: str, keywords: List[str]) -> float:
    """Keyword density score 0–2 added to source priority during ranking."""
    if not keywords or not text:
        return 0.0
    t = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in t)
    return min(hits / max(len(keywords), 1) * 2, 2.0)


_YF_TICKER_MAP: Dict[str, str] = {
    "XAUUSD": "GC=F",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "SPX500": "^GSPC",
    "NAS100": "^NDX",
}


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
        "ForexLive": "📰",
        "Guardian Business": "📰",
        "Mining.com": "⛏️",
        "Business Wire": "📡",
    }

    SOURCE_PRIORITY = {
        "Reuters": 10,
        "Bloomberg": 10,
        "ForexLive": 9,
        "CNBC": 9,
        "FXStreet": 8,
        "Forex.com": 8,
        "Guardian Business": 7,
        "Kitco": 7,
        "Mining.com": 7,
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
        "Business Wire": 6,
    }

    # Direct RSS feed configuration — 20 feeds (blocked feeds replaced with working alternatives)
    RSS_FEEDS: Dict[str, Dict] = {
        # reuters.com and ft.com dropped public RSS — replaced with ForexLive + Guardian
        "ForexLive": {
            "url": "https://www.forexlive.com/feed/news",
            "icon": "📰", "priority": 9,
        },
        "Guardian Business": {
            "url": "https://www.theguardian.com/uk/business/rss",
            "icon": "📰", "priority": 7,
        },
        "FXStreet": {
            "url": "https://www.fxstreet.com/rss/news",
            "icon": "💱", "priority": 8,
        },
        "Kitco": {
            "url": "https://www.kitco.com/rss/news.xml",
            "icon": "🥇", "priority": 7,
        },
        # investing.com blocks scrapers via Cloudflare — replaced with Mining.com for metals
        "Mining.com": {
            "url": "https://www.mining.com/feed/",
            "icon": "⛏️", "priority": 7,
        },
        "DailyFX": {
            "url": "https://www.dailyfx.com/feeds/all",
            "icon": "📊", "priority": 6,
        },
        "CNBC Markets": {
            "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "icon": "📺", "priority": 9,
        },
        "CNBC Finance": {
            "url": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
            "icon": "📺", "priority": 9,
        },
        "MarketWatch": {
            "url": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
            "icon": "⌚", "priority": 6,
        },
        "Benzinga": {
            "url": "https://www.benzinga.com/feed",
            "icon": "📰", "priority": 6,
        },
        "ZeroHedge": {
            "url": "https://feeds.feedburner.com/zerohedge/feed",
            "icon": "⚠️", "priority": 5,
        },
        "Nasdaq": {
            "url": "https://www.nasdaq.com/feed/rssoutbound?category=Markets",
            "icon": "💻", "priority": 7,
        },
        "MyFXBook": {
            "url": "https://www.myfxbook.com/rss/forex-news-rss",
            "icon": "📒", "priority": 7,
        },
        "ForexFactory": {
            "url": "https://www.forexfactory.com/ff_calendar.php?week=this&ctype=impact&timezone=NY&format=xml",
            "icon": "🏭", "priority": 7,
        },
        "CoinDesk": {
            "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "icon": "₿", "priority": 7,
        },
        "CoinTelegraph": {
            "url": "https://cointelegraph.com/rss",
            "icon": "📱", "priority": 7,
        },
        "Decrypt": {
            "url": "https://decrypt.co/feed",
            "icon": "🔑", "priority": 6,
        },
        "The Block": {
            "url": "https://www.theblock.co/rss.xml",
            "icon": "⛓️", "priority": 6,
        },
        "Yahoo Finance": {
            "url": "https://finance.yahoo.com/rss/topfinstories",
            "icon": "🟣", "priority": 6,
        },
        "Business Wire": {
            "url": "https://www.businesswire.com/rss/home/?rss=G1",
            "icon": "📡", "priority": 6,
        },
    }

    # Which RSS sources are relevant for each symbol
    SYMBOL_RSS_MAP: Dict[str, List[str]] = {
        "XAUUSD": ["ForexLive", "FXStreet", "Kitco", "Mining.com", "MyFXBook", "ForexFactory", "CNBC Markets"],
        "BTCUSD": ["CoinDesk", "CoinTelegraph", "Decrypt", "The Block", "ForexLive", "CNBC Markets"],
        "ETHUSD": ["CoinDesk", "CoinTelegraph", "Decrypt", "The Block", "ForexLive"],
        "EURUSD": ["FXStreet", "ForexLive", "DailyFX", "MyFXBook", "ForexFactory", "Guardian Business"],
        "GBPUSD": ["FXStreet", "ForexLive", "DailyFX", "MyFXBook", "Guardian Business"],
        "USDJPY": ["FXStreet", "ForexLive", "DailyFX", "MyFXBook", "CNBC Markets"],
        "SPX500": ["MarketWatch", "ForexLive", "CNBC Markets", "Guardian Business", "Nasdaq", "Business Wire"],
        "NAS100": ["MarketWatch", "CNBC Markets", "Nasdaq", "ForexLive", "Guardian Business", "Benzinga"],
    }
    
    _HIGH_IMPACT = [
        'breaking', 'crash', 'surprise', 'fomc', 'rate decision', 'rate hike', 'rate cut',
        'bankruptcy', 'recession', 'war', 'fed chair', 'powell', 'lagarde', 'bailey',
        'emergency', 'crisis', 'plunge', 'surges', 'record high', 'record low',
        'nfp', 'cpi', 'inflation', 'gdp', 'jobs report', 'unemployment', 'default',
        'sanctions', 'tariff', 'geopolitical', 'escalation', 'ceasefire',
    ]
    _LOW_IMPACT = [
        'analysis', 'outlook', 'forecast', 'technical', 'review',
        'weekly', 'monthly', 'seasonal', 'preview', 'wrap-up',
    ]

    def __init__(self):
        self.newsapi_key = os.getenv("NEWSAPI_KEY", "")
        self.marketaux_api_key = os.getenv("MARKETAUX_API_KEY", "")
        self.alphavantage_key = os.getenv("ALPHAVANTAGE_KEY", "")
        self._symbol_keywords = {
            "XAUUSD": ["gold price", "gold futures", "precious metals", "XAUUSD", "gold rally",
                       "gold demand", "gold ETF", "XAU", "safe haven", "gold market"],
            "BTCUSD": ["bitcoin price", "BTC", "cryptocurrency", "BTCUSD", "bitcoin rally",
                       "bitcoin ETF", "crypto market", "bitcoin halving", "digital asset"],
            "ETHUSD": ["ethereum price", "ETH", "ether", "ETHUSD", "ethereum upgrade",
                       "DeFi", "smart contract", "layer 2", "ethereum network"],
            "EURUSD": ["EUR USD", "euro forex", "eurozone", "EURUSD", "ECB", "euro rate",
                       "european central bank", "euro inflation", "EUR/USD"],
            "GBPUSD": ["GBP USD", "british pound", "sterling forex", "GBPUSD", "BOE",
                       "bank of england", "pound sterling", "UK economy", "GBP/USD"],
            "USDJPY": ["USD JPY", "yen", "japanese yen", "USDJPY", "BOJ", "bank of japan",
                       "yen intervention", "japan inflation", "USD/JPY"],
            "SPX500": ["S&P 500", "US stock market", "US equities", "S&P500", "SPX",
                       "wall street", "dow jones", "market rally", "stock market crash"],
            "NAS100": ["NASDAQ", "tech stocks", "nasdaq", "NAS100", "technology sector",
                       "big tech", "nasdaq composite", "QQQ", "semiconductor"],
        }

        self._financial_domains = "reuters.com,bloomberg.com,cnbc.com,fxstreet.com,kitco.com,investing.com,yahoofinance.com,marketwatch.com,wsj.com,ft.com"

        self._sentiment_cache: Dict[str, Dict] = {}
        self._sentiment_cache_lock = threading.Lock()
        self._SENTIMENT_CACHE_TTL = 300

        self._max_workers = 4
    
    def get_news(self, symbol: str, max_items: int = 15) -> List[Dict]:
        """
        Fetch news from all sources in parallel, deduplicate, and return top items.
        """
        keywords = self._symbol_keywords.get(symbol, [symbol])
        
        results = self._fetch_all_sources_parallel(symbol, keywords)
        
        if not results:
            return []
        
        deduplicated = self._deduplicate_and_rank(results, keywords)
        
        for item in deduplicated[:max_items]:
            item["sentiment_label"] = item.get("sentiment_label") or self._get_cached_sentiment(item["headline"])
        
        return deduplicated[:max_items]
    
    def _fetch_all_sources_parallel(self, symbol: str, keywords: List[str]) -> List[Dict]:
        """
        Fetch from all sources in parallel — RSS, yfinance, Alpha Vantage, NewsAPI, Marketaux.
        Uses as_completed() with 20s wall-clock budget.
        """
        all_results = []
        relevant_rss = self.SYMBOL_RSS_MAP.get(symbol, list(self.RSS_FEEDS.keys())[:4])
        max_workers = self._max_workers + len(relevant_rss) + 3
        _silent_sources = {"forexcom", "google"}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: Dict = {}

            # Tier 1: paid APIs
            if self.newsapi_key:
                futures[executor.submit(self._fetch_newsapi, keywords)] = "newsapi"
            if self.marketaux_api_key:
                futures[executor.submit(self._fetch_marketaux, symbol)] = "marketaux"
            if self.alphavantage_key:
                futures[executor.submit(self._fetch_alphavantage, symbol, keywords)] = "alphavantage"

            # Tier 2: direct RSS
            for src_name in relevant_rss:
                feed_cfg = self.RSS_FEEDS.get(src_name, {})
                feed_url = feed_cfg.get("url", "")
                if feed_url:
                    futures[executor.submit(
                        self._fetch_rss_direct, src_name, feed_url, keywords
                    )] = f"rss_{src_name}"

            # Tier 3: yfinance JSON (no API key required)
            futures[executor.submit(self._fetch_yfinance_news, symbol, keywords)] = "yfinance"

            # Tier 4: Google News scraping (unreliable — silent on failure)
            futures[executor.submit(self._fetch_forexcom_news, keywords)] = "forexcom"
            futures[executor.submit(self._fetch_google_news, keywords, symbol)] = "google"

            try:
                for future in as_completed(futures, timeout=20):
                    source = futures[future]
                    try:
                        items = future.result()
                        if items:
                            for item in items:
                                item["_source"] = source
                            all_results.extend(items)
                    except Exception as e:
                        if source not in _silent_sources:
                            print(f"[UnifiedNews] {source} failed ({type(e).__name__}): {e}")
            except TimeoutError:
                print(f"[UnifiedNews] Partial timeout — {len(all_results)} items collected")

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

            for entry in feed.entries[:30]:
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
    
    def _fetch_yfinance_news(self, symbol: str, keywords: List[str]) -> List[Dict]:
        """Fetch news from yfinance Ticker.news — no API key required."""
        ticker_sym = _YF_TICKER_MAP.get(symbol, symbol)
        try:
            import yfinance as yf
            ticker = yf.Ticker(ticker_sym)
            raw = ticker.news or []
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(days=3)
            items = []
            for art in raw[:30]:
                title = art.get("title", "")
                if not title or len(title) < 20:
                    continue
                pub_ts = art.get("providerPublishTime") or art.get("published_at")
                time_ago_str = "Live"
                if pub_ts:
                    try:
                        pub_dt = datetime.fromtimestamp(int(pub_ts), tz=timezone.utc)
                        if pub_dt < cutoff:
                            continue
                        diff_min = (now - pub_dt).total_seconds() / 60
                        if diff_min < 60:
                            time_ago_str = f"{int(diff_min)}m ago"
                        elif diff_min < 1440:
                            time_ago_str = f"{int(diff_min/60)}h ago"
                        else:
                            time_ago_str = f"{int(diff_min/1440)}d ago"
                    except Exception:
                        pass
                publisher = art.get("publisher", "Yahoo Finance")
                sentiment_result = self._analyze_with_cache(title)
                impact = self._analyze_impact(title)
                items.append({
                    "headline": title[:250],
                    "sentiment": sentiment_result["score"],
                    "sentiment_label": sentiment_result["sentiment"],
                    "confidence": sentiment_result["confidence"],
                    "impact": impact,
                    "time_ago": time_ago_str,
                    "source": publisher,
                    "source_icon": self.SOURCE_ICONS.get(publisher, "🟣"),
                    "url": art.get("link", ""),
                    "impact_timing": "Market hours",
                })
            return items
        except Exception as e:
            print(f"[UnifiedNews] yfinance news error ({ticker_sym}): {e}")
            return []

    def _fetch_alphavantage(self, symbol: str, keywords: List[str]) -> List[Dict]:
        """Fetch from Alpha Vantage News & Sentiment API (free: 25 req/day)."""
        try:
            av_symbol = _YF_TICKER_MAP.get(symbol, symbol).replace("=X", "").replace("=F", "")
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": av_symbol,
                "limit": 10,
                "sort": "LATEST",
                "apikey": self.alphavantage_key,
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                return []
            data = resp.json()
            feed = data.get("feed", [])
            now = datetime.now(timezone.utc)
            items = []
            for art in feed:
                title = art.get("title", "")
                if not title or len(title) < 20:
                    continue
                raw_score = float(art.get("overall_sentiment_score", 0))
                raw_label = art.get("overall_sentiment_label", "Neutral").lower()
                if "bullish" in raw_label:
                    sentiment_label = "bullish"
                elif "bearish" in raw_label:
                    sentiment_label = "bearish"
                else:
                    sentiment_label = "neutral"
                # Parse time format: 20240422T120000
                time_str = art.get("time_published", "")
                time_ago_str = "Live"
                try:
                    pub_dt = datetime.strptime(time_str, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
                    diff_min = (now - pub_dt).total_seconds() / 60
                    if diff_min < 60:
                        time_ago_str = f"{int(diff_min)}m ago"
                    elif diff_min < 1440:
                        time_ago_str = f"{int(diff_min/60)}h ago"
                    else:
                        time_ago_str = f"{int(diff_min/1440)}d ago"
                except Exception:
                    pass
                source = art.get("source", "AlphaVantage")
                impact = self._analyze_impact(title)
                items.append({
                    "headline": title[:250],
                    "sentiment": raw_score,
                    "sentiment_label": sentiment_label,
                    "confidence": abs(raw_score),
                    "impact": impact,
                    "time_ago": time_ago_str,
                    "source": source,
                    "source_icon": self.SOURCE_ICONS.get(source, "📰"),
                    "url": art.get("url", ""),
                    "impact_timing": "Market hours",
                })
            return items
        except Exception as e:
            print(f"[UnifiedNews] Alpha Vantage error: {e}")
            return []

    def _deduplicate_and_rank(self, items: List[Dict], keywords: Optional[List[str]] = None) -> List[Dict]:
        """
        Deduplicate using Jaccard token similarity (≥0.55 = duplicate).
        Rank by: source_priority + recency_bonus (0/1/3) + relevance*2.
        Token sets are stored for O(1) pre-filter before full Jaccard check.
        """
        if not items:
            return []

        seen_token_sets: List[set] = []   # parallel list to seen_headlines
        seen_headlines: List[str] = []
        seen_urls: set = set()
        unique_items: List[Dict] = []
        now = datetime.now(timezone.utc)

        for item in items:
            headline = item.get("headline", "")
            url = item.get("url", "")

            if not headline or len(headline) < 20:
                continue
            if url and url in seen_urls:
                continue

            htokens = set(headline.lower().split())
            is_dup = False
            for i, seen in enumerate(seen_headlines):
                # Fast pre-filter: if token sets share no words, Jaccard=0
                if not htokens & seen_token_sets[i]:
                    continue
                if _jaccard(headline, seen) >= 0.55:
                    is_dup = True
                    break
            if is_dup:
                continue

            seen_headlines.append(headline)
            seen_token_sets.append(htokens)
            if url:
                seen_urls.add(url)

            # Recency bonus from time_ago string
            recency_bonus = 0
            ta = item.get("time_ago", "")
            try:
                if ta.endswith("m ago"):
                    mins = int(ta.split("m")[0])
                    if mins <= 60:
                        recency_bonus = 3
                    elif mins <= 120:
                        recency_bonus = 1
                elif ta.endswith("h ago"):
                    hrs = int(ta.split("h")[0])
                    if hrs <= 2:
                        recency_bonus = 1
            except Exception:
                pass

            src_priority = self.SOURCE_PRIORITY.get(item.get("source", ""), 5)
            rel = _relevance(headline, keywords or []) if keywords else 0.0
            item["_score"] = src_priority + recency_bonus + rel * 2
            unique_items.append(item)

        unique_items.sort(key=lambda x: x.get("_score", 0), reverse=True)
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
            
            now = datetime.now(timezone.utc)
            news_items = []
            for article in articles:
                headline = article.get("title", "")
                if not headline or headline == "[Removed]":
                    continue

                published_at = article.get("publishedAt", "")
                time_ago_str = "Live"
                try:
                    pub_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    diff_min = (now - pub_dt).total_seconds() / 60
                    if diff_min < 60:
                        time_ago_str = f"{int(diff_min)}m ago"
                    elif diff_min < 1440:
                        time_ago_str = f"{int(diff_min / 60)}h ago"
                    else:
                        time_ago_str = f"{int(diff_min / 1440)}d ago"
                except Exception:
                    pass

                sentiment_result = self._analyze_with_cache(headline)
                impact = self._analyze_impact(headline)
                source = article.get("source", {}).get("name", "News")

                news_items.append({
                    "headline": headline[:250],
                    "sentiment": sentiment_result["score"],
                    "sentiment_label": sentiment_result["sentiment"],
                    "confidence": sentiment_result["confidence"],
                    "impact": impact,
                    "time_ago": time_ago_str,
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
        t = text.lower()
        if any(w in t for w in UnifiedNewsService._HIGH_IMPACT):
            return "HIGH"
        if any(w in t for w in UnifiedNewsService._LOW_IMPACT):
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
