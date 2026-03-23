"""
Intelligent News Service - AI-Powered Financial News
Uses DeepSeek LLM for intelligent sentiment analysis
Falls back to NewsAPI.org + keyword-based sentiment
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import requests
import random
from typing import List, Dict, Optional

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# Import DeepSeek sentiment analyzer
from services.deepseek_sentiment import analyze_sentiment, get_aggregate_sentiment, sentiment_analyzer


class IntelligentNewsService:
    """
    AI-powered news fetching using Marketaux API.
    
    Features:
    - Marketaux API (FREE) - intelligent search + AI sentiment
    - NewsAPI fallback - more sources, local sentiment
    - Proper caching
    - No fake "Visit..." fallback articles
    """
    
    SOURCE_ICONS = {
        "Reuters": "📰",
        "Bloomberg": "💼",
        "CNBC": "📺",
        "FX Street": "💱",
        "FXStreet": "💱",
        "Forex Factory": "🏭",
        "Forex.com": "💹",
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
    }
    
    def __init__(self):
        self.marketaux_api_key = os.getenv("MARKETAUX_API_KEY", "")
        self.newsapi_key = os.getenv("NEWSAPI_KEY", "")
        self._symbol_keywords = {
            "XAUUSD": ["gold price", "gold futures", "precious metals"],
            "BTCUSD": ["bitcoin price", "BTC", "cryptocurrency"],
            "ETHUSD": ["ethereum price", "ETH", "ether"],
            "EURUSD": ["EUR USD", "euro forex", "eurozone"],
            "GBPUSD": ["GBP USD", "british pound", "sterling forex"],
            "USDJPY": ["USD JPY", "yen", "japanese yen"],
            "SPX500": ["S&P 500", "US stock market", "US equities"],
            "NAS100": ["NASDAQ", "tech stocks", "nasdaq"],
        }
        
        self._financial_domains = "reuters.com,bloomberg.com,cnbc.com,fxstreet.com,kitco.com,investing.com,yahoofinance.com,marketwatch.com,wsj.com,ft.com,forex.com"
    
    def get_news(self, symbol: str) -> List[Dict]:
        """Fetch news for a symbol using intelligent search."""
        
        # Try NewsAPI as primary (large source pool)
        if self.newsapi_key:
            results = self._fetch_newsapi(symbol)
            if results:
                return results
        
        # Try Forex.com via Google News RSS
        results = self._fetch_forexcom_news(symbol)
        if results:
            return results
        
        # Try DuckDuckGo news search as free alternative
        results = self._fetch_duckduckgo_news(symbol)
        if results:
            return results
        
        # Return empty list - no fake articles
        return []
    
    def get_news_with_aggregate(self, symbol: str) -> Dict:
        """Get news with aggregate sentiment analysis."""
        news_items = self.get_news(symbol)
        
        if not news_items:
            return {"items": [], "aggregate": {"sentiment": "neutral", "score": 0.0, "confidence": 0.0}}
        
        headlines = [item["headline"] for item in news_items]
        aggregate = get_aggregate_sentiment(headlines)
        
        return {
            "items": news_items,
            "aggregate": aggregate
        }
    
    def _fetch_marketaux(self, symbol: str) -> Optional[List[Dict]]:
        """Fetch from Marketaux API - has built-in sentiment!"""
        try:
            keywords = self._symbol_keywords.get(symbol, [symbol])
            query = " OR ".join(keywords[:2])
            
            url = "https://api.marketaux.com/v1/news/all"
            params = {
                "api_key": self.marketaux_api_key,
                "symbols": symbol,
                "limit": 10,
                "sort": "published_at:desc"
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            if "data" not in data:
                return None
            
            news_items = []
            for item in data.get("data", []):
                sentiment = item.get("sentiment_score", 0)
                
                # Convert sentiment to our format
                if sentiment > 0.3:
                    impact = "HIGH"
                elif sentiment < -0.3:
                    impact = "HIGH"
                elif sentiment != 0:
                    impact = "MEDIUM"
                else:
                    impact = "MEDIUM"
                
                source = item.get("source", "Marketaux")
                
                news_items.append({
                    "headline": item.get("title", "")[:250],
                    "sentiment": sentiment,
                    "impact": impact,
                    "time_ago": self._time_ago(item.get("published_at")),
                    "source": source,
                    "source_icon": self.SOURCE_ICONS.get(source, "📰"),
                    "url": item.get("url", ""),
                    "impact_timing": "Market hours"
                })
            
            return news_items[:10] if news_items else None
            
        except Exception as e:
            print(f"[IntelligentNews] Marketaux error: {e}")
            return None
    
    def _fetch_newsapi(self, symbol: str) -> Optional[List[Dict]]:
        """Fetch from NewsAPI.org with financial sources."""
        try:
            keywords = self._symbol_keywords.get(symbol, [symbol])
            query = " OR ".join([f'"{k}"' for k in keywords[:3]])
            
            url = "https://newsapi.org/v2/everything"
            params = {
                "apiKey": self.newsapi_key,
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10,
                "domains": self._financial_domains,
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            articles = data.get("articles", [])
            
            if not articles:
                params["domains"] = None
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    articles = response.json().get("articles", [])
            
            news_items = []
            for article in articles:
                headline = article.get("title", "")
                
                ai_sentiment = analyze_sentiment(headline)
                impact = self._analyze_impact(headline)
                
                source = article.get("source", {}).get("name", "News")
                
                news_items.append({
                    "headline": headline[:250],
                    "sentiment": ai_sentiment["score"],
                    "sentiment_label": ai_sentiment["sentiment"],
                    "confidence": ai_sentiment["confidence"],
                    "impact": impact,
                    "time_ago": "Live",
                    "source": source,
                    "source_icon": self.SOURCE_ICONS.get(source, "📰"),
                    "url": article.get("url", ""),
                    "impact_timing": "Market hours"
                })
            
            return news_items[:10] if news_items else None
            
        except Exception as e:
            print(f"[IntelligentNews] NewsAPI error: {e}")
            return None
    
    def _fetch_duckduckgo_news(self, symbol: str) -> Optional[List[Dict]]:
        """Fetch from DuckDuckGo news search (free, no API key needed)."""
        try:
            keywords = self._symbol_keywords.get(symbol, [symbol])
            
            url = f"https://duckduckgo.com/?q={keywords[0]}+news&format=json"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code != 200:
                return None
            
            # DuckDuckGo returns HTML with embedded JSON
            text = response.text
            start = text.find('DDG.rrd = ')
            if start == -1:
                return None
            
            start += 10
            end = text.find(';', start)
            if end == -1:
                return None
            
            import json
            data = json.loads(text[start:end])
            
            results = data.get('results', [])
            
            news_items = []
            for r in results[:10]:
                title = r.get('title', '')
                if len(title) < 20:
                    continue
                
                ai_sentiment = analyze_sentiment(title)
                impact = self._analyze_impact(title)
                
                news_items.append({
                    "headline": title[:250],
                    "sentiment": ai_sentiment["score"],
                    "sentiment_label": ai_sentiment["sentiment"],
                    "confidence": ai_sentiment["confidence"],
                    "impact": impact,
                    "time_ago": "Live",
                    "source": r.get('source', 'Web'),
                    "source_icon": self.SOURCE_ICONS.get(r.get('source', ''), "🔍"),
                    "url": r.get('url', ''),
                    "impact_timing": "Market hours"
                })
            
            return news_items[:10] if news_items else None
            
        except Exception as e:
            print(f"[IntelligentNews] DuckDuckGo error: {e}")
            return None
    
    def _fetch_forexcom_news(self, symbol: str) -> Optional[List[Dict]]:
        """Fetch forex.com news via Google News RSS."""
        try:
            import xml.etree.ElementTree as ET
            
            keywords = self._symbol_keywords.get(symbol, [symbol])
            query = "+".join(keywords[:2])
            
            url = f"https://news.google.com/rss/search?q=site:forex.com+{query}&hl=en-US&gl=US&ceid=US:en"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                return None
            
            root = ET.fromstring(response.text)
            items = root.findall('.//item')
            
            if not items:
                return None
            
            news_items = []
            for item in items[:10]:
                title_elem = item.find('title')
                link_elem = item.find('link')
                pubdate_elem = item.find('pubDate')
                
                title = title_elem.text if title_elem is not None else ""
                link = str(link_elem.text) if link_elem is not None else ""
                pubdate = str(pubdate_elem.text) if pubdate_elem is not None else ""
                
                if not title or len(title) < 20:
                    continue
                
                ai_sentiment = analyze_sentiment(title)
                impact = self._analyze_impact(title)
                
                news_items.append({
                    "headline": title[:250],
                    "sentiment": ai_sentiment["score"],
                    "sentiment_label": ai_sentiment["sentiment"],
                    "confidence": ai_sentiment["confidence"],
                    "impact": impact,
                    "time_ago": self._parse_pubdate(pubdate),
                    "source": "Forex.com",
                    "source_icon": "💹",
                    "url": self._extract_article_url(link),
                    "impact_timing": "Market hours"
                })
            
            return news_items[:10] if news_items else None
            
        except Exception as e:
            print(f"[IntelligentNews] Forex.com error: {e}")
            return None
    
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
    
    def _analyze_impact(self, text: str) -> str:
        """Analyze news impact level."""
        if not text:
            return "MEDIUM"
        
        text_lower = text.lower()
        
        high_impact = [
            'breaking', 'crash', 'surprise', 'fomc', 'rate decision',
            'bankruptcy', 'recession', 'war', 'fed chair', 'powell'
        ]
        
        low_impact = [
            'analysis', 'outlook', 'forecast', 'technical', 'review'
        ]
        
        if any(word in text_lower for word in high_impact):
            return "HIGH"
        elif any(word in text_lower for word in low_impact):
            return "LOW"
        return "MEDIUM"
    
    def _time_ago(self, dt_str: str) -> str:
        """Convert datetime to 'time ago' string."""
        if not dt_str:
            return "Live"
        
        try:
            from datetime import datetime, timedelta
            
            # Handle different formats
            if 'T' in dt_str:
                dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            else:
                return "Live"
            
            diff = datetime.now(dt.tzinfo) - dt
            minutes = diff.total_seconds() / 60
            
            if minutes < 60:
                return f"{int(minutes)}m ago"
            elif minutes < 1440:
                return f"{int(minutes/60)}h ago"
            else:
                return f"{int(minutes/1440)}d ago"
                
        except:
            return "Live"


# Global instance
news_service = IntelligentNewsService()


def get_intelligent_news(symbol: str) -> List[Dict]:
    """Convenience function to get intelligent news."""
    return news_service.get_news(symbol)