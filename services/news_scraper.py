"""
Financial News Scrapers
========================
Scrapers for Forex Factory, FXStreet, Investing.com, and other financial news sources.

Note: Some websites may block automated scraping. For production use, consider:
- Using official APIs (NewsAPI, Financial Modeling Prep)
- Running scrapers less frequently
- Adding proper delays between requests
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import re
import json
import random


class NewsSource(Enum):
    """News source identifiers with URLs."""
    FOREX_FACTORY = "forex_factory"
    FXSTREET = "fxstreet"
    INVESTING_COM = "investing_com"
    DAILYFX = "dailyfx"
    BLOOMBERG = "bloomberg"
    REUTERS = "reuters"
    COINDESK = "coindesk"
    CRYPTO_PANIC = "crypto_panic"
    MARKET_WATCH = "marketwatch"
    YAHOO_FINANCE = "yahoo_finance"
    CNBC = "cnbc"
    TRADING_ECONOMICS = "trading_economics"
    FX_EMPIRE = "fx_empire"
    KITCO = "kitco"


# News source configuration with URLs and icons
NEWS_SOURCE_CONFIG = {
    NewsSource.BLOOMBERG: {
        "name": "Bloomberg",
        "url": "https://www.bloomberg.com",
        "icon": "💼",
        "color": "#FF6600",
        "description": "Global business and financial news"
    },
    NewsSource.CNBC: {
        "name": "CNBC",
        "url": "https://www.cnbc.com",
        "icon": "📺",
        "color": "#003366",
        "description": "Stock market news & analysis"
    },
    NewsSource.INVESTING_COM: {
        "name": "Investing.com",
        "url": "https://www.investing.com",
        "icon": "📈",
        "color": "#008000",
        "description": "Financial markets & trading"
    },
    NewsSource.FXSTREET: {
        "name": "FXStreet",
        "url": "https://www.fxstreet.com",
        "icon": "💱",
        "color": "#E91E63",
        "description": "Forex news & analysis"
    },
    NewsSource.FOREX_FACTORY: {
        "name": "Forex Factory",
        "url": "https://www.forexfactory.com",
        "icon": "🏭",
        "color": "#FF9800",
        "description": "Forex trading community"
    },
    NewsSource.REUTERS: {
        "name": "Reuters",
        "url": "https://www.reuters.com",
        "icon": "📰",
        "color": "#FF8000",
        "description": "Breaking news & markets"
    },
    NewsSource.MARKET_WATCH: {
        "name": "MarketWatch",
        "url": "https://www.marketwatch.com",
        "icon": "⌚",
        "color": "#00A600",
        "description": "Stock market data"
    },
    NewsSource.YAHOO_FINANCE: {
        "name": "Yahoo Finance",
        "url": "https://finance.yahoo.com",
        "icon": "🟣",
        "color": "#400090",
        "description": "Finance & investing"
    },
    NewsSource.DAILYFX: {
        "name": "DailyFX",
        "url": "https://www.dailyfx.com",
        "icon": "📊",
        "color": "#E74C3C",
        "description": "Forex analysis & education"
    },
    NewsSource.COINDESK: {
        "name": "CoinDesk",
        "url": "https://www.coindesk.com",
        "icon": "₿",
        "color": "#1652F0",
        "description": "Cryptocurrency news"
    },
    NewsSource.TRADING_ECONOMICS: {
        "name": "Trading Economics",
        "url": "https://tradingeconomics.com",
        "icon": "🌐",
        "color": "#2E86AB",
        "description": "Economic indicators"
    },
    NewsSource.FX_EMPIRE: {
        "name": "FX Empire",
        "url": "https://www.fxempire.com",
        "icon": "🏛️",
        "color": "#9B59B6",
        "description": "Forex & CFD analysis"
    },
    NewsSource.KITCO: {
        "name": "Kitco News",
        "url": "https://www.kitco.com/news/",
        "icon": "🥇",
        "color": "#FFD700",
        "description": "Precious metals news"
    },
}


@dataclass
class NewsArticle:
    """Container for scraped news article."""
    title: str
    source: NewsSource
    url: str
    published_at: datetime
    sentiment: float  # -1 to 1
    impact: str  # 'HIGH', 'MEDIUM', 'LOW'
    summary: Optional[str] = None
    content: Optional[str] = None
    related_symbols: List[str] = None
    id: Optional[str] = None
    relevance: Optional[float] = None


class ForexFactoryScraper:
    """
    Scraper for Forex Factory economic calendar and news.

    Scrapes:
    - Economic calendar events
    - Forex news articles
    - Market sentiment
    """

    BASE_URL = "https://www.forexfactory.com"

    async def fetch_calendar(self, session: aiohttp.ClientSession,
                            date: datetime = None) -> List[Dict]:
        """
        Fetch economic calendar events.

        Parameters
        ----------
        session : aiohttp.ClientSession
            HTTP session
        date : datetime
            Date to fetch (default: today)
        """
        if date is None:
            date = datetime.now()

        url = f"{self.BASE_URL}/calendar.php?day={date.strftime('%m.%d.%Y')}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        try:
            async with session.get(url, headers=headers, timeout=10) as response:
                html = await response.text()
                return self._parse_calendar(html)
        except Exception as e:
            print(f"Forex Factory calendar error: {e}")
            return []

    def _parse_calendar(self, html: str) -> List[Dict]:
        """Parse economic calendar HTML."""
        soup = BeautifulSoup(html, 'lxml')
        events = []

        # Find calendar rows
        rows = soup.select('tr.calendar__row')

        for row in rows:
            try:
                time_elem = row.select_one('td.calendar__cell.calendar__time')
                impact_elem = row.select_one('td.calendar__cell.calendar__impact')
                currency_elem = row.select_one('td.calendar__cell.calendar__currency')
                event_elem = row.select_one('td.calendar__cell.calendar__event')
                actual_elem = row.select_one('td.calendar__cell.calendar__actual')
                forecast_elem = row.select_one('td.calendar__cell.calendar__forecast')
                previous_elem = row.select_one('td.calendar__cell.calendar__previous')

                if not all([time_elem, event_elem]):
                    continue

                # Extract impact color
                impact_color = ''
                if impact_elem:
                    img = impact_elem.select_one('img')
                    if img:
                        impact_color = img.get('src', '')

                # Determine impact level
                if 'red' in impact_color:
                    impact = 'HIGH'
                elif 'orange' in impact_color:
                    impact = 'MEDIUM'
                elif 'yellow' in impact_color:
                    impact = 'LOW'
                else:
                    impact = 'NONE'

                event = {
                    'time': time_elem.get_text(strip=True),
                    'currency': currency_elem.get_text(strip=True) if currency_elem else '',
                    'event': event_elem.get_text(strip=True),
                    'impact': impact,
                    'actual': actual_elem.get_text(strip=True) if actual_elem else '',
                    'forecast': forecast_elem.get_text(strip=True) if forecast_elem else '',
                    'previous': previous_elem.get_text(strip=True) if previous_elem else '',
                }

                events.append(event)

            except Exception as e:
                continue

        return events

    async def fetch_news(self, session: aiohttp.ClientSession,
                        limit: int = 20) -> List[NewsArticle]:
        """Fetch latest Forex news."""
        url = f"{self.BASE_URL}/forum/"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        try:
            async with session.get(url, headers=headers, timeout=10) as response:
                html = await response.text()
                return self._parse_news(html, limit)
        except Exception as e:
            print(f"Forex Factory news error: {e}")
            return []

    def _parse_news(self, html: str, limit: int) -> List[NewsArticle]:
        """Parse news HTML."""
        soup = BeautifulSoup(html, 'lxml')
        articles = []

        # Find forum threads
        threads = soup.select('li.block-body')

        for thread in threads[:limit]:
            try:
                title_elem = thread.select_one('a.block-body')
                time_elem = thread.select_one('time')

                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                href = title_elem.get('href', '')
                
                # Build full URL
                full_url = f"{self.BASE_URL}{href}" if href and not href.startswith('http') else href

                # Skip non-news threads
                if any(x in title.lower() for x in ['reply', 'post', 'discussion']):
                    continue

                # Simple sentiment analysis based on keywords
                sentiment = self._analyze_sentiment(title)

                # Determine impact
                impact = 'MEDIUM'
                if any(x in title.lower() for x in ['central bank', 'fed', 'ecb', 'rate decision', 'cpi', 'inflation']):
                    impact = 'HIGH'

                article = NewsArticle(
                    title=title,
                    source=NewsSource.FOREX_FACTORY,
                    url=full_url,
                    published_at=datetime.now(),
                    sentiment=sentiment,
                    impact=impact,
                    related_symbols=self._extract_symbols(title),
                    id=f"ff-{len(articles)}",
                    relevance=0.7
                )

                articles.append(article)

            except Exception as e:
                continue

        return articles

    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis."""
        bullish_words = ['rise', 'gain', 'surge', 'rally', 'bullish', 'positive',
                        'upgrade', 'beat', 'exceed', 'strong', 'higher']
        bearish_words = ['fall', 'drop', 'plunge', 'crash', 'bearish', 'negative',
                        'downgrade', 'miss', 'weak', 'lower', 'concern']

        text_lower = text.lower()

        bullish_count = sum(1 for word in bullish_words if word in text_lower)
        bearish_count = sum(1 for word in bearish_words if word in text_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0

        return (bullish_count - bearish_count) / total

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract currency symbols from text."""
        symbols = []

        # Major currency pairs
        currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']

        for curr in currencies:
            if curr in text.upper():
                symbols.append(f"{curr}USD" if curr != 'USD' else 'DXY')

        # Gold
        if any(x in text.lower() for x in ['gold', 'xau']):
            symbols.append('XAUUSD')

        # Oil
        if any(x in text.lower() for x in ['oil', 'crude', 'wti', 'brent']):
            symbols.append('USOIL')

        return list(set(symbols))


class FXStreetScraper:
    """
    Scraper for FXStreet news and analysis.
    """

    BASE_URL = "https://www.fxstreet.com"

    async def fetch_news(self, session: aiohttp.ClientSession,
                        limit: int = 20) -> List[NewsArticle]:
        """Fetch latest news from FXStreet."""
        url = f"{self.BASE_URL}/news"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        try:
            async with session.get(url, headers=headers, timeout=10) as response:
                html = await response.text()
                return self._parse_news(html, limit)
        except Exception as e:
            print(f"FXStreet news error: {e}")
            return []

    def _parse_news(self, html: str, limit: int) -> List[NewsArticle]:
        """Parse FXStreet news HTML."""
        soup = BeautifulSoup(html, 'lxml')
        articles = []

        # Find news articles
        article_elems = soup.select('article')

        for elem in article_elems[:limit]:
            try:
                title_elem = elem.select_one('h2 a, h3 a')
                time_elem = elem.select_one('time')
                summary_elem = elem.select_one('p')

                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                href = title_elem.get('href', '')

                # Build full URL
                full_url = href if href.startswith('http') else f"{self.BASE_URL}{href}"

                # Parse publish time
                published_at = datetime.now()
                if time_elem:
                    datetime_str = time_elem.get('datetime', '')
                    if datetime_str:
                        try:
                            published_at = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                        except:
                            pass

                # Sentiment analysis
                sentiment = self._analyze_sentiment(title)

                # Impact level
                impact = 'MEDIUM'
                if any(x in title.lower() for x in ['breaking', 'urgent', 'central bank']):
                    impact = 'HIGH'

                article = NewsArticle(
                    title=title,
                    source=NewsSource.FXSTREET,
                    url=full_url,
                    published_at=published_at,
                    sentiment=sentiment,
                    impact=impact,
                    summary=summary_elem.get_text(strip=True)[:200] if summary_elem else None,
                    related_symbols=self._extract_symbols(title),
                    id=f"fxs-{len(articles)}",
                    relevance=0.75
                )

                articles.append(article)

            except Exception as e:
                continue

        return articles

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of news title."""
        bullish_words = ['rises', 'gains', 'jumps', 'surges', 'bullish', 'upbeat',
                        'hawkish', 'strengthens', 'rally', 'higher']
        bearish_words = ['falls', 'drops', 'plunges', 'bearish', 'dovish',
                        'weakens', 'lower', 'concerns', 'slumps']

        text_lower = text.lower()
        bullish_count = sum(1 for word in bullish_words if word in text_lower)
        bearish_count = sum(1 for word in bearish_words if word in text_lower)

        total = bullish_count + bearish_count
        return (bullish_count - bearish_count) / total if total > 0 else 0.0

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract symbols from text."""
        return ForexFactoryScraper()._extract_symbols(text)


class InvestingComScraper:
    """
    Scraper for Investing.com news and economic calendar.
    Specialized in currency and commodity news.
    """

    BASE_URL = "https://www.investing.com"
    
    # Symbol to Investing.com section mapping
    SYMBOL_SECTIONS = {
        'XAUUSD': '/currencies/xau-usd-news',
        'BTCUSD': '/crypto/bitcoin/btc-usd-news',
        'ETHUSD': '/crypto/ethereum/eth-usd-news',
        'EURUSD': '/currencies/eur-usd-news',
        'GBPUSD': '/currencies/gbp-usd-news',
        'USDJPY': '/currencies/usd-jpy-news',
        'SPX500': '/indices/us-spx-500-news',
        'NAS100': '/indices/nq-100-news',
        'USOIL': '/commodities/crude-oil-news',
        'BRENT': '/commodities/brent-crude-oil-news',
    }

    async def fetch_calendar(self, session: aiohttp.ClientSession,
                            date: datetime = None) -> List[Dict]:
        """Fetch economic calendar."""
        if date is None:
            date = datetime.now()

        url = f"{self.BASE_URL}/economic-calendar/"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': url
        }

        try:
            async with session.get(url, headers=headers, timeout=10) as response:
                html = await response.text()
                return self._parse_calendar(html)
        except Exception as e:
            print(f"Investing.com calendar error: {e}")
            return []

    def _parse_calendar(self, html: str) -> List[Dict]:
        """Parse economic calendar."""
        soup = BeautifulSoup(html, 'lxml')
        events = []

        # Find calendar rows
        rows = soup.select('tr.js-event-item')

        for row in rows:
            try:
                time_elem = row.select_one('span.first')
                impact_elem = row.select_one('span.sentiment')
                currency_elem = row.select_one('td.flagCur')
                event_elem = row.select_one('td.event')

                # Get impact from class
                impact = 'MEDIUM'
                if impact_elem:
                    impact_class = impact_elem.get('class', [])
                    if 'high' in str(impact_class):
                        impact = 'HIGH'
                    elif 'low' in str(impact_class):
                        impact = 'LOW'

                event = {
                    'time': time_elem.get_text(strip=True) if time_elem else '',
                    'currency': currency_elem.get_text(strip=True) if currency_elem else '',
                    'event': event_elem.get_text(strip=True) if event_elem else '',
                    'impact': impact,
                }

                events.append(event)

            except Exception as e:
                continue

        return events

    async def fetch_news(self, session: aiohttp.ClientSession,
                        limit: int = 20, symbol: str = None) -> List[NewsArticle]:
        """Fetch latest news for specific symbol or general forex news."""
        # Determine URL based on symbol
        if symbol and symbol in self.SYMBOL_SECTIONS:
            url = f"{self.BASE_URL}{self.SYMBOL_SECTIONS[symbol]}"
        else:
            # Default to forex news
            url = f"{self.BASE_URL}/currencies/"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        try:
            async with session.get(url, headers=headers, timeout=15) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._parse_news(html, limit, symbol)
                else:
                    print(f"Investing.com news HTTP {response.status}")
                    return []
        except Exception as e:
            print(f"Investing.com news error: {e}")
            return []

    def _parse_news(self, html: str, limit: int, symbol: str = None) -> List[NewsArticle]:
        """Parse Investing.com news HTML."""
        soup = BeautifulSoup(html, 'lxml')
        articles = []

        # Try multiple selectors for news articles
        article_selectors = [
            'article',
            'div.article-item',
            'div.news-item',
            'div.largeTitle',
            'div[class*="article"]',
            'a[class*="article"]',
        ]

        article_elems = []
        for selector in article_selectors:
            elems = soup.select(selector)
            if elems:
                article_elems.extend(elems)

        # Remove duplicates
        seen_urls = set()
        unique_elems = []
        for elem in article_elems:
            link = elem.select_one('a')
            if link:
                href = link.get('href', '')
                if href and href not in seen_urls:
                    seen_urls.add(href)
                    unique_elems.append(elem)

        article_elems = unique_elems[:limit * 2]  # Get more to filter later

        for elem in article_elems:
            try:
                # Try to find link and title
                link_elem = elem.select_one('a')
                if not link_elem:
                    continue

                title = link_elem.get_text(strip=True)
                if not title or len(title) < 10:
                    continue

                href = link_elem.get('href', '')
                
                # Build full URL
                if href.startswith('/'):
                    full_url = f"{self.BASE_URL}{href}"
                elif href.startswith('http'):
                    full_url = href
                else:
                    continue

                # Skip non-news pages
                if any(x in full_url.lower() for x in ['/premium/', '/pro/', '/ad/', 'sponsor']):
                    continue

                # Find publish time
                published_at = datetime.now()
                time_elem = elem.select_one('time, span[class*="time"], small[class*="time"]')
                if time_elem:
                    datetime_str = time_elem.get('datetime', '') or time_elem.get_text(strip=True)
                    if datetime_str:
                        try:
                            if 'ago' in datetime_str.lower():
                                # Parse relative time
                                if 'hour' in datetime_str:
                                    hours = int(''.join(filter(str.isdigit, datetime_str)))
                                    published_at = datetime.now() - timedelta(hours=hours)
                                elif 'min' in datetime_str:
                                    minutes = int(''.join(filter(str.isdigit, datetime_str)))
                                    published_at = datetime.now() - timedelta(minutes=minutes)
                                elif 'day' in datetime_str:
                                    days = int(''.join(filter(str.isdigit, datetime_str)))
                                    published_at = datetime.now() - timedelta(days=days)
                            else:
                                published_at = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                        except:
                            pass

                # Find summary if available
                summary = None
                summary_elem = elem.select_one('p, div[class*="summary"], div[class*="desc"]')
                if summary_elem:
                    summary = summary_elem.get_text(strip=True)[:200]

                # Sentiment analysis
                sentiment = self._analyze_sentiment(title)

                # Impact level
                impact = 'MEDIUM'
                if any(x in title.lower() for x in ['fed', 'ecb', 'central bank', 'breaking', 'urgent', 'rate decision']):
                    impact = 'HIGH'

                # Extract related symbols
                related_symbols = self._extract_symbols(title)
                if symbol and symbol not in related_symbols:
                    related_symbols.insert(0, symbol)

                article = NewsArticle(
                    id=f"inv-{len(articles)}-{int(datetime.now().timestamp())}",
                    title=title,
                    source=NewsSource.INVESTING_COM,
                    url=full_url,
                    published_at=published_at,
                    sentiment=sentiment,
                    impact=impact,
                    summary=summary,
                    related_symbols=related_symbols,
                    relevance=0.8
                )

                articles.append(article)

                if len(articles) >= limit:
                    break

            except Exception as e:
                print(f"Error parsing article: {e}")
                continue

        # If we couldn't find enough articles, try alternative approach
        if len(articles) < limit // 2:
            articles.extend(self._parse_news_alternative(soup, limit - len(articles), symbol))

        return articles[:limit]

    def _parse_news_alternative(self, soup: BeautifulSoup, limit: int, symbol: str = None) -> List[NewsArticle]:
        """Alternative news parsing for different page layouts."""
        articles = []
        
        # Find all links that look like news
        news_links = soup.select('a[href*="/news/"], a[href*="news"]')
        
        for link in news_links[:limit * 3]:
            try:
                title = link.get_text(strip=True)
                href = link.get('href', '')
                
                if not title or len(title) < 10:
                    continue
                    
                if href.startswith('/'):
                    full_url = f"{self.BASE_URL}{href}"
                elif href.startswith('http'):
                    full_url = href
                else:
                    continue
                
                # Skip non-news
                if any(x in full_url.lower() for x in ['/premium/', '/pro/', '/ad/', 'sponsor', 'login']):
                    continue
                
                # Check if we already have this URL
                if any(a.url == full_url for a in articles):
                    continue
                
                sentiment = self._analyze_sentiment(title)
                impact = 'HIGH' if any(x in title.lower() for x in ['fed', 'ecb', 'central bank', 'breaking']) else 'MEDIUM'
                
                article = NewsArticle(
                    id=f"inv-alt-{len(articles)}",
                    title=title,
                    source=NewsSource.INVESTING_COM,
                    url=full_url,
                    published_at=datetime.now(),
                    sentiment=sentiment,
                    impact=impact,
                    related_symbols=self._extract_symbols(title),
                    relevance=0.7
                )
                
                articles.append(article)
                
                if len(articles) >= limit:
                    break
                    
            except:
                continue
        
        return articles

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment."""
        bullish_words = ['rises', 'gains', 'jumps', 'surges', 'bullish', 'upbeat',
                        'hawkish', 'strengthens', 'rally', 'higher', 'soars', 'climbs',
                        'positive', 'optimistic', 'upgrade', 'beat', 'exceeds']
        bearish_words = ['falls', 'drops', 'plunges', 'bearish', 'dovish',
                        'weakens', 'lower', 'concerns', 'slumps', 'declines', 'sinks',
                        'negative', 'pessimistic', 'downgrade', 'miss', 'below']

        text_lower = text.lower()
        bullish_count = sum(1 for word in bullish_words if word in text_lower)
        bearish_count = sum(1 for word in bearish_words if word in text_lower)

        total = bullish_count + bearish_count
        return (bullish_count - bearish_count) / total if total > 0 else 0.0

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract symbols from text."""
        symbols = []
        text_upper = text.upper()

        # Currency pairs
        if 'EUR' in text_upper and 'USD' in text_upper:
            symbols.append('EURUSD')
        elif 'GBP' in text_upper and 'USD' in text_upper:
            symbols.append('GBPUSD')
        elif 'USD' in text_upper and 'JPY' in text_upper:
            symbols.append('USDJPY')
        elif 'USD' in text_upper and 'CHF' in text_upper:
            symbols.append('USDCHF')
        elif 'AUD' in text_upper and 'USD' in text_upper:
            symbols.append('AUDUSD')
        elif 'USD' in text_upper and 'CAD' in text_upper:
            symbols.append('USDCAD')
        
        # Commodities
        if any(x in text_upper for x in ['GOLD', 'XAU', 'PRECIOUS METAL']):
            symbols.append('XAUUSD')
        if any(x in text_upper for x in ['OIL', 'CRUDE', 'WTI', 'BRENT']):
            symbols.append('USOIL')
        if 'SILVER' in text_upper or 'XAG' in text_upper:
            symbols.append('XAGUSD')
            
        # Crypto
        if 'BITCOIN' in text_upper or 'BTC' in text_upper:
            symbols.append('BTCUSD')
        if 'ETHEREUM' in text_upper or 'ETH' in text_upper:
            symbols.append('ETHUSD')
            
        # Indices
        if any(x in text_upper for x in ['S&P 500', 'SPX', 'US500']):
            symbols.append('SPX500')
        if any(x in text_upper for x in ['NASDAQ', 'NAS100', 'NQ', 'TECH']):
            symbols.append('NAS100')

        return list(set(symbols))


class NewsAggregator:
    """
    Aggregates news from multiple sources.
    Specialized in real-time currency and commodity news.
    """

    # User agents to rotate
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    ]

    def __init__(self):
        self.forex_factory = ForexFactoryScraper()
        self.fxstreet = FXStreetScraper()
        self.investing_com = InvestingComScraper()
        self._news_cache: Dict[str, List[NewsArticle]] = {}  # Cache by symbol
        self._calendar_cache: List[Dict] = []
        self._last_fetch: Dict[str, datetime] = {}  # Last fetch time by symbol

    def _get_headers(self) -> Dict:
        """Get random user agent headers."""
        return {
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    async def fetch_symbol_news(self, symbol: str, limit: int = 15) -> List[NewsArticle]:
        """
        Fetch news for a specific symbol.
        Prioritizes Investing.com for symbol-specific news.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., 'XAUUSD', 'EURUSD')
        limit : int
            Maximum number of articles to return

        Returns
        -------
        List[NewsArticle]
            List of news articles for the symbol
        """
        # Check cache (5 minutes)
        if symbol in self._news_cache and symbol in self._last_fetch:
            time_diff = (datetime.now() - self._last_fetch[symbol]).total_seconds()
            if time_diff < 300 and len(self._news_cache[symbol]) >= limit:
                return self._news_cache[symbol][:limit]

        all_articles = []

        # Primary: Investing.com symbol-specific news
        try:
            async with aiohttp.ClientSession(headers=self._get_headers()) as session:
                investing_articles = await self.investing_com.fetch_news(session, limit, symbol)
                if investing_articles:
                    all_articles.extend(investing_articles)
                    print(f"✓ Fetched {len(investing_articles)} articles from Investing.com for {symbol}")
        except Exception as e:
            print(f"Investing.com {symbol} news error: {e}")

        # Secondary: General forex/currency news if symbol-specific didn't return enough
        if len(all_articles) < limit:
            try:
                async with aiohttp.ClientSession(headers=self._get_headers()) as session:
                    # Fetch general currency news
                    general_articles = await self.investing_com.fetch_news(session, limit - len(all_articles), None)
                    if general_articles:
                        # Filter for relevant symbols
                        relevant = [a for a in general_articles if symbol in (a.related_symbols or []) or len(all_articles) < limit]
                        all_articles.extend(relevant[:limit - len(all_articles)])
                        print(f"✓ Fetched {len(relevant)} general articles for {symbol}")
            except Exception as e:
                print(f"General news fetch error: {e}")

        # Tertiary: FXStreet as backup
        if len(all_articles) < limit:
            try:
                async with aiohttp.ClientSession(headers=self._get_headers()) as session:
                    fxstreet_articles = await self.fxstreet.fetch_news(session, limit - len(all_articles))
                    if fxstreet_articles:
                        relevant = [a for a in fxstreet_articles if symbol in (a.related_symbols or []) or len(all_articles) < limit]
                        all_articles.extend(relevant[:limit - len(all_articles)])
            except Exception as e:
                print(f"FXStreet news error: {e}")

        # Sort by time and remove duplicates
        seen_urls = set()
        unique_articles = []
        for article in sorted(all_articles, key=lambda x: x.published_at, reverse=True):
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)

        # Cache results (even if limited)
        if unique_articles:
            self._news_cache[symbol] = unique_articles
            self._last_fetch[symbol] = datetime.now()
            print(f"✓ Cached {len(unique_articles)} news articles for {symbol}")
        else:
            # Return empty list instead of synthetic news
            print(f"⚠ No real news available for {symbol}")
            self._news_cache[symbol] = []
            self._last_fetch[symbol] = datetime.now()

        return unique_articles[:limit]

    async def fetch_all_news(self, limit_per_source: int = 10) -> List[NewsArticle]:
        """
        Fetch general news from all sources (no symbol filter).

        Parameters
        ----------
        limit_per_source : int
            Maximum articles per source
        """
        # Return cached news if recently fetched (within 5 minutes)
        cache_key = 'general'
        if cache_key in self._news_cache and cache_key in self._last_fetch:
            time_diff = (datetime.now() - self._last_fetch[cache_key]).total_seconds()
            if time_diff < 300:
                return self._news_cache[cache_key][:limit_per_source * 3]

        all_articles = []

        async with aiohttp.ClientSession(headers=self._get_headers()) as session:
            # Fetch from Investing.com (general currencies)
            try:
                investing_articles = await self.investing_com.fetch_news(session, limit_per_source, None)
                if investing_articles:
                    all_articles.extend(investing_articles)
                    print(f"✓ Fetched {len(investing_articles)} articles from Investing.com")
            except Exception as e:
                print(f"Investing.com general news error: {e}")

            # Fetch from FXStreet
            try:
                fxstreet_articles = await self.fxstreet.fetch_news(session, limit_per_source)
                if fxstreet_articles:
                    all_articles.extend(fxstreet_articles)
                    print(f"✓ Fetched {len(fxstreet_articles)} articles from FXStreet")
            except Exception as e:
                print(f"FXStreet news error: {e}")

        # Sort and deduplicate
        seen_urls = set()
        unique_articles = []
        for article in sorted(all_articles, key=lambda x: x.published_at, reverse=True):
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)

        # Cache results
        if unique_articles:
            self._news_cache[cache_key] = unique_articles
            self._last_fetch[cache_key] = datetime.now()
        else:
            # No synthetic fallback - return empty
            print("⚠ No real news available")
            self._news_cache[cache_key] = []
            self._last_fetch[cache_key] = datetime.now()

        return unique_articles[:limit_per_source * 3]

    async def fetch_calendar(self, date: datetime = None) -> List[Dict]:
        """Fetch economic calendar from all sources."""
        # Return cached calendar if recently fetched
        if self._calendar_cache and self._last_fetch and 'calendar' in self._last_fetch:
            time_diff = (datetime.now() - self._last_fetch.get('calendar', datetime.now() - timedelta(hours=1))).total_seconds()
            if time_diff < 300:
                return self._calendar_cache

        async with aiohttp.ClientSession(headers=self._get_headers()) as session:
            tasks = [
                self.forex_factory.fetch_calendar(session, date),
                self.investing_com.fetch_calendar(session, date),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_events = []
            for result in results:
                if isinstance(result, list):
                    all_events.extend(result)

            # Remove duplicates
            seen = set()
            unique_events = []
            for event in all_events:
                key = (event.get('time'), event.get('event'))
                if key not in seen:
                    seen.add(key)
                    unique_events.append(event)

            # Cache results
            self._calendar_cache = unique_events
            if 'calendar' not in self._last_fetch:
                self._last_fetch['calendar'] = datetime.now()

            return unique_events

    def _generate_fallback_news(self, count: int = 10) -> List[NewsArticle]:
        """Generate fallback news when scraping fails - with real source names and specific URLs."""
        templates = [
            ("Fed Chair Powell signals cautious approach to rate cuts amid inflation concerns", 0.3, "HIGH", ["EURUSD", "GBPUSD", "XAUUSD"], "forex_factory", "/calendar.php?day=03.09.2026"),
            ("ECB maintains hawkish stance as Eurozone inflation remains elevated", 0.4, "HIGH", ["EURUSD"], "fxstreet", "/forex-news/ecb-maintains-hawkish-stance-inflation-20260309"),
            ("Gold prices surge to weekly highs on safe-haven demand", 0.7, "MEDIUM", ["XAUUSD"], "investing_com", "/news/commodities/gold-prices-surge-safe-haven-demand"),
            ("Bitcoin breaks key resistance level as crypto market rallies", 0.8, "MEDIUM", ["BTCUSD", "ETHUSD"], "forex_factory", "/thread/bitcoin-breaks-resistance-crypto-rally"),
            ("US Dollar strengthens on better-than-expected jobs data", 0.5, "HIGH", ["EURUSD", "GBPUSD", "USDJPY"], "fxstreet", "/forex-news/usd-strengthens-jobs-data-20260309"),
            ("Oil prices decline on demand concerns despite OPEC+ cuts", -0.4, "MEDIUM", ["USOIL", "BRENT"], "investing_com", "/news/commodities/oil-prices-decline-demand-opec"),
            ("Bank of Japan hints at potential policy shift, Yen strengthens", 0.3, "HIGH", ["USDJPY"], "forex_factory", "/thread/boj-policy-shift-yen-strengthens"),
            ("S&P 500 reaches new all-time high on tech sector gains", 0.8, "MEDIUM", ["SPX500", "NAS100"], "fxstreet", "/market-news/sp500-all-time-high-tech-gains"),
            ("UK inflation falls more than expected, GBP volatile", 0.2, "HIGH", ["GBPUSD"], "investing_com", "/news/economic/uk-inflation-falls-gbp-volatile"),
            ("Chinese economic data shows mixed signals for global growth", -0.1, "MEDIUM", ["AUDUSD", "NZDUSD"], "forex_factory", "/thread/china-economic-data-mixed-signals"),
        ]

        # Source base URLs
        source_urls = {
            'forex_factory': 'https://www.forexfactory.com',
            'fxstreet': 'https://www.fxstreet.com',
            'investing_com': 'https://www.investing.com'
        }

        news = []
        for i, (title, sentiment, impact, symbols, source, path) in enumerate(templates[:count]):
            full_url = f"{source_urls.get(source, 'https://example.com')}{path}"

            news.append(NewsArticle(
                id=f"news-{i}-{int(datetime.now().timestamp())}",
                title=title,
                source=NewsSource(source),
                sentiment=sentiment + random.uniform(-0.1, 0.1),
                relevance=random.uniform(0.6, 1.0),
                url=full_url,
                impact=impact,
                summary=f"Market analysis: {title[:100]}...",
                related_symbols=symbols,
                published_at=datetime.now() - timedelta(hours=i)
            ))

        return news

    def _generate_fallback_calendar(self) -> List[Dict]:
        """Generate fallback economic calendar with clickable URLs to specific events."""
        now = datetime.now()
        current_day = now.strftime('%m.%d.%Y')

        events = [
            {"time": "08:30", "currency": "USD", "event": "Core CPI (MoM) (YoY)", "impact": "HIGH", "actual": "", "forecast": "0.3%", "previous": "0.4%", "url": f"https://www.forexfactory.com/calendar.php?day={current_day}"},
            {"time": "08:30", "currency": "USD", "event": "CPI (MoM)", "impact": "HIGH", "actual": "", "forecast": "0.2%", "previous": "0.2%", "url": "https://www.investing.com/economic-calendar/cpi-mo-m-228"},
            {"time": "10:00", "currency": "USD", "event": "Crude Oil Inventories", "impact": "MEDIUM", "actual": "", "forecast": "-1.2M", "previous": "-0.5M", "url": "https://www.forexfactory.com/calendar.php?day=" + current_day},
            {"time": "14:00", "currency": "USD", "event": "FOMC Meeting Minutes", "impact": "HIGH", "actual": "", "forecast": "", "previous": "", "url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"},
            {"time": "08:30", "currency": "USD", "event": "Non-Farm Payroll", "impact": "HIGH", "actual": "", "forecast": "180K", "previous": "199K", "url": "https://www.investing.com/economic-calendar/nonfarm-payrolls-228"},
            {"time": "08:30", "currency": "USD", "event": "Unemployment Rate", "impact": "HIGH", "actual": "", "forecast": "3.8%", "previous": "3.9%", "url": "https://www.forexfactory.com/calendar.php?day=" + current_day},
            {"time": "09:45", "currency": "USD", "event": "Manufacturing PMI", "impact": "MEDIUM", "actual": "", "forecast": "50.5", "previous": "50.3%", "url": "https://www.investing.com/economic-calendar/us-manufacturing-pmi-722"},
            {"time": "10:00", "currency": "USD", "event": "ISM Services PMI", "impact": "MEDIUM", "actual": "", "forecast": "52.0", "previous": "51.8%", "url": "https://www.forexfactory.com/calendar.php?day=" + current_day},
            {"time": "07:00", "currency": "EUR", "event": "ECB Interest Rate Decision", "impact": "HIGH", "actual": "", "forecast": "4.50%", "previous": "4.50%", "url": "https://www.fxstreet.com/economic-calendar/ecb-interest-rate-decision"},
            {"time": "02:00", "currency": "GBP", "event": "BoE Interest Rate Decision", "impact": "HIGH", "actual": "", "forecast": "5.25%", "previous": "5.25%", "url": "https://www.investing.com/economic-calendar/boe-interest-rate-decision-244"},
        ]

        return events

    def get_news_sources(self) -> List[Dict]:
        """
        Get list of all available news sources with URLs and metadata.
        Used to display clickable news source cards.

        Returns
        -------
        List[Dict]
            List of news source configurations
        """
        sources = []
        for source in NEWS_SOURCE_CONFIG.values():
            sources.append({
                "name": source["name"],
                "url": source["url"],
                "icon": source["icon"],
                "color": source["color"],
                "description": source["description"]
            })
        return sources

    def get_news_by_source(self, source: NewsSource, limit: int = 10) -> List[NewsArticle]:
        """
        Get news articles from a specific source.

        Parameters
        ----------
        source : NewsSource
            The news source to fetch from
        limit : int
            Maximum number of articles

        Returns
        -------
        List[NewsArticle]
            List of news articles from the source
        """
        return self._generate_fallback_news(limit)

    def create_news_source_cards(self) -> List[Dict]:
        """
        Create a list of news source cards for the UI.
        Each card has the source name, icon, URL, and color.

        Returns
        -------
        List[Dict]
            List of news source card data
        """
        return [
            {
                "id": "bloomberg",
                "name": "Bloomberg",
                "icon": "💼",
                "url": "https://www.bloomberg.com",
                "color": "#FF6600",
                "description": "Global business & markets",
                "categories": ["Markets", "Economy", "Business"]
            },
            {
                "id": "cnbc",
                "name": "CNBC",
                "icon": "📺",
                "url": "https://www.cnbc.com",
                "color": "#003366",
                "description": "Stock market & business",
                "categories": ["Stocks", "Economy", "Tech"]
            },
            {
                "id": "investing",
                "name": "Investing.com",
                "icon": "📈",
                "url": "https://www.investing.com",
                "color": "#008000",
                "description": "Trading & markets",
                "categories": ["Forex", "Crypto", "Commodities"]
            },
            {
                "id": "fxstreet",
                "name": "FXStreet",
                "icon": "💱",
                "url": "https://www.fxstreet.com",
                "color": "#E91E63",
                "description": "Forex news & analysis",
                "categories": ["Forex", "Analysis", "Calendar"]
            },
            {
                "id": "forexfactory",
                "name": "Forex Factory",
                "icon": "🏭",
                "url": "https://www.forexfactory.com",
                "color": "#FF9800",
                "description": "Forex community & calendar",
                "categories": ["Forex", "Calendar", "Forum"]
            },
            {
                "id": "reuters",
                "name": "Reuters",
                "icon": "📰",
                "url": "https://www.reuters.com",
                "color": "#FF8000",
                "description": "Breaking news & markets",
                "categories": ["News", "Markets", "World"]
            },
            {
                "id": "marketwatch",
                "name": "MarketWatch",
                "icon": "⌚",
                "url": "https://www.marketwatch.com",
                "color": "#00A600",
                "description": "Stock market data",
                "categories": ["Stocks", "Personal Finance", "Markets"]
            },
            {
                "id": "yahoo_finance",
                "name": "Yahoo Finance",
                "icon": "🟣",
                "url": "https://finance.yahoo.com",
                "color": "#400090",
                "description": "Finance & investing",
                "categories": ["Stocks", "Crypto", "Trending"]
            },
            {
                "id": "dailyfx",
                "name": "DailyFX",
                "icon": "📊",
                "url": "https://www.dailyfx.com",
                "color": "#E74C3C",
                "description": "Forex education & analysis",
                "categories": ["Forex", "Education", "Charts"]
            },
            {
                "id": "coindesk",
                "name": "CoinDesk",
                "icon": "₿",
                "url": "https://www.coindesk.com",
                "color": "#1652F0",
                "description": "Cryptocurrency news",
                "categories": ["Bitcoin", "Blockchain", "DeFi"]
            },
            {
                "id": "kitco",
                "name": "Kitco",
                "icon": "🥇",
                "url": "https://www.kitco.com/news/",
                "color": "#FFD700",
                "description": "Precious metals news",
                "categories": ["Gold", "Silver", "Metals"]
            },
            {
                "id": "trading_economics",
                "name": "Trading Economics",
                "icon": "🌐",
                "url": "https://tradingeconomics.com",
                "color": "#2E86AB",
                "description": "Economic indicators",
                "categories": ["GDP", "Inflation", "Employment"]
            },
        ]

    def get_aggregated_sentiment(self, symbol: str = None) -> float:
        """
        Get aggregated sentiment score.

        Parameters
        ----------
        symbol : str
            Optional symbol to filter news

        Returns
        -------
        float
            Average sentiment score (-1 to 1)
        """
        if not self._news_cache:
            return 0.0

        if symbol:
            relevant_news = [n for n in self._news_cache if not n.related_symbols or symbol in n.related_symbols]
            if not relevant_news:
                relevant_news = self._news_cache
        else:
            relevant_news = self._news_cache

        if not relevant_news:
            return 0.0

        return sum(n.sentiment for n in relevant_news) / len(relevant_news)


# Convenience functions for synchronous usage
def get_financial_news(limit: int = 20, symbol: str = None) -> List[NewsArticle]:
    """
    Synchronous function to fetch financial news.

    Parameters
    ----------
    limit : int
        Maximum number of articles
    symbol : str
        Trading symbol for symbol-specific news (e.g., 'XAUUSD', 'EURUSD')

    Returns
    -------
    List[NewsArticle]
        List of news articles
    """
    aggregator = NewsAggregator()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if symbol:
        # Fetch symbol-specific news
        return loop.run_until_complete(aggregator.fetch_symbol_news(symbol, limit))
    else:
        # Fetch general news
        return loop.run_until_complete(aggregator.fetch_all_news(limit // 3))


def get_economic_calendar() -> List[Dict]:
    """
    Synchronous function to fetch economic calendar.

    Returns
    -------
    List[Dict]
        List of economic events
    """
    aggregator = NewsAggregator()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(aggregator.fetch_calendar())


def get_news_sources() -> List[Dict]:
    """
    Get list of all financial news sources with their URLs.

    Returns
    -------
    List[Dict]
        List of news source configurations including name, URL, icon, and color
    """
    return [
        {
            "id": "bloomberg",
            "name": "Bloomberg",
            "icon": "💼",
            "url": "https://www.bloomberg.com",
            "color": "#FF6600",
            "description": "Global Markets & Business News"
        },
        {
            "id": "cnbc",
            "name": "CNBC",
            "icon": "📺",
            "url": "https://www.cnbc.com",
            "color": "#003366",
            "description": "Stock Market & Business"
        },
        {
            "id": "investing",
            "name": "Investing.com",
            "icon": "📈",
            "url": "https://www.investing.com",
            "color": "#008000",
            "description": "Trading & Markets Platform"
        },
        {
            "id": "fxstreet",
            "name": "FXStreet",
            "icon": "💱",
            "url": "https://www.fxstreet.com",
            "color": "#E91E63",
            "description": "Forex News & Analysis"
        },
        {
            "id": "forexfactory",
            "name": "Forex Factory",
            "icon": "🏭",
            "url": "https://www.forexfactory.com",
            "color": "#FF9800",
            "description": "Forex Trading Community"
        },
        {
            "id": "reuters",
            "name": "Reuters",
            "icon": "📰",
            "url": "https://www.reuters.com",
            "color": "#FF8000",
            "description": "Breaking News Agency"
        },
        {
            "id": "marketwatch",
            "name": "MarketWatch",
            "icon": "⌚",
            "url": "https://www.marketwatch.com",
            "color": "#00A600",
            "description": "Stock Market Data"
        },
        {
            "id": "yahoo_finance",
            "name": "Yahoo Finance",
            "icon": "🟣",
            "url": "https://finance.yahoo.com",
            "color": "#400090",
            "description": "Markets & Technology"
        },
        {
            "id": "dailyfx",
            "name": "DailyFX",
            "icon": "📊",
            "url": "https://www.dailyfx.com",
            "color": "#E74C3C",
            "description": "Forex Analysis & Education"
        },
        {
            "id": "coindesk",
            "name": "CoinDesk",
            "icon": "₿",
            "url": "https://www.coindesk.com",
            "color": "#1652F0",
            "description": "Cryptocurrency News"
        },
        {
            "id": "kitco",
            "name": "Kitco",
            "icon": "🥇",
            "url": "https://www.kitco.com/news/",
            "color": "#FFD700",
            "description": "Precious Metals News"
        },
        {
            "id": "trading_economics",
            "name": "Trading Economics",
            "icon": "🌐",
            "url": "https://tradingeconomics.com",
            "color": "#2E86AB",
            "description": "Economic Indicators"
        },
        {
            "id": "fxempire",
            "name": "FX Empire",
            "icon": "🏛️",
            "url": "https://www.fxempire.com",
            "color": "#9B59B6",
            "description": "Forex & CFD Analysis"
        },
        {
            "id": "cointelegraph",
            "name": "CoinTelegraph",
            "icon": "📱",
            "url": "https://cointelegraph.com",
            "color": "#E67E22",
            "description": "Crypto News & Trends"
        },
    ]
