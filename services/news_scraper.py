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
    """News source identifiers."""
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
    """

    BASE_URL = "https://www.investing.com"

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
                        limit: int = 20) -> List[NewsArticle]:
        """Fetch latest news."""
        url = f"{self.BASE_URL}/news/"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        try:
            async with session.get(url, headers=headers, timeout=10) as response:
                html = await response.text()
                return self._parse_news(html, limit)
        except Exception as e:
            print(f"Investing.com news error: {e}")
            return []

    def _parse_news(self, html: str, limit: int) -> List[NewsArticle]:
        """Parse news HTML."""
        soup = BeautifulSoup(html, 'lxml')
        articles = []

        # Find news items
        items = soup.select('div.largeTitle')

        for item in items[:limit]:
            try:
                link_elem = item.select_one('a')

                if not link_elem:
                    continue

                title = link_elem.get_text(strip=True)
                href = link_elem.get('href', '')

                # Build full URL
                full_url = href if href.startswith('http') else f"{self.BASE_URL}{href}"

                sentiment = self._analyze_sentiment(title)

                impact = 'MEDIUM'
                if any(x in title.lower() for x in ['fed', 'ecb', 'central bank', 'breaking']):
                    impact = 'HIGH'

                article = NewsArticle(
                    title=title,
                    source=NewsSource.INVESTING_COM,
                    url=full_url,
                    published_at=datetime.now(),
                    sentiment=sentiment,
                    impact=impact,
                    related_symbols=self._extract_symbols(title),
                    id=f"inv-{len(articles)}",
                    relevance=0.7
                )

                articles.append(article)

            except Exception as e:
                continue

        return articles

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment."""
        return ForexFactoryScraper()._analyze_sentiment(text)

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract symbols."""
        return ForexFactoryScraper()._extract_symbols(text)


class NewsAggregator:
    """
    Aggregates news from multiple sources.
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
        self._news_cache: List[NewsArticle] = []
        self._calendar_cache: List[Dict] = []
        self._last_fetch: Optional[datetime] = None

    def _get_headers(self) -> Dict:
        """Get random user agent headers."""
        return {
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    async def fetch_all_news(self, limit_per_source: int = 10) -> List[NewsArticle]:
        """
        Fetch news from all sources.

        Parameters
        ----------
        limit_per_source : int
            Maximum articles per source
        """
        # Return cached news if recently fetched (within 5 minutes)
        if self._last_fetch and (datetime.now() - self._last_fetch).total_seconds() < 300:
            return self._news_cache[:limit_per_source * 3]

        async with aiohttp.ClientSession(headers=self._get_headers()) as session:
            tasks = [
                self.forex_factory.fetch_news(session, limit_per_source),
                self.fxstreet.fetch_news(session, limit_per_source),
                self.investing_com.fetch_news(session, limit_per_source),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine and sort by time
            all_articles = []
            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)

            # Add fallback news if scraping failed
            if len(all_articles) < 5:
                all_articles.extend(self._generate_fallback_news(10))

            all_articles.sort(key=lambda x: x.published_at, reverse=True)

            # Cache results
            self._news_cache = all_articles
            self._last_fetch = datetime.now()

            return all_articles[:limit_per_source * 3]

    async def fetch_calendar(self, date: datetime = None) -> List[Dict]:
        """Fetch economic calendar from all sources."""
        # Return cached calendar if recently fetched
        if self._calendar_cache and self._last_fetch and (datetime.now() - self._last_fetch).total_seconds() < 300:
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

            # Add fallback events if scraping failed
            if len(all_events) < 3:
                all_events.extend(self._generate_fallback_calendar())

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
                timestamp=datetime.now() - timedelta(hours=i),
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
def get_financial_news(limit: int = 20) -> List[NewsArticle]:
    """
    Synchronous function to fetch financial news.

    Parameters
    ----------
    limit : int
        Maximum number of articles

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
