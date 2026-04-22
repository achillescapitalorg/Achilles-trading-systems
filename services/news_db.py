"""
News Database Module - SQLite Storage with 90-day Retention
==========================================================
Features:
- Store all news articles in SQLite database
- Query by date range, instrument, sentiment
- Auto-cleanup of news older than 90 days
- Full-text search on headlines
"""

import sqlite3
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import json

DATABASE_PATH = Path(__file__).parent.parent / "data" / "news.db"


class NewsDatabase:
    """
    SQLite database for persistent news storage.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    headline TEXT NOT NULL,
                    url TEXT,
                    source TEXT,
                    source_icon TEXT DEFAULT '📰',
                    sentiment_score REAL DEFAULT 0,
                    sentiment_label TEXT DEFAULT 'neutral',
                    confidence REAL DEFAULT 0,
                    impact TEXT DEFAULT 'MEDIUM',
                    instrument TEXT,
                    topic TEXT,
                    published_at TEXT,
                    cached_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_instrument ON news(instrument)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_published ON news(published_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment ON news(sentiment_label)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cached ON news(cached_at)
            """)
            
            conn.commit()
            conn.close()
    
    def save_news(self, news_items: List[Dict], instrument: str = None):
        """
        Save news items to database.
        
        Args:
            news_items: List of news item dictionaries
            instrument: Optional instrument symbol
        """
        if not news_items:
            return 0
        
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            saved = 0
            for item in news_items:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO news 
                        (headline, url, source, source_icon, sentiment_score, sentiment_label, 
                         confidence, impact, instrument, topic, published_at, cached_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        item.get('headline', ''),
                        item.get('url', ''),
                        item.get('source', ''),
                        item.get('source_icon', '📰'),
                        item.get('sentiment', 0),
                        item.get('sentiment_label', 'neutral'),
                        item.get('confidence', 0),
                        item.get('impact', 'MEDIUM'),
                        instrument or item.get('instrument', ''),
                        item.get('topic', ''),
                        item.get('pubdate', ''),
                        datetime.now().isoformat()
                    ))
                    if cursor.rowcount > 0:
                        saved += 1
                except Exception as e:
                    continue
            
            conn.commit()
            conn.close()
            return saved
    
    def get_news_by_date_range(self, start_date: datetime, end_date: datetime, 
                          instrument: str = None, limit: int = 100) -> List[Dict]:
        """
        Get news within a date range.
        
        Args:
            start_date: Start datetime
            end_date: End datetime
            instrument: Optional instrument filter
            limit: Maximum results
        
        Returns:
            List of news items
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            if instrument:
                cursor.execute("""
                    SELECT id, headline, url, source, source_icon, sentiment_score, 
                           sentiment_label, confidence, impact, instrument, topic, 
                           published_at, cached_at
                    FROM news
                    WHERE published_at >= ? AND published_at <= ?
                    AND instrument = ?
                    ORDER BY published_at DESC
                    LIMIT ?
                """, (start_date.isoformat(), end_date.isoformat(), instrument, limit))
            else:
                cursor.execute("""
                    SELECT id, headline, url, source, source_icon, sentiment_score, 
                           sentiment_label, confidence, impact, instrument, topic, 
                           published_at, cached_at
                    FROM news
                    WHERE published_at >= ? AND published_at <= ?
                    ORDER BY published_at DESC
                    LIMIT ?
                """, (start_date.isoformat(), end_date.isoformat(), limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_dict(row) for row in rows]
    
    def get_recent_news(self, minutes: int = 60, instrument: str = None, 
                        limit: int = 50) -> List[Dict]:
        """
        Get recent news within specified minutes.
        
        Args:
            minutes: Number of minutes to look back
            instrument: Optional instrument filter
            limit: Maximum results
        
        Returns:
            List of news items
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return self.get_news_by_date_range(cutoff, datetime.now(), instrument, limit)
    
    def get_news_by_instrument(self, instrument: str, days: int = 7, 
                                limit: int = 100) -> List[Dict]:
        """Get news for an instrument within specified days."""
        start = datetime.now() - timedelta(days=days)
        return self.get_news_by_date_range(start, datetime.now(), instrument, limit)
    
    def search_news(self, query: str, limit: int = 50) -> List[Dict]:
        """
        Full-text search on headlines.
        
        Args:
            query: Search query string
            limit: Maximum results
        
        Returns:
            List of matching news items
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, headline, url, source, source_icon, sentiment_score, 
                       sentiment_label, confidence, impact, instrument, topic, 
                       published_at, cached_at
                FROM news
                WHERE headline LIKE ?
                ORDER BY published_at DESC
                LIMIT ?
            """, (f"%{query}%", limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_dict(row) for row in rows]
    
    def get_sentiment_trend(self, instrument: str, days: int = 7) -> List[Dict]:
        """
        Get daily sentiment trend for an instrument.
        
        Args:
            instrument: Instrument symbol
            days: Number of days to look back
        
        Returns:
            List of daily sentiment summaries
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DATE(published_at) as date,
                       AVG(sentiment_score) as avg_sentiment,
                       COUNT(*) as count,
                       SUM(CASE WHEN sentiment_label = 'bullish' THEN 1 ELSE 0 END) as bullish,
                       SUM(CASE WHEN sentiment_label = 'bearish' THEN 1 ELSE 0 END) as bearish
                FROM news
                WHERE instrument = ?
                AND published_at >= ?
                GROUP BY DATE(published_at)
                ORDER BY date DESC
            """, (instrument, (datetime.now() - timedelta(days=days)).isoformat()))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'date': row[0],
                    'avg_sentiment': row[1],
                    'count': row[2],
                    'bullish': row[3],
                    'bearish': row[4]
                }
                for row in rows
            ]
    
    def get_all_news(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get all news with pagination."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, headline, url, source, source_icon, sentiment_score, 
                       sentiment_label, confidence, impact, instrument, topic, 
                       published_at, cached_at
                FROM news
                ORDER BY published_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_dict(row) for row in rows]
    
    def cleanup_old_news(self, days: int = 90):
        """
        Remove news older than specified days.
        
        Args:
            days: Number of days to retain
        """
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cutoff = datetime.now() - timedelta(days=days)
            cursor.execute("""
                DELETE FROM news WHERE cached_at < ?
            """, (cutoff.isoformat(),))
            
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            return deleted
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM news")
            total = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM news WHERE cached_at >= ?
            """, ((datetime.now() - timedelta(days=1)).isoformat(),))
            last_24h = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM news WHERE cached_at >= ?
            """, ((datetime.now() - timedelta(days=7)).isoformat(),))
            last_7d = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT instrument) FROM news")
            instruments = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_news': total,
                'last_24h': last_24h,
                'last_7d': last_7d,
                'instruments': instruments,
                'retention_days': 90
            }
    
    def _row_to_dict(self, row: tuple) -> Dict:
        """Convert database row to dictionary."""
        return {
            'id': row[0],
            'headline': row[1],
            'url': row[2],
            'source': row[3],
            'source_icon': row[4],
            'sentiment': row[5],
            'sentiment_label': row[6],
            'confidence': row[7],
            'impact': row[8],
            'instrument': row[9],
            'topic': row[10],
            'published_at': row[11],
            'cached_at': row[12]
        }


# Singleton instance
_news_db = None
_db_lock = threading.Lock()


def get_news_db() -> NewsDatabase:
    """Get singleton instance of NewsDatabase."""
    global _news_db
    if _news_db is None:
        with _db_lock:
            if _news_db is None:
                _news_db = NewsDatabase()
    return _news_db


def save_news_to_db(news_items: List[Dict], instrument: str = None) -> int:
    """Convenience function to save news to database."""
    return get_news_db().save_news(news_items, instrument)


def get_recent_news(minutes: int = 60, instrument: str = None, limit: int = 50) -> List[Dict]:
    """Convenience function to get recent news."""
    return get_news_db().get_recent_news(minutes, instrument, limit)


def get_news_by_instrument(instrument: str, days: int = 7, limit: int = 100) -> List[Dict]:
    """Convenience function to get news by instrument."""
    return get_news_db().get_news_by_instrument(instrument, days, limit)


def search_news(query: str, limit: int = 50) -> List[Dict]:
    """Convenience function to search news."""
    return get_news_db().search_news(query, limit)


def get_sentiment_trend(instrument: str, days: int = 7) -> List[Dict]:
    """Convenience function to get sentiment trend."""
    return get_news_db().get_sentiment_trend(instrument, days)


def cleanup_old_news(days: int = 90) -> int:
    """Convenience function to cleanup old news."""
    return get_news_db().cleanup_old_news(days)