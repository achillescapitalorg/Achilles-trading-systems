"""
News Cache Module - Persistent cache with background refresh
Provides thread-safe caching of news data with JSON file persistence.
"""
import json
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class CachedNewsItem:
    """A single cached news item."""
    headline: str
    sentiment: float
    impact: str
    time_ago: str
    source: str
    source_icon: str
    url: str
    impact_timing: str
    cached_at: str


@dataclass
class SymbolCache:
    """Cache entry for a single symbol."""
    items: List[Dict]
    timestamp: str
    item_count: int


@dataclass
class NewsCacheData:
    """Root cache data structure."""
    version: int = 1
    last_full_refresh: str = ""
    news: Dict[str, SymbolCache] = None
    
    def __post_init__(self):
        if self.news is None:
            self.news = {}


class NewsCache:
    """
    Thread-safe persistent news cache with background refresh.
    
    Features:
    - JSON file persistence (survives app restarts)
    - TTL-based cache invalidation (5 minutes default)
    - Background refresh capability
    - Thread-safe operations
    """
    
    CACHE_VERSION = 1
    DEFAULT_TTL_MINUTES = 2
    MAX_ITEMS_PER_SYMBOL = 15  # Reduced from 25 to save memory
    
    def __init__(self, cache_dir: str = None):
        """Initialize the news cache."""
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data'
            )
        
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / 'news_cache.json'
        self.lock = threading.RLock()
        self._cache: Optional[NewsCacheData] = None
        self._refresh_in_progress: Dict[str, bool] = {}
        
        self._ensure_cache_dir()
        self._load_cache()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_cache(self) -> NewsCacheData:
        """Load cache from disk. Thread-safe."""
        with self.lock:
            if self._cache is not None:
                return self._cache
            
            if not self.cache_file.exists():
                self._cache = NewsCacheData()
                return self._cache
            
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                news_dict = {}
                for symbol, cache_data in data.get('news', {}).items():
                    news_dict[symbol] = SymbolCache(
                        items=cache_data.get('items', []),
                        timestamp=cache_data.get('timestamp', ''),
                        item_count=len(cache_data.get('items', []))
                    )
                
                self._cache = NewsCacheData(
                    version=data.get('version', self.CACHE_VERSION),
                    last_full_refresh=data.get('last_full_refresh', ''),
                    news=news_dict
                )
                print(f"[NewsCache] Loaded cache with {len(news_dict)} symbols")
                
            except json.JSONDecodeError as e:
                print(f"[NewsCache] Corrupted cache file: {e}")
                self._cache = NewsCacheData()
            except Exception as e:
                print(f"[NewsCache] Error loading cache: {e}")
                self._cache = NewsCacheData()
            
            return self._cache
    
    def _save_cache(self):
        """Save cache to disk. Thread-safe."""
        with self.lock:
            if self._cache is None:
                return
            
            try:
                news_dict = {}
                for symbol, cache_data in self._cache.news.items():
                    news_dict[symbol] = {
                        'items': cache_data.items,
                        'timestamp': cache_data.timestamp
                    }
                
                data = {
                    'version': self.CACHE_VERSION,
                    'last_full_refresh': self._cache.last_full_refresh,
                    'news': news_dict
                }
                
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                print(f"[NewsCache] Error saving cache: {e}")
    
    def get(self, symbol: str) -> Optional[List[Dict]]:
        """
        Get cached news for a symbol.
        
        Returns:
            List of news items if cache exists, None otherwise.
        """
        with self.lock:
            cache = self._load_cache()
            if symbol not in cache.news:
                return None
            
            return cache.news[symbol].items.copy()
    
    def get_all(self) -> Dict[str, List[Dict]]:
        """Get all cached news for all symbols."""
        with self.lock:
            cache = self._load_cache()
            return {
                symbol: cache_data.items.copy()
                for symbol, cache_data in cache.news.items()
            }
    
    def set(self, symbol: str, items: List[Dict]):
        """
        Set cached news for a symbol.
        
        Args:
            symbol: Instrument symbol (e.g., 'XAUUSD')
            items: List of news item dictionaries
        """
        with self.lock:
            cache = self._load_cache()
            cache.news[symbol] = SymbolCache(
                items=items[:self.MAX_ITEMS_PER_SYMBOL],
                timestamp=datetime.now().isoformat(),
                item_count=len(items)
            )
            self._save_cache()
    
    def set_all(self, news_by_symbol: Dict[str, List[Dict]]):
        """Set cached news for all symbols at once."""
        with self.lock:
            cache = self._load_cache()
            now = datetime.now().isoformat()
            
            for symbol, items in news_by_symbol.items():
                cache.news[symbol] = SymbolCache(
                    items=items[:self.MAX_ITEMS_PER_SYMBOL],
                    timestamp=now,
                    item_count=len(items)
                )
            
            cache.last_full_refresh = now
            self._save_cache()
            print(f"[NewsCache] Saved news for {len(news_by_symbol)} symbols")
    
    def is_stale(self, symbol: str, ttl_minutes: int = None) -> bool:
        """
        Check if cache for a symbol is stale.
        
        Args:
            symbol: Instrument symbol
            ttl_minutes: TTL in minutes (default: 5)
            
        Returns:
            True if cache is stale or missing, False if fresh.
        """
        if ttl_minutes is None:
            ttl_minutes = self.DEFAULT_TTL_MINUTES
        
        with self.lock:
            cache = self._load_cache()
            if symbol not in cache.news:
                return True
            
            timestamp_str = cache.news[symbol].timestamp
            if not timestamp_str:
                return True
            
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                age = datetime.now() - timestamp
                return age.total_seconds() > (ttl_minutes * 60)
            except (ValueError, TypeError):
                return True
    
    def is_full_refresh_stale(self, ttl_minutes: int = None) -> bool:
        """Check if the full cache refresh is stale."""
        if ttl_minutes is None:
            ttl_minutes = self.DEFAULT_TTL_MINUTES
        
        with self.lock:
            cache = self._load_cache()
            if not cache.last_full_refresh:
                return True
            
            try:
                timestamp = datetime.fromisoformat(cache.last_full_refresh)
                age = datetime.now() - timestamp
                return age.total_seconds() > (ttl_minutes * 60)
            except (ValueError, TypeError):
                return True
    
    def get_last_updated(self, symbol: str = None) -> Optional[str]:
        """Get last update timestamp for a symbol or full refresh."""
        with self.lock:
            cache = self._load_cache()
            
            if symbol:
                if symbol in cache.news:
                    return cache.news[symbol].timestamp
                return None
            else:
                return cache.last_full_refresh
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            cache = self._load_cache()
            
            total_items = sum(c.item_count for c in cache.news.values())
            symbols_cached = len(cache.news)
            
            stale_count = 0
            for symbol in cache.news:
                if self.is_stale(symbol):
                    stale_count += 1
            
            return {
                'symbols_cached': symbols_cached,
                'total_items': total_items,
                'stale_count': stale_count,
                'last_full_refresh': cache.last_full_refresh,
                'cache_file': str(self.cache_file),
                'cache_size_bytes': self.cache_file.stat().st_size if self.cache_file.exists() else 0
            }
    
    def clear(self):
        """Clear all cached news."""
        with self.lock:
            self._cache = NewsCacheData()
            self._save_cache()
            print("[NewsCache] Cache cleared")
    
    def refresh_symbol(self, symbol: str, fetch_func=None) -> Tuple[bool, Optional[List[Dict]]]:
        """
        Refresh cache for a single symbol.
        
        Args:
            symbol: Instrument symbol to refresh
            fetch_func: Optional function to fetch news (should return List[Dict])
            
        Returns:
            Tuple of (success, items)
        """
        if fetch_func is None:
            return False, None
        
        with self.lock:
            if self._refresh_in_progress.get(symbol, False):
                return False, self.get(symbol)
            self._refresh_in_progress[symbol] = True
        
        try:
            items = fetch_func(symbol)
            if items:
                self.set(symbol, items)
                return True, items
            return False, None
        except Exception as e:
            print(f"[NewsCache] Error refreshing {symbol}: {e}")
            return False, None
        finally:
            with self.lock:
                self._refresh_in_progress[symbol] = False


def run_background_news_refresh(cache: NewsCache, symbols: List[str], fetch_func):
    """
    Refresh news for all symbols in parallel (capped at 3 workers).

    Each symbol's fetch is itself heavy (multi-source aggregation), so a small
    pool keeps total memory bounded while still finishing in seconds rather
    than minutes. Forces a GC pass at the end to release transient allocations.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print(f"[NewsCache] Starting parallel background refresh for {len(symbols)} symbols")
    start_time = time.time()

    news_by_symbol = {}
    # Cap at 3 — each worker spawns its own internal source pool, so the
    # multiplicative concurrency is what matters.
    max_workers = min(len(symbols), 3) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sym = {executor.submit(fetch_func, sym): sym for sym in symbols}
        for fut in as_completed(future_to_sym):
            sym = future_to_sym[fut]
            try:
                items = fut.result(timeout=25)
                if items:
                    news_by_symbol[sym] = items
                    print(f"[NewsCache] Fetched {len(items)} items for {sym}")
            except Exception as e:
                print(f"[NewsCache] Error fetching {sym}: {e}")

    if news_by_symbol:
        cache.set_all(news_by_symbol)

    import gc
    gc.collect()

    elapsed = time.time() - start_time
    print(f"[NewsCache] Background refresh completed in {elapsed:.2f}s")


def start_background_refresh_thread(cache: NewsCache, symbols: List[str], fetch_func):
    """Start background refresh in a daemon thread."""
    thread = threading.Thread(
        target=run_background_news_refresh,
        args=(cache, symbols, fetch_func),
        daemon=True,
        name="NewsCacheRefresh"
    )
    thread.start()
    return thread
