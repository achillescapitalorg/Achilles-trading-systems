"""
Trading Memory - Auto-captures trading decisions using GraphMem
================================================================
Captures:
- Trading signals generated
- Backtest results
- Pattern discoveries
- Regime changes

Disabled by default. Enable via: ENABLE_TRADING_MEMORY=true
Uses Ollama (llama3.2:1b) for entity extraction.
Falls back gracefully if Ollama unavailable.
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from threading import Lock
import json

MEMORY_DB_PATH = Path(__file__).parent.parent / "trading_memory.db"

_memory_lock = Lock()
_memory_instance = None
_graphmem_available = False
_graphmem_memory = None


def _init_db():
    """Initialize SQLite for simple memory storage (fallback)."""
    conn = sqlite3.connect(str(MEMORY_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trading_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            data TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            asset TEXT,
            importance REAL DEFAULT 0.5
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp ON trading_memory(timestamp)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_asset ON trading_memory(asset)
    """)
    conn.commit()
    return conn


def _try_init_graphmem():
    """Try to initialize GraphMem with Ollama."""
    global _graphmem_available, _graphmem_memory

    try:
        from graphmem import GraphMem, MemoryConfig

        config = MemoryConfig(
            llm_provider="openai_compatible",
            llm_model="llama3.2:1b",
            llm_api_base="http://localhost:11434/v1",
            llm_api_key="ollama",
            embedding_provider="openai_compatible",
            embedding_model="nomic-embed-text",
            embedding_api_base="http://localhost:11434/v1",
            embedding_api_key="ollama",
            turso_db_path=str(MEMORY_DB_PATH)
        )

        _graphmem_memory = GraphMem(
            config,
            user_id="trading_terminal",
            memory_id="signals"
        )
        _graphmem_available = True
        print("✅ Trading memory: GraphMem initialized with Ollama")
        return True
    except Exception as e:
        print(f"⚠️ Trading memory: GraphMem unavailable ({e}). Using SQLite fallback.")
        _graphmem_available = False
        return False


class TradingMemory:
    """
    Singleton trading memory with GraphMem + SQLite fallback.
    Enable via: ENABLE_TRADING_MEMORY=true
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._enabled = os.getenv("ENABLE_TRADING_MEMORY", "false").lower() == "true"
        self._conn = None

        if self._enabled:
            _try_init_graphmem()
            self._conn = _init_db()
            print(f"✅ Trading memory enabled (GraphMem: {_graphmem_available})")
        else:
            print("ℹ️ Trading memory disabled. Set ENABLE_TRADING_MEMORY=true to enable.")

        self._initialized = True

    def capture_signal(self, data: Dict[str, Any]):
        """Log trading signal."""
        if not self._enabled:
            return

        text = f"Signal: {data['asset']} {data['direction']}, " \
               f"Confidence: {data.get('confidence', 'N/A')}, " \
               f"VPIN: {data.get('vpin', 'N/A')}, " \
               f"Regime: {data.get('regime', 'N/A')}, " \
               f"Entry: {data.get('entry', 'N/A')}, " \
               f"Time: {data.get('timestamp', datetime.now().isoformat())}"

        self._save_memory("signal", text, data.get("asset"))

    def capture_backtest(self, data: Dict[str, Any]):
        """Log backtest result."""
        if not self._enabled:
            return

        text = f"Backtest: {data.get('strategy', 'Unknown')} over {data.get('period', 'N/A')}, " \
               f"Sharpe: {data.get('sharpe', 'N/A')}, " \
               f"Max DD: {data.get('max_drawdown', 'N/A')}%, " \
               f"Win Rate: {data.get('win_rate', 'N/A')}%"

        self._save_memory("backtest", text, data.get("asset"))

    def capture_pattern(self, data: Dict[str, Any]):
        """Log pattern discovery."""
        if not self._enabled:
            return

        text = f"Pattern: {data.get('name', 'Unknown')}, " \
               f"Description: {data.get('description', 'N/A')}, " \
               f"Success Rate: {data.get('success_rate', 'N/A')}"

        self._save_memory("pattern", text, data.get("asset"))

    def capture_regime_change(self, data: Dict[str, Any]):
        """Log regime change."""
        if not self._enabled:
            return

        text = f"Regime Change: {data['asset']}, " \
               f"From: {data.get('from_regime', 'N/A')}, " \
               f"To: {data.get('to_regime', 'N/A')}, " \
               f"Time: {data.get('timestamp', datetime.now().isoformat())}"

        self._save_memory("regime", text, data.get("asset"))

    def _save_memory(self, mem_type: str, text: str, asset: Optional[str] = None):
        """Save to GraphMem or SQLite fallback."""
        with _memory_lock:
            if _graphmem_available and _graphmem_memory:
                try:
                    _graphmem_memory.ingest(text)
                except Exception as e:
                    print(f"⚠️ GraphMem ingest failed: {e}")
                    self._save_to_sqlite(mem_type, text, asset)
            else:
                self._save_to_sqlite(mem_type, text, asset)

    def _save_to_sqlite(self, mem_type: str, text: str, asset: Optional[str]):
        """Save to SQLite fallback."""
        try:
            self._conn.execute(
                "INSERT INTO trading_memory (type, data, timestamp, asset) VALUES (?, ?, ?, ?)",
                (mem_type, text, datetime.now().isoformat(), asset)
            )
            self._conn.commit()
        except Exception as e:
            print(f"⚠️ SQLite save failed: {e}")

    def query(self, query_str: str, limit: int = 10) -> str:
        """Query memory - GraphMem or SQLite."""
        if not self._enabled:
            return "Memory disabled"

        if _graphmem_available and _graphmem_memory:
            try:
                return _graphmem_memory.query(query_str)
            except Exception as e:
                print(f"⚠️ GraphMem query failed: {e}")

        return self._query_sqlite(query_str, limit)

    def _query_sqlite(self, query_str: str, limit: int) -> str:
        """Query SQLite fallback."""
        try:
            results = self._conn.execute(
                """
                SELECT type, data, timestamp, asset
                FROM trading_memory
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()

            if not results:
                return "No memories found."

            lines = ["Trading Memory Results:"]
            for r in results:
                lines.append(f"\n[{r[0]}] {r[3]}: {r[1]} @ {r[2]}")

            return "\n".join(lines)
        except Exception as e:
            return f"Query failed: {e}"

    def get_recent(self, mem_type: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """Get recent memories."""
        if not self._enabled:
            return []

        try:
            if mem_type:
                results = self._conn.execute(
                    """
                    SELECT type, data, timestamp, asset
                    FROM trading_memory
                    WHERE type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (mem_type, limit)
                ).fetchall()
            else:
                results = self._conn.execute(
                    """
                    SELECT type, data, timestamp, asset
                    FROM trading_memory
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,)
                ).fetchall()

            return [
                {"type": r[0], "data": r[1], "timestamp": r[2], "asset": r[3]}
                for r in results
            ]
        except Exception as e:
            return []

    def get_by_asset(self, asset: str, limit: int = 20) -> List[Dict]:
        """Get memories for specific asset."""
        if not self._enabled:
            return []

        try:
            results = self._conn.execute(
                """
                SELECT type, data, timestamp, asset
                FROM trading_memory
                WHERE asset = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (asset, limit)
            ).fetchall()

            return [
                {"type": r[0], "data": r[1], "timestamp": r[2], "asset": r[3]}
                for r in results
            ]
        except Exception as e:
            return []

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        if not self._enabled:
            return {"enabled": False}

        try:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM trading_memory"
            ).fetchone()[0]

            by_type = dict(self._conn.execute(
                "SELECT type, COUNT(*) FROM trading_memory GROUP BY type"
            ).fetchall())

            by_asset = dict(self._conn.execute(
                "SELECT asset, COUNT(*) FROM trading_memory WHERE asset IS NOT NULL GROUP BY asset"
            ).fetchall())

            return {
                "enabled": True,
                "graphmem_available": _graphmem_available,
                "total_memories": total,
                "by_type": by_type,
                "by_asset": by_asset
            }
        except Exception as e:
            return {"enabled": True, "error": str(e)}


def get_memory() -> TradingMemory:
    """Get singleton instance."""
    return TradingMemory()