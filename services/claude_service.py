"""
Claude AI Service
=================
Primary AI backend using the Anthropic Claude API.

Fallback chain: Claude → Ollama → keyword-based engine

Design decisions:
- Lazy import of `anthropic` so the app runs even if the package is not installed
- Singleton pattern — one client instance per process
- `is_available()` checks both SDK presence and API key; re-verified every 60s
  so setting ANTHROPIC_API_KEY at runtime (without restart) is supported
"""

import os
import threading
import time
from typing import Optional, Dict, TYPE_CHECKING

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

if TYPE_CHECKING:
    from services.market_context_builder import MarketContext


# ---------------------------------------------------------------------------
# System prompt — tells Claude how to behave in the trading terminal context
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an expert quantitative analyst and trading advisor embedded \
in a professional trading terminal. You have access to live data: technical indicators, \
stochastic volatility model parameters, market-regime detection, GARCH volatility forecasts, \
risk metrics, Monte Carlo simulations, and aggregated FinBERT news sentiment.

Guidelines:
- Ground EVERY claim in the numeric data provided in the context block
- Cite specific values (e.g., "RSI is at 34.2, approaching oversold territory")
- When indicators conflict, flag the conflict explicitly
- Distinguish between technical signals, model forecasts, and sentiment-driven factors
- When entry/SL/TP are available from the unified recommendation, reference them
- Keep responses concise and actionable (3–6 sentences or a short bullet list)
- Never fabricate numbers that are not in the context block
- If a value shows "Not available", acknowledge the data gap rather than guessing
"""


class ClaudeService:
    """
    Thin wrapper around the Anthropic Messages API.

    Usage:
        svc = get_claude_service()
        if svc.is_available():
            result = svc.chat_with_context(user_msg, ctx, formatted_prompt)
    """

    _client = None
    _client_lock = threading.Lock()
    _available: Optional[bool] = None
    _avail_lock = threading.Lock()
    _last_check: float = 0.0
    _CHECK_INTERVAL: float = 60.0

    def __init__(self) -> None:
        self.model = "claude-sonnet-4-6"
        self.max_tokens = 1024
        self.api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """
        Return True if the anthropic package is installed AND an API key is set.
        Result is cached for 60 seconds to avoid import overhead on every request.
        """
        now = time.time()
        with self._avail_lock:
            if self._available is not None and (now - self._last_check) < self._CHECK_INTERVAL:
                return self._available

            # Re-read key in case it was added to .env after startup
            self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
            try:
                import anthropic  # noqa: F401 — just checking availability
                self._available = bool(self.api_key)
            except ImportError:
                self._available = False
            self._last_check = time.time()
            return self._available

    # ------------------------------------------------------------------
    # Client singleton
    # ------------------------------------------------------------------

    def _get_client(self):
        """Lazy-init the Anthropic client (thread-safe)."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    import anthropic
                    self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def chat_with_context(
        self,
        user_message: str,
        context: "MarketContext",
        formatted_prompt: str,
    ) -> Dict:
        """
        Send a user question to Claude with the full market context prepended.

        Args:
            user_message:     Raw question typed by the user in the AI chat box
            context:          MarketContext dataclass (used for metadata / logging)
            formatted_prompt: Pre-built context block from format_for_prompt()

        Returns:
            Dict with keys:
                response  (str)           — Claude's answer
                success   (bool)          — True on successful API call
                error     (str | None)    — Error message on failure
                source    (str)           — Always "claude"
        """
        if not self.is_available():
            return {
                "response": "",
                "success": False,
                "error": "Claude not available (no API key or package not installed)",
                "source": "claude",
            }

        try:
            client = self._get_client()
            full_user_content = (
                f"{formatted_prompt}\n\n"
                f"---\n"
                f"User question: {user_message}"
            )
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": full_user_content}],
            )
            response_text = message.content[0].text
            return {
                "response": response_text,
                "success": True,
                "error": None,
                "source": "claude",
            }

        except Exception as exc:
            print(f"[ClaudeService] API error: {exc}")
            return {
                "response": "",
                "success": False,
                "error": str(exc),
                "source": "claude",
            }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        """Return a dict suitable for the UI status bar."""
        return {
            "claude_available": self.is_available(),
            "claude_model": self.model if self.is_available() else None,
            "api_key_set": bool(os.getenv("ANTHROPIC_API_KEY", "")),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[ClaudeService] = None
_instance_lock = threading.Lock()


def get_claude_service() -> ClaudeService:
    """Return the process-wide ClaudeService singleton."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ClaudeService()
    return _instance
