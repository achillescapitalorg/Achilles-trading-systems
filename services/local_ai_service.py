"""
Local AI Service for Trading Terminal
=======================================
Provides:
1. FinBERT - Financial sentiment analysis (HuggingFace)
2. Ollama - Local LLM for market analysis and Q&A
"""

import os
import time
import json
import threading
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

SENTIMENT_CACHE = {}
SENTIMENT_CACHE_TTL = 300


class FinBERTSentiment:
    """FinBERT-based sentiment analysis for financial news."""
    
    _model = None
    _loading = False
    _lock = threading.Lock()
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if FinBERT can be loaded."""
        try:
            import transformers
            return True
        except ImportError:
            return False
    
    @classmethod
    def _load_model(cls):
        """Load FinBERT model (singleton)."""
        if cls._model is not None:
            return cls._model
            
        with cls._lock:
            if cls._loading:
                return None
            cls._loading = True
            
            try:
                import transformers
                from transformers import pipeline
                cls._model = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    device=-1,  # CPU
                    truncation=True,
                    max_length=512
                )
                print("[FinBERT] Model loaded successfully")
                return cls._model
            except Exception as e:
                print(f"[FinBERT] Failed to load: {e}")
                cls._model = None
                cls._loading = False
                return None
    
    @classmethod
    def analyze(cls, text: str) -> Dict:
        """
        Analyze sentiment of financial text.
        
        Returns:
            Dict with: sentiment (bullish/bearish/neutral), score (-1 to 1), confidence (0 to 1)
        """
        cache_key = hash(text)
        if cache_key in SENTIMENT_CACHE:
            cached = SENTIMENT_CACHE[cache_key]
            if time.time() - cached["timestamp"] < SENTIMENT_CACHE_TTL:
                return cached["result"]
        
        if cls._model is None:
            cls._load_model()
        
        if cls._model is None:
            return cls._keyword_fallback(text)
        
        try:
            result = cls._model(text[:512])[0]
            
            label = result["label"].lower()
            score = result["score"]
            
            if label == "positive":
                sentiment = "bullish"
                normalized_score = score
            elif label == "negative":
                sentiment = "bearish"
                normalized_score = -score
            else:
                sentiment = "neutral"
                normalized_score = 0.0
            
            output = {
                "sentiment": sentiment,
                "score": normalized_score,
                "confidence": score
            }
            
            SENTIMENT_CACHE[cache_key] = {
                "result": output,
                "timestamp": time.time()
            }
            
            return output
            
        except Exception as e:
            print(f"[FinBERT] Analysis error: {e}")
            return cls._keyword_fallback(text)
    
    @classmethod
    def _keyword_fallback(cls, text: str) -> Dict:
        """Fallback keyword-based sentiment analysis."""
        text_lower = text.lower()
        
        bullish_words = [
            "surge", "rise", "gain", "bull", "up", "growth", "rally", "high",
            "profit", "positive", "increase", "breakout", "optimistic", "buy",
            "recover", "boom", "strength", "support", "break higher"
        ]
        
        bearish_words = [
            "fall", "drop", "bear", "down", "loss", "decline", "crash", "low",
            "negative", "decrease", "breakdown", "pessimistic", "sell",
            "recession", "weakness", "resistance", "break lower", "plunge"
        ]
        
        bullish_count = sum(1 for word in bullish_words if word in text_lower)
        bearish_count = sum(1 for word in bearish_words if word in text_lower)
        
        if bullish_count > bearish_count:
            score = min(bullish_count / max(bullish_count + bearish_count, 1), 1.0)
            return {"sentiment": "bullish", "score": score, "confidence": 0.5}
        elif bearish_count > bullish_count:
            score = min(bearish_count / max(bullish_count + bearish_count, 1), 1.0)
            return {"sentiment": "bearish", "score": -score, "confidence": 0.5}
        else:
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.3}
    
    @classmethod
    def analyze_batch(cls, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts."""
        return [cls.analyze(text) for text in texts]


class OllamaClient:
    """Ollama client for local LLM inference."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama3.2:1b"
        self._available = None
    
    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        if self._available is not None:
            return self._available
        
        try:
            import requests
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=2
            )
            self._available = response.status_code == 200
            if self._available:
                models = response.json().get("models", [])
                print(f"[Ollama] Available models: {[m['name'] for m in models]}")
            return self._available
        except Exception as e:
            print(f"[Ollama] Not available: {e}")
            self._available = False
            return False
    
    def set_model(self, model_name: str):
        """Set the model to use."""
        self.model = model_name
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: int = 500) -> Dict:
        """
        Generate text response from LLM.
        
        Returns:
            Dict with: response (str), success (bool), error (str)
        """
        if not self.is_available():
            return {
                "response": "Ollama is not running. Please start Ollama to use AI analysis.",
                "success": False,
                "error": "Ollama not available"
            }
        
        try:
            import requests
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=90    # llama3.2:1b on CPU takes ~25s even for short replies
            )
            
            print(f"[Ollama] Response status: {response.status_code}")
            print(f"[Ollama] Response text: {response.text[:200]}...")
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "")
                print(f"[Ollama] Got response: {content[:100]}...")
                return {
                    "response": content,
                    "success": True,
                    "error": None
                }
            else:
                return {
                    "response": "",
                    "success": False,
                    "error": f"API error: {response.status_code}"
                }
                
        except Exception as e:
            print(f"[Ollama] Exception: {e}")
            return {
                "response": "",
                "success": False,
                "error": str(e)
            }
    
    def analyze_market(self, symbol: str, news_headlines: List[str], 
                       price_data: Dict) -> Dict:
        """
        Analyze market for a symbol using LLM.
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD')
            news_headlines: List of recent news headlines
            price_data: Dict with current price, trend, etc.
        
        Returns:
            Dict with analysis summary
        """
        news_summary = "\n".join(f"- {h}" for h in news_headlines[:5])
        
        prompt = f"""You are a professional financial analyst for a trading terminal.

Current Symbol: {symbol}
Current Price: ${price_data.get('current_price', 'N/A')}
Recent Trend: {price_data.get('trend', 'N/A')}
Change: {price_data.get('change_pct', 'N/A')}%

Recent News:
{news_summary}

Based on the above information, provide a brief analysis (2-3 sentences):
1. What is your sentiment for {symbol} in the coming days?
2. What key factors should traders consider?
3. What is your recommendation (BUY/SELL/HOLD) with brief reason?

Keep your response concise and actionable."""
        
        result = self.generate(
            prompt=prompt,
            system_prompt="You are an expert financial analyst. Provide clear, actionable trading insights."
        )
        
        return result
    
    def analyze_market_with_context(
        self,
        symbol: str,
        context,           # MarketContext — typed loosely to avoid circular import
        formatted_prompt: str,
    ) -> Dict:
        """
        Analyze market using the full MarketContext string already formatted by
        market_context_builder.format_for_prompt().

        This is the preferred method over analyze_market() because it receives
        ALL platform data (indicators, Heston, Markov, GARCH, risk metrics,
        Monte Carlo, FinBERT sentiment) rather than just 5 headlines.

        Args:
            symbol:           Trading symbol
            context:          MarketContext dataclass (used for metadata)
            formatted_prompt: Pre-built context block from format_for_prompt()

        Returns:
            Dict with: response (str), success (bool), error (str|None), source ("ollama")
        """
        # Use the compact prompt for Ollama (small local LLMs time out on long inputs)
        try:
            from services.market_context_builder import format_for_ollama_prompt
            compact = format_for_ollama_prompt(context)
        except Exception:
            compact = formatted_prompt[:500]   # hard truncate as last resort

        prompt = (
            f"{compact}\n\n"
            f"Give a short BUY/SELL/HOLD analysis for {symbol} based on the data above. "
            f"3-4 sentences max."
        )
        result = self.generate(
            prompt=prompt,
            system_prompt="You are a concise trading analyst. Be brief and cite the numbers.",
            max_tokens=250,
        )
        result["source"] = "ollama"
        return result

    def answer_question(self, question: str, context: Optional[str] = None) -> Dict:
        """Answer a general trading question."""
        prompt = question
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}"

        return self.generate(
            prompt=prompt,
            system_prompt="You are a helpful trading assistant. Answer questions about financial markets, trading, and investments."
        )


class LocalAIService:
    """
    Unified local AI service combining FinBERT and Ollama.
    """
    
    def __init__(self):
        self.sentiment_analyzer = FinBERTSentiment()
        self.ollama = OllamaClient()
    
    def get_sentiment(self, text: str) -> Dict:
        """Get sentiment for a single text."""
        return self.sentiment_analyzer.analyze(text)
    
    def get_sentiment_batch(self, texts: List[str]) -> List[Dict]:
        """Get sentiment for multiple texts."""
        return self.sentiment_analyzer.analyze_batch(texts)
    
    def analyze_market(self, symbol: str, news: List[str], price_data: Dict) -> Dict:
        """Analyze market using Ollama LLM."""
        return self.ollama.analyze_market(symbol, news, price_data)
    
    def chat(self, message: str, context: Optional[str] = None) -> Dict:
        """Chat with the LLM."""
        return self.ollama.answer_question(message, context)
    
    def get_status(self) -> Dict:
        """Get status of all AI services."""
        return {
            "finbert_available": self.sentiment_analyzer.is_available(),
            "ollama_available": self.ollama.is_available(),
            "ollama_model": self.ollama.model if self.ollama.is_available() else None
        }


# Singleton instance
_local_ai_service = None
_service_lock = threading.Lock()


def get_local_ai_service() -> LocalAIService:
    """Get singleton instance of LocalAIService."""
    global _local_ai_service
    if _local_ai_service is None:
        with _service_lock:
            if _local_ai_service is None:
                _local_ai_service = LocalAIService()
    return _local_ai_service


def analyze_sentiment(text: str) -> Dict:
    """Convenience function for sentiment analysis."""
    return get_local_ai_service().get_sentiment(text)


def analyze_sentiment_batch(texts: List[str]) -> List[Dict]:
    """Convenience function for batch sentiment analysis."""
    return get_local_ai_service().get_sentiment_batch(texts)


def analyze_market(symbol: str, news: List[str], price_data: Dict) -> Dict:
    """Convenience function for market analysis."""
    return get_local_ai_service().analyze_market(symbol, news, price_data)


def chat_with_ai(message: str, context: Optional[str] = None) -> Dict:
    """Convenience function for chatting with AI."""
    return get_local_ai_service().chat(message, context)


def get_ai_status() -> Dict:
    """Get status of AI services."""
    return get_local_ai_service().get_status()


def get_ai_status() -> Dict:
    """Get status of AI services."""
    return get_local_ai_service().get_status()