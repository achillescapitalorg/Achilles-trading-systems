"""
DeepSeek LLM Sentiment Analysis Service
Uses DeepSeek API for intelligent financial news sentiment classification.
"""
import os
import json
import time
import requests
from typing import List, Dict, Optional
from functools import lru_cache
from datetime import datetime, timedelta

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

SENTIMENT_CACHE = {}
CACHE_TTL_SECONDS = 300


class DeepSeekSentiment:
    """DeepSeek LLM-powered sentiment analysis for financial news."""
    
    SYSTEM_PROMPT = """You are a professional financial news analyst specializing in trading and investments.

Classify each news headline into ONE of these categories:
- BULLISH: Indicates prices going up, positive sentiment, buy signals, recovery, growth
- BEARISH: Indicates prices going down, negative sentiment, sell signals, decline, crash
- NEUTRAL: Factual news without clear direction, mixed signals, technical analysis
- HOLD: Uncertain, wait-and-see, conflicting signals, low confidence

Return your classification as a JSON object with these fields:
{
  "sentiment": "bullish|bearish|neutral|hold",
  "score": float (-1.0 to 1.0, negative=bearish, positive=bullish, 0=neutral),
  "confidence": float (0.0 to 1.0)
}

Be precise and consistent in your classifications."""
    
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self._fallback = KeywordSentiment()
    
    def is_available(self) -> bool:
        """Check if DeepSeek API key is configured."""
        return bool(self.api_key)
    
    def analyze_single(self, headline: str) -> Dict:
        """Analyze a single headline."""
        cache_key = hash(headline)
        
        if cache_key in SENTIMENT_CACHE:
            cached = SENTIMENT_CACHE[cache_key]
            if time.time() - cached["timestamp"] < CACHE_TTL_SECONDS:
                return cached["result"]
        
        if not self.is_available():
            return self._fallback.analyze(headline)
        
        try:
            result = self._call_deepseek_single(headline)
            SENTIMENT_CACHE[cache_key] = {
                "result": result,
                "timestamp": time.time()
            }
            return result
        except Exception as e:
            error_str = str(e)
            if "402" in error_str or "Payment Required" in error_str:
                print(f"[DeepSeek] API requires payment - using keyword fallback")
                self.api_key = ""
            return self._fallback.analyze(headline)
    
    def analyze_batch(self, headlines: List[str]) -> List[Dict]:
        """Analyze multiple headlines in one API call."""
        if not headlines:
            return []
        
        if not self.is_available():
            return [self._fallback.analyze(h) for h in headlines]
        
        try:
            return self._call_deepseek_batch(headlines)
        except Exception as e:
            error_str = str(e)
            if "402" in error_str or "Payment Required" in error_str:
                print(f"[DeepSeek] Batch API requires payment - using keyword fallback")
                self.api_key = ""
            return [self._fallback.analyze(h) for h in headlines]
    
    def _call_deepseek_single(self, headline: str) -> Dict:
        """Call DeepSeek API for single headline."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this financial news headline:\n\n\"{headline}\""}
            ],
            "temperature": 0.1,
            "max_tokens": 100,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=3
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")
        
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        
        result = json.loads(content)
        return self._normalize_result(result)
    
    def _call_deepseek_batch(self, headlines: List[str]) -> List[Dict]:
        """Call DeepSeek API for batch headline analysis."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        headlines_text = "\n\n".join([f"{i+1}. \"{h}\"" for i, h in enumerate(headlines)])
        
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze these {len(headlines)} financial news headlines. Return a JSON array with one analysis per headline:\n\n{headlines_text}"}
            ],
            "temperature": 0.1,
            "max_tokens": 500,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=5
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")
        
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        
        parsed = json.loads(content)
        
        if isinstance(parsed, list):
            results = [self._normalize_result(r) for r in parsed]
        elif "analyses" in parsed:
            results = [self._normalize_result(r) for r in parsed["analyses"]]
        else:
            raise Exception("Unexpected response format")
        
        while len(results) < len(headlines):
            results.append({"sentiment": "neutral", "score": 0.0, "confidence": 0.5})
        
        return results[:len(headlines)]
    
    def _normalize_result(self, result: Dict) -> Dict:
        """Normalize DeepSeek response to standard format."""
        sentiment_map = {
            "bullish": "bullish",
            "bearish": "bearish", 
            "neutral": "neutral",
            "hold": "hold"
        }
        
        sentiment = result.get("sentiment", "neutral").lower()
        sentiment = sentiment_map.get(sentiment, "neutral")
        
        score = float(result.get("score", 0.0))
        score = max(-1.0, min(1.0, score))
        
        confidence = float(result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "sentiment": sentiment,
            "score": score,
            "confidence": confidence
        }
    
    def get_aggregate_sentiment(self, headlines: List[str]) -> Dict:
        """Calculate aggregate sentiment from multiple headlines."""
        if not headlines:
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0, "total": 0}
        
        analyses = self.analyze_batch(headlines)
        
        if not analyses:
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0, "total": 0}
        
        total_score = sum(a["score"] * a["confidence"] for a in analyses)
        total_confidence = sum(a["confidence"] for a in analyses)
        
        if total_confidence > 0:
            avg_score = total_score / total_confidence
        else:
            avg_score = 0.0
        
        sentiment_counts = {"bullish": 0, "bearish": 0, "neutral": 0, "hold": 0}
        for a in analyses:
            sentiment_counts[a["sentiment"]] = sentiment_counts.get(a["sentiment"], 0) + 1
        
        dominant = max(sentiment_counts, key=sentiment_counts.get)
        
        return {
            "sentiment": dominant,
            "score": avg_score,
            "confidence": min(1.0, total_confidence / len(analyses)),
            "total": len(analyses),
            "breakdown": sentiment_counts
        }


class KeywordSentiment:
    """Fallback keyword-based sentiment when DeepSeek is unavailable."""
    
    BULLISH_KEYWORDS = [
        'rise', 'gain', 'surge', 'rally', 'bullish', 'positive', 'upgrade',
        'strong', 'higher', 'growth', 'boom', 'recovery', 'breakout',
        'beat', 'exceed', 'optimistic', 'support', 'up', 'soar', 'jump',
        'climb', 'advance', 'increase', 'profit', 'win', 'success', 'boom'
    ]
    
    BEARISH_KEYWORDS = [
        'fall', 'drop', 'crash', 'bearish', 'negative', 'downgrade',
        'weak', 'lower', 'decline', 'loss', 'plunge', 'recession',
        'concern', 'warning', 'risk', 'sell', 'down', 'volatile',
        'slump', 'sink', 'tumble', 'fear', 'panic', 'crisis', 'warn'
    ]
    
    HOLD_KEYWORDS = [
        'uncertain', 'wait', 'caution', 'unclear', 'mixed', 'maybe',
        'watch', 'monitor', 'pending', 'speculation', 'might', 'could'
    ]
    
    def analyze(self, text: str) -> Dict:
        """Analyze text using keyword matching."""
        if not text:
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.3}
        
        text_lower = text.lower()
        
        bullish_count = sum(1 for word in self.BULLISH_KEYWORDS if word in text_lower)
        bearish_count = sum(1 for word in self.BEARISH_KEYWORDS if word in text_lower)
        hold_count = sum(1 for word in self.HOLD_KEYWORDS if word in text_lower)
        
        total_keywords = bullish_count + bearish_count + hold_count
        
        if total_keywords == 0:
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.3}
        
        if bullish_count > bearish_count and bullish_count > hold_count:
            score = 0.3 + (bullish_count / total_keywords) * 0.5
            confidence = 0.4 + (bullish_count / total_keywords) * 0.3
            return {"sentiment": "bullish", "score": min(1.0, score), "confidence": min(0.9, confidence)}
        
        if bearish_count > bullish_count and bearish_count > hold_count:
            score = -0.3 - (bearish_count / total_keywords) * 0.5
            confidence = 0.4 + (bearish_count / total_keywords) * 0.3
            return {"sentiment": "bearish", "score": max(-1.0, score), "confidence": min(0.9, confidence)}
        
        if hold_count > 0:
            return {"sentiment": "hold", "score": 0.0, "confidence": 0.4}
        
        return {"sentiment": "neutral", "score": 0.0, "confidence": 0.3}


sentiment_analyzer = DeepSeekSentiment()


def analyze_sentiment(text: str) -> Dict:
    """Analyze sentiment of a single text."""
    return sentiment_analyzer.analyze_single(text)


def analyze_batch_sentiment(texts: List[str]) -> List[Dict]:
    """Analyze sentiment of multiple texts."""
    return sentiment_analyzer.analyze_batch(texts)


def get_aggregate_sentiment(texts: List[str]) -> Dict:
    """Get aggregate sentiment from multiple texts."""
    return sentiment_analyzer.get_aggregate_sentiment(texts)
