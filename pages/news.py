"""
News Page - Comprehensive News Aggregation View with Async Loading
Uses persistent cache for instant loading and background refresh.
"""
import dash
from dash import dcc, html, Input, Output, callback, no_update
import dash_bootstrap_components as dbc
from datetime import datetime
import random
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import COLORS, INSTRUMENTS, _fetch_news_from_sources, news_cache

# ── Performance state ────────────────────────────────────────────────────────
# Sentiment-aggregate cache (recomputed at most once per `_AGG_TTL` seconds —
# the previous version called FinBERT on every callback, which OOM'd the box).
_AGG_TTL = 60.0
_agg_cache = {"data": None, "ts": 0.0}
_agg_lock = threading.Lock()

# Background news-refresh tracker — only one in flight, throttled.
_bg_refresh_active = False
_bg_refresh_lock = threading.Lock()
_bg_last_run = 0.0
_BG_MIN_INTERVAL = 90.0

# AI summary (Ollama) cache — Ollama call takes 30-90s. Compute once in a
# daemon thread and serve cached HTML for `_AI_SUMMARY_TTL` seconds.
_ai_summary_cache = {"html": None, "ts": 0.0}
_ai_summary_lock = threading.Lock()
_ai_summary_in_flight = False
_AI_SUMMARY_TTL = 300.0


TOPIC_ORDER = ["Fed/Rate", "Geopolitical", "Europe", "UK", "Japan", "Asia", "Crypto", "Metals", "Equities", "Technical", "Trade", "General"]
TOPIC_ICONS = {"Fed/Rate": "🏦", "Europe": "🇪🇺", "UK": "🇬🇧", "Japan": "🇯🇵", "Asia": "🌏", "Geopolitical": "🌍", "Crypto": "₿", "Metals": "🥇", "Equities": "📈", "Technical": "📊", "Trade": "📦", "General": "📰"}


def _detect_topic(headline, source):
    headline_lower = headline.lower()
    if any(word in headline_lower for word in ['fed', 'federal reserve', 'interest rate', 'inflation', 'cpi', 'pce', 'fomc', 'powell', 'monetary']):
        return "Fed/Rate"
    elif any(word in headline_lower for word in ['ecb', 'europe', 'eurozone', 'german', 'french']):
        return "Europe"
    elif any(word in headline_lower for word in ['bank of england', 'boe', 'uk', 'british', 'pound']):
        return "UK"
    elif any(word in headline_lower for word in ['bank of japan', 'boj', 'japan', 'yen', 'japanese']):
        return "Japan"
    elif any(word in headline_lower for word in ['china', 'chinese', 'asian', 'yuan']):
        return "Asia"
    elif any(word in headline_lower for word in ['war', 'conflict', 'geopolitical', 'middle east', 'ukraine', 'russia', 'tension']):
        return "Geopolitical"
    elif any(word in headline_lower for word in ['bitcoin', 'btc', 'ether', 'eth', 'crypto', 'blockchain', 'satoshi']):
        return "Crypto"
    elif any(word in headline_lower for word in ['gold', 'xau', 'silver', 'metal', 'precious']):
        return "Metals"
    elif any(word in headline_lower for word in ['stock', 's&p', 'nasdaq', 'dow', 'equity', 'market']):
        return "Equities"
    elif any(word in headline_lower for word in ['technical', 'chart', 'pattern', 'resistance', 'support', 'indicator']):
        return "Technical"
    elif any(word in headline_lower for word in ['trade', 'tariff', 'export', 'import', 'economy']):
        return "Trade"
    return "General"


def _get_sentiment_badge(sentiment_label, sentiment_score, confidence):
    sentiment_label = sentiment_label or "neutral"
    
    if sentiment_label == "bullish":
        color, emoji = COLORS["success"], "🟢"
        label = "BULLISH"
    elif sentiment_label == "bearish":
        color, emoji = COLORS["danger"], "🔴"
        label = "BEARISH"
    elif sentiment_label == "hold":
        color, emoji = COLORS["warning"], "🟡"
        label = "HOLD"
    else:
        color, emoji = COLORS["text_secondary"], "⚪"
        label = "NEUTRAL"
    
    score_str = f"{sentiment_score:+.2f}" if sentiment_score else "0.00"
    return html.Span([
        html.Span(emoji, style={"marginRight": "4px"}),
        html.Span(f"{label}", style={"fontWeight": "bold", "fontSize": "10px"}),
        html.Span(f" ({score_str})", style={"fontSize": "9px", "opacity": 0.8})
    ],
        style={"display": "inline-flex", "alignItems": "center", "backgroundColor": f"{color}20", "color": color, "padding": "3px 8px", "borderRadius": "4px", "marginRight": "8px", "fontSize": "11px"})


def _render_news_item(item, show_instrument=True):
    topic = _detect_topic(item.get("headline", ""), item.get("source", ""))
    return html.Div([
        html.Div([html.A(item.get("headline", "No headline"), href=item.get("url", "#"), target="_blank",
              style={"color": COLORS["text"], "textDecoration": "none", "fontSize": "13px", "fontWeight": "500", "lineHeight": "1.4"})], style={"marginBottom": "8px"}),
        html.Div([
            html.Span(f"{item.get('source_icon', '📰')} {item.get('source', 'Unknown')}", style={"color": COLORS["text_secondary"], "fontSize": "11px", "marginRight": "12px"}),
            html.Span(f"⏱ {item.get('time_ago', 'Live')}", style={"color": COLORS["text_secondary"], "fontSize": "11px", "marginRight": "12px"}),
            html.Span(f"📊 {topic}", style={"color": COLORS["info"], "fontSize": "11px", "marginRight": "12px"}),
            html.Span(f"⚡ {item.get('impact', 'MEDIUM')}", style={"color": COLORS["warning"] if item.get('impact') == 'MEDIUM' else COLORS["danger"], "fontSize": "11px"}),
        ], style={"marginBottom": "6px"}),
        html.Div([
            _get_sentiment_badge(item.get("sentiment_label", "neutral"), item.get("sentiment", 0), item.get("confidence", 0)),
            html.Span(f"Confidence: {item.get('confidence', 0):.0%}", style={"color": COLORS["text_secondary"], "fontSize": "9px", "marginRight": "8px"}),
            html.Span(f"📍 {item.get('instrument', '')}" if show_instrument else "", style={"color": COLORS["accent"], "fontSize": "10px"})
        ]),
        html.Hr(style={"borderColor": COLORS["border"], "margin": "12px 0"})
    ], style={"padding": "10px", "backgroundColor": COLORS["surface_light"], "borderRadius": "6px", "marginBottom": "10px"})


def _fetch_and_cache(symbol: str, force: bool = False):
    """Fetch news for one symbol, bypassing cache when force=True."""
    if not force:
        cached = news_cache.get(symbol)
        if cached is not None:
            return cached
    try:
        items = _fetch_news_from_sources(symbol)
        if items:
            news_cache.set(symbol, items)
            return items
    except Exception:
        pass
    # Fall back to cache even on forced refresh if fetch fails
    return news_cache.get(symbol) or []


def _kick_background_refresh():
    """Refresh ALL symbols in parallel inside a daemon thread. Non-blocking,
    throttled to one run per `_BG_MIN_INTERVAL` seconds.
    """
    global _bg_refresh_active, _bg_last_run
    with _bg_refresh_lock:
        now = _time.time()
        if _bg_refresh_active or (now - _bg_last_run) < _BG_MIN_INTERVAL:
            return
        _bg_refresh_active = True
        _bg_last_run = now

    def _bg():
        try:
            with ThreadPoolExecutor(max_workers=min(len(INSTRUMENTS), 3)) as ex:
                list(ex.map(lambda inst: _fetch_and_cache(inst["symbol"], force=False),
                            INSTRUMENTS))
            with _agg_lock:
                _agg_cache["ts"] = 0.0   # invalidate so next callback picks up
        except Exception as e:
            print(f"[News] Background refresh error: {e}")
        finally:
            global _bg_refresh_active
            with _bg_refresh_lock:
                _bg_refresh_active = False

    threading.Thread(target=_bg, daemon=True, name="NewsBgRefresh").start()


def _get_all_news(force_refresh: bool = False):
    """Read all symbols' news. force_refresh fans out in parallel. Cache path
    is instant (in-memory dict reads).
    """
    all_news = []

    if force_refresh:
        with ThreadPoolExecutor(max_workers=min(len(INSTRUMENTS), 3)) as ex:
            results = list(ex.map(
                lambda inst: (inst, _fetch_and_cache(inst["symbol"], force=True)),
                INSTRUMENTS,
            ))
    else:
        # Cache-only path — instant
        results = [(inst, news_cache.get(inst["symbol"]) or []) for inst in INSTRUMENTS]

    for inst, news_items in results:
        if news_items:
            for item in news_items:
                item["instrument"] = inst["symbol"]
                item["instrument_name"] = inst["name"]
            all_news.extend(news_items)

    if not all_news:
        for inst in INSTRUMENTS:
            all_news.append({"headline": f"Latest {inst['name']} market analysis",
                             "sentiment": 0, "sentiment_label": "neutral",
                             "impact": "MEDIUM", "time_ago": "Live",
                             "source": "VibeTrading", "source_icon": "📈",
                             "url": "#", "instrument": inst["symbol"],
                             "instrument_name": inst["name"]})
    random.seed(42)
    random.shuffle(all_news)
    return all_news


def _get_news_with_aggregate(force_refresh: bool = False):
    """Get news and aggregate sentiment for each instrument."""
    result = {}
    for inst in INSTRUMENTS:
        symbol = inst["symbol"]
        news_items = _fetch_and_cache(symbol, force=force_refresh)
        headlines = [item.get("headline", "") for item in news_items] if news_items else []
        result[symbol] = {
            "news": news_items or [],
            "headlines": headlines,
        }
    return result


def _calculate_instrument_sentiment():
    """Aggregate per-instrument sentiment from already-cached items.

    Reuses the ``sentiment_label`` and ``sentiment`` fields stored on each
    item by the news pipeline — NOT recomputed here. This is O(N) over cached
    items: no model loads, no GPU, no network. Result is memoized for
    `_AGG_TTL` seconds so repeated callbacks within the same minute are free.
    """
    now = _time.time()
    with _agg_lock:
        if _agg_cache["data"] is not None and (now - _agg_cache["ts"]) < _AGG_TTL:
            return _agg_cache["data"]

    sentiment_data = {}
    for inst in INSTRUMENTS:
        symbol = inst["symbol"]
        items = news_cache.get(symbol) or []

        if not items:
            sentiment_data[symbol] = {
                "name": inst["name"],
                "avg_sentiment": 0, "ai_sentiment": "neutral",
                "ai_score": 0.0, "ai_confidence": 0.0,
                "news_count": 0,
                "bullish_count": 0, "bearish_count": 0, "neutral_count": 0,
                "breakdown": {},
            }
            continue

        labels = [it.get("sentiment_label", "neutral") for it in items]
        scores = [float(it.get("sentiment", 0) or 0) for it in items]
        positive = sum(1 for l in labels if l == "bullish")
        negative = sum(1 for l in labels if l == "bearish")
        neutral  = len(labels) - positive - negative
        avg_score = sum(scores) / len(scores) if scores else 0.0

        if positive > negative:
            agg = "bullish"
        elif negative > positive:
            agg = "bearish"
        else:
            agg = "neutral"

        sentiment_data[symbol] = {
            "name":          inst["name"],
            "avg_sentiment": avg_score,
            "ai_sentiment":  agg,
            "ai_score":      avg_score,
            "ai_confidence": abs(avg_score),
            "news_count":    len(items),
            "bullish_count": positive,
            "bearish_count": negative,
            "neutral_count": neutral,
            "breakdown": {"bullish": positive, "bearish": negative, "neutral": neutral},
        }

    with _agg_lock:
        _agg_cache["data"] = sentiment_data
        _agg_cache["ts"]   = now
    return sentiment_data


def _render_sentiment_meter(sentiment_data):
    """Render a sentiment overview bar for all instruments with AI sentiment."""
    items = []
    for inst in INSTRUMENTS:
        symbol = inst["symbol"]
        data = sentiment_data.get(symbol, {"name": inst["name"], "avg_sentiment": 0, "news_count": 0, "ai_sentiment": "neutral"})
        
        ai_sentiment = data.get("ai_sentiment", "neutral")
        
        if ai_sentiment == "bullish":
            color = COLORS["success"]
            label = "🟢 BULLISH"
            emoji = "📈"
        elif ai_sentiment == "bearish":
            color = COLORS["danger"]
            label = "🔴 BEARISH"
            emoji = "📉"
        elif ai_sentiment == "hold":
            color = COLORS["warning"]
            label = "🟡 HOLD"
            emoji = "⏸️"
        else:
            color = COLORS["text_secondary"]
            label = "⚪ NEUTRAL"
            emoji = "⏸️"
        
        ai_score = data.get("ai_score", 0)
        bar_width = abs(ai_score) * 50
        
        items.append(html.Div([
            html.Div([
                html.Span(f"{symbol}", style={"fontWeight": "bold", "fontSize": "11px", "color": COLORS["text"]}),
                html.Span(f" {emoji}", style={"marginLeft": "5px"}),
            ], style={"marginBottom": "4px"}),
            html.Div([
                html.Span(label, style={"fontSize": "9px", "color": color, "fontWeight": "bold"})
            ], style={"marginBottom": "4px"}),
            html.Div([
                html.Div(style={
                    "width": f"{bar_width}%",
                    "backgroundColor": color,
                    "height": "6px",
                    "borderRadius": "3px",
                    "position": "relative",
                    "left": f"{50 if ai_score >= 0 else 50 - bar_width}%"
                }),
                html.Div(style={"position": "absolute", "left": "50%", "top": "0", "bottom": "0", "width": "1px", "backgroundColor": COLORS["border"]})
            ], style={"position": "relative", "width": "100%", "height": "6px", "backgroundColor": COLORS["surface_light"], "borderRadius": "3px", "overflow": "hidden", "marginBottom": "4px"}),
            html.Span(f"{data['news_count']} news", style={"fontSize": "9px", "color": COLORS["text_secondary"]})
        ], style={"width": "12%", "display": "inline-block", "marginRight": "1%", "verticalAlign": "top", "padding": "8px", "backgroundColor": COLORS["surface"], "borderRadius": "6px"}))
    
    return html.Div([
        html.H5([html.Span("🤖 ", style={"fontSize": "16px"}), "AI Market Sentiment", 
                 html.Span(" (FinBERT)", style={"fontSize": "10px", "color": COLORS["text_secondary"]})], 
                style={"color": COLORS["text"], "fontSize": "14px", "marginBottom": "10px", "fontWeight": "bold"}),
        html.Div(items, style={"marginBottom": "10px"}),
        html.Div([
            html.Span("← Bearish", style={"color": COLORS["danger"], "fontSize": "10px", "marginRight": "20px"}),
            html.Span("Neutral", style={"color": COLORS["warning"], "fontSize": "10px", "marginRight": "20px"}),
            html.Span("Bullish →", style={"color": COLORS["success"], "fontSize": "10px"})
        ], style={"textAlign": "center", "marginTop": "5px"})
    ], style={"padding": "15px", "backgroundColor": COLORS["surface"], "borderRadius": "8px", "marginBottom": "20px"})


def _build_news_tabs(force_refresh: bool = False):
    all_news = _get_all_news(force_refresh=force_refresh)
    news_by_instrument = {inst["symbol"]: [item for item in all_news if item.get("instrument") == inst["symbol"]] for inst in INSTRUMENTS}
    news_by_topic = {}
    for item in all_news:
        topic = _detect_topic(item.get("headline", ""), item.get("source", ""))
        if topic not in news_by_topic:
            news_by_topic[topic] = []
        news_by_topic[topic].append(item)
    
    last_updated = news_cache.get_last_updated() or "Never"
    if last_updated and last_updated != "Never":
        try:
            dt = datetime.fromisoformat(last_updated)
            diff = (datetime.now() - dt).total_seconds()
            last_updated = f"Updated {int(diff)}s ago" if diff < 60 else f"Updated {int(diff/60)}m ago" if diff < 3600 else f"Updated {int(diff/3600)}h ago"
        except:
            pass
    
    # All News Tab
    all_items = []
    for inst in INSTRUMENTS:
        symbol = inst["symbol"]
        inst_news = news_by_instrument.get(symbol, [])
        all_items.append(html.Div([
            html.H6([f"📈 {inst['name']} ({symbol})", html.Span(f" - {len(inst_news)} articles", style={"color": COLORS["text_secondary"], "fontSize": "11px", "marginLeft": "8px"})],
                   style={"color": COLORS["accent"], "marginBottom": "12px", "fontSize": "14px"}),
            *[_render_news_item(item, show_instrument=True) for item in inst_news]
        ], style={"marginBottom": "20px", "paddingBottom": "15px", "borderBottom": f"1px dashed {COLORS['border']}"}))
    
    # By Instrument Tab
    by_inst_items = []
    for inst in INSTRUMENTS:
        symbol = inst["symbol"]
        inst_news = news_by_instrument.get(symbol, [])
        content = html.Div([
            html.H5([f"📈 {inst['name']} ({symbol})", html.Span(f" - {len(inst_news)} articles", style={"color": COLORS["text_secondary"], "fontSize": "11px"})],
                    style={"marginBottom": "12px", "paddingBottom": "8px", "borderBottom": f"1px solid {COLORS['border']}"}),
            *[_render_news_item(item, show_instrument=False) for item in inst_news]
        ], style={"marginBottom": "20px"}) if inst_news else html.P(f"No news for {inst['name']}", style={"color": COLORS["text_secondary"], "padding": "10px"})
        by_inst_items.append(dbc.AccordionItem(content, title=f"{inst['name']} ({symbol})", item_id=f"instrument-{symbol}"))
    
    # By Topic Tab
    by_topic_items = []
    for topic in TOPIC_ORDER:
        topic_news = news_by_topic.get(topic, [])
        if topic_news:
            content = html.Div([
                html.H5([TOPIC_ICONS.get(topic, "📰"), html.Span(f" {topic}", style={"color": COLORS["accent"], "fontSize": "14px", "marginLeft": "8px"}),
                        html.Span(f" - {len(topic_news)} articles", style={"color": COLORS["text_secondary"], "fontSize": "11px", "marginLeft": "8px"})],
                        style={"marginBottom": "12px", "paddingBottom": "8px", "borderBottom": f"1px solid {COLORS['border']}"}),
                *[_render_news_item(item, show_instrument=True) for item in topic_news]
            ], style={"marginBottom": "20px"})
            by_topic_items.append(dbc.AccordionItem(content, title=f"{topic}", item_id=f"topic-{topic}"))
    
    # Economic Calendar Tab
    calendar_events = _get_economic_calendar()
    calendar_items = []
    for event in calendar_events[:15]:
        impact_color = {'HIGH': COLORS['danger'], 'MEDIUM': COLORS['warning'], 'LOW': COLORS['info']}.get(event.get('impact', 'LOW'), COLORS['text_secondary'])
        calendar_items.append(html.Div([
            html.Div([
                html.Span(f"🕐 {event.get('time', '')}", style={"color": COLORS["text_secondary"], "fontSize": "11px", "marginRight": "10px"}),
                html.Span(f"{event.get('currency', '')}", style={"color": COLORS["accent"], "fontSize": "11px", "fontWeight": "bold", "marginRight": "10px"}),
                html.Span(f"🔴 {event.get('impact', '')}", style={"color": impact_color, "fontSize": "10px", "fontWeight": "bold"}),
            ], style={"marginBottom": "4px"}),
            html.A(event.get('event', 'Unknown Event'), href=event.get('url', '#'), target='_blank', 
                  style={"color": COLORS["text"], "fontSize": "12px", "fontWeight": "500", "textDecoration": "none"}),
            html.Div([
                html.Span(f"Forecast: {event.get('forecast', '-')}", style={"color": COLORS["success"], "fontSize": "10px", "marginRight": "10px"}),
                html.Span(f"Previous: {event.get('previous', '-')}", style={"color": COLORS["text_secondary"], "fontSize": "10px"}),
            ], style={"marginTop": "4px"}),
        ], style={"padding": "10px", "backgroundColor": COLORS["surface_light"], "borderRadius": "6px", "marginBottom": "8px"}))
    
    calendar_content = html.Div([
        html.H5("📅 Upcoming Economic Events", style={"color": COLORS["text"], "marginBottom": "15px", "fontSize": "14px"}),
        html.Div(calendar_items) if calendar_items else html.P("No upcoming events", style={"color": COLORS["text_secondary"]})
    ], style={"padding": "10px"})
    
    # AI Analysis Tab - Market Summary
    ai_summary = _get_ai_market_summary()
    ai_content = html.Div([
        html.H5("🤖 AI Market Analysis", style={"color": COLORS["text"], "marginBottom": "15px", "fontSize": "14px"}),
        ai_summary
    ], style={"padding": "10px"})
    
    tabs = dbc.Tabs([
        dbc.Tab([html.Div(all_items, style={"padding": "10px"})], label="📰 All News", tab_id="tab-all", label_style={"color": COLORS["text"], "fontSize": "12px"}),
        dbc.Tab([html.Div([dbc.Accordion(by_inst_items, active_item="instrument-XAUUSD", flush=True)], style={"padding": "10px"})],
                label="📈 By Instrument", tab_id="tab-instrument", label_style={"color": COLORS["text"], "fontSize": "12px"}),
        dbc.Tab(html.Div([dbc.Accordion(by_topic_items, active_item="topic-Fed/Rate", flush=True)] if by_topic_items else html.P("No news", style={"color": COLORS["text_secondary"]}), style={"padding": "10px"}),
                label="🏷️ By Topic", tab_id="tab-topic", label_style={"color": COLORS["text"], "fontSize": "12px"}),
        dbc.Tab(calendar_content, label="📅 Economic Calendar", tab_id="tab-calendar", label_style={"color": COLORS["text"], "fontSize": "12px"}),
        dbc.Tab(ai_content, label="🤖 AI Analysis", tab_id="tab-ai", label_style={"color": COLORS["text"], "fontSize": "12px"}),
    ], active_tab="tab-all", style={"backgroundColor": COLORS["surface"], "borderRadius": "6px", "padding": "10px"})
    
    return tabs, len(all_news), last_updated


def _get_economic_calendar():
    """Get economic calendar events."""
    from datetime import datetime, timedelta
    
    events = [
        {"time": "08:30", "currency": "USD", "event": "Core CPI (MoM)", "impact": "HIGH", "forecast": "0.3%", "previous": "0.4%", "url": "https://www.investing.com/economic-calendar/cpi-mo-m-228"},
        {"time": "08:30", "currency": "USD", "event": "Non-Farm Payrolls", "impact": "HIGH", "forecast": "180K", "previous": "199K", "url": "https://www.investing.com/economic-calendar/nonfarm-payrolls-228"},
        {"time": "14:00", "currency": "USD", "event": "FOMC Meeting Minutes", "impact": "HIGH", "forecast": "-", "previous": "-", "url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"},
        {"time": "07:00", "currency": "EUR", "event": "ECB Interest Rate Decision", "impact": "HIGH", "forecast": "4.50%", "previous": "4.50%", "url": "https://www.ecb.europa.eu/"},
        {"time": "07:30", "currency": "EUR", "event": "ECB Press Conference", "impact": "HIGH", "forecast": "-", "previous": "-", "url": "https://www.ecb.europa.eu/"},
        {"time": "02:00", "currency": "GBP", "event": "BoE Interest Rate Decision", "impact": "HIGH", "forecast": "5.25%", "previous": "5.25%", "url": "https://www.bankofengland.co.uk/"},
        {"time": "02:00", "currency": "JPY", "event": "BoJ Interest Rate Decision", "impact": "HIGH", "forecast": "0.1%", "previous": "0.1%", "url": "https://www.boj.or.jp/"},
        {"time": "10:00", "currency": "USD", "event": "ISM Manufacturing PMI", "impact": "MEDIUM", "forecast": "50.5", "previous": "50.3", "url": "https://www.investing.com/economic-calendar/us-ism-manufacturing-pmi-722"},
        {"time": "10:00", "currency": "USD", "event": "ISM Services PMI", "impact": "MEDIUM", "forecast": "52.0", "previous": "51.8", "url": "https://www.investing.com/economic-calendar/us-ism-services-pmi-724"},
        {"time": "08:30", "currency": "USD", "event": "GDP (QoQ)", "impact": "HIGH", "forecast": "2.1%", "previous": "2.8%", "url": "https://www.bea.gov/"},
        {"time": "08:30", "currency": "USD", "event": "Retail Sales (MoM)", "impact": "MEDIUM", "forecast": "0.3%", "previous": "0.4%", "url": "https://www.census.gov/"},
        {"time": "08:00", "currency": "EUR", "event": "German CPI (YoY)", "impact": "MEDIUM", "forecast": "2.3%", "previous": "2.6%", "url": "https://www.destatis.de/"},
    ]
    return events


def _kick_ai_summary_refresh():
    """Compute the Ollama market summary in a daemon thread. No-op if cached
    or already in flight. Avoids blocking the news callback for 30-90 seconds.
    """
    global _ai_summary_in_flight
    with _ai_summary_lock:
        now = _time.time()
        if _ai_summary_in_flight:
            return
        if _ai_summary_cache["html"] is not None and (now - _ai_summary_cache["ts"]) < _AI_SUMMARY_TTL:
            return
        _ai_summary_in_flight = True

    def _compute():
        global _ai_summary_in_flight
        try:
            from services.local_ai_service import get_local_ai_service
            ai_service = get_local_ai_service()
            status = ai_service.get_status()
            if not status.get("ollama_available"):
                rendered = html.P(
                    "🦙 Ollama not available — start it with `ollama run llama3.2:1b`",
                    style={"color": COLORS["warning"], "fontSize": "12px",
                           "padding": "10px",
                           "backgroundColor": f"{COLORS['warning']}20",
                           "borderRadius": "6px"})
            else:
                all_news = _get_all_news()[:10]
                headlines = [it.get("headline", "") for it in all_news if it.get("headline")]
                news_summary = "\n".join([f"- {h}" for h in headlines[:5]])
                prompt = (
                    "You are a professional financial analyst. Based on these recent news "
                    f"headlines, provide a brief market summary:\n\n{news_summary}\n\n"
                    "Cover: overall market sentiment, key themes, short-term trading implications. "
                    "Keep it concise (2-3 sentences)."
                )
                result = ai_service.chat(prompt)
                if result.get("success"):
                    rendered = html.Div([
                        html.Div([
                            html.Span("🦙 Ollama: ",
                                      style={"color": COLORS["success"], "fontSize": "10px",
                                             "fontWeight": "bold"}),
                            html.Span("Connected",
                                      style={"color": COLORS["success"], "fontSize": "10px"}),
                        ], style={"marginBottom": "10px"}),
                        html.Div([
                            html.Span("🤖 AI Summary:",
                                      style={"color": COLORS["accent"], "fontSize": "12px",
                                             "fontWeight": "bold", "marginBottom": "8px",
                                             "display": "block"}),
                            html.P(result.get("response", "No analysis available"),
                                   style={"color": COLORS["text"], "fontSize": "12px",
                                          "lineHeight": "1.6"})
                        ], style={"padding": "15px",
                                  "backgroundColor": COLORS["surface_light"],
                                  "borderRadius": "8px"})
                    ])
                else:
                    rendered = html.P(
                        f"AI analysis unavailable: {result.get('error', 'Unknown')}",
                        style={"color": COLORS["text_secondary"]})
            with _ai_summary_lock:
                _ai_summary_cache["html"] = rendered
                _ai_summary_cache["ts"]   = _time.time()
        except Exception as e:
            print(f"[News] AI summary compute error: {e}")
        finally:
            with _ai_summary_lock:
                _ai_summary_in_flight = False

    threading.Thread(target=_compute, daemon=True, name="NewsAISummary").start()


def _get_ai_market_summary():
    """
    Return cached AI summary instantly. NEVER blocks on Ollama.
    First call kicks a daemon thread; subsequent calls within TTL serve cache.
    """
    with _ai_summary_lock:
        cached_html = _ai_summary_cache["html"]
        ts = _ai_summary_cache["ts"]
        in_flight = _ai_summary_in_flight

    now = _time.time()
    fresh = cached_html is not None and (now - ts) < _AI_SUMMARY_TTL
    if not fresh and not in_flight:
        _kick_ai_summary_refresh()
    if fresh:
        return cached_html
    return html.Div([
        html.Span("🤖 ", style={"fontSize": "14px"}),
        html.Span("AI summary is being generated…",
                  style={"color": COLORS["text_secondary"], "fontSize": "12px"}),
        html.Br(),
        html.Span("(Refresh in ~30 seconds)",
                  style={"color": COLORS["text_secondary"], "fontSize": "10px",
                         "fontStyle": "italic"}),
    ], style={"padding": "15px", "backgroundColor": COLORS["surface_light"],
              "borderRadius": "8px"})


# ── No module-level fetches — placeholders only ──────────────────────────────
# Previously this file ran _build_news_tabs() and _calculate_instrument_sentiment()
# at module-import time, which loaded FinBERT and blocked the page for 30-60s.
# The callback below populates everything from the cache instantly on first
# render and kicks a background refresh.
_loading_placeholder = html.Div(
    "Loading news…",
    style={"color": COLORS["text_secondary"], "fontSize": "13px",
           "textAlign": "center", "padding": "40px"},
)
tabs_content   = _loading_placeholder
article_count  = 0
last_updated   = ""
sentiment_data = {}
sentiment_meter = html.Div(style={"height": "60px"})

layout = dbc.Container(fluid=True, style={"backgroundColor": COLORS["background"], "minHeight": "100vh", "padding": "20px"}, children=[
    html.H3([html.Span("📰 ", style={"fontSize": "24px"}), "News Terminal"], style={"color": COLORS["text"], "marginBottom": "20px", "fontWeight": "bold"}),
    
    html.Div(id="sentiment-meter", children=sentiment_meter),
    
    dbc.Row([
        dbc.Col([html.Div([html.Span("📊 ", style={"color": COLORS["accent"]}), html.Span(f"{article_count} articles", id="news-count", style={"color": COLORS["text_secondary"], "fontSize": "12px"})])], width=8),
        dbc.Col([html.Div([
            html.Span(f"⏰ {last_updated}", id="news-updated", style={"color": COLORS["text_secondary"], "fontSize": "10px", "marginRight": "10px"}),
            dbc.Button("🔄 Refresh", id="refresh-btn", color="primary", size="sm", style={"float": "right", "fontSize": "11px"})
        ])], width=4),
    ], className="mb-4"),
    html.Div(id="news-tabs-container", children=tabs_content),
    dcc.Interval(id="refresh-interval", interval=60000),
])


@callback(
    Output("news-tabs-container", "children"),
    Output("news-count", "children"),
    Output("news-updated", "children"),
    Output("sentiment-meter", "children"),
    Input("url", "pathname"),
    Input("refresh-btn", "n_clicks"),
    Input("refresh-interval", "n_intervals"),
)
def update_news(pathname, n_clicks, n_intervals):
    from dash import callback_context
    if pathname != "/news":
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    try:
        triggered = callback_context.triggered_id if callback_context.triggered else None
        force = (triggered == "refresh-btn")

        # Kick a non-blocking parallel cache refresh on every render. The UI
        # itself is built from cache instantly; the daemon thread updates the
        # cache so the next interval tick (60s) shows fresher items.
        if not force:
            _kick_background_refresh()

        tabs, count, updated = _build_news_tabs(force_refresh=force)
        sent = _calculate_instrument_sentiment()
        meter = _render_sentiment_meter(sent)
        return tabs, f"{count} articles", f"⏰ {updated}", meter
    except Exception as e:
        # Surface the error in the UI instead of leaving a perpetual spinner.
        import traceback
        traceback.print_exc()
        err_div = html.Div([
            html.H5("⚠️ News failed to load",
                    style={"color": COLORS["danger"], "fontSize": "14px"}),
            html.P(f"{type(e).__name__}: {e}",
                   style={"color": COLORS["text_secondary"], "fontSize": "11px"}),
            html.P("Check server log for traceback.",
                   style={"color": COLORS["text_secondary"], "fontSize": "10px",
                          "fontStyle": "italic"}),
        ], style={"padding": "20px"})
        return err_div, "Error", "⚠️ Error loading", html.Div(style={"height": "60px"})
