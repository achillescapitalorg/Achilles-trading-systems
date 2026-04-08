"""
News Page - Comprehensive News Aggregation View with Async Loading
Uses persistent cache for instant loading and background refresh.
"""
import dash
from dash import dcc, html, Input, Output, callback, no_update
import dash_bootstrap_components as dbc
from datetime import datetime
import random

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import COLORS, INSTRUMENTS, _fetch_news_from_sources, news_cache


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


def _get_all_news():
    all_news = []
    for inst in INSTRUMENTS:
        symbol = inst["symbol"]
        news_items = news_cache.get(symbol)
        if news_items is None:
            try:
                news_items = _fetch_news_from_sources(symbol)
                if news_items:
                    news_cache.set(symbol, news_items)
            except:
                news_items = []
        if news_items:
            for item in news_items:
                item["instrument"] = symbol
                item["instrument_name"] = inst["name"]
            all_news.extend(news_items)
    if not all_news:
        for inst in INSTRUMENTS:
            all_news.append({"headline": f"Latest {inst['name']} market analysis", "sentiment": 0, "sentiment_label": "neutral", "impact": "MEDIUM", "time_ago": "Live",
                           "source": "VibeTrading", "source_icon": "📈", "url": "#", "instrument": inst["symbol"], "instrument_name": inst["name"]})
    random.seed(42)
    random.shuffle(all_news)
    return all_news


def _get_news_with_aggregate():
    """Get news and aggregate sentiment for each instrument."""
    result = {}
    for inst in INSTRUMENTS:
        symbol = inst["symbol"]
        news_items = news_cache.get(symbol)
        if news_items is None:
            try:
                news_items = _fetch_news_from_sources(symbol)
                if news_items:
                    news_cache.set(symbol, news_items)
            except:
                news_items = []
        
        headlines = [item.get("headline", "") for item in news_items] if news_items else []
        
        result[symbol] = {
            "news": news_items or [],
            "headlines": headlines,
        }
    
    return result


def _calculate_instrument_sentiment():
    """Calculate aggregated sentiment for each instrument using local AI (FinBERT)."""
    from services.local_ai_service import get_local_ai_service
    
    ai_service = get_local_ai_service()
    sentiment_data = {}
    
    for inst in INSTRUMENTS:
        symbol = inst["symbol"]
        news_items = news_cache.get(symbol)
        
        if news_items and len(news_items) > 0:
            headlines = [item.get("headline", "") for item in news_items]
            
            try:
                results = ai_service.get_sentiment_batch(headlines)
                scores = [r.get("score", 0) for r in results]
                sentiments = [r.get("sentiment", "neutral") for r in results]
                
                avg_score = sum(scores) / len(scores) if scores else 0
                
                positive = sum(1 for s in sentiments if s == "bullish")
                negative = sum(1 for s in sentiments if s == "bearish")
                
                if positive > negative:
                    agg_sentiment = "bullish"
                elif negative > positive:
                    agg_sentiment = "bearish"
                else:
                    agg_sentiment = "neutral"
                
                aggregate = {
                    "sentiment": agg_sentiment,
                    "score": avg_score,
                    "confidence": abs(avg_score),
                    "breakdown": {"bullish": positive, "bearish": negative, "neutral": len(sentiments) - positive - negative}
                }
            except Exception as e:
                print(f"Sentiment error: {e}")
                aggregate = {"sentiment": "neutral", "score": 0.0, "confidence": 0.0, "breakdown": {}}
            
            sentiments = [item.get("sentiment", 0) for item in news_items]
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            sentiment_data[symbol] = {
                "name": inst["name"],
                "avg_sentiment": avg_sentiment,
                "ai_sentiment": aggregate.get("sentiment", "neutral"),
                "ai_score": aggregate.get("score", 0.0),
                "ai_confidence": aggregate.get("confidence", 0.0),
                "news_count": len(news_items),
                "bullish_count": sum(1 for s in sentiments if s > 0),
                "bearish_count": sum(1 for s in sentiments if s < 0),
                "neutral_count": sum(1 for s in sentiments if s == 0),
                "breakdown": aggregate.get("breakdown", {})
            }
        else:
            sentiment_data[symbol] = {
                "name": inst["name"],
                "avg_sentiment": 0,
                "ai_sentiment": "neutral",
                "ai_score": 0.0,
                "ai_confidence": 0.0,
                "news_count": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "breakdown": {}
            }
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
                 html.Span(" (DeepSeek)", style={"fontSize": "10px", "color": COLORS["text_secondary"]})], 
                style={"color": COLORS["text"], "fontSize": "14px", "marginBottom": "10px", "fontWeight": "bold"}),
        html.Div(items, style={"marginBottom": "10px"}),
        html.Div([
            html.Span("← Bearish", style={"color": COLORS["danger"], "fontSize": "10px", "marginRight": "20px"}),
            html.Span("Neutral", style={"color": COLORS["warning"], "fontSize": "10px", "marginRight": "20px"}),
            html.Span("Bullish →", style={"color": COLORS["success"], "fontSize": "10px"})
        ], style={"textAlign": "center", "marginTop": "5px"})
    ], style={"padding": "15px", "backgroundColor": COLORS["surface"], "borderRadius": "8px", "marginBottom": "20px"})


def _build_news_tabs():
    all_news = _get_all_news()
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
    
    all_items = []
    for inst in INSTRUMENTS:
        symbol = inst["symbol"]
        inst_news = news_by_instrument.get(symbol, [])
        all_items.append(html.Div([
            html.H6([f"📈 {inst['name']} ({symbol})", html.Span(f" - {len(inst_news)} articles", style={"color": COLORS["text_secondary"], "fontSize": "11px", "marginLeft": "8px"})],
                   style={"color": COLORS["accent"], "marginBottom": "12px", "fontSize": "14px"}),
            *[_render_news_item(item, show_instrument=True) for item in inst_news]
        ], style={"marginBottom": "20px", "paddingBottom": "15px", "borderBottom": f"1px dashed {COLORS['border']}"}))
    
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
    
    tabs = dbc.Tabs([
        dbc.Tab([html.Div(all_items, style={"padding": "10px"})], label="📰 All News", tab_id="tab-all", label_style={"color": COLORS["text"], "fontSize": "12px"}),
        dbc.Tab([html.Div([dbc.Accordion(by_inst_items, active_item="instrument-XAUUSD", flush=True)], style={"padding": "10px"})],
                label="📈 By Instrument", tab_id="tab-instrument", label_style={"color": COLORS["text"], "fontSize": "12px"}),
        dbc.Tab(html.Div([dbc.Accordion(by_topic_items, active_item="topic-Fed/Rate", flush=True)] if by_topic_items else html.P("No news", style={"color": COLORS["text_secondary"]}), style={"padding": "10px"}),
                label="🏷️ By Topic", tab_id="tab-topic", label_style={"color": COLORS["text"], "fontSize": "12px"}),
    ], active_tab="tab-all", style={"backgroundColor": COLORS["surface"], "borderRadius": "6px", "padding": "10px"})
    
    return tabs, len(all_news), last_updated


tabs_content, article_count, last_updated = _build_news_tabs()
sentiment_data = _calculate_instrument_sentiment()
sentiment_meter = _render_sentiment_meter(sentiment_data)

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
    if pathname != "/news":
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    tabs, count, updated = _build_news_tabs()
    sentiment_data = _calculate_instrument_sentiment()
    sentiment_meter = _render_sentiment_meter(sentiment_data)
    return tabs, f"{count} articles", f"⏰ {updated}", sentiment_meter
