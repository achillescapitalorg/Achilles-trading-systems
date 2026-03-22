"""
News Page - Comprehensive News Aggregation View
"""
import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
from datetime import datetime
import random

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import (
    COLORS, INSTRUMENTS, _fetch_news_for_symbol, _get_source_icon
)


def _aggregate_all_news():
    """Fetch news for all instruments."""
    all_news = []
    
    for inst in INSTRUMENTS:
        symbol = inst["symbol"]
        news_items, _ = _fetch_news_for_symbol(symbol)
        
        for item in news_items:
            item["instrument"] = symbol
            item["instrument_name"] = inst["name"]
            item["instrument_type"] = inst["type"]
        
        all_news.extend(news_items)
    
    random.seed(42)
    random.shuffle(all_news)
    
    return all_news


def _detect_topic(headline, source):
    """Detect the topic category for a news item."""
    headline_lower = headline.lower()
    source_lower = source.lower()
    
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
    else:
        return "General"


def _get_sentiment_badge(sentiment, source):
    """Generate a sentiment badge for the source."""
    if sentiment > 0:
        color = COLORS["success"]
        emoji = "🟢"
    elif sentiment < 0:
        color = COLORS["danger"]
        emoji = "🔴"
    else:
        color = COLORS["warning"]
        emoji = "🟡"
    
    return html.Span([
        html.Span(emoji, style={"marginRight": "4px"}),
        html.Span(source, style={"fontSize": "10px"})
    ], style={
        "display": "inline-block",
        "backgroundColor": f"{color}20",
        "color": color,
        "padding": "2px 8px",
        "borderRadius": "4px",
        "marginRight": "4px",
        "marginBottom": "4px",
        "fontSize": "10px"
    })


def _render_news_item(item, show_instrument=True):
    """Render a single news item with source-specific sentiment."""
    topic = _detect_topic(item.get("headline", ""), item.get("source", ""))
    
    time_str = item.get("time_ago", "Live")
    
    return html.Div([
        html.Div([
            html.A(
                item.get("headline", "No headline"),
                href=item.get("url", "#"),
                target="_blank",
                style={
                    "color": COLORS["text"],
                    "textDecoration": "none",
                    "fontSize": "13px",
                    "fontWeight": "500",
                    "lineHeight": "1.4"
                }
            ),
        ], style={"marginBottom": "8px"}),
        
        html.Div([
            html.Span(
                f"{item.get('source_icon', '📰')} {item.get('source', 'Unknown')}",
                style={
                    "color": COLORS["text_secondary"],
                    "fontSize": "11px",
                    "marginRight": "12px"
                }
            ),
            html.Span(
                f"⏱ {time_str}",
                style={
                    "color": COLORS["text_secondary"],
                    "fontSize": "11px",
                    "marginRight": "12px"
                }
            ),
            html.Span(
                f"📊 {topic}",
                style={
                    "color": COLORS["info"],
                    "fontSize": "11px",
                    "marginRight": "12px"
                }
            ),
            html.Span(
                f"⚡ {item.get('impact', 'MEDIUM')}",
                style={
                    "color": COLORS["warning"] if item.get('impact') == 'MEDIUM' else COLORS["danger"],
                    "fontSize": "11px"
                }
            ),
        ], style={"marginBottom": "6px"}),
        
        html.Div([
            _get_sentiment_badge(item.get("sentiment", 0), item.get("source", "Unknown")),
            html.Span(
                f"📍 {item.get('instrument', '')}" if show_instrument else "",
                style={
                    "color": COLORS["accent"],
                    "fontSize": "10px",
                    "marginLeft": "8px"
                }
            ),
        ]),
        
        html.Hr(style={"borderColor": COLORS["border"], "margin": "12px 0"})
    ], style={"padding": "10px", "backgroundColor": COLORS["surface_light"], "borderRadius": "6px", "marginBottom": "10px"})


def _render_instrument_section(symbol, name, news_items):
    """Render a section for a single instrument with its news."""
    if not news_items:
        return html.Div([
            html.P(f"No news available for {name}", style={"color": COLORS["text_secondary"], "fontSize": "12px", "padding": "10px"})
        ], style={"backgroundColor": COLORS["surface_light"], "borderRadius": "6px", "marginBottom": "10px"})
    
    return html.Div([
        html.H5([
            html.Span(f"📈 {name} ({symbol})", style={"color": COLORS["accent"], "fontSize": "14px"}),
            html.Span(f" - {len(news_items)} articles", style={"color": COLORS["text_secondary"], "fontSize": "11px"})
        ], style={"marginBottom": "12px", "paddingBottom": "8px", "borderBottom": f"1px solid {COLORS['border']}"}),
        
        *[ _render_news_item(item, show_instrument=False) for item in news_items[:5] ]
    ], style={"marginBottom": "20px"})


def _render_topic_section(topic, news_items):
    """Render a section for a topic category."""
    if not news_items:
        return html.Div()
    
    topic_icons = {
        "Fed/Rate": "🏦",
        "Europe": "🇪🇺",
        "UK": "🇬🇧",
        "Japan": "🇯🇵",
        "Asia": "🌏",
        "Geopolitical": "🌍",
        "Crypto": "₿",
        "Metals": "🥇",
        "Equities": "📈",
        "Technical": "📊",
        "Trade": "📦",
        "General": "📰"
    }
    
    return html.Div([
        html.H5([
            html.Span(topic_icons.get(topic, "📰"), style={"marginRight": "8px"}),
            html.Span(topic, style={"color": COLORS["accent"], "fontSize": "14px"}),
            html.Span(f" - {len(news_items)} articles", style={"color": COLORS["text_secondary"], "fontSize": "11px", "marginLeft": "8px"})
        ], style={"marginBottom": "12px", "paddingBottom": "8px", "borderBottom": f"1px solid {COLORS['border']}"}),
        
        *[ _render_news_item(item, show_instrument=True) for item in news_items[:5] ]
    ], style={"marginBottom": "20px"})


def _generate_instrument_summary(symbol, name, news_items):
    """Generate a summary for an instrument based on its news."""
    if not news_items:
        return {
            "headlines": [f"Visit news sources for latest {name} updates"],
            "key_themes": ["Market news"],
            "sentiment": "NEUTRAL"
        }
    
    headlines = [item.get("headline", "") for item in news_items[:3]]
    
    topics = set()
    for item in news_items:
        topic = _detect_topic(item.get("headline", ""), item.get("source", ""))
        topics.add(topic)
    
    sentiment_value = sum(item.get("sentiment", 0) for item in news_items)
    if sentiment_value > 0:
        sentiment = "BULLISH"
    elif sentiment_value < 0:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"
    
    return {
        "headlines": headlines,
        "key_themes": list(topics)[:3],
        "sentiment": sentiment
    }


def get_news_layout():
    """Return the news page layout."""
    
    all_news = _aggregate_all_news()
    
    news_by_instrument = {}
    for inst in INSTRUMENTS:
        symbol = inst["symbol"]
        news_items = [item for item in all_news if item.get("instrument") == symbol]
        news_by_instrument[symbol] = news_items
    
    news_by_topic = {}
    for item in all_news:
        topic = _detect_topic(item.get("headline", ""), item.get("source", ""))
        if topic not in news_by_topic:
            news_by_topic[topic] = []
        news_by_topic[topic].append(item)
    
    topic_order = ["Fed/Rate", "Geopolitical", "Europe", "UK", "Japan", "Asia", "Crypto", "Metals", "Equities", "Technical", "Trade", "General"]
    
    return dbc.Container(fluid=True, style={"backgroundColor": COLORS["background"], "minHeight": "100vh", "padding": "20px"}, children=[
        html.H3([
            html.Span("📰 ", style={"fontSize": "24px"}),
            "News Terminal"
        ], style={"color": COLORS["text"], "marginBottom": "20px", "fontWeight": "bold"}),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("📊 ", style={"color": COLORS["accent"]}),
                    html.Span(f"{len(all_news)} articles across {len(INSTRUMENTS)} instruments", style={"color": COLORS["text_secondary"], "fontSize": "12px"})
                ])
            ], width=12)
        ], className="mb-4"),
        
        dbc.Tabs([
            dbc.Tab([
                html.Div([
                    html.H5("📰 All News (Chronological)", style={"color": COLORS["text"], "marginBottom": "20px", "fontSize": "16px"}),
                    
                    html.Div([
                        html.Div([
                            html.H6([
                                f"📈 {inst['name']} ({inst['symbol']})",
                                html.Span(f" - {len(news_by_instrument.get(inst['symbol'], []))} articles", 
                                         style={"color": COLORS["text_secondary"], "fontSize": "11px", "marginLeft": "8px"})
                            ], style={"color": COLORS["accent"], "marginBottom": "12px", "fontSize": "14px"}),
                            
                            *[_render_news_item(item, show_instrument=True) for item in news_by_instrument.get(inst["symbol"], [])[:3]]
                        ], style={"marginBottom": "20px", "paddingBottom": "15px", "borderBottom": f"1px dashed {COLORS['border']}"})
                        for inst in INSTRUMENTS
                    ])
                ], style={"padding": "10px"})
            ], label="📰 All News", tab_id="tab-all", label_style={"color": COLORS["text"], "fontSize": "12px"}),
            
            dbc.Tab([
                html.Div([
                    html.H5("📈 News by Instrument", style={"color": COLORS["text"], "marginBottom": "20px", "fontSize": "16px"}),
                    
                    dbc.Accordion([
                        dbc.AccordionItem([
                            _render_instrument_section(inst["symbol"], inst["name"], news_by_instrument.get(inst["symbol"], []))
                        ], title=f"{inst['name']} ({inst['symbol']}) - {len(news_by_instrument.get(inst['symbol'], []))} articles", 
                          item_id=f"instrument-{inst['symbol']}",
                          style={"backgroundColor": COLORS["surface"], "color": COLORS["text"], "border": f"1px solid {COLORS['border']}"})
                        for inst in INSTRUMENTS
                    ], active_item="instrument-XAUUSD", flush=True)
                ], style={"padding": "10px"})
            ], label="📈 By Instrument", tab_id="tab-instrument", label_style={"color": COLORS["text"], "fontSize": "12px"}),
            
            dbc.Tab([
                html.Div([
                    html.H5("🏷️ News by Topic", style={"color": COLORS["text"], "marginBottom": "20px", "fontSize": "16px"}),
                    
                    dbc.Accordion([
                        dbc.AccordionItem([
                            _render_topic_section(topic, news_by_topic.get(topic, []))
                        ], title=f"{topic} - {len(news_by_topic.get(topic, []))} articles",
                          item_id=f"topic-{topic}",
                          style={"backgroundColor": COLORS["surface"], "color": COLORS["text"], "border": f"1px solid {COLORS['border']}"})
                        for topic in topic_order if topic in news_by_topic
                    ], active_item="topic-Fed/Rate", flush=True)
                ], style={"padding": "10px"})
            ], label="🏷️ By Topic", tab_id="tab-topic", label_style={"color": COLORS["text"], "fontSize": "12px"}),
        ], id="news-tabs", active_tab="tab-all", style={"backgroundColor": COLORS["surface"], "borderRadius": "6px", "padding": "10px"}),
    ])


layout = get_news_layout()