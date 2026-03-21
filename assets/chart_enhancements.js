/**
 * Enhanced Chart Interactions for Trading Terminal
 * =================================================
 * Adds TradingView-like drawing tools and interactions
 */

(function() {
    'use strict';

    // Wait for Dash/Plotly to be ready
    document.addEventListener('DOMContentLoaded', function() {
        initChartEnhancements();
    });

    function initChartEnhancements() {
        // Wait for the chart to render
        setTimeout(function() {
            setupPlotlyDrawingTools();
            setupChartKeyboardShortcuts();
            setupCrosshairEnhancements();
        }, 2000);
    }

    function setupPlotlyDrawingTools() {
        // Enable drawing tools in modebar
        const chartElement = document.querySelector('[id="price-chart"]');
        if (!chartElement) return;

        // Get the plotly graph div
        const plotDiv = document.querySelector('.js-plotly-plot');
        if (!plotDiv) return;

        // Add custom drawing mode buttons
        const modeBar = document.querySelector('.modebar');
        if (modeBar) {
            // Add drawline button if not present
            if (!modeBar.querySelector('[data-title="Straight Line"]')) {
                addDrawingButton(modeBar, 'Straight Line', drawLineIcon, activateLineMode);
            }
            
            // Add drawrect button if not present
            if (!modeBar.querySelector('[data-title="Rectangle"]')) {
                addDrawingButton(modeBar, 'Rectangle', drawRectIcon, activateRectMode);
            }
        }
    }

    function addDrawingButton(container, title, icon, callback) {
        const btn = document.createElement('button');
        btn.className = 'modebar-btn';
        btn.setAttribute('data-title', title);
        btn.innerHTML = icon;
        btn.addEventListener('click', callback);
        container.insertBefore(btn, container.firstChild);
    }

    const drawLineIcon = '<svg height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="5" y1="19" x2="19" y2="5"/></svg>';
    const drawRectIcon = '<svg height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/></svg>';

    function activateLineMode() {
        const plotDiv = document.querySelector('.js-plotly-plot');
        if (plotDiv && plotDiv._fullLayout) {
            Plotly.relayout(plotDiv, {dragmode: 'drawline'});
        }
    }

    function activateRectMode() {
        const plotDiv = document.querySelector('.js-plotly-plot');
        if (plotDiv && plotDiv._fullLayout) {
            Plotly.relayout(plotDiv, {dragmode: 'drawrect'});
        }
    }

    function setupChartKeyboardShortcuts() {
        document.addEventListener('keydown', function(e) {
            // Only trigger if chart is focused/hovered
            const chartHovered = e.target.closest('.js-plotly-plot');
            if (!chartHovered) return;

            const plotDiv = document.querySelector('.js-plotly-plot');
            if (!plotDiv) return;

            switch(e.key) {
                case 'Escape':
                    // Exit drawing mode
                    Plotly.relayout(plotDiv, {dragmode: 'pan'});
                    break;
                case 'z':
                    if (e.ctrlKey || e.metaKey) {
                        // Undo (Ctrl+Z)
                        e.preventDefault();
                        // Plotly doesn't have native undo, but we can reset view
                        Plotly.relayout(plotDiv, {'xaxis.autorange': true, 'yaxis.autorange': true});
                    }
                    break;
                case '0':
                    // Reset zoom to fit all data
                    Plotly.relayout(plotDiv, {'xaxis.autorange': true, 'yaxis.autorange': true});
                    break;
                case '+':
                case '=':
                    // Zoom in
                    zoomChart(plotDiv, 0.8);
                    break;
                case '-':
                    // Zoom out
                    zoomChart(plotDiv, 1.25);
                    break;
            }
        });
    }

    function zoomChart(plotDiv, factor) {
        const layout = plotDiv._fullLayout;
        if (!layout.xaxis.range || !layout.yaxis.range) return;

        const xRange = layout.xaxis.range;
        const yRange = layout.yaxis.range;

        const xCenter = (xRange[0] + xRange[1]) / 2;
        const yCenter = (yRange[0] + yRange[1]) / 2;

        const xSpan = (xRange[1] - xRange[0]) * factor / 2;
        const ySpan = (yRange[1] - yRange[0]) * factor / 2;

        Plotly.relayout(plotDiv, {
            'xaxis.range': [xCenter - xSpan, xCenter + xSpan],
            'yaxis.range': [yCenter - ySpan, yCenter + ySpan]
        });
    }

    function setupCrosshairEnhancements() {
        // Add price line indicator
        const chartElement = document.querySelector('[id="price-chart"]');
        if (!chartElement) return;

        const plotDiv = document.querySelector('.js-plotly-plot');
        if (!plotDiv) return;

        // Subscribe to hover events
        plotDiv.on('plotly_hover', function(data) {
            // Add custom hover info
            const point = data.points[0];
            if (point) {
                showPriceTooltip(point);
            }
        });

        plotDiv.on('plotly_unhover', function() {
            hidePriceTooltip();
        });
    }

    function showPriceTooltip(point) {
        let tooltip = document.getElementById('chart-price-tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'chart-price-tooltip';
            tooltip.className = 'chart-tooltip';
            tooltip.style.cssText = `
                position: absolute;
                background: rgba(30, 34, 45, 0.95);
                color: #d1d4dc;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 11px;
                font-family: 'Roboto Mono', monospace;
                pointer-events: none;
                z-index: 1000;
                border: 1px solid #2a2e39;
            `;
            document.body.appendChild(tooltip);
        }

        const price = typeof point.y === 'number' ? point.y.toFixed(2) : point.y;
        const time = point.x;

        tooltip.innerHTML = `
            <div style="margin-bottom: 4px; color: #787b86;">Price</div>
            <div style="font-size: 14px; font-weight: bold;">${price}</div>
            <div style="margin-top: 4px; color: #787b86;">Time</div>
            <div>${new Date(time).toLocaleString()}</div>
        `;

        tooltip.style.display = 'block';
    }

    function hidePriceTooltip() {
        const tooltip = document.getElementById('chart-price-tooltip');
        if (tooltip) {
            tooltip.style.display = 'none';
        }
    }

    // Mouse move handler for tooltip positioning
    document.addEventListener('mousemove', function(e) {
        const tooltip = document.getElementById('chart-price-tooltip');
        if (tooltip && tooltip.style.display !== 'none') {
            tooltip.style.left = (e.clientX + 15) + 'px';
            tooltip.style.top = (e.clientY + 15) + 'px';
        }
    });

    // Auto-refresh indicator
    function showRefreshIndicator() {
        const chartCard = document.querySelector('[id="price-chart"]')?.closest('.card');
        if (chartCard) {
            const header = chartCard.querySelector('.card-header');
            if (header && !header.querySelector('.refresh-dot')) {
                const dot = document.createElement('span');
                dot.className = 'refresh-dot';
                dot.style.cssText = `
                    display: inline-block;
                    width: 8px;
                    height: 8px;
                    background: #00ff88;
                    border-radius: 50%;
                    margin-left: 8px;
                    animation: pulse 2s infinite;
                `;
                header.appendChild(dot);

                // Add CSS animation
                if (!document.getElementById('refresh-animation-style')) {
                    const style = document.createElement('style');
                    style.id = 'refresh-animation-style';
                    style.textContent = `
                        @keyframes pulse {
                            0%, 100% { opacity: 1; }
                            50% { opacity: 0.3; }
                        }
                    `;
                    document.head.appendChild(style);
                }
            }
        }
    }

    // Show indicator after initial load
    setTimeout(showRefreshIndicator, 3000);

})();
