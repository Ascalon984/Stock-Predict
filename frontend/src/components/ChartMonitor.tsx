'use client';

import dynamic from 'next/dynamic';
import React, { useState, useMemo } from 'react';
import { PredictionResponse } from '../app/types';
import { CandlestickChart, LineChart, BarChart3, Layers, Clock, AlertTriangle } from 'lucide-react';
import { motion } from 'framer-motion';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export type ChartRange = '1D' | '5D' | '1M' | '6M' | 'YTD' | '1Y' | '5Y';

interface ChartMonitorProps {
    data: PredictionResponse;
    activeRange: string;
    onRangeChange: (range: string) => void;
}

const CHART_RANGES: { id: ChartRange; label: string }[] = [
    { id: '1D', label: '1D' },
    { id: '5D', label: '5D' },
    { id: '1M', label: '1M' },
    { id: '6M', label: '6M' },
    { id: 'YTD', label: 'YTD' },
    { id: '1Y', label: '1Y' },
    { id: '5Y', label: '5Y' },
];

export const ChartMonitor: React.FC<ChartMonitorProps> = ({ data, activeRange, onRangeChange }) => {
    const [chartMode, setChartMode] = useState<'candlestick' | 'line'>('candlestick');
    const [showBands, setShowBands] = useState(true);
    const [showMA, setShowMA] = useState(true);

    const histDates = data.historical_data.dates;
    const histClose = data.historical_data.close;
    const histOpen = data.historical_data.open;
    const histHigh = data.historical_data.high;
    const histLow = data.historical_data.low;
    const lastPrice = histClose[histClose.length - 1];

    // Determine if intraday logic needs to be applied
    const isIntradayView = ['1D', '5D'].includes(activeRange);

    // Smart Forecast Selection
    // Dynamically choose appropriate forecast horizon based on selected time range
    const selectedForecast = useMemo(() => {
        if (!data.forecasts || data.forecasts.length === 0) return null;

        // Find specific horizons
        const f30 = data.forecasts.find(f => f.horizon === 30);
        const f7 = data.forecasts.find(f => f.horizon === 7);
        const f1 = data.forecasts.find(f => f.horizon === 1);

        // Select base forecast based on range
        let baseForecast = null;
        if (['6M', 'YTD', '1Y', '5Y'].includes(activeRange)) {
            baseForecast = f30 || f7 || f1;
        } else if (activeRange === '1M') {
            baseForecast = f7 || f30 || f1;
        }

        // Rebase forecast to match current real-time price precisely
        // This prevents the "jumping line" visual when real-time price drifts from cached forecast base
        if (baseForecast && lastPrice && baseForecast.last_price) {
            const ratio = lastPrice / baseForecast.last_price;
            return {
                ...baseForecast,
                forecast: baseForecast.forecast.map(v => v * ratio),
                confidence_upper: baseForecast.confidence_upper.map(v => v * ratio),
                confidence_lower: baseForecast.confidence_lower.map(v => v * ratio)
            };
        }

        return baseForecast;
    }, [data.forecasts, activeRange, lastPrice]);

    // Construct forecast dates
    const forecastDates = useMemo(() => {
        if (!selectedForecast || !histDates.length) return [];
        const lastDateStr = histDates[histDates.length - 1];
        // Handle ISO string correctly
        const lastDate = new Date(lastDateStr.includes('T') ? lastDateStr : lastDateStr + 'T00:00:00');

        return selectedForecast.forecast.map((_, i) => {
            const d = new Date(lastDate);
            d.setDate(d.getDate() + i + 1);
            return d.toISOString().split('T')[0];
        });
    }, [selectedForecast, histDates]);

    // Calculate moving averages
    const calculateMA = (period: number) => {
        return histClose.map((_, i) => {
            if (i < period - 1) return null;
            const slice = histClose.slice(i - period + 1, i + 1);
            return slice.reduce((a, b) => a + b, 0) / period;
        });
    };

    const ma20 = calculateMA(20);
    const ma50 = calculateMA(50);

    // Build chart traces
    const traces: Plotly.Data[] = [];

    // 1. Main Price Chart
    if (chartMode === 'candlestick') {
        traces.push({
            x: histDates,
            open: histOpen,
            high: histHigh,
            low: histLow,
            close: histClose,
            type: 'candlestick',
            name: 'Price',
            increasing: { line: { color: '#22c55e', width: 1 }, fillcolor: '#22c55e' },
            decreasing: { line: { color: '#ef4444', width: 1 }, fillcolor: '#ef4444' },
            hoverlabel: { bgcolor: '#1a1a1a' }
        } as Plotly.Data);
    } else {
        traces.push({
            x: histDates,
            y: histClose,
            type: 'scatter',
            mode: 'lines',
            name: 'Close Price',
            line: { color: '#3b82f6', width: 2 },
            fill: 'tozeroy', // Nice gradient effect
            fillcolor: 'rgba(59, 130, 246, 0.1)',
            connectgaps: true // Fix gaps in line chart
        } as Plotly.Data);
    }

    // 2. Moving Averages
    if (showMA) {
        traces.push({
            x: histDates,
            y: ma20,
            type: 'scatter',
            mode: 'lines',
            name: 'MA 20',
            line: { color: '#f59e0b', width: 1.5 },
            opacity: 0.8,
            connectgaps: true
        } as Plotly.Data);

        traces.push({
            x: histDates,
            y: ma50,
            type: 'scatter',
            mode: 'lines',
            name: 'MA 50',
            line: { color: '#8b5cf6', width: 1.5 },
            opacity: 0.8,
            connectgaps: true
        } as Plotly.Data);
    }

    // 3. Bollinger Bands (only in Line mode to avoid clutter)
    if (showBands && chartMode === 'line') {
        const bbUpper = histClose.map((_, i) => {
            if (i < 19) return null;
            const slice = histClose.slice(i - 19, i + 1);
            const mean = slice.reduce((a, b) => a + b, 0) / 20;
            const std = Math.sqrt(slice.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / 20);
            return mean + 2 * std;
        });

        const bbLower = histClose.map((_, i) => {
            if (i < 19) return null;
            const slice = histClose.slice(i - 19, i + 1);
            const mean = slice.reduce((a, b) => a + b, 0) / 20;
            const std = Math.sqrt(slice.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / 20);
            return mean - 2 * std;
        });

        traces.push({
            x: histDates,
            y: bbUpper,
            type: 'scatter',
            mode: 'lines',
            name: 'BB Upper',
            line: { width: 0 },
            showlegend: false,
            hoverinfo: 'skip',
            connectgaps: true
        } as Plotly.Data);

        traces.push({
            x: histDates,
            y: bbLower,
            type: 'scatter',
            mode: 'lines',
            name: 'Bollinger Bands',
            fill: 'tonexty',
            fillcolor: 'rgba(139, 92, 246, 0.05)',
            line: { width: 0 },
            hoverinfo: 'skip',
            connectgaps: true
        } as Plotly.Data);
    }

    // 4. Forecast Lines
    if (selectedForecast && !isIntradayView) {
        // Connect history to forecast
        const xConnect = [histDates[histDates.length - 1], ...forecastDates];
        const yConnect = [lastPrice, ...selectedForecast.forecast];

        traces.push({
            x: xConnect,
            y: yConnect,
            type: 'scatter',
            mode: 'lines+markers',
            name: `AI Forecast (${selectedForecast.horizon}D)`,
            line: { color: '#10b981', width: 2, dash: 'dash' },
            marker: { size: 3, color: '#10b981', symbol: 'circle' },
            connectgaps: true
        } as Plotly.Data);

        // Confidence interval
        // Ensure confidence bounds also start at last price for continuity
        const upperConnect = [lastPrice, ...selectedForecast.confidence_upper];
        const lowerConnect = [lastPrice, ...selectedForecast.confidence_lower];

        traces.push({
            x: xConnect,
            y: upperConnect,
            type: 'scatter',
            mode: 'lines',
            name: 'Upper Bound',
            line: { width: 0 },
            showlegend: false,
            hoverinfo: 'skip',
            connectgaps: true
        } as Plotly.Data);

        traces.push({
            x: xConnect,
            y: lowerConnect,
            type: 'scatter',
            mode: 'lines',
            name: '95% Confidence',
            fill: 'tonexty',
            fillcolor: 'rgba(16, 185, 129, 0.06)',
            line: { width: 0 },
            hoverinfo: 'skip',
            connectgaps: true
        } as Plotly.Data);
    }

    // Formatting configurations based on range
    const getXAxisFormat = () => {
        if (['1D', '5D'].includes(activeRange)) return '%I:%M %p'; // 10:30 AM
        if (['1M', '6M', 'YTD'].includes(activeRange)) return '%d %b'; // 05 Feb
        return '%b %Y'; // Feb 2026
    };

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-4"
        >
            {/* Chart Controls Bar */}
            <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 p-2 bg-neutral-900/60 backdrop-blur-md rounded-2xl border border-neutral-800/60 shadow-lg">
                {/* Left: Chart Type */}
                <div className="flex items-center gap-1 p-1 bg-neutral-950/50 rounded-xl border border-neutral-800">
                    <button
                        onClick={() => setChartMode('candlestick')}
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 ${chartMode === 'candlestick'
                            ? 'bg-neutral-800 text-white shadow-sm ring-1 ring-white/5'
                            : 'text-neutral-500 hover:text-neutral-300 hover:bg-neutral-800/50'
                            }`}
                    >
                        <CandlestickChart className="w-4 h-4" />
                        <span className="hidden sm:inline">Candles</span>
                    </button>
                    <button
                        onClick={() => setChartMode('line')}
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 ${chartMode === 'line'
                            ? 'bg-neutral-800 text-white shadow-sm ring-1 ring-white/5'
                            : 'text-neutral-500 hover:text-neutral-300 hover:bg-neutral-800/50'
                            }`}
                    >
                        <LineChart className="w-4 h-4" />
                        <span className="hidden sm:inline">Line</span>
                    </button>
                </div>

                {/* Center: Range Selector */}
                <div className="flex items-center gap-1 p-1 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-xl border border-blue-500/20">
                    <Clock className="w-4 h-4 text-blue-400 ml-2" />
                    {CHART_RANGES.map((range) => (
                        <button
                            key={range.id}
                            onClick={() => onRangeChange(range.id)}
                            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all duration-200 ${activeRange === range.id
                                ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/30'
                                : 'text-neutral-400 hover:text-white hover:bg-neutral-800'
                                }`}
                        >
                            {range.label}
                        </button>
                    ))}
                </div>

                {/* Right: Indicators */}
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => setShowMA(!showMA)}
                        className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium transition-all ${showMA ? 'text-amber-400 bg-amber-500/10' : 'text-neutral-500 hover:bg-neutral-800'
                            }`}
                    >
                        <BarChart3 className="w-3.5 h-3.5" />
                        MA
                    </button>
                    <button
                        onClick={() => setShowBands(!showBands)}
                        className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium transition-all ${showBands ? 'text-purple-400 bg-purple-500/10' : 'text-neutral-500 hover:bg-neutral-800'
                            }`}
                    >
                        <Layers className="w-3.5 h-3.5" />
                        BB
                    </button>
                </div>
            </div>

            {/* Intraday Info Banner */}
            {isIntradayView && (
                <div className="flex items-center gap-3 p-3 bg-blue-500/10 border border-blue-500/20 rounded-xl">
                    <Clock className="w-5 h-5 text-blue-400 shrink-0" />
                    <div className="flex-1">
                        <span className="text-sm text-blue-300 font-medium">
                            Intraday View ({activeRange})
                        </span>
                        <span className="text-xs text-blue-400/70 ml-2">
                            • Real-time Market Data • Gaps indicate non-trading hours
                        </span>
                    </div>
                </div>
            )}

            {/* Main Chart Container */}
            <div className="relative group">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-2xl blur opacity-30 group-hover:opacity-50 transition duration-500"></div>

                <div className="relative w-full h-[500px] bg-[#0f0f13] rounded-2xl border border-neutral-800 p-1 overflow-hidden shadow-2xl">
                    <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 pointer-events-none"></div>
                    <div className="relative w-full h-full p-2">
                        <Plot
                            data={traces}
                            layout={{
                                autosize: true,
                                paper_bgcolor: 'rgba(0,0,0,0)',
                                plot_bgcolor: 'rgba(0,0,0,0)',
                                font: { color: '#737373', family: 'Inter, system-ui, sans-serif', size: 11 },
                                xaxis: {
                                    gridcolor: '#333333',
                                    showgrid: true,
                                    minor: {
                                        showgrid: true,
                                        gridcolor: '#1f1f1f',
                                        gridwidth: 0.5
                                    },
                                    zeroline: false,
                                    rangeslider: { visible: false },
                                    showspikes: true,
                                    spikethickness: 1,
                                    spikecolor: '#404040',
                                    spikemode: 'across',
                                    tickformat: getXAxisFormat(),
                                    nticks: 10,
                                    hoverformat: isIntradayView ? '%d %b, %I:%M %p' : '%d %b %Y',
                                    type: 'date',
                                    // Range Breaks to hide non-trading hours and weekends
                                    rangebreaks: isIntradayView ? [
                                        { pattern: 'hour', bounds: [16, 9] }, // Hide 16:00 - 09:00 (Check timezone if needed)
                                        { bounds: ["sat", "mon"] } // Hide weekends
                                        // Note: Lunch break gap left visible to be accurate to data, can add if requested
                                    ] : [
                                        { bounds: ["sat", "mon"] }
                                    ]
                                },
                                yaxis: {
                                    gridcolor: '#333333',
                                    showgrid: true,
                                    minor: {
                                        showgrid: true,
                                        gridcolor: '#1f1f1f',
                                        gridwidth: 0.5
                                    },
                                    zeroline: false,
                                    tickprefix: 'Rp ',
                                    tickformat: ',.0f',
                                    side: 'right',
                                    showspikes: true,
                                    spikethickness: 1,
                                    spikecolor: '#404040',
                                },
                                margin: { l: 20, r: 80, t: 20, b: 40 },
                                showlegend: true,
                                legend: {
                                    orientation: 'h',
                                    y: 1.02,
                                    x: 0,
                                    bgcolor: 'rgba(15, 15, 20, 0.85)',
                                    bordercolor: 'rgba(64, 64, 64, 0.3)',
                                    borderwidth: 1,
                                    font: { size: 10, color: '#a3a3a3' },
                                },
                                hovermode: 'x unified',
                                hoverlabel: {
                                    bgcolor: '#171717',
                                    bordercolor: '#404040',
                                },
                                dragmode: 'pan',
                            }}
                            config={{
                                displayModeBar: false,
                                responsive: true,
                                scrollZoom: true,
                            }}
                            useResizeHandler={true}
                            style={{ width: '100%', height: '100%' }}
                        />
                    </div>
                </div>
            </div>

            {/* Volume Chart */}
            <div className="w-full h-[180px] bg-neutral-900/30 rounded-xl border border-neutral-800/50 p-4 backdrop-blur-sm">
                <div className="flex items-center justify-between mb-2">
                    <h4 className="text-xs font-semibold text-neutral-500 uppercase tracking-widest">Volume Analysis</h4>
                </div>
                <Plot
                    data={[
                        {
                            x: histDates.slice(-60),
                            y: data.historical_data.volume.slice(-60),
                            type: 'bar',
                            marker: {
                                color: data.historical_data.close.slice(-60).map((c, i) =>
                                    i === 0 ? '#3b82f6' : c > data.historical_data.close.slice(-60)[i - 1] ? '#22c55e' : '#ef4444'
                                ),
                                opacity: 0.8
                            },
                        },
                    ]}
                    layout={{
                        autosize: true,
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: '#525252', size: 10 },
                        xaxis: { showgrid: false, showticklabels: false, type: 'date', rangebreaks: isIntradayView ? [{ bounds: ["sat", "mon"] }] : [] },
                        yaxis: {
                            showgrid: false,
                            tickformat: '.2s',
                            side: 'right',
                        },
                        margin: { l: 0, r: 50, t: 10, b: 20 }, // Increased bottom margin
                        showlegend: false,
                        bargap: 0.2,
                        hovermode: 'x',
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '100%' }}
                />
            </div>
        </motion.div>
    );
};
