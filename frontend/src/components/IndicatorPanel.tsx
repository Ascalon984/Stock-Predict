'use client';

import React, { useState } from 'react';
import { PredictionResponse } from '../app/types';
import { Activity, TrendingUp, TrendingDown, Gauge, BarChart2, Waves, Info, Zap, Target, ArrowUp, ArrowDown } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface IndicatorPanelProps {
    data: PredictionResponse;
}

// Get interpretation for RSI
const getRSIInterpretation = (value: number) => {
    if (value > 70) return { text: 'Overbought — potential reversal or pullback', type: 'warning' };
    if (value > 60) return { text: 'Strong momentum, approaching resistance', type: 'bullish' };
    if (value < 30) return { text: 'Oversold — bounce potential', type: 'opportunity' };
    if (value < 40) return { text: 'Weakening momentum, watch for support', type: 'bearish' };
    return { text: 'Neutral range — no strong bias', type: 'neutral' };
};

// Get interpretation for MACD
const getMACDInterpretation = (macd: { value: number; signal_line: number; histogram: number; interpretation: string }) => {
    if (macd.interpretation === 'Bullish') {
        if (macd.histogram > 0 && macd.value > 0) return 'Strong bullish momentum building';
        return 'Bullish crossover — potential upside';
    }
    if (macd.histogram < 0 && macd.value < 0) return 'Strong bearish pressure continues';
    return 'Bearish divergence — weakness expected';
};

// Get interpretation for ATR
const getATRInterpretation = (atrPercent: number) => {
    if (atrPercent > 4) return 'Very high volatility — significant price swings expected';
    if (atrPercent > 3) return 'Elevated volatility — wider stops recommended';
    if (atrPercent > 2) return 'Moderate volatility — normal trading range';
    return 'Low volatility — tight consolidation';
};

// RSI Visualization - Enhanced
const RSIIndicator: React.FC<{ value: number; signal: string }> = ({ value, signal }) => {
    const getZone = () => {
        if (value > 70) return { label: 'Overbought', color: 'text-red-400', zone: 'danger' };
        if (value < 30) return { label: 'Oversold', color: 'text-green-400', zone: 'opportunity' };
        return { label: 'Neutral', color: 'text-yellow-400', zone: 'neutral' };
    };

    const zone = getZone();
    const position = Math.min(Math.max((value / 100) * 100, 0), 100);
    const interpretation = getRSIInterpretation(value);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-4"
        >
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <Gauge className="w-4 h-4 text-blue-400" />
                    <span className="text-sm font-semibold text-neutral-300">RSI (14)</span>
                </div>
                <span className={`text-sm font-bold ${zone.color}`}>{zone.label}</span>
            </div>

            {/* RSI Bar */}
            <div className="relative h-3 bg-neutral-800 rounded-full overflow-hidden mb-2">
                {/* Zone backgrounds */}
                <div className="absolute inset-y-0 left-0 w-[30%] bg-green-500/20" />
                <div className="absolute inset-y-0 left-[30%] right-[30%] bg-yellow-500/10" />
                <div className="absolute inset-y-0 right-0 w-[30%] bg-red-500/20" />

                {/* Marker */}
                <motion.div
                    initial={{ left: '50%' }}
                    animate={{ left: `${position}%` }}
                    transition={{ type: 'spring', stiffness: 100 }}
                    className="absolute top-0 bottom-0 w-1 bg-white rounded-full shadow-lg"
                    style={{ transform: 'translateX(-50%)' }}
                />
            </div>

            {/* Labels */}
            <div className="flex justify-between text-[10px] text-neutral-500 mb-3">
                <span>0</span>
                <span>30</span>
                <span>50</span>
                <span>70</span>
                <span>100</span>
            </div>

            <div className="flex items-center justify-between">
                <span className="text-3xl font-extrabold text-white">{value.toFixed(1)}</span>
            </div>

            {/* Interpretation Summary */}
            <div className={`mt-3 p-2 rounded-lg text-xs ${interpretation.type === 'bullish' || interpretation.type === 'opportunity'
                ? 'bg-green-500/10 text-green-400'
                : interpretation.type === 'bearish' || interpretation.type === 'warning'
                    ? 'bg-red-500/10 text-red-400'
                    : 'bg-neutral-800 text-neutral-400'
                }`}>
                <span className="font-medium">{interpretation.text}</span>
            </div>
        </motion.div>
    );
};

// MACD Visualization - Enhanced
const MACDIndicator: React.FC<{ macd: { value: number; signal_line: number; histogram: number; interpretation: string } }> = ({ macd }) => {
    const isBullish = macd.interpretation === 'Bullish';
    const interpretation = getMACDInterpretation(macd);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-4"
        >
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-purple-400" />
                    <span className="text-sm font-semibold text-neutral-300">MACD</span>
                </div>
                <span className={`text-sm font-bold ${isBullish ? 'text-green-400' : 'text-red-400'}`}>
                    {isBullish ? 'Bullish Momentum' : 'Bearish Pressure'}
                </span>
            </div>

            <div className="grid grid-cols-3 gap-3 text-center">
                <div>
                    <div className="text-xs text-neutral-500 mb-1">MACD</div>
                    <div className={`text-lg font-extrabold ${macd.value >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {macd.value.toFixed(1)}
                    </div>
                </div>
                <div>
                    <div className="text-xs text-neutral-500 mb-1">Signal</div>
                    <div className="text-lg font-bold text-blue-400">
                        {macd.signal_line.toFixed(1)}
                    </div>
                </div>
                <div>
                    <div className="text-xs text-neutral-500 mb-1">Histogram</div>
                    <div className={`text-lg font-extrabold ${macd.histogram >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {macd.histogram.toFixed(1)}
                    </div>
                </div>
            </div>

            {/* Histogram visualization */}
            <HistogramBars macd={macd} />

            {/* Interpretation Summary */}
            <div className={`mt-3 p-2 rounded-lg text-xs ${isBullish ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'
                }`}>
                <span className="font-medium">{interpretation}</span>
            </div>
        </motion.div>
    );
};

const HistogramBars: React.FC<{ macd: { histogram: number } }> = ({ macd }) => {
    const bars = React.useMemo(() => {
        return Array.from({ length: 20 }).map((_, i) => ({
            height: Math.random() * 100,
            isPositive: i > 10 ? macd.histogram > 0 : Math.random() > 0.5
        }));
    }, [macd.histogram]);

    return (
        <div className="mt-3 h-8 flex items-end justify-center gap-0.5">
            {bars.map((bar, i) => (
                <motion.div
                    key={i}
                    initial={{ height: 0 }}
                    animate={{ height: `${20 + bar.height * 0.6}%` }}
                    transition={{ delay: i * 0.02 }}
                    className={`w-1.5 rounded-t ${bar.isPositive ? 'bg-green-500/50' : 'bg-red-500/50'}`}
                />
            ))}
        </div>
    );
};

// Bollinger Bands - Enhanced
const BollingerBands: React.FC<{ bb: { upper: number; middle: number; lower: number; width: number; position: string }; close: number }> = ({ bb, close }) => {
    const range = bb.upper - bb.lower;
    const closePosition = ((close - bb.lower) / range) * 100;

    const getPositionInterpretation = () => {
        if (bb.position === 'Upper') return 'Near upper band — potential resistance';
        if (bb.position === 'Lower') return 'Near lower band — potential support';
        return 'Mid-range — no immediate band pressure';
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-4"
        >
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <Waves className="w-4 h-4 text-cyan-400" />
                    <span className="text-sm font-semibold text-neutral-300">Bollinger Bands</span>
                </div>
                <span className={`text-xs px-2 py-1 rounded font-medium ${bb.position === 'Upper' ? 'bg-red-500/20 text-red-400' :
                    bb.position === 'Lower' ? 'bg-green-500/20 text-green-400' :
                        'bg-neutral-500/20 text-neutral-400'
                    }`}>
                    {bb.position}
                </span>
            </div>

            {/* Visual representation */}
            <div className="relative h-20 my-3">
                {/* Band lines */}
                <div className="absolute top-0 left-0 right-0 h-px bg-red-500/50" />
                <div className="absolute top-1/2 left-0 right-0 h-px bg-neutral-500 -translate-y-1/2" />
                <div className="absolute bottom-0 left-0 right-0 h-px bg-green-500/50" />

                {/* Band fill */}
                <div className="absolute inset-0 bg-gradient-to-b from-red-500/10 via-transparent to-green-500/10" />

                {/* Price position */}
                <motion.div
                    initial={{ top: '50%' }}
                    animate={{ top: `${100 - closePosition}%` }}
                    transition={{ type: 'spring', stiffness: 100 }}
                    className="absolute left-1/2 -translate-x-1/2 -translate-y-1/2"
                >
                    <div className="w-4 h-4 rounded-full bg-blue-500 shadow-lg shadow-blue-500/50 flex items-center justify-center">
                        <div className="w-2 h-2 rounded-full bg-white" />
                    </div>
                </motion.div>

                {/* Labels */}
                <div className="absolute top-0 right-0 text-[10px] text-red-400 font-medium">Rp {bb.upper.toLocaleString()}</div>
                <div className="absolute top-1/2 right-0 -translate-y-1/2 text-[10px] text-neutral-400">Rp {bb.middle.toLocaleString()}</div>
                <div className="absolute bottom-0 right-0 text-[10px] text-green-400 font-medium">Rp {bb.lower.toLocaleString()}</div>
            </div>

            <div className="text-center text-xs text-neutral-500 mb-2">
                Band Width: <span className="font-bold text-neutral-300">{bb.width.toFixed(2)}%</span>
            </div>

            {/* Interpretation */}
            <div className="p-2 rounded-lg text-xs bg-neutral-800 text-neutral-400">
                <span className="font-medium">{getPositionInterpretation()}</span>
            </div>
        </motion.div>
    );
};

export const IndicatorPanel: React.FC<IndicatorPanelProps> = ({ data }) => {
    const { indicators } = data;

    return (
        <div className="space-y-4">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <BarChart2 className="w-5 h-5 text-blue-400" />
                Technical Indicators
            </h3>

            <div className="grid gap-4">
                {/* RSI */}
                <RSIIndicator value={indicators.rsi.value} signal={indicators.rsi.signal} />

                {/* MACD */}
                <MACDIndicator macd={indicators.macd} />

                {/* Bollinger Bands */}
                <BollingerBands bb={indicators.bollinger_bands} close={indicators.close} />
            </div>

            {/* Moving Averages Summary */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-4"
            >
                <div className="flex items-center gap-2 mb-3">
                    <TrendingUp className="w-4 h-4 text-amber-400" />
                    <span className="text-sm font-semibold text-neutral-300">Moving Averages</span>
                    <span className={`ml-auto text-sm font-bold ${indicators.moving_averages.trend === 'Bullish' ? 'text-green-400' : 'text-red-400'
                        }`}>
                        {indicators.moving_averages.trend === 'Bullish' ? 'Bullish Momentum' : 'Bearish Pressure'}
                    </span>
                </div>

                <div className="space-y-2">
                    {[
                        { label: 'SMA 20', value: indicators.moving_averages.sma_20, color: 'bg-amber-500' },
                        { label: 'SMA 50', value: indicators.moving_averages.sma_50, color: 'bg-purple-500' },
                        { label: 'SMA 200', value: indicators.moving_averages.sma_200, color: 'bg-cyan-500' },
                        { label: 'EMA 20', value: indicators.moving_averages.ema_20, color: 'bg-pink-500' },
                    ].map((ma) => {
                        const isAbove = indicators.close > ma.value;
                        return (
                            <div key={ma.label} className="flex items-center justify-between text-sm">
                                <div className="flex items-center gap-2">
                                    <div className={`w-2 h-2 rounded-full ${ma.color}`} />
                                    <span className="text-neutral-400">{ma.label}</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="font-medium text-white">Rp {ma.value.toLocaleString()}</span>
                                    {isAbove ? (
                                        <ArrowUp className="w-3 h-3 text-green-400" />
                                    ) : (
                                        <ArrowDown className="w-3 h-3 text-red-400" />
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>

                {/* Price vs MA comparison */}
                <div className="mt-3 pt-3 border-t border-neutral-800">
                    <div className="flex items-center justify-between text-sm">
                        <span className="text-neutral-500">Current Price</span>
                        <span className="font-extrabold text-white">Rp {indicators.close.toLocaleString()}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm mt-1">
                        <span className="text-neutral-500">vs SMA 50</span>
                        <span className={`font-bold ${indicators.close > indicators.moving_averages.sma_50 ? 'text-green-400' : 'text-red-400'
                            }`}>
                            {((indicators.close - indicators.moving_averages.sma_50) / indicators.moving_averages.sma_50 * 100).toFixed(2)}%
                        </span>
                    </div>
                </div>

                {/* MA Interpretation */}
                <div className={`mt-3 p-2 rounded-lg text-xs ${indicators.moving_averages.trend === 'Bullish'
                    ? 'bg-green-500/10 text-green-400'
                    : 'bg-red-500/10 text-red-400'
                    }`}>
                    <span className="font-medium">
                        {indicators.close > indicators.moving_averages.sma_200
                            ? 'Above MA200 — long-term uptrend intact'
                            : 'Below MA200 — long-term downtrend pressure'}
                    </span>
                </div>
            </motion.div>

            {/* Volatility - Enhanced */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-4"
            >
                <div className="flex items-center gap-2 mb-3">
                    <Activity className="w-4 h-4 text-orange-400" />
                    <span className="text-sm font-semibold text-neutral-300">Volatility (ATR)</span>
                </div>

                <div className="flex items-center justify-between">
                    <div>
                        <div className="text-2xl font-extrabold text-white">Rp {indicators.volatility.atr.toLocaleString()}</div>
                        <div className="text-sm text-neutral-500">{indicators.volatility.atr_percent.toFixed(2)}% of price</div>
                    </div>
                    <div className={`px-3 py-1.5 rounded-lg text-sm font-bold ${indicators.volatility.atr_percent > 3 ? 'bg-red-500/20 text-red-400' :
                        indicators.volatility.atr_percent > 2 ? 'bg-yellow-500/20 text-yellow-400' :
                            'bg-green-500/20 text-green-400'
                        }`}>
                        {indicators.volatility.atr_percent > 3 ? 'High' :
                            indicators.volatility.atr_percent > 2 ? 'Moderate' : 'Low'}
                    </div>
                </div>

                {/* ATR Interpretation */}
                <div className={`mt-3 p-2 rounded-lg text-xs ${indicators.volatility.atr_percent > 3
                    ? 'bg-orange-500/10 text-orange-400'
                    : 'bg-neutral-800 text-neutral-400'
                    }`}>
                    <span className="font-medium">{getATRInterpretation(indicators.volatility.atr_percent)}</span>
                </div>
            </motion.div>

            {/* OBV (On-Balance Volume) - New */}
            {indicators.obv && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.55 }}
                    className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-4"
                >
                    <div className="flex items-center gap-2 mb-3">
                        <BarChart2 className="w-4 h-4 text-cyan-400" />
                        <span className="text-sm font-semibold text-neutral-300">On-Balance Volume (OBV)</span>
                    </div>

                    <div className="flex items-center justify-between">
                        <div>
                            <div className="text-2xl font-extrabold text-white">
                                {(indicators.obv.value / 1000000).toFixed(2)}M
                            </div>
                            <div className="text-sm text-neutral-500">
                                {indicators.obv.change_percent >= 0 ? '+' : ''}{indicators.obv.change_percent.toFixed(1)}% trend
                            </div>
                        </div>
                        <div className={`px-3 py-1.5 rounded-lg text-sm font-bold ${indicators.obv.trend === 'Accumulation' ? 'bg-green-500/20 text-green-400' :
                                indicators.obv.trend === 'Distribution' ? 'bg-red-500/20 text-red-400' :
                                    'bg-neutral-800 text-neutral-400'
                            }`}>
                            {indicators.obv.trend}
                        </div>
                    </div>

                    <div className={`mt-3 p-2 rounded-lg text-xs ${indicators.obv.trend === 'Accumulation' ? 'bg-green-500/10 text-green-400' :
                            indicators.obv.trend === 'Distribution' ? 'bg-red-500/10 text-red-400' :
                                'bg-neutral-800 text-neutral-400'
                        }`}>
                        <span className="font-medium">
                            {indicators.obv.trend === 'Accumulation'
                                ? 'Institutional buying pressure detected — bullish signal'
                                : indicators.obv.trend === 'Distribution'
                                    ? 'Selling pressure increasing — bearish warning'
                                    : 'Volume neutral — no clear accumulation or distribution'}
                        </span>
                    </div>
                </motion.div>
            )}

            {/* Summary Panel */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
                className="bg-gradient-to-br from-blue-500/10 to-purple-500/10 border border-blue-500/20 rounded-xl p-4"
            >
                <div className="flex items-center gap-2 mb-3">
                    <Target className="w-4 h-4 text-blue-400" />
                    <span className="text-sm font-semibold text-neutral-300">Technical Summary</span>
                </div>
                <div className="space-y-2 text-sm text-neutral-400">
                    <p>
                        <span className="font-medium text-white">RSI</span> indicates {
                            indicators.rsi.value > 70 ? 'overbought conditions with potential reversal risk' :
                                indicators.rsi.value < 30 ? 'oversold conditions with bounce potential' :
                                    'neutral momentum with no extreme readings'
                        }.
                    </p>
                    <p>
                        <span className="font-medium text-white">MACD</span> shows {
                            indicators.macd.interpretation === 'Bullish' ? 'bullish momentum with positive trend strength' :
                                'bearish pressure with weakening trend'
                        }{indicators.macd.crossover_detected && indicators.macd.crossover_type
                            ? ` — ${indicators.macd.crossover_type} crossover just detected!`
                            : ''}.
                    </p>
                    <p>
                        <span className="font-medium text-white">Volatility</span> is {
                            indicators.volatility.regime || (
                                indicators.volatility.atr_percent > 3 ? 'High' :
                                    indicators.volatility.atr_percent > 2 ? 'Moderate' : 'Low'
                            )
                        } — {
                            indicators.volatility.atr_percent > 3 ? 'expect wider price swings' :
                                indicators.volatility.atr_percent > 2 ? 'normal trading conditions' :
                                    'tight consolidation phase'
                        }.
                    </p>
                    {indicators.obv && (
                        <p>
                            <span className="font-medium text-white">Volume Flow</span> shows {
                                indicators.obv.trend === 'Accumulation' ? 'institutional accumulation — smart money buying' :
                                    indicators.obv.trend === 'Distribution' ? 'distribution pattern — potential selling pressure' :
                                        'balanced activity — no clear directional bias'
                            }.
                        </p>
                    )}
                </div>
            </motion.div>
        </div>
    );
};
