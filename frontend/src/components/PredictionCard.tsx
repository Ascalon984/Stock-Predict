'use client';

import React, { useState } from 'react';
import { PredictionResponse, Sentiment } from '../app/types';
import { ArrowUp, ArrowDown, Minus, AlertTriangle, TrendingUp, TrendingDown, Activity, Brain, Cpu, Info, Zap, Target, BarChart3, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface PredictionCardProps {
    data: PredictionResponse;
}

// Get market context badge
const getMarketContext = (sentiment: Sentiment, trend: { short_term: string; long_term: string }) => {
    const { rsi, macd, moving_averages, volatility } = sentiment.component_scores as any || {};
    const hasHighVolatility = sentiment.rationale?.some(r => r.toLowerCase().includes('volatil')) || false;
    const isMeanReversion = sentiment.rationale?.some(r => r.toLowerCase().includes('overbought') || r.toLowerCase().includes('oversold')) || false;
    const isMomentum = trend.short_term === trend.long_term && trend.short_term !== 'Sideways';

    if (hasHighVolatility) return { label: 'High Volatility', color: 'bg-orange-500/20 text-orange-400 border-orange-500/30', bias: 'Caution Advised' };
    if (isMeanReversion) return { label: 'Mean Reversion', color: 'bg-purple-500/20 text-purple-400 border-purple-500/30', bias: 'Counter-trend' };
    if (isMomentum) return { label: 'Momentum', color: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30', bias: 'Trend-following' };
    return { label: 'Consolidation', color: 'bg-neutral-500/20 text-neutral-400 border-neutral-500/30', bias: 'Range-bound' };
};

// Professional sentiment labels
const getSentimentLabel = (level: string) => {
    switch (level) {
        case 'SANGAT_BULLISH': return 'Strong Bullish Momentum';
        case 'BULLISH': return 'Bullish Momentum';
        case 'NEUTRAL': return 'Neutral Bias';
        case 'BEARISH': return 'Bearish Pressure';
        case 'SANGAT_BEARISH': return 'Strong Bearish Pressure';
        default: return 'Neutral Bias';
    }
};

// Sentiment Gauge Component - Enhanced
const SentimentGauge: React.FC<{ sentiment: Sentiment; trend: { short_term: string; long_term: string } }> = ({ sentiment, trend }) => {
    const getGaugeColor = () => {
        switch (sentiment.level) {
            case 'SANGAT_BULLISH': return { from: '#22c55e', to: '#16a34a', bg: 'from-green-500/20 to-green-600/20' };
            case 'BULLISH': return { from: '#4ade80', to: '#22c55e', bg: 'from-green-400/20 to-green-500/20' };
            case 'NEUTRAL': return { from: '#a3a3a3', to: '#737373', bg: 'from-neutral-400/20 to-neutral-500/20' };
            case 'BEARISH': return { from: '#f87171', to: '#ef4444', bg: 'from-red-400/20 to-red-500/20' };
            case 'SANGAT_BEARISH': return { from: '#ef4444', to: '#dc2626', bg: 'from-red-500/20 to-red-600/20' };
            default: return { from: '#a3a3a3', to: '#737373', bg: 'from-neutral-400/20 to-neutral-500/20' };
        }
    };

    const colors = getGaugeColor();
    const rotation = (sentiment.score - 0.5) * 180;
    const marketContext = getMarketContext(sentiment, trend);
    const hasSignalConflict = trend.short_term !== trend.long_term;

    return (
        <div className={`relative p-6 rounded-2xl bg-gradient-to-br ${colors.bg} border border-neutral-800 overflow-hidden`}>
            {/* Decorative glow */}
            <div className="absolute inset-0 bg-gradient-to-b from-white/5 to-transparent pointer-events-none" />

            <div className="relative text-center">
                <div className="flex items-center justify-center gap-2 mb-2">
                    <span className="text-xs uppercase tracking-widest font-semibold text-neutral-500">
                        Market Sentiment
                    </span>
                    {hasSignalConflict && (
                        <div className="relative group">
                            <AlertCircle className="w-4 h-4 text-amber-500 animate-pulse cursor-help" />
                            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-neutral-900 border border-neutral-700 rounded-lg text-xs text-neutral-300 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10 shadow-xl">
                                Short-term and long-term signals diverge
                            </div>
                        </div>
                    )}
                </div>

                {/* Gauge Visualization */}
                <div className="relative w-48 h-24 mx-auto mb-4 mt-2">
                    <svg className="absolute inset-0 w-full h-full" viewBox="0 0 200 100">
                        <defs>
                            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stopColor="#ef4444" />
                                <stop offset="25%" stopColor="#f97316" />
                                <stop offset="50%" stopColor="#eab308" />
                                <stop offset="75%" stopColor="#84cc16" />
                                <stop offset="100%" stopColor="#22c55e" />
                            </linearGradient>
                        </defs>
                        <path
                            d="M 20 90 A 80 80 0 0 1 180 90"
                            fill="none"
                            stroke="url(#gaugeGradient)"
                            strokeWidth="12"
                            strokeLinecap="round"
                            className="opacity-30"
                        />
                        <path
                            d="M 20 90 A 80 80 0 0 1 180 90"
                            fill="none"
                            stroke="url(#gaugeGradient)"
                            strokeWidth="12"
                            strokeLinecap="round"
                            strokeDasharray={`${sentiment.score * 251} 251`}
                            className="drop-shadow-lg"
                        />
                    </svg>

                    {/* Needle */}
                    <motion.div
                        className="absolute bottom-0 left-1/2 w-1 h-16 -ml-0.5 origin-bottom"
                        initial={{ rotate: -90 }}
                        animate={{ rotate: rotation }}
                        transition={{ type: 'spring', stiffness: 50, damping: 15 }}
                    >
                        <div className="w-full h-full bg-gradient-to-t from-white to-transparent rounded-full" />
                        <div className="absolute bottom-0 left-1/2 -ml-2 w-4 h-4 bg-white rounded-full shadow-lg" />
                    </motion.div>
                </div>

                {/* Sentiment Label - TIER 1 Priority */}
                <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.3 }}
                    className="text-3xl font-extrabold tracking-tight"
                    style={{ color: colors.from }}
                >
                    {getSentimentLabel(sentiment.level)}
                </motion.div>

                {/* Context Badge */}
                <div className="mt-4 flex items-center justify-center gap-3">
                    <div className={`px-3 py-1.5 rounded-full text-xs font-semibold border ${marketContext.color}`}>
                        {marketContext.label}
                    </div>
                    <div className="text-xs text-neutral-500">
                        <span className="font-medium text-neutral-400">{marketContext.bias}</span>
                    </div>
                </div>

                {/* Confidence */}
                <div className="mt-4 flex items-center justify-center gap-4 text-sm">
                    <div className="flex items-center gap-1.5">
                        <div className="w-2 h-2 rounded-full bg-blue-500" />
                        <span className="text-neutral-400">Confidence:</span>
                        <span className="font-bold text-white">{Math.round(sentiment.confidence * 100)}%</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

// Signal Summary Component - Enhanced
const SignalSummary: React.FC<{ sentiment: Sentiment }> = ({ sentiment }) => {
    const { bullish_signals, bearish_signals, neutral_signals } = sentiment.component_scores;
    const total = bullish_signals + bearish_signals + neutral_signals;

    return (
        <div className="grid grid-cols-3 gap-2 mt-4">
            <div className="bg-green-500/10 border border-green-500/20 rounded-xl p-3 text-center relative overflow-hidden">
                <div className="absolute bottom-0 left-0 right-0 h-1 bg-green-500/30">
                    <motion.div
                        className="h-full bg-green-500"
                        initial={{ width: 0 }}
                        animate={{ width: `${(bullish_signals / total) * 100}%` }}
                        transition={{ duration: 1, delay: 0.2 }}
                    />
                </div>
                <TrendingUp className="w-4 h-4 text-green-400 mx-auto mb-1" />
                <div className="text-lg font-bold text-green-400">{bullish_signals}</div>
                <div className="text-[10px] text-green-500/70 uppercase font-medium">Bullish</div>
            </div>
            <div className="bg-neutral-500/10 border border-neutral-500/20 rounded-xl p-3 text-center relative overflow-hidden">
                <div className="absolute bottom-0 left-0 right-0 h-1 bg-neutral-500/30">
                    <motion.div
                        className="h-full bg-neutral-500"
                        initial={{ width: 0 }}
                        animate={{ width: `${(neutral_signals / total) * 100}%` }}
                        transition={{ duration: 1, delay: 0.3 }}
                    />
                </div>
                <Activity className="w-4 h-4 text-neutral-400 mx-auto mb-1" />
                <div className="text-lg font-bold text-neutral-400">{neutral_signals}</div>
                <div className="text-[10px] text-neutral-500/70 uppercase font-medium">Neutral</div>
            </div>
            <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-3 text-center relative overflow-hidden">
                <div className="absolute bottom-0 left-0 right-0 h-1 bg-red-500/30">
                    <motion.div
                        className="h-full bg-red-500"
                        initial={{ width: 0 }}
                        animate={{ width: `${(bearish_signals / total) * 100}%` }}
                        transition={{ duration: 1, delay: 0.4 }}
                    />
                </div>
                <TrendingDown className="w-4 h-4 text-red-400 mx-auto mb-1" />
                <div className="text-lg font-bold text-red-400">{bearish_signals}</div>
                <div className="text-[10px] text-red-500/70 uppercase font-medium">Bearish</div>
            </div>
        </div>
    );
};

// Forecast micro-explanations
const getForecastExplanation = (horizon: number, change: number, indicators: any) => {
    const rsi = indicators?.rsi?.value || 50;
    const macdInterpretation = indicators?.macd?.interpretation || 'Neutral';
    const maTrend = indicators?.moving_averages?.trend || 'Neutral';
    const close = indicators?.close || 0;
    const sma200 = indicators?.moving_averages?.sma_200 || close;

    if (horizon === 1) {
        if (Math.abs(change) < 0.5) return 'Noise-driven • Low conviction';
        if (rsi > 70) return 'Overbought reversal risk';
        if (rsi < 30) return 'Oversold bounce potential';
        return change > 0 ? 'Short-term momentum' : 'Near-term weakness';
    }

    if (horizon === 7) {
        if (macdInterpretation === 'Bullish') return 'MACD bullish crossover';
        if (macdInterpretation === 'Bearish') return 'MACD bearish divergence';
        return maTrend === 'Bullish' ? 'Above key MA levels' : 'Below key MA levels';
    }

    if (horizon === 30) {
        if (close > sma200) return 'Above MA200 • Structural strength';
        if (close < sma200) return 'Below MA200 • Structural weakness';
        return 'Testing major resistance';
    }

    return 'Model projection';
};

// Model Contribution Component - Enhanced with tooltip
const ModelContribution: React.FC<{ data: PredictionResponse }> = ({ data }) => {
    const [showTooltip, setShowTooltip] = useState(false);
    const contribution = data.model_contribution || { sarima_weight: 0.5, lstm_weight: 0.5, ensemble_method: 'weighted_average' };
    const sarimaPercent = Math.round(contribution.sarima_weight * 100);
    const lstmPercent = Math.round(contribution.lstm_weight * 100);

    return (
        <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-4 mt-4">
            <div className="flex items-center gap-2 mb-3">
                <Brain className="w-4 h-4 text-purple-400" />
                <span className="text-sm font-semibold text-neutral-300">Model Contribution</span>
                <div className="relative ml-auto">
                    <Info
                        className="w-4 h-4 text-neutral-500 cursor-help hover:text-neutral-400 transition-colors"
                        onMouseEnter={() => setShowTooltip(true)}
                        onMouseLeave={() => setShowTooltip(false)}
                    />
                    <AnimatePresence>
                        {showTooltip && (
                            <motion.div
                                initial={{ opacity: 0, y: 5 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: 5 }}
                                className="absolute right-0 top-full mt-2 w-64 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-xs text-neutral-300 z-20 shadow-xl"
                            >
                                <p className="leading-relaxed">
                                    Weights adjusted dynamically based on volatility regime detection and recent prediction accuracy.
                                </p>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>

            <div className="space-y-3">
                {/* SARIMA Bar - Cool Blue */}
                <div>
                    <div className="flex justify-between text-xs mb-1">
                        <span className="text-neutral-400 flex items-center gap-1">
                            <Cpu className="w-3 h-3 text-blue-400" /> SARIMA (Statistical)
                        </span>
                        <span className="text-blue-400 font-bold">{sarimaPercent}%</span>
                    </div>
                    <div className="h-2.5 bg-neutral-800 rounded-full overflow-hidden">
                        <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${sarimaPercent}%` }}
                            transition={{ duration: 1, delay: 0.2 }}
                            className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full"
                        />
                    </div>
                </div>

                {/* LSTM Bar - Subtle Purple */}
                <div>
                    <div className="flex justify-between text-xs mb-1">
                        <span className="text-neutral-400 flex items-center gap-1">
                            <Brain className="w-3 h-3 text-purple-400" /> LSTM (Deep Learning)
                        </span>
                        <span className="text-purple-400 font-bold">{lstmPercent}%</span>
                    </div>
                    <div className="h-2.5 bg-neutral-800 rounded-full overflow-hidden">
                        <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${lstmPercent}%` }}
                            transition={{ duration: 1, delay: 0.4 }}
                            className="h-full bg-gradient-to-r from-purple-600 to-purple-400 rounded-full"
                        />
                    </div>
                </div>
            </div>

            <div className="mt-3 text-xs text-neutral-500 text-center border-t border-neutral-800 pt-3">
                Ensemble: <span className="text-neutral-400 font-medium">{contribution.ensemble_method.replace('_', ' ')}</span>
            </div>
        </div>
    );
};

// Indicator Impact Toggle Component - Uses actual backend data
const IndicatorImpact: React.FC<{ sentiment: Sentiment }> = ({ sentiment }) => {
    const [showImpact, setShowImpact] = useState(false);

    // Use actual indicator_impact from backend sentiment calculation
    const impacts = sentiment.indicator_impact || {
        RSI: 20,
        MACD: 20,
        'Moving Avg': 20,
        'Bollinger': 20,
        'Forecast': 20
    };

    // Get color based on impact value
    const getImpactColor = (impact: number) => {
        if (impact >= 30) return 'from-cyan-500 to-cyan-400';
        if (impact >= 20) return 'from-blue-500 to-blue-400';
        if (impact >= 10) return 'from-purple-500 to-purple-400';
        return 'from-neutral-600 to-neutral-500';
    };

    return (
        <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-4 mt-4">
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-cyan-400" />
                    <span className="text-sm font-semibold text-neutral-300">Indicator Impact</span>
                </div>
                <button
                    onClick={() => setShowImpact(!showImpact)}
                    className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${showImpact
                        ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                        : 'bg-neutral-800 text-neutral-400 hover:bg-neutral-700'
                        }`}
                >
                    {showImpact ? 'Hide' : 'Show'} Impact
                </button>
            </div>

            <AnimatePresence>
                {showImpact && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="space-y-2"
                    >
                        {Object.entries(impacts).map(([indicator, impact], i) => (
                            <div key={indicator} className="flex items-center gap-3">
                                <span className="text-xs text-neutral-400 w-20">{indicator}</span>
                                <div className="flex-1 h-2 bg-neutral-800 rounded-full overflow-hidden">
                                    <motion.div
                                        initial={{ width: 0 }}
                                        animate={{ width: `${impact as number}%` }}
                                        transition={{ duration: 0.5, delay: i * 0.1 }}
                                        className={`h-full bg-gradient-to-r ${getImpactColor(impact as number)} rounded-full`}
                                    />
                                </div>
                                <span className="text-xs font-medium text-cyan-400 w-10 text-right">{impact}%</span>
                            </div>
                        ))}
                        <p className="text-[10px] text-neutral-500 mt-2 text-center">
                            Real-time contribution to sentiment score (from AI model)
                        </p>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export const PredictionCard: React.FC<PredictionCardProps> = ({ data }) => {
    const { sentiment, forecasts, trend, indicators } = data;

    const dailyForecast = forecasts.find(f => f.horizon === 1);
    const weeklyForecast = forecasts.find(f => f.horizon === 7);
    const monthlyForecast = forecasts.find(f => f.horizon === 30);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-4"
        >
            {/* Sentiment Gauge - Enhanced with conflict indicator */}
            <SentimentGauge sentiment={sentiment} trend={trend} />

            {/* Signal Summary */}
            <SignalSummary sentiment={sentiment} />

            {/* Forecast Cards - Enhanced with micro-explanations */}
            <div className="grid grid-cols-3 gap-2">
                {[
                    { label: '1D', forecast: dailyForecast, horizon: 1 },
                    { label: '7D', forecast: weeklyForecast, horizon: 7 },
                    { label: '1M', forecast: monthlyForecast, horizon: 30 },
                ].map(({ label, forecast, horizon }) => {
                    const change = forecast?.predicted_change_percent || 0;
                    const isPositive = change >= 0;
                    const explanation = getForecastExplanation(horizon, change, indicators);

                    return (
                        <div
                            key={label}
                            className={`relative p-3 rounded-xl border overflow-hidden ${isPositive
                                ? 'bg-green-500/5 border-green-500/20'
                                : 'bg-red-500/5 border-red-500/20'
                                }`}
                        >
                            <div className="text-xs text-neutral-500 font-medium">{label} Forecast</div>
                            <div className={`text-xl font-extrabold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                                {isPositive ? '+' : ''}{change.toFixed(2)}%
                            </div>
                            {/* Micro-explanation */}
                            <div className="text-[10px] text-neutral-500 mt-1 leading-tight min-h-[24px]">
                                {explanation}
                            </div>
                            {isPositive ? (
                                <ArrowUp className="absolute right-2 top-2 w-4 h-4 text-green-500/30" />
                            ) : (
                                <ArrowDown className="absolute right-2 top-2 w-4 h-4 text-red-500/30" />
                            )}
                        </div>
                    );
                })}
            </div>

            {/* Trend Analysis - Enhanced labels */}
            <div className="grid grid-cols-2 gap-3">
                <div className="bg-neutral-900/50 p-4 rounded-xl border border-neutral-800">
                    <div className="text-xs text-neutral-500 mb-1 uppercase tracking-wide">Short Term</div>
                    <div className="flex items-center space-x-2 text-base font-bold text-neutral-200">
                        {trend.short_term === 'Uptrend' && <ArrowUp className="w-5 h-5 text-green-500" />}
                        {trend.short_term === 'Downtrend' && <ArrowDown className="w-5 h-5 text-red-500" />}
                        {trend.short_term === 'Sideways' && <Minus className="w-5 h-5 text-yellow-500" />}
                        <span>{trend.short_term}</span>
                    </div>
                </div>

                <div className="bg-neutral-900/50 p-4 rounded-xl border border-neutral-800">
                    <div className="text-xs text-neutral-500 mb-1 uppercase tracking-wide">Long Term</div>
                    <div className="flex items-center space-x-2 text-base font-bold text-neutral-200">
                        {trend.long_term === 'Uptrend' && <ArrowUp className="w-5 h-5 text-green-500" />}
                        {trend.long_term === 'Downtrend' && <ArrowDown className="w-5 h-5 text-red-500" />}
                        {trend.long_term === 'Sideways' && <Minus className="w-5 h-5 text-yellow-500" />}
                        <span>{trend.long_term}</span>
                    </div>
                </div>
            </div>

            {/* Model Contribution - Enhanced with tooltip */}
            <ModelContribution data={data} />

            {/* Indicator Impact Toggle */}
            <IndicatorImpact sentiment={sentiment} />

            {/* Analysis Rationale - Tier 3 Info */}
            <div className="bg-neutral-900/50 p-4 rounded-xl border border-neutral-800">
                <div className="text-sm font-semibold text-neutral-300 mb-3 flex items-center gap-2">
                    <Target className="w-4 h-4 text-amber-400" />
                    Analysis Rationale
                </div>
                <ul className="space-y-2">
                    {sentiment.rationale.slice(0, 5).map((r, i) => (
                        <motion.li
                            key={i}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.1 }}
                            className="flex items-start text-sm text-neutral-400"
                        >
                            <span className="mr-2 mt-1.5 w-1.5 h-1.5 rounded-full bg-blue-500 shrink-0" />
                            {r}
                        </motion.li>
                    ))}
                </ul>
            </div>

            {/* Warning - Enhanced copy */}
            {trend.inconsistency_warning && (
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex items-start p-4 bg-yellow-900/20 rounded-xl border border-yellow-900/50 text-yellow-500 text-sm"
                >
                    <AlertTriangle className="w-5 h-5 mr-3 shrink-0 mt-0.5" />
                    <div>
                        <div className="font-semibold mb-1">Signal Divergence Detected</div>
                        <p className="text-yellow-500/80">Short-term and long-term signals show meaningful divergence. Consider this as increased uncertainty; manage risk accordingly.</p>
                    </div>
                </motion.div>
            )}

            {/* Disclaimer - Professional tone */}
            <div className="text-[10px] text-neutral-600 text-center py-2">
                Informational analysis only • Not financial advice
            </div>
        </motion.div>
    );
};
