'use client';

import React, { useEffect, useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
    TrendingUp,
    TrendingDown,
    Zap,
    Brain,
    BarChart3,
    Activity,
    ChevronRight,
    Sparkles,
    Shield,
    Target
} from 'lucide-react';
import { BLUE_CHIP_STOCKS, STOCK_LIST, formatPrice } from '../data/stocks';
import { StockLogo } from './StockLogo';

interface HeroLandingProps {
    onSelectStock: (ticker: string) => void;
}

// Live Market Ticker
const LiveTicker: React.FC<{ stocks: typeof BLUE_CHIP_STOCKS }> = ({ stocks }) => {
    return (
        <div className="overflow-hidden py-3">
            <motion.div
                className="flex gap-6"
                animate={{ x: [0, -1000] }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            >
                {[...stocks, ...stocks].map((stock, i) => (
                    <div
                        key={`${stock.ticker}-${i}`}
                        className="flex items-center gap-3 px-4 py-2 bg-neutral-800/30 rounded-lg border border-neutral-700/30 shrink-0"
                    >
                        <span className="font-bold text-white text-sm">{stock.ticker.replace('.JK', '')}</span>
                        <span className="text-neutral-400 font-mono text-sm">Rp {formatPrice(stock.lastPrice || 0)}</span>
                        <span className={`flex items-center gap-1 text-xs font-semibold ${(stock.change || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
                            }`}>
                            {(stock.change || 0) >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                            {(stock.change || 0) >= 0 ? '+' : ''}{stock.change?.toFixed(2)}%
                        </span>
                    </div>
                ))}
            </motion.div>
        </div>
    );
};

// Floating Orb Component
const FloatingOrb: React.FC<{ color: string; size: number; delay: number; duration: number; className?: string }> = ({
    color, size, delay, duration, className
}) => (
    <motion.div
        className={`absolute rounded-full blur-xl ${className}`}
        style={{
            background: color,
            width: size,
            height: size,
        }}
        animate={{
            y: [0, -20, 0],
            x: [0, 10, 0],
            scale: [1, 1.1, 1],
            opacity: [0.3, 0.6, 0.3],
        }}
        transition={{
            duration,
            delay,
            repeat: Infinity,
            ease: "easeInOut",
        }}
    />
);

// Stats Card
const StatCard: React.FC<{ icon: React.ReactNode; label: string; value: string; subValue: string }> = ({
    icon, label, value, subValue
}) => (
    <motion.div
        className="bg-neutral-900/50 backdrop-blur-sm p-4 rounded-2xl border border-neutral-800/50 hover:border-neutral-700/50 transition-all group"
        whileHover={{ y: -5, scale: 1.02 }}
        transition={{ type: "spring", stiffness: 300 }}
    >
        <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 text-blue-400 group-hover:from-blue-500/30 group-hover:to-purple-500/30 transition-colors">
                {icon}
            </div>
            <span className="text-xs text-neutral-500 uppercase tracking-wider font-medium">{label}</span>
        </div>
        <div className="text-2xl font-bold text-white">{value}</div>
        <div className="text-xs text-neutral-500 mt-1">{subValue}</div>
    </motion.div>
);

// Quick Action Stock Button
const QuickStockButton: React.FC<{
    stock: typeof BLUE_CHIP_STOCKS[0];
    onClick: () => void;
    index: number;
}> = ({ stock, onClick, index }) => (
    <motion.button
        onClick={onClick}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 + index * 0.1 }}
        whileHover={{ scale: 1.05, y: -3 }}
        whileTap={{ scale: 0.98 }}
        className="group relative overflow-hidden px-5 py-3 bg-gradient-to-br from-neutral-800/80 to-neutral-900/80 hover:from-neutral-700/80 hover:to-neutral-800/80 border border-neutral-700/50 hover:border-blue-500/30 rounded-2xl transition-all duration-300"
    >
        {/* Hover gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-r from-blue-500/0 via-blue-500/5 to-purple-500/0 opacity-0 group-hover:opacity-100 transition-opacity" />

        <div className="relative flex items-center gap-3">
            {/* Stock Logo */}
            <div className="w-10 h-10 rounded-xl overflow-hidden shadow-sm">
                <StockLogo ticker={stock.ticker} className="w-full h-full text-xs" />
            </div>

            <div className="text-left">
                <div className="flex items-center gap-2">
                    <span className="font-bold text-white group-hover:text-blue-400 transition-colors">
                        {stock.ticker.replace('.JK', '')}
                    </span>
                    {stock.sector && (
                        <span className="text-[9px] px-2 py-0.5 bg-blue-500/10 text-blue-400 rounded-full font-medium">
                            {stock.sector}
                        </span>
                    )}
                </div>
                <div className="flex items-center gap-2 mt-0.5">
                    <span className="text-neutral-400 text-sm font-mono">
                        Rp {formatPrice(stock.lastPrice || 0)}
                    </span>
                    <span className={`text-xs font-semibold flex items-center gap-0.5 ${(stock.change || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
                        }`}>
                        {(stock.change || 0) >= 0 ? '+' : ''}{stock.change?.toFixed(2)}%
                    </span>
                </div>
            </div>

            <ChevronRight className="w-4 h-4 text-neutral-600 group-hover:text-blue-400 group-hover:translate-x-1 transition-all ml-2" />
        </div>
    </motion.button>
);

export const HeroLanding: React.FC<HeroLandingProps> = ({ onSelectStock }) => {
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    if (!mounted) return null;

    return (
        <div className="relative min-h-[700px] flex flex-col">
            {/* Floating Orbs Background */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <FloatingOrb color="rgba(59, 130, 246, 0.15)" size={300} delay={0} duration={8} className="top-10 -left-20" />
                <FloatingOrb color="rgba(139, 92, 246, 0.12)" size={250} delay={2} duration={10} className="top-40 right-10" />
                <FloatingOrb color="rgba(6, 182, 212, 0.1)" size={200} delay={4} duration={7} className="bottom-20 left-1/3" />
                <FloatingOrb color="rgba(236, 72, 153, 0.08)" size={180} delay={1} duration={9} className="bottom-40 right-1/4" />
            </div>

            {/* Main Content */}
            <div className="relative flex-1 flex flex-col">
                {/* Hero Section */}
                <div className="flex-1 flex items-center justify-center py-12">
                    <div className="w-full max-w-5xl mx-auto">

                        {/* Hero Text */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.3 }}
                            className="text-center mb-10"
                        >
                            <h2 className="text-4xl md:text-5xl font-bold mb-4">
                                <span className="text-white">AI-Powered</span>
                                <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400"> Stock Predictions</span>
                            </h2>
                            <p className="text-neutral-400 text-lg max-w-2xl mx-auto leading-relaxed">
                                Harness the power of <span className="text-blue-400 font-semibold">SARIMA</span> and <span className="text-purple-400 font-semibold">LSTM</span> neural networks
                                to analyze <span className="text-white font-semibold">{STOCK_LIST.length} IDX stocks</span> with real-time technical indicators
                            </p>
                        </motion.div>

                        {/* Stats Row */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.4 }}
                            className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10"
                        >
                            <StatCard
                                icon={<BarChart3 className="w-5 h-5" />}
                                label="Stocks"
                                value={`${STOCK_LIST.length}+`}
                                subValue="IDX Listed"
                            />
                            <StatCard
                                icon={<Brain className="w-5 h-5" />}
                                label="Models"
                                value="Hybrid"
                                subValue="SARIMA + LSTM"
                            />
                            <StatCard
                                icon={<Activity className="w-5 h-5" />}
                                label="Indicators"
                                value="10+"
                                subValue="Technical Signals"
                            />
                            <StatCard
                                icon={<Target className="w-5 h-5" />}
                                label="Horizons"
                                value="3"
                                subValue="1D, 7D, 30D"
                            />
                        </motion.div>

                        {/* Quick Start Section */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.5 }}
                            className="text-center"
                        >
                            <div className="flex items-center justify-center gap-2 text-sm text-neutral-500 mb-6">
                                <Sparkles className="w-4 h-4 text-amber-400" />
                                <span>Quick Start — Select a stock to begin analysis</span>
                            </div>

                            <div className="flex flex-wrap justify-center gap-3">
                                {BLUE_CHIP_STOCKS.slice(0, 5).map((stock, index) => (
                                    <QuickStockButton
                                        key={stock.ticker}
                                        stock={stock}
                                        onClick={() => onSelectStock(stock.ticker)}
                                        index={index}
                                    />
                                ))}
                            </div>
                        </motion.div>
                    </div>
                </div>

                {/* Live Ticker Bar */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.8 }}
                    className="border-t border-neutral-800/50 bg-neutral-900/30 backdrop-blur-sm"
                >
                    <LiveTicker stocks={BLUE_CHIP_STOCKS} />
                </motion.div>

                {/* Feature Pills */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 1 }}
                    className="py-6 flex flex-wrap justify-center gap-3"
                >
                    {[
                        { icon: Shield, label: 'Confidence Intervals', color: 'from-emerald-500/20 to-emerald-500/5' },
                        { icon: Zap, label: 'Real-time Updates', color: 'from-amber-500/20 to-amber-500/5' },
                        { icon: Brain, label: 'Neural Networks', color: 'from-purple-500/20 to-purple-500/5' },
                        { icon: Activity, label: 'Technical Analysis', color: 'from-blue-500/20 to-blue-500/5' },
                    ].map((feature) => (
                        <div
                            key={feature.label}
                            className={`flex items-center gap-2 px-4 py-2 bg-gradient-to-r ${feature.color} rounded-full border border-neutral-800/50`}
                        >
                            <feature.icon className="w-4 h-4 text-neutral-400" />
                            <span className="text-sm text-neutral-400 font-medium">{feature.label}</span>
                        </div>
                    ))}
                </motion.div>
            </div>
        </div>
    );
};
