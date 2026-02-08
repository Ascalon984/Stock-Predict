'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { StockSelector } from '../components/StockSelector';
import { StockLogo } from '../components/StockLogo';
import { ChartMonitor } from '../components/ChartMonitor';
import { PredictionCard } from '../components/PredictionCard';
import { IndicatorPanel } from '../components/IndicatorPanel';
import { HeroLanding } from '../components/HeroLanding';
import { generateDummyPrediction } from '../data/dummyData';
import { STOCK_LIST, BLUE_CHIP_STOCKS, formatPrice, formatMarketCap } from '../data/stocks';
import { PredictionResponse } from './types';
import {
  TrendingUp,
  Activity,
  BarChart3,
  Brain,
  Shield,
  Clock,
  AlertCircle,
  ChevronRight,
  Sparkles,
  Zap,
  LineChart,
  Target,
  ArrowUpRight,
  ArrowDownRight,
  Cpu,
  Layers,
  PieChart,
  Globe,
  TrendingDown
} from 'lucide-react';

// Market stats component
const MarketStats: React.FC = () => {
  const [time, setTime] = useState<Date | null>(null);

  useEffect(() => {
    setTime(new Date());
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  if (!time) return <div className="hidden lg:flex items-center gap-4 animate-pulse h-6 w-32 bg-neutral-800/50 rounded-lg" />;

  const isMarketOpen = () => {
    const hours = time.getHours();
    const day = time.getDay();
    // IDX market hours: Mon-Fri, 9:00 AM - 3:30 PM WIB
    return day > 0 && day < 6 && hours >= 9 && hours < 16;
  };

  return (
    <div className="hidden lg:flex items-center gap-4">
      <div className="flex items-center gap-2">
        <Clock className="w-4 h-4 text-neutral-500" />
        <span className="text-neutral-400 font-mono text-sm">
          {time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true })}
        </span>
        <span className="text-neutral-600 text-xs">Jakarta</span>
      </div>
      <div className="h-4 w-px bg-neutral-800" />
      <div className={`flex items-center gap-2 px-2.5 py-1 rounded-full text-xs font-medium ${isMarketOpen()
        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
        : 'bg-neutral-800/50 text-neutral-500 border border-neutral-700/50'
        }`}>
        <div className={`w-1.5 h-1.5 rounded-full ${isMarketOpen() ? 'bg-emerald-400 animate-pulse' : 'bg-neutral-500'}`} />
        {isMarketOpen() ? 'Market Open' : 'Market Closed'}
      </div>
    </div>
  );
};

// Quick stat card
const QuickStat: React.FC<{
  label: string;
  value: string | number;
  subValue?: string;
  icon: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
}> = ({ label, value, subValue, icon, trend }) => (
  <div className="bg-neutral-900/50 backdrop-blur-sm p-4 rounded-xl border border-neutral-800/50 hover:border-neutral-700/50 transition-all group">
    <div className="flex items-start justify-between">
      <div className="text-xs text-neutral-500 uppercase tracking-wider font-semibold">{label}</div>
      <div className="p-1.5 rounded-lg bg-neutral-800/50 text-neutral-400 group-hover:text-blue-400 transition-colors">
        {icon}
      </div>
    </div>
    <div className={`text-2xl font-extrabold mt-2 ${trend === 'up' ? 'text-emerald-400' :
      trend === 'down' ? 'text-red-400' :
        'text-white'
      }`}>
      {value}
    </div>
    {subValue && (
      <div className="text-xs text-neutral-400 mt-1 font-medium">{subValue}</div>
    )}
  </div>
);

export default function Home() {
  const [isSelectingStock, setIsSelectingStock] = useState(false);
  const [ticker, setTicker] = useState<string>('');
  const [activeTab, setActiveTab] = useState<'chart' | 'indicators'>('chart');
  const [chartRange, setChartRange] = useState<string>('1M');

  const getFetchParams = (range: string) => {
    switch (range) {
      case '1D': return { interval: '5m', period: '1d' };
      case '5D': return { interval: '15m', period: '5d' };
      case '1M': return { interval: '1d', period: '1mo' };
      case '6M': return { interval: '1d', period: '6mo' };
      case 'YTD': return { interval: '1d', period: 'ytd' };
      case '1Y': return { interval: '1d', period: '1y' };
      case '5Y': return { interval: '1wk', period: '5y' };
      default: return { interval: '1d', period: '1mo' };
    }
  };

  const { interval: currentInterval, period: currentPeriod } = getFetchParams(chartRange);

  // Use TanStack Query for real-time data with auto-polling
  const { useChartData, useRealtimeQuote, useMarketStatus } = require('../hooks/useRealtimeData');

  // Market status for header display
  const { data: marketStatus } = useMarketStatus();

  // Real-time quote with auto-polling (10-30 seconds based on market status)
  const {
    data: quoteData,
    isLoading: quoteLoading,
    isRefetching: quoteRefetching
  } = useRealtimeQuote(ticker || null);

  // Chart data with auto-polling
  const {
    data: chartData,
    isLoading: chartLoading,
    isRefetching: chartRefetching
  } = useChartData(ticker || null, currentInterval, currentPeriod);

  // Combine loading states
  const loading = quoteLoading || chartLoading || isSelectingStock;
  const isRefreshing = quoteRefetching || chartRefetching;

  // Build data object for components (compatible with existing structure)
  const [data, setData] = useState<PredictionResponse | null>(null);

  // Update data when chart data changes
  useEffect(() => {
    const updateData = async () => {
      if (chartData?.success && chartData.historical_data) {
        // For daily data, also try to get predictions
        let predictionData = null;
        if (currentInterval === '1d' || currentInterval === '1wk') {
          try {
            const predResponse = await fetch(
              `http://localhost:8000/api/v1/predict/${ticker}?period=1y`
            );
            if (predResponse.ok) {
              predictionData = await predResponse.json();
            }
          } catch {
            console.log('Prediction not available');
          }
        }

        if (predictionData?.success) {
          setData({
            ...predictionData,
            historical_data: chartData.historical_data,
            is_intraday: chartData.is_intraday,
            interval: currentInterval,
            current_price: chartData.current_price,
            change_percent: chartData.change_percent
          });
        } else {
          // Use dummy prediction structure with real historical data
          const result = generateDummyPrediction(ticker);
          setData({
            ...result,
            historical_data: chartData.historical_data,
            is_intraday: chartData.is_intraday,
            interval: currentInterval,
            current_price: chartData.current_price,
            change_percent: chartData.change_percent,
            // Add quote data for more accurate real-time info
            ...(quoteData && {
              current_price: quoteData.last_price,
              change_percent: quoteData.change_percent
            })
          });
        }

        // Data ready, turn off selecting state
        setIsSelectingStock(false);
      }
    };

    if (ticker && chartData) {
      updateData();
    }
  }, [chartData, quoteData, ticker, currentInterval]);

  const handleSelectStock = useCallback((selectedTicker: string) => {
    setIsSelectingStock(true);
    setTicker(selectedTicker);
  }, []);

  // Handle range change from chart
  const handleRangeChange = useCallback((range: string) => {
    setChartRange(range);
  }, []);

  const selectedStock = STOCK_LIST.find(s => s.ticker === ticker);

  return (
    <main className="min-h-screen bg-[#07070a] text-neutral-200 selection:bg-blue-500/30 overflow-x-hidden">
      {/* Animated Background */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        {/* Gradient orbs */}
        <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-blue-600/8 rounded-full blur-[100px] animate-glow-pulse" />
        <div className="absolute bottom-0 right-1/4 w-[400px] h-[400px] bg-purple-600/8 rounded-full blur-[100px] animate-glow-pulse" style={{ animationDelay: '1.5s' }} />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-cyan-600/5 rounded-full blur-[120px] animate-glow-pulse" style={{ animationDelay: '3s' }} />

        {/* Subtle grid */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.01)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.01)_1px,transparent_1px)] bg-[size:80px_80px]" />
      </div>

      {/* Header */}
      <header className="relative border-b border-neutral-800/50 bg-neutral-950/80 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-[1800px] mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            {/* Logo with glow */}
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl blur-xl opacity-40 group-hover:opacity-60 transition-opacity" />
              <div className="relative w-10 h-10 rounded-xl flex items-center justify-center shadow-lg overflow-hidden bg-neutral-900">
                <img src="/logo-app.jpeg" alt="StockPredict.AI" className="w-full h-full object-cover" />
              </div>
            </div>
            <div>
              <h1 className="font-bold text-xl tracking-tight flex items-center">
                StockPredict
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400">.AI</span>
              </h1>
              <div className="hidden sm:flex items-center gap-2 text-[10px] font-mono text-neutral-500 tracking-widest uppercase">
                <Cpu className="w-3 h-3" />
                Hybrid SARIMA-LSTM Engine v2.0
              </div>
            </div>
          </div>

          <div className="flex items-center gap-6">
            <MarketStats />

            {/* Stock count badge */}
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-neutral-900/50 border border-neutral-800 rounded-full">
              <Globe className="w-3.5 h-3.5 text-blue-400" />
              <span className="text-xs font-medium text-neutral-400">
                IDX • <span className="text-white">{STOCK_LIST.length}</span> Stocks
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="relative max-w-[1800px] mx-auto px-6 py-8">
        <div className="flex flex-col xl:flex-row gap-8">

          {/* Left Sidebar: Search & Prediction */}
          <div className="w-full xl:w-[420px] shrink-0 space-y-6">
            {/* Search Card */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="relative overflow-hidden"
            >
              {/* Gradient border glow */}
              <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/20 via-purple-500/10 to-cyan-500/20 rounded-2xl blur-sm" />

              <div className="relative glass-premium p-6 rounded-2xl">
                <div className="flex items-center gap-3 mb-5">
                  <div className="p-2.5 rounded-xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 border border-blue-500/20">
                    <Sparkles className="w-5 h-5 text-blue-400" />
                  </div>
                  <div>
                    <h2 className="text-lg font-bold text-white">AI Analysis</h2>
                    <p className="text-xs text-neutral-500">Select a stock to analyze</p>
                  </div>
                </div>

                <StockSelector
                  onSelect={handleSelectStock}
                  currentTicker={ticker}
                  currentQuote={quoteData}
                />

                <div className="mt-5 flex items-start gap-3 p-3.5 bg-amber-500/5 border border-amber-500/10 rounded-xl">
                  <AlertCircle className="w-4 h-4 text-amber-500 mt-0.5 shrink-0" />
                  <p className="text-xs text-neutral-400 leading-relaxed">
                    AI-powered predictions using hybrid SARIMA-LSTM.
                    <span className="text-amber-400 font-medium"> Not financial advice.</span>
                  </p>
                </div>
              </div>
            </motion.div>

            {/* Loading State */}
            <AnimatePresence>
              {loading && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className="relative overflow-hidden glass-premium p-8 rounded-2xl"
                >
                  <div className="flex flex-col items-center justify-center space-y-6">
                    {/* Neural network animation */}
                    <div className="relative w-28 h-28">
                      <div className="absolute inset-0 border-4 border-blue-500/20 rounded-full animate-ping" style={{ animationDuration: '2s' }} />
                      <div className="absolute inset-3 border-4 border-purple-500/20 rounded-full animate-ping" style={{ animationDelay: '0.3s', animationDuration: '2s' }} />
                      <div className="absolute inset-6 border-4 border-cyan-500/30 rounded-full animate-ping" style={{ animationDelay: '0.6s', animationDuration: '2s' }} />
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center">
                          <Brain className="w-7 h-7 text-blue-400 animate-pulse" />
                        </div>
                      </div>
                    </div>

                    <div className="text-center">
                      <div className="text-lg font-semibold text-white mb-2">
                        Analyzing {ticker.replace('.JK', '')}
                      </div>
                      <div className="flex items-center gap-2 text-sm text-neutral-500">
                        <Zap className="w-4 h-4 text-blue-400 animate-pulse" />
                        <span>Running Neural Engine...</span>
                      </div>
                    </div>

                    {/* Progress steps */}
                    <div className="w-full space-y-2.5 text-xs">
                      {['Fetching market data...', 'Computing SARIMA forecast...', 'Running LSTM inference...', 'Generating signals...'].map((step, i) => (
                        <motion.div
                          key={step}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.3 }}
                          className="flex items-center gap-3 text-neutral-500"
                        >
                          <div className="w-5 h-5 rounded-full bg-neutral-800/50 flex items-center justify-center">
                            <div className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
                          </div>
                          {step}
                        </motion.div>
                      ))}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Prediction Results */}
            <AnimatePresence>
              {data && !loading && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-4"
                >
                  <PredictionCard data={data} />
                </motion.div>
              )}
            </AnimatePresence>

            {/* Empty State Features */}
            {!data && !loading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="space-y-4"
              >
                <div className="text-sm font-semibold text-neutral-500 uppercase tracking-wider flex items-center gap-2">
                  <Layers className="w-4 h-4" />
                  Platform Features
                </div>
                {[
                  { icon: Brain, title: 'Hybrid AI Model', desc: 'SARIMA + LSTM ensemble for accurate forecasts', color: 'from-blue-500/20 to-purple-500/20' },
                  { icon: BarChart3, title: 'Technical Analysis', desc: 'RSI, MACD, Bollinger Bands, Moving Averages', color: 'from-green-500/20 to-emerald-500/20' },
                  { icon: Activity, title: 'Multi-Horizon Forecasts', desc: '1 Day, 7 Days, and 1 Month predictions', color: 'from-orange-500/20 to-amber-500/20' },
                  { icon: Shield, title: 'Uncertainty Aware', desc: 'Confidence intervals that expand with volatility', color: 'from-purple-500/20 to-pink-500/20' },
                ].map((feature, i) => (
                  <motion.div
                    key={feature.title}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 + i * 0.1 }}
                    className="flex items-start gap-4 p-4 bg-neutral-900/30 rounded-xl border border-neutral-800/50 hover:bg-neutral-900/50 hover:border-neutral-700/50 transition-all duration-300 cursor-default group"
                  >
                    <div className={`p-2.5 rounded-xl bg-gradient-to-br ${feature.color} border border-neutral-700/50 group-hover:border-neutral-600/50 transition-colors`}>
                      <feature.icon className="w-5 h-5 text-white" />
                    </div>
                    <div className="flex-1">
                      <div className="font-medium text-white group-hover:text-blue-400 transition-colors">{feature.title}</div>
                      <div className="text-sm text-neutral-500 mt-0.5">{feature.desc}</div>
                    </div>
                    <ChevronRight className="w-4 h-4 text-neutral-600 group-hover:text-neutral-400 group-hover:translate-x-1 transition-all" />
                  </motion.div>
                ))}
              </motion.div>
            )}
          </div>

          {/* Main Content Area */}
          <div className="flex-1 min-w-0">
            {data && !loading ? (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                {/* Stock Header */}
                <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
                  <div className="flex items-center gap-4">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500/20 via-purple-500/15 to-cyan-500/20 border border-neutral-700/50 flex items-center justify-center shadow-lg p-2">
                      <StockLogo ticker={data.ticker} className="w-full h-full object-contain" />
                    </div>
                    <div>
                      <h2 className="text-3xl font-bold text-white tracking-tight flex items-center gap-3">
                        {data.ticker.replace('.JK', '')}
                        <span className="text-xs font-medium text-neutral-500 bg-neutral-800/80 px-2.5 py-1 rounded-lg">
                          IDX
                        </span>
                        {selectedStock?.sector && (
                          <span className="text-xs font-medium text-blue-400 bg-blue-500/10 px-2.5 py-1 rounded-lg border border-blue-500/20">
                            {selectedStock.sector}
                          </span>
                        )}
                      </h2>
                      {selectedStock && (
                        <p className="text-neutral-500 text-sm mt-1">{selectedStock.name}</p>
                      )}
                    </div>
                  </div>

                  <div className="flex flex-col items-end">
                    <div className="text-3xl font-mono font-bold text-white">
                      Rp {(data.current_price || data.historical_data.close[data.historical_data.close.length - 1]).toLocaleString()}
                    </div>
                    <div className="flex items-center gap-3 text-sm mt-1">
                      {data.change_percent !== undefined && (
                        <span className={`flex items-center gap-1 font-medium ${data.change_percent >= 0 ? 'text-emerald-400' : 'text-red-400'
                          }`}>
                          {data.change_percent >= 0 ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
                          {data.change_percent >= 0 ? '+' : ''}{data.change_percent.toFixed(2)}%
                        </span>
                      )}
                      <span className="text-neutral-600">•</span>
                      <span className="text-neutral-500">
                        {new Date(data.timestamp).toLocaleString('id-ID', {
                          day: 'numeric',
                          month: 'short',
                          hour: '2-digit',
                          minute: '2-digit'
                        })}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Tab Navigation */}
                <div className="flex items-center gap-2 p-1.5 bg-neutral-900/50 rounded-xl border border-neutral-800/50 w-fit backdrop-blur-sm">
                  <button
                    onClick={() => setActiveTab('chart')}
                    className={`flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium text-sm transition-all duration-200 ${activeTab === 'chart'
                      ? 'bg-gradient-to-r from-blue-500/20 to-cyan-500/10 text-blue-400 shadow-lg shadow-blue-500/5 border border-blue-500/20'
                      : 'text-neutral-400 hover:text-neutral-300 hover:bg-neutral-800/50'
                      }`}
                  >
                    <TrendingUp className="w-4 h-4" />
                    Price Chart
                  </button>
                  <button
                    onClick={() => setActiveTab('indicators')}
                    className={`flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium text-sm transition-all duration-200 ${activeTab === 'indicators'
                      ? 'bg-gradient-to-r from-purple-500/20 to-pink-500/10 text-purple-400 shadow-lg shadow-purple-500/5 border border-purple-500/20'
                      : 'text-neutral-400 hover:text-neutral-300 hover:bg-neutral-800/50'
                      }`}
                  >
                    <BarChart3 className="w-4 h-4" />
                    Technical Indicators
                  </button>
                </div>

                {/* Tab Content */}
                <AnimatePresence mode="wait">
                  {activeTab === 'chart' ? (
                    <motion.div
                      key="chart"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 20 }}
                      transition={{ duration: 0.2 }}
                    >
                      <ChartMonitor data={data} activeRange={chartRange} onRangeChange={handleRangeChange} />
                    </motion.div>
                  ) : (
                    <motion.div
                      key="indicators"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      transition={{ duration: 0.2 }}
                    >
                      <IndicatorPanel data={data} />
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Quick Stats Row */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <QuickStat
                    label="RSI (14)"
                    value={data.indicators.rsi.value.toFixed(1)}
                    subValue={data.indicators.rsi.value > 70 ? 'Overbought Zone' : data.indicators.rsi.value < 30 ? 'Oversold Zone' : 'Neutral Range'}
                    icon={<Activity className="w-4 h-4" />}
                    trend={data.indicators.rsi.value > 70 ? 'down' : data.indicators.rsi.value < 30 ? 'up' : 'neutral'}
                  />
                  <QuickStat
                    label="MACD"
                    value={data.indicators.macd.interpretation === 'Bullish' ? 'Bullish Momentum' : 'Bearish Pressure'}
                    subValue={data.indicators.macd.crossover_detected ? 'Crossover Signal' : 'No Crossover'}
                    icon={<TrendingUp className="w-4 h-4" />}
                    trend={data.indicators.macd.interpretation === 'Bullish' ? 'up' : 'down'}
                  />
                  <QuickStat
                    label="Trend (SMA 50)"
                    value={data.indicators.moving_averages.trend === 'Bullish' ? 'Bullish Momentum' : 'Bearish Pressure'}
                    subValue={`${data.indicators.close > data.indicators.moving_averages.sma_50 ? 'Above' : 'Below'} SMA 50`}
                    icon={<Target className="w-4 h-4" />}
                    trend={data.indicators.moving_averages.trend === 'Bullish' ? 'up' : 'down'}
                  />
                  <QuickStat
                    label="Volatility (ATR)"
                    value={data.indicators.volatility.atr_percent > 3 ? 'High' : data.indicators.volatility.atr_percent > 2 ? 'Moderate' : 'Low'}
                    subValue={`${data.indicators.volatility.atr_percent.toFixed(2)}% daily range`}
                    icon={<PieChart className="w-4 h-4" />}
                    trend={data.indicators.volatility.atr_percent > 3 ? 'down' : 'neutral'}
                  />
                </div>

                {/* Disclaimer */}
                <div className="p-4 bg-neutral-900/30 rounded-xl border border-neutral-800/50 text-xs text-neutral-500 leading-relaxed">
                  <strong className="text-neutral-400">Disclaimer:</strong> Informational analysis only. This is not financial advice. Past performance does not guarantee future results.
                </div>
              </motion.div>
            ) : !loading && (
              /* Premium Hero Landing */
              <HeroLanding onSelectStock={handleSelectStock} />
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="relative border-t border-neutral-800/50 bg-neutral-950/50 backdrop-blur-xl mt-12">
        <div className="max-w-[1800px] mx-auto px-6 py-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-neutral-500">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg overflow-hidden flex items-center justify-center bg-neutral-900 border border-neutral-800">
                <img src="/logo-app.jpeg" alt="logo" className="w-full h-full object-cover" />
              </div>
              <div>
                <span className="font-semibold text-neutral-300">StockPredict.AI</span>
                <span className="text-neutral-600 mx-2">•</span>
                <span>Hybrid SARIMA-LSTM Predictive Analytics</span>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1.5">
                <Globe className="w-3.5 h-3.5 text-blue-400" />
                IDX Market
              </span>
              <span className="text-neutral-700">•</span>
              <span>{STOCK_LIST.length} Listed Stocks</span>
              <span className="text-neutral-700">•</span>
              <span className="flex items-center gap-1.5 text-amber-500">
                <Sparkles className="w-3.5 h-3.5" />
                Demo Mode
              </span>
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
}
