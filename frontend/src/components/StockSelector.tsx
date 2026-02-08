import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, TrendingUp, TrendingDown, X, ChevronRight, History, Sparkles, Command, Ghost, ArrowUpRight, ArrowDownRight, Flame, Star } from 'lucide-react';
import {
    STOCK_LIST, searchStocks, BLUE_CHIP_STOCKS, TRENDING_STOCKS,
    Stock, formatPrice, getTopGainers, getTopLosers, SECTORS
} from '../data/stocks';
import { useBatchQuotes } from '../hooks/useRealtimeData';
import { QuoteData } from '../hooks/useRealtimeData';

interface StockSelectorProps {
    onSelect: (ticker: string) => void;
    currentTicker?: string;
    currentQuote?: QuoteData | null; // Add prop for current quote
}

// Imported from components/StockLogo
import { StockLogo } from './StockLogo';

// Stock Item Component - Updated to accept realPrice
const StockItem: React.FC<{
    stock: Stock;
    isActive: boolean;
    isSelected?: boolean;
    onClick: () => void;
    realtimeData?: QuoteData | null; // Add realtime data prop
}> = ({ stock, isActive, isSelected, onClick, realtimeData }) => {
    // Use realtime data if available, otherwise fallback to static
    const price = realtimeData ? realtimeData.last_price : stock.lastPrice;
    const change = realtimeData ? realtimeData.change_percent : stock.change;

    const isPositive = (change || 0) >= 0;
    const tickerDisplay = stock.ticker.replace('.JK', '');

    return (
        <button
            onClick={onClick}
            className={`
                w-full flex items-center justify-between px-4 py-3.5 rounded-xl text-left transition-all duration-200 group
                ${isActive
                    ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/30'
                    : isSelected
                        ? 'bg-blue-500/10 border-2 border-blue-500/30'
                        : 'bg-neutral-800/50 hover:bg-neutral-700/70 border-2 border-transparent hover:border-neutral-600'
                }
            `}
        >
            <div className="flex items-center gap-3 min-w-0 flex-1">
                {/* Stock Logo */}
                <StockLogo
                    ticker={stock.ticker}
                    className={`w-12 h-12 rounded-xl shrink-0 transition-opacity ${isActive ? 'opacity-100' : 'opacity-90 group-hover:opacity-100'}`}
                />

                {/* Stock Info */}
                <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                        <span className={`text-base font-bold ${isActive ? 'text-white' : 'text-white'}`}>
                            {tickerDisplay}
                        </span>
                        {realtimeData && (
                            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" title="Real-time data" />
                        )}
                        {stock.sector && (
                            <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold uppercase tracking-wide
                                ${isActive
                                    ? 'bg-white/20 text-white'
                                    : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                                }`}>
                                {stock.sector}
                            </span>
                        )}
                    </div>
                    <div className={`text-sm truncate mt-0.5 ${isActive ? 'text-white/70' : 'text-neutral-400'}`}>
                        {stock.name}
                    </div>
                </div>
            </div>

            {/* Price & Change */}
            {price !== undefined && (
                <div className="text-right ml-3 shrink-0">
                    <div className={`text-base font-mono font-bold ${isActive ? 'text-white' : 'text-white'}`}>
                        Rp {formatPrice(price)}
                    </div>
                    <div className={`text-sm font-semibold flex items-center justify-end gap-0.5 
                        ${isActive
                            ? 'text-white/80'
                            : isPositive ? 'text-emerald-400' : 'text-rose-400'
                        }`}>
                        {isPositive ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
                        {isPositive ? '+' : ''}{change?.toFixed(2)}%
                    </div>
                </div>
            )}
        </button>
    );
};

const SectorChip: React.FC<{
    sector: { name: string; icon: any };
    isActive: boolean;
    onClick: () => void;
}> = ({ sector, isActive, onClick }) => {
    const Icon = sector.icon;
    return (
        <button
            onClick={onClick}
            className={`
                flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-bold whitespace-nowrap transition-all border-2
                ${isActive
                    ? 'bg-blue-500 text-white border-blue-500'
                    : 'bg-neutral-800 text-neutral-400 border-neutral-700 hover:text-white hover:border-neutral-600'
                }
            `}
        >
            <Icon className="w-3.5 h-3.5" />
            {sector.name}
        </button>
    );
};

export const StockSelector: React.FC<StockSelectorProps> = ({ onSelect, currentTicker, currentQuote }) => {
    // ... state declarations ...
    const [query, setQuery] = useState('');
    const [isOpen, setIsOpen] = useState(false);
    const [activeIndex, setActiveIndex] = useState(-1);
    const [recentSearches, setRecentSearches] = useState<string[]>([]);
    const [selectedSector, setSelectedSector] = useState<string | null>(null);
    const [viewMode, setViewMode] = useState<'trending' | 'gainers' | 'losers' | 'all'>('trending');
    const inputRef = useRef<HTMLInputElement>(null);
    const listRef = useRef<HTMLDivElement>(null);

    // ... imports / memos ...
    // Search results (memoized)
    const searchResults = useMemo(() => {
        if (query.length < 1) return [];
        return searchStocks(query, 15);
    }, [query]);

    // Filtered stocks by sector (memoized)
    const sectorStocks = useMemo(() => {
        if (!selectedSector) return [];
        return STOCK_LIST.filter(s => s.sector === selectedSector).slice(0, 15);
    }, [selectedSector]);

    // Hot stocks lists
    const topGainers = useMemo(() => getTopGainers(8), []);
    const topLosers = useMemo(() => getTopLosers(8), []);

    // Flatten for keyboard navigation
    const flatList = useMemo(() => {
        if (query) return searchResults;
        if (selectedSector) return sectorStocks;
        switch (viewMode) {
            case 'gainers': return topGainers;
            case 'losers': return topLosers;
            case 'trending': return TRENDING_STOCKS;
            default: return BLUE_CHIP_STOCKS;
        }
    }, [query, searchResults, selectedSector, sectorStocks, viewMode, topGainers, topLosers]);

    // FETCH REAL-TIME QUOTES FOR VISIBLE LIST
    // We prioritize checking recent searches to keep history accurate
    const { data: recentQuotes } = useBatchQuotes(recentSearches, { pollingEnabled: isOpen });

    // Helper to find quote for a ticker
    const getRealtimeQuote = useCallback((ticker: string) => {
        if (ticker === currentTicker && currentQuote) return currentQuote;
        if (recentQuotes) {
            const found = recentQuotes.find((q: any) => q && q.ticker === ticker);
            if (found && 'last_price' in found) return found as QuoteData;
        }
        return null;
    }, [currentTicker, currentQuote, recentQuotes]);

    useEffect(() => {
        const saved = localStorage.getItem('recentStocks');
        if (saved) {
            setRecentSearches(JSON.parse(saved).slice(0, 5));
        }
    }, []);

    // ... rest of useEffects ...
    useEffect(() => {
        setActiveIndex(-1);
        if (query && !isOpen) setIsOpen(true);
        if (query) setSelectedSector(null);
    }, [query]);

    const handleSelect = useCallback((stock: Stock) => {
        onSelect(stock.ticker);
        setQuery('');
        setIsOpen(false);
        setActiveIndex(-1);
        setSelectedSector(null);

        setRecentSearches(prev => {
            const newRecents = [stock.ticker, ...prev.filter(s => s !== stock.ticker)].slice(0, 5);
            localStorage.setItem('recentStocks', JSON.stringify(newRecents));
            return newRecents;
        });
    }, [onSelect]);

    // ... keyboard & scroll effects ...
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (!isOpen) return;
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                setActiveIndex(prev => (prev < flatList.length - 1 ? prev + 1 : 0));
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                setActiveIndex(prev => (prev > 0 ? prev - 1 : flatList.length - 1));
            } else if (e.key === 'Enter') {
                e.preventDefault();
                if (activeIndex >= 0 && flatList[activeIndex]) {
                    handleSelect(flatList[activeIndex]);
                } else if (flatList.length > 0) {
                    handleSelect(flatList[0]);
                }
            } else if (e.key === 'Escape') {
                setIsOpen(false);
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [isOpen, flatList, activeIndex, handleSelect]);

    useEffect(() => {
        if (activeIndex >= 0 && listRef.current) {
            const activeItem = listRef.current.querySelector(`[data-index="${activeIndex}"]`);
            if (activeItem) {
                activeItem.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
            }
        }
    }, [activeIndex]);

    const selectedStock = currentTicker ? STOCK_LIST.find(s => s.ticker === currentTicker) : null;

    // Override display logic for Selected Stock Panel
    const displayPrice = currentQuote ? currentQuote.last_price : selectedStock?.lastPrice;
    const displayChange = currentQuote ? currentQuote.change_percent : selectedStock?.change;
    const isPositiveDisplay = (displayChange || 0) >= 0;

    return (
        <div className="relative w-full">
            {/* ... search input section ... */}
            <div className="relative">
                <div className={`
                    relative bg-neutral-900 border-2 rounded-2xl flex items-center overflow-hidden transition-all duration-300
                    ${isOpen
                        ? 'border-blue-500 ring-4 ring-blue-500/20'
                        : 'border-neutral-700 hover:border-neutral-600'
                    }
                `}>
                    <div className="pl-4 pr-3">
                        <Search className={`w-5 h-5 transition-colors duration-200 ${isOpen ? 'text-blue-400' : 'text-neutral-500'}`} />
                    </div>

                    <input
                        ref={inputRef}
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onFocus={() => setIsOpen(true)}
                        placeholder="Search ticker (e.g. BBCA, TLKM)..."
                        className="w-full bg-transparent border-none text-white placeholder-neutral-500 focus:outline-none py-4 text-base font-medium"
                    />

                    <div className="pr-4 flex items-center gap-2">
                        {query && (
                            <button
                                onClick={() => {
                                    setQuery('');
                                    inputRef.current?.focus();
                                }}
                                className="p-2 rounded-full bg-neutral-800 text-neutral-400 hover:text-white hover:bg-neutral-700 transition-all"
                            >
                                <X className="w-4 h-4" />
                            </button>
                        )}
                    </div>
                </div>
            </div>

            {/* Full Screen Modal Overlay */}
            <AnimatePresence>
                {isOpen && (
                    <>
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={() => setIsOpen(false)}
                            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[9998]"
                        />
                        <motion.div
                            initial={{ opacity: 0, y: 20, scale: 0.98 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, y: 20, scale: 0.98 }}
                            transition={{ duration: 0.2 }}
                            className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[95vw] max-w-2xl max-h-[85vh] bg-neutral-900 border-2 border-neutral-700 rounded-3xl shadow-2xl overflow-hidden z-[9999] flex flex-col"
                        >
                            {/* ... Modal Header ... */}
                            <div className="p-4 border-b-2 border-neutral-800 bg-neutral-900">
                                <div className="relative bg-neutral-800 border-2 border-neutral-700 rounded-xl flex items-center">
                                    <div className="pl-4 pr-3">
                                        <Search className="w-5 h-5 text-blue-400" />
                                    </div>
                                    <input
                                        type="text"
                                        value={query}
                                        onChange={(e) => setQuery(e.target.value)}
                                        placeholder="Search ticker..."
                                        className="w-full bg-transparent text-white placeholder-neutral-500 focus:outline-none py-3.5 text-base font-medium"
                                        autoFocus
                                    />
                                    <button
                                        onClick={() => setIsOpen(false)}
                                        className="mr-2 p-2 rounded-lg bg-neutral-700 text-neutral-400 hover:text-white hover:bg-neutral-600 transition-all"
                                    >
                                        <X className="w-4 h-4" />
                                    </button>
                                </div>
                                {/* Sector Chips */}
                                <div className="mt-4 flex gap-2 overflow-x-auto pb-2 hide-scrollbar">
                                    {SECTORS.slice(0, 8).map(sector => (
                                        <SectorChip
                                            key={sector.name}
                                            sector={sector}
                                            isActive={selectedSector === sector.name}
                                            onClick={() => {
                                                setSelectedSector(selectedSector === sector.name ? null : sector.name);
                                                setQuery('');
                                            }}
                                        />
                                    ))}
                                </div>
                            </div>

                            <div ref={listRef} className="flex-1 overflow-y-auto p-4">
                                {query ? (
                                    <div className="space-y-3">
                                        {/* Search Results */}
                                        {searchResults.map((stock, idx) => (
                                            <div key={stock.ticker} data-index={idx}>
                                                <StockItem
                                                    stock={stock}
                                                    isActive={idx === activeIndex}
                                                    isSelected={stock.ticker === currentTicker}
                                                    onClick={() => handleSelect(stock)}
                                                    realtimeData={getRealtimeQuote(stock.ticker)}
                                                />
                                            </div>
                                        ))}
                                    </div>
                                ) : selectedSector ? (
                                    // Sector Results
                                    <div className="space-y-3">
                                        {sectorStocks.map((stock, idx) => (
                                            <div key={stock.ticker} data-index={idx}>
                                                <StockItem
                                                    stock={stock}
                                                    isActive={idx === activeIndex}
                                                    isSelected={stock.ticker === currentTicker}
                                                    onClick={() => handleSelect(stock)}
                                                    realtimeData={getRealtimeQuote(stock.ticker)}
                                                />
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <div className="space-y-6">
                                        {/* Recent Searches */}
                                        {recentSearches.length > 0 && (
                                            <div>
                                                <div className="flex items-center gap-2 text-sm font-semibold text-neutral-400 uppercase tracking-wider mb-3">
                                                    <History className="w-4 h-4" />
                                                    Recent Searches
                                                </div>
                                                <div className="flex flex-wrap gap-2">
                                                    {recentSearches.map(ticker => {
                                                        const stock = STOCK_LIST.find(s => s.ticker === ticker);
                                                        const quote = getRealtimeQuote(ticker);
                                                        if (!stock) return null;

                                                        const pChange = quote ? quote.change_percent : stock.change;

                                                        return (
                                                            <button
                                                                key={ticker}
                                                                onClick={() => handleSelect(stock)}
                                                                className="flex items-center gap-2 px-4 py-2.5 bg-neutral-800 hover:bg-neutral-700 border-2 border-neutral-700 hover:border-neutral-600 rounded-xl transition-all group"
                                                            >
                                                                {/* Minimal logo for recents */}
                                                                <StockLogo ticker={ticker} className="w-5 h-5 rounded-md" />
                                                                <span className="text-sm font-bold text-white">
                                                                    {ticker.replace('.JK', '')}
                                                                </span>
                                                                {pChange !== undefined && (
                                                                    <span className={`text-xs font-bold ${pChange >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                                                        {pChange >= 0 ? '+' : ''}{pChange.toFixed(1)}%
                                                                    </span>
                                                                )}
                                                            </button>
                                                        );
                                                    })}
                                                </div>
                                            </div>
                                        )}

                                        {/* View Mode Tabs */}
                                        <div className="flex gap-2 p-1.5 bg-neutral-800 rounded-xl border-2 border-neutral-700">
                                            {[
                                                { id: 'trending', label: 'Trending', icon: Flame },
                                                { id: 'gainers', label: 'Top Gainers', icon: TrendingUp },
                                                { id: 'losers', label: 'Top Losers', icon: TrendingDown },
                                                { id: 'all', label: 'Blue Chips', icon: Star },
                                            ].map(tab => (
                                                <button
                                                    key={tab.id}
                                                    onClick={() => setViewMode(tab.id as typeof viewMode)}
                                                    className={`flex-1 flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-sm font-bold transition-all ${viewMode === tab.id
                                                        ? 'bg-blue-500 text-white shadow-lg'
                                                        : 'text-neutral-400 hover:text-white hover:bg-neutral-700'
                                                        }`}
                                                >
                                                    <tab.icon className="w-4 h-4" />
                                                    <span className="hidden sm:inline">{tab.label}</span>
                                                </button>
                                            ))}
                                        </div>

                                        {/* Stock List */}
                                        <div className="space-y-3">
                                            {flatList.map((stock, idx) => (
                                                <div key={stock.ticker} data-index={idx}>
                                                    <StockItem
                                                        stock={stock}
                                                        isActive={idx === activeIndex}
                                                        isSelected={stock.ticker === currentTicker}
                                                        onClick={() => handleSelect(stock)}
                                                        realtimeData={getRealtimeQuote(stock.ticker)}
                                                    />
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Modal Footer */}
                            {/* ... footer code ... */}
                            <div className="px-4 py-3 bg-neutral-800 border-t-2 border-neutral-700">
                                <div className="flex items-center justify-between text-xs text-neutral-500 font-medium">
                                    <div className="flex items-center gap-2">
                                        <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                                        <span>Market Open</span>
                                        <span className="text-neutral-600">•</span>
                                        <span>Real-time Data Active</span>
                                    </div>
                                    <div className="flex gap-4">
                                        <span className="flex items-center gap-1.5">
                                            <kbd className="px-2 py-1 bg-neutral-700 rounded text-neutral-400 font-mono text-[10px]">ESC</kbd>
                                            Close
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    </>
                )}
            </AnimatePresence>

            {/* Currently Selected Stock Display */}
            {selectedStock && !isOpen && (
                <motion.div
                    initial={{ opacity: 0, y: -5 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4 bg-neutral-800/50 rounded-2xl border-2 border-neutral-700 p-4"
                >
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            {/* Selected Stock Logo */}
                            <StockLogo ticker={selectedStock.ticker} className="w-14 h-14 rounded-2xl border-2 border-blue-500/30 shadow-lg shadow-blue-500/10" />

                            <div>
                                <div className="flex items-center gap-2">
                                    <span className="text-xl font-bold text-white">
                                        {selectedStock.ticker.replace('.JK', '')}
                                    </span>
                                    {currentQuote && (
                                        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                                    )}
                                    {selectedStock.sector && (
                                        <span className="text-xs bg-blue-500/10 border border-blue-500/20 text-blue-400 px-2.5 py-1 rounded-full font-semibold">
                                            {selectedStock.sector}
                                        </span>
                                    )}
                                </div>
                                <div className="text-sm text-neutral-400 mt-0.5">
                                    {selectedStock.name}
                                </div>
                            </div>
                        </div>
                        {displayPrice !== undefined && (
                            <div className="text-right">
                                <div className="text-2xl font-mono font-bold text-white">
                                    Rp {formatPrice(displayPrice)}
                                </div>
                                <div className={`text-base font-semibold flex items-center justify-end gap-1 ${isPositiveDisplay ? 'text-emerald-400' : 'text-rose-400'
                                    }`}>
                                    {isPositiveDisplay
                                        ? <ArrowUpRight className="w-5 h-5" />
                                        : <ArrowDownRight className="w-5 h-5" />
                                    }
                                    {isPositiveDisplay ? '+' : ''}{displayChange?.toFixed(2)}%
                                </div>
                            </div>
                        )}
                    </div>
                </motion.div>
            )}
        </div>
    );
};
