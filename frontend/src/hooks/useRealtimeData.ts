/**
 * Real-time Stock Data Hooks using TanStack Query
 * =================================================
 * Provides automatic data fetching and polling for stock data.
 * 
 * Features:
 * - Automatic refetching every 10-30 seconds
 * - Stale-while-revalidate pattern
 * - Error handling with fallback
 * - Market status awareness (pauses polling when market is closed)
 */

import { useQuery, useQueryClient, UseQueryOptions } from '@tanstack/react-query';
import { useEffect, useCallback, useState } from 'react';

// API Base URL
const API_BASE = 'http://localhost:8000/api/v1';

// Types
export interface MarketStatus {
    is_open: boolean;
    session: 'session_1' | 'session_2' | 'break' | 'closed';
    current_time: string;
    day_of_week: string;
}

export interface QuoteData {
    ticker: string;
    name: string;
    last_price: number;
    previous_close: number | null;
    change: number;
    change_percent: number;
    open: number;
    high: number;
    low: number;
    volume: number;
    last_update: string;
    timestamp: string;
    interval_used: '1m' | '1d';
    market_status: MarketStatus;
    intraday?: {
        timestamps: string[];
        prices: number[];
        volumes: number[];
    };
}

export interface HistoricalData {
    dates: string[];
    open: number[];
    high: number[];
    low: number[];
    close: number[];
    volume: number[];
}

export interface ChartData {
    success: boolean;
    ticker: string;
    interval: string;
    period: string;
    is_intraday: boolean;
    data_points: number;
    current_price: number;
    previous_close: number | null;
    change: number;
    change_percent: number;
    last_update: string;
    historical_data: HistoricalData;
    error?: string;
}

// Polling intervals in milliseconds
const POLLING_INTERVALS = {
    MARKET_OPEN: 10000,      // 10 seconds when market is open
    MARKET_BREAK: 30000,     // 30 seconds during break
    MARKET_CLOSED: 60000,    // 1 minute when market is closed
    INTRADAY: 15000,         // 15 seconds for intraday charts
    DAILY: 60000,            // 1 minute for daily charts
};

/**
 * Fetch market status
 */
async function fetchMarketStatus(): Promise<MarketStatus & { market: string }> {
    const response = await fetch(`${API_BASE}/market/status`);
    if (!response.ok) {
        throw new Error('Failed to fetch market status');
    }
    return response.json();
}

/**
 * Fetch real-time quote for a single stock
 */
async function fetchQuote(ticker: string): Promise<QuoteData> {
    const response = await fetch(`${API_BASE}/stocks/${ticker}/quote`);
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch quote');
    }
    return response.json();
}

/**
 * Fetch historical/chart data
 */
async function fetchChartData(
    ticker: string,
    interval: string = '1d',
    period?: string
): Promise<ChartData> {
    const params = new URLSearchParams({ interval });
    if (period) {
        params.append('period', period);
    }

    const response = await fetch(`${API_BASE}/stocks/${ticker}/history?${params}`);
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch chart data');
    }
    return response.json();
}

/**
 * Hook: Market Status with auto-polling
 */
export function useMarketStatus() {
    return useQuery({
        queryKey: ['market-status'],
        queryFn: fetchMarketStatus,
        refetchInterval: 30000, // Check every 30 seconds
        staleTime: 10000,
    });
}

/**
 * Hook: Real-time Quote with smart polling
 * 
 * Automatically adjusts polling interval based on market status
 */
export function useRealtimeQuote(
    ticker: string | null,
    options?: {
        pollingEnabled?: boolean;
    }
) {
    const { data: marketStatus } = useMarketStatus();
    const pollingEnabled = options?.pollingEnabled ?? true;

    // Determine polling interval based on market status
    const getPollingInterval = useCallback(() => {
        if (!pollingEnabled) return false;

        if (marketStatus?.is_open) {
            if (marketStatus.session === 'break') {
                return POLLING_INTERVALS.MARKET_BREAK;
            }
            return POLLING_INTERVALS.MARKET_OPEN;
        }
        return POLLING_INTERVALS.MARKET_CLOSED;
    }, [marketStatus, pollingEnabled]);

    return useQuery({
        queryKey: ['quote', ticker],
        queryFn: () => fetchQuote(ticker!),
        enabled: !!ticker,
        refetchInterval: getPollingInterval(),
        staleTime: 5000,
        retry: 2,
        retryDelay: 1000,
    });
}

/**
 * Hook: Chart Data with polling
 * 
 * Polls more frequently for intraday data when market is open
 * Uses range-specific stale times for optimal caching during tab switches
 */
export function useChartData(
    ticker: string | null,
    interval: string = '1d',
    period?: string,
    options?: {
        pollingEnabled?: boolean;
    }
) {
    const { data: marketStatus } = useMarketStatus();
    const pollingEnabled = options?.pollingEnabled ?? true;
    const isIntraday = ['1m', '5m', '15m', '1h'].includes(interval);

    // Determine polling interval
    const getPollingInterval = useCallback(() => {
        if (!pollingEnabled) return false;

        if (isIntraday && marketStatus?.is_open) {
            return POLLING_INTERVALS.INTRADAY;
        }

        if (marketStatus?.is_open) {
            return POLLING_INTERVALS.MARKET_OPEN;
        }

        return POLLING_INTERVALS.DAILY;
    }, [isIntraday, marketStatus, pollingEnabled]);

    // Range-specific stale times for smoother tab switching
    // Intraday data: 5s stale (changes rapidly)
    // Daily data: 60s stale (changes less frequently)
    // Weekly/longer: 5min stale (changes very infrequently)
    const getStaleTime = useCallback(() => {
        if (isIntraday) return 5000; // 5 seconds
        if (interval === '1wk') return 300000; // 5 minutes
        return 60000; // 1 minute for daily
    }, [interval, isIntraday]);

    // Use gcTime to keep data in cache longer during tab switches
    const getGcTime = useCallback(() => {
        if (isIntraday) return 60000; // 1 minute garbage collection
        return 600000; // 10 minutes for daily/weekly
    }, [isIntraday]);

    return useQuery({
        queryKey: ['chart', ticker, interval, period],
        queryFn: () => fetchChartData(ticker!, interval, period),
        enabled: !!ticker,
        refetchInterval: getPollingInterval(),
        staleTime: getStaleTime(),
        gcTime: getGcTime(),
        retry: 2,
        retryDelay: 1000,
        // Keep previous data during refetch for smoother UX
        placeholderData: (previousData) => previousData,
    });
}

/**
 * Hook: Multiple Quotes (for watchlist)
 */
export function useBatchQuotes(
    tickers: string[],
    options?: {
        pollingEnabled?: boolean;
    }
) {
    const { data: marketStatus } = useMarketStatus();
    const pollingEnabled = options?.pollingEnabled ?? true;

    const getPollingInterval = useCallback(() => {
        if (!pollingEnabled || tickers.length === 0) return false;

        if (marketStatus?.is_open) {
            return POLLING_INTERVALS.MARKET_OPEN;
        }
        return POLLING_INTERVALS.MARKET_CLOSED;
    }, [marketStatus, pollingEnabled, tickers.length]);

    return useQuery({
        queryKey: ['batch-quotes', tickers],
        queryFn: async () => {
            // Fetch quotes in parallel
            const promises = tickers.map(ticker =>
                fetchQuote(ticker).catch(err => ({
                    ticker,
                    error: err.message,
                    success: false
                }))
            );
            return Promise.all(promises);
        },
        enabled: tickers.length > 0,
        refetchInterval: getPollingInterval(),
        staleTime: 5000,
    });
}

/**
 * Hook: Smart Data Fetcher
 * 
 * Combines quote and chart data with intelligent caching
 * This is the main hook to use in the application
 */
export function useStockData(
    ticker: string | null,
    interval: string = '1d',
    period?: string
) {
    const queryClient = useQueryClient();
    const isIntraday = ['1m', '5m', '15m', '1h'].includes(interval);

    // Fetch market status first
    const marketStatus = useMarketStatus();

    // Fetch quote data
    const quote = useRealtimeQuote(ticker, {
        pollingEnabled: true
    });

    // Fetch chart data
    const chart = useChartData(ticker, interval, period, {
        pollingEnabled: true
    });

    // Manual refresh function
    const refresh = useCallback(() => {
        if (ticker) {
            queryClient.invalidateQueries({ queryKey: ['quote', ticker] });
            queryClient.invalidateQueries({ queryKey: ['chart', ticker] });
        }
    }, [queryClient, ticker]);

    return {
        ticker,
        isIntraday,
        marketStatus: marketStatus.data,
        quote: quote.data,
        chart: chart.data,
        isLoading: quote.isLoading || chart.isLoading,
        isRefetching: quote.isRefetching || chart.isRefetching,
        error: quote.error || chart.error,
        lastUpdated: quote.dataUpdatedAt || chart.dataUpdatedAt,
        refresh
    };
}

/**
 * Hook: Connection Status
 * 
 * Monitors API connectivity
 */
export function useConnectionStatus() {
    const [isOnline, setIsOnline] = useState(true);
    const queryClient = useQueryClient();

    useEffect(() => {
        const checkConnection = async () => {
            try {
                const response = await fetch(`${API_BASE}/health`, {
                    method: 'GET',
                    cache: 'no-cache'
                });
                setIsOnline(response.ok);
            } catch {
                setIsOnline(false);
            }
        };

        // Check immediately
        checkConnection();

        // Check every 30 seconds
        const interval = setInterval(checkConnection, 30000);

        // Listen for online/offline events
        const handleOnline = () => {
            setIsOnline(true);
            queryClient.invalidateQueries();
        };

        const handleOffline = () => setIsOnline(false);

        window.addEventListener('online', handleOnline);
        window.addEventListener('offline', handleOffline);

        return () => {
            clearInterval(interval);
            window.removeEventListener('online', handleOnline);
            window.removeEventListener('offline', handleOffline);
        };
    }, [queryClient]);

    return isOnline;
}
