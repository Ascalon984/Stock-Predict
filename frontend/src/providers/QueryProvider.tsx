'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState, ReactNode } from 'react';

interface QueryProviderProps {
    children: ReactNode;
}

export default function QueryProvider({ children }: QueryProviderProps) {
    const [queryClient] = useState(
        () =>
            new QueryClient({
                defaultOptions: {
                    queries: {
                        // Stale time: how long data is considered fresh
                        staleTime: 10 * 1000, // 10 seconds
                        // Cache time: how long to keep inactive data
                        gcTime: 5 * 60 * 1000, // 5 minutes (previously cacheTime)
                        // Retry failed requests
                        retry: 2,
                        retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 10000),
                        // Refetch on window focus (useful for real-time data)
                        refetchOnWindowFocus: true,
                    },
                },
            })
    );

    return (
        <QueryClientProvider client={queryClient}>
            {children}
        </QueryClientProvider>
    );
}
