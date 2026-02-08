import React, { useState, useEffect } from 'react';
import { Image as ImageIcon } from 'lucide-react';

// Cache for broken images to prevent repeated 404 requests
const brokenImageCache = new Set<string>();

export const StockLogo: React.FC<{ ticker: string; className?: string }> = ({ ticker, className }) => {
    const [error, setError] = useState(false);
    const [loaded, setLoaded] = useState(false);

    // Safe check if ticker is undefined
    const displayTicker = ticker || '';
    const tickerName = displayTicker.replace('.JK', '');

    // Handle specific file extensions
    const src = ['AMMAN', 'AMMN'].includes(tickerName)
        ? `/logos/${tickerName}.jfif`
        : `/logos/${tickerName}.png`;

    useEffect(() => {
        // Reset state when ticker changes
        if (brokenImageCache.has(tickerName)) {
            setError(true);
        } else {
            setError(false);
            setLoaded(false);
        }
    }, [tickerName]);

    const handleError = () => {
        brokenImageCache.add(tickerName);
        setError(true);
    };

    if (error || !ticker) {
        // Generate consistent gradient based on ticker string
        const getGradient = (str: string) => {
            const colors = [
                'from-blue-600 to-blue-400',
                'from-emerald-600 to-emerald-400',
                'from-purple-600 to-purple-400',
                'from-orange-600 to-orange-400',
                'from-cyan-600 to-cyan-400',
                'from-indigo-600 to-indigo-400'
            ];
            const index = str.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % colors.length;
            return colors[index];
        };

        return (
            <div className={`flex items-center justify-center font-bold text-white shadow-inner ${className} bg-gradient-to-br ${getGradient(tickerName)}`}>
                <span className="drop-shadow-sm text-[85%]">{tickerName.slice(0, 2)}</span>
            </div>
        );
    }

    return (
        <div className={`relative overflow-hidden bg-white ${className} flex items-center justify-center`}>
            {!loaded && (
                <div className="absolute inset-0 bg-neutral-200 animate-pulse" />
            )}
            <img
                src={src}
                alt={tickerName}
                className={`w-[85%] h-[85%] object-contain transition-opacity duration-300 ${loaded ? 'opacity-100' : 'opacity-0'}`}
                onLoad={() => setLoaded(true)}
                onError={handleError}
                loading="lazy"
            />
        </div>
    );
};
