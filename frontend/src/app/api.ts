import { StockListResponse, PredictionResponse } from './types';

const API_BASE_URL = 'http://localhost:8000/api/v1';

export async function fetchStocks(search: string = '', limit: number = 50): Promise<StockListResponse> {
    const params = new URLSearchParams({
        limit: limit.toString(),
    });
    if (search) {
        params.append('search', search);
    }

    const response = await fetch(`${API_BASE_URL}/stocks?${params.toString()}`);
    if (!response.ok) {
        throw new Error('Failed to fetch stocks');
    }
    return response.json();
}

export async function fetchPrediction(ticker: string, period: string = '1y'): Promise<PredictionResponse> {
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ticker, period }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch prediction');
    }
    return response.json();
}
