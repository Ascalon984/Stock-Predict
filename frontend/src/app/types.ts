export type Stock = {
    ticker: string;
    name: string;
    sector?: string;
    industry?: string;
};

export type StockListResponse = {
    total: number;
    stocks: Stock[];
};

export type IndicatorSummary = {
    close: number;
    rsi: {
        value: number;
        signal: string;
        zone?: 'overbought' | 'oversold' | 'neutral';
    };
    macd: {
        value: number;
        signal_line: number;
        histogram: number;
        interpretation: string;
        crossover_detected: boolean;
        crossover_type?: 'bullish' | 'bearish' | null;
        strength?: number;
    };
    moving_averages: {
        sma_20: number;
        sma_50: number;
        sma_200: number;
        ema_20: number;
        trend: string;
        price_vs_sma50_pct?: number;
        price_vs_sma200_pct?: number;
    };
    bollinger_bands: {
        upper: number;
        middle: number;
        lower: number;
        width: number;
        position: string;
        percent_b?: number;
    };
    volatility: {
        atr: number;
        atr_percent: number;
        regime?: 'Extreme' | 'High' | 'Moderate' | 'Low' | 'Very Low';
    };
    obv?: {
        value: number;
        trend: 'Accumulation' | 'Distribution' | 'Neutral';
        change_percent: number;
    };
};

export type Forecast = {
    horizon: number;
    horizon_label: string;
    forecast: number[];
    confidence_lower: number[];
    confidence_upper: number[];
    last_price: number;
    predicted_change_percent: number;
};

export type Sentiment = {
    level: string;
    label_id: string;
    label_en: string;
    score: number;
    raw_score: number;
    confidence: number;
    rationale: string[];
    component_scores: {
        total_signals: number;
        bullish_signals: number;
        bearish_signals: number;
        neutral_signals: number;
    };
    indicator_impact?: Record<string, number>;
};

export type ModelContribution = {
    sarima_weight: number;
    lstm_weight: number;
    ensemble_method: string;
};

export type PredictionResponse = {
    success: boolean;
    ticker: string;
    timestamp: string;
    processing_time_seconds: number;
    data_quality: {
        is_valid: boolean;
        warnings: string[];
        data_freshness_hours: number;
    };
    historical_data: {
        dates: string[];
        open: number[];
        high: number[];
        low: number[];
        close: number[];
        volume: number[];
    };
    indicators: IndicatorSummary;
    forecasts: Forecast[];
    trend: {
        short_term: string;
        long_term: string;
        overall: string;
        inconsistency_warning: boolean;
        interpretation: string;
    };
    sentiment: Sentiment;
    model_contribution?: ModelContribution;
    disclaimer: string;
    // Intraday support
    is_intraday?: boolean;
    interval?: string;
    current_price?: number;
    change_percent?: number;
};
