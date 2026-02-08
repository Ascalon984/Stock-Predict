import { PredictionResponse } from '../app/types';

// Generate realistic OHLCV data
function generateHistoricalData(days: number = 252, basePrice: number = 5000) {
    const dates: string[] = [];
    const open: number[] = [];
    const high: number[] = [];
    const low: number[] = [];
    const close: number[] = [];
    const volume: number[] = [];

    let currentPrice = basePrice;
    const today = new Date();

    for (let i = days; i >= 0; i--) {
        const date = new Date(today);
        date.setDate(date.getDate() - i);

        // Skip weekends
        if (date.getDay() === 0 || date.getDay() === 6) continue;

        dates.push(date.toISOString().split('T')[0]);

        // Generate realistic price movement
        const volatility = 0.02;
        const trend = 0.0003; // Slight upward trend
        const randomWalk = (Math.random() - 0.5) * 2 * volatility;
        const priceChange = currentPrice * (trend + randomWalk);

        const openPrice = currentPrice;
        const closePrice = currentPrice + priceChange;
        const dayRange = Math.abs(priceChange) + currentPrice * volatility * Math.random();
        const highPrice = Math.max(openPrice, closePrice) + dayRange * Math.random();
        const lowPrice = Math.min(openPrice, closePrice) - dayRange * Math.random();

        open.push(Math.round(openPrice));
        close.push(Math.round(closePrice));
        high.push(Math.round(highPrice));
        low.push(Math.round(lowPrice));
        volume.push(Math.round(1000000 + Math.random() * 50000000));

        currentPrice = closePrice;
    }

    return { dates, open, high, low, close, volume };
}

// Generate forecast data
function generateForecast(lastPrice: number, horizon: number, trend: 'up' | 'down' | 'sideways') {
    const forecast: number[] = [];
    const confidenceLower: number[] = [];
    const confidenceUpper: number[] = [];

    let price = lastPrice;
    const trendFactor = trend === 'up' ? 0.002 : trend === 'down' ? -0.002 : 0;
    const volatility = 0.015;

    for (let i = 0; i < horizon; i++) {
        const change = price * (trendFactor + (Math.random() - 0.5) * volatility);
        price += change;
        forecast.push(Math.round(price));

        const uncertainty = price * 0.02 * (1 + i * 0.1); // Uncertainty grows with time
        confidenceLower.push(Math.round(price - uncertainty));
        confidenceUpper.push(Math.round(price + uncertainty));
    }

    return { forecast, confidenceLower, confidenceUpper };
}

// Stock-specific data presets
const STOCK_PRESETS: Record<string, { basePrice: number; trend: 'up' | 'down' | 'sideways'; sentiment: string }> = {
    'BBCA.JK': { basePrice: 7875, trend: 'up', sentiment: 'BULLISH' },
    'BBRI.JK': { basePrice: 5125, trend: 'up', sentiment: 'SANGAT_BULLISH' },
    'BMRI.JK': { basePrice: 6450, trend: 'sideways', sentiment: 'NEUTRAL' },
    'TLKM.JK': { basePrice: 3820, trend: 'down', sentiment: 'BEARISH' },
    'ASII.JK': { basePrice: 5275, trend: 'up', sentiment: 'BULLISH' },
    'BBNI.JK': { basePrice: 5550, trend: 'up', sentiment: 'BULLISH' },
    'ICBP.JK': { basePrice: 11250, trend: 'sideways', sentiment: 'NEUTRAL' },
    'INDF.JK': { basePrice: 7175, trend: 'sideways', sentiment: 'NEUTRAL' },
    'KLBF.JK': { basePrice: 1565, trend: 'up', sentiment: 'BULLISH' },
    'ADRO.JK': { basePrice: 2980, trend: 'up', sentiment: 'SANGAT_BULLISH' },
    'ANTM.JK': { basePrice: 1820, trend: 'sideways', sentiment: 'NEUTRAL' },
    'PGAS.JK': { basePrice: 1445, trend: 'down', sentiment: 'BEARISH' },
    'PTBA.JK': { basePrice: 2860, trend: 'up', sentiment: 'BULLISH' },
    'SMGR.JK': { basePrice: 4120, trend: 'down', sentiment: 'SANGAT_BEARISH' },
};

export function generateDummyPrediction(ticker: string): PredictionResponse {
    const preset = STOCK_PRESETS[ticker] || {
        basePrice: 1000 + Math.random() * 10000,
        trend: ['up', 'down', 'sideways'][Math.floor(Math.random() * 3)] as 'up' | 'down' | 'sideways',
        sentiment: ['BULLISH', 'BEARISH', 'NEUTRAL'][Math.floor(Math.random() * 3)]
    };

    const historical = generateHistoricalData(252, preset.basePrice);
    const lastPrice = historical.close[historical.close.length - 1];

    // Calculate technical indicators
    const closes = historical.close;
    const sma20 = closes.slice(-20).reduce((a, b) => a + b, 0) / 20;
    const sma50 = closes.slice(-50).reduce((a, b) => a + b, 0) / 50;
    const sma200 = closes.slice(-200).reduce((a, b) => a + b, 0) / 200;

    // RSI calculation (simplified)
    const gains: number[] = [];
    const losses: number[] = [];
    for (let i = closes.length - 15; i < closes.length; i++) {
        const change = closes[i] - closes[i - 1];
        if (change > 0) {
            gains.push(change);
            losses.push(0);
        } else {
            gains.push(0);
            losses.push(Math.abs(change));
        }
    }
    const avgGain = gains.reduce((a, b) => a + b, 0) / 14;
    const avgLoss = losses.reduce((a, b) => a + b, 0) / 14;
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    const rsi = 100 - (100 / (1 + rs));

    // Bollinger Bands
    const bbPeriod = closes.slice(-20);
    const bbMean = bbPeriod.reduce((a, b) => a + b, 0) / 20;
    const bbStd = Math.sqrt(bbPeriod.map(x => Math.pow(x - bbMean, 2)).reduce((a, b) => a + b, 0) / 20);
    const bbUpper = bbMean + 2 * bbStd;
    const bbLower = bbMean - 2 * bbStd;

    // ATR calculation (simplified)
    const atrPeriod = 14;
    let atrSum = 0;
    for (let i = closes.length - atrPeriod; i < closes.length; i++) {
        const tr = Math.max(
            historical.high[i] - historical.low[i],
            Math.abs(historical.high[i] - closes[i - 1]),
            Math.abs(historical.low[i] - closes[i - 1])
        );
        atrSum += tr;
    }
    const atr = atrSum / atrPeriod;

    // MACD
    const ema12 = closes.slice(-12).reduce((a, b) => a + b, 0) / 12;
    const ema26 = closes.slice(-26).reduce((a, b) => a + b, 0) / 26;
    const macdLine = ema12 - ema26;
    const signalLine = macdLine * 0.85; // Simplified

    // Generate forecasts
    const forecast1d = generateForecast(lastPrice, 1, preset.trend);
    const forecast7d = generateForecast(lastPrice, 7, preset.trend);
    const forecast30d = generateForecast(lastPrice, 30, preset.trend);

    // Sentiment mapping
    const sentimentMap: Record<string, { label_id: string; label_en: string; score: number }> = {
        'SANGAT_BULLISH': { label_id: 'Sangat Bullish', label_en: 'Very Bullish', score: 0.85 },
        'BULLISH': { label_id: 'Bullish', label_en: 'Bullish', score: 0.65 },
        'NEUTRAL': { label_id: 'Netral', label_en: 'Neutral', score: 0.50 },
        'BEARISH': { label_id: 'Bearish', label_en: 'Bearish', score: 0.35 },
        'SANGAT_BEARISH': { label_id: 'Sangat Bearish', label_en: 'Very Bearish', score: 0.15 },
    };

    const sentimentData = sentimentMap[preset.sentiment] || sentimentMap['NEUTRAL'];

    // Generate rationale
    const rationales: string[] = [];
    if (rsi > 70) rationales.push('RSI indicates overbought conditions (>70)');
    else if (rsi < 30) rationales.push('RSI indicates oversold conditions (<30)');
    else rationales.push(`RSI at ${rsi.toFixed(1)} suggests neutral momentum`);

    if (lastPrice > sma50) rationales.push('Price trading above SMA 50 - bullish signal');
    else rationales.push('Price trading below SMA 50 - bearish signal');

    if (macdLine > signalLine) rationales.push('MACD bullish crossover detected');
    else rationales.push('MACD bearish crossover detected');

    if (lastPrice < bbLower) rationales.push('Price near lower Bollinger Band - potential bounce');
    else if (lastPrice > bbUpper) rationales.push('Price near upper Bollinger Band - resistance expected');

    rationales.push(`ATR volatility at ${((atr / lastPrice) * 100).toFixed(2)}% - ${atr / lastPrice > 0.03 ? 'high' : 'moderate'} volatility`);

    return {
        success: true,
        ticker,
        timestamp: new Date().toISOString(),
        processing_time_seconds: 2.5 + Math.random() * 2,
        data_quality: {
            is_valid: true,
            warnings: [],
            data_freshness_hours: Math.random() * 2,
        },
        historical_data: historical,
        indicators: {
            close: lastPrice,
            rsi: {
                value: rsi,
                signal: rsi > 70 ? 'Overbought' : rsi < 30 ? 'Oversold' : 'Neutral',
            },
            macd: {
                value: macdLine,
                signal_line: signalLine,
                histogram: macdLine - signalLine,
                interpretation: macdLine > signalLine ? 'Bullish' : 'Bearish',
                crossover_detected: Math.abs(macdLine - signalLine) < 10,
            },
            moving_averages: {
                sma_20: Math.round(sma20),
                sma_50: Math.round(sma50),
                sma_200: Math.round(sma200),
                ema_20: Math.round(sma20 * 0.98),
                trend: lastPrice > sma50 ? 'Bullish' : 'Bearish',
            },
            bollinger_bands: {
                upper: Math.round(bbUpper),
                middle: Math.round(bbMean),
                lower: Math.round(bbLower),
                width: ((bbUpper - bbLower) / bbMean * 100),
                position: lastPrice > bbUpper ? 'Above' : lastPrice < bbLower ? 'Below' : 'Within',
            },
            volatility: {
                atr: Math.round(atr),
                atr_percent: (atr / lastPrice) * 100,
            },
        },
        forecasts: [
            {
                horizon: 1,
                horizon_label: '1 Day',
                forecast: forecast1d.forecast,
                confidence_lower: forecast1d.confidenceLower,
                confidence_upper: forecast1d.confidenceUpper,
                last_price: lastPrice,
                predicted_change_percent: ((forecast1d.forecast[0] - lastPrice) / lastPrice) * 100,
            },
            {
                horizon: 7,
                horizon_label: '7 Days',
                forecast: forecast7d.forecast,
                confidence_lower: forecast7d.confidenceLower,
                confidence_upper: forecast7d.confidenceUpper,
                last_price: lastPrice,
                predicted_change_percent: ((forecast7d.forecast[6] - lastPrice) / lastPrice) * 100,
            },
            {
                horizon: 30,
                horizon_label: '1 Month',
                forecast: forecast30d.forecast,
                confidence_lower: forecast30d.confidenceLower,
                confidence_upper: forecast30d.confidenceUpper,
                last_price: lastPrice,
                predicted_change_percent: ((forecast30d.forecast[29] - lastPrice) / lastPrice) * 100,
            },
        ],
        trend: {
            short_term: preset.trend === 'up' ? 'Uptrend' : preset.trend === 'down' ? 'Downtrend' : 'Sideways',
            long_term: lastPrice > sma200 ? 'Uptrend' : 'Downtrend',
            overall: preset.trend === 'up' ? 'Uptrend' : preset.trend === 'down' ? 'Downtrend' : 'Sideways',
            inconsistency_warning: (preset.trend === 'up' && lastPrice < sma200) || (preset.trend === 'down' && lastPrice > sma200),
            interpretation: `Stock showing ${preset.trend === 'up' ? 'bullish' : preset.trend === 'down' ? 'bearish' : 'consolidating'} pattern with ${lastPrice > sma50 ? 'strong' : 'weak'} momentum`,
        },
        sentiment: {
            level: preset.sentiment,
            label_id: sentimentData.label_id,
            label_en: sentimentData.label_en,
            score: sentimentData.score,
            raw_score: sentimentData.score,
            confidence: 0.7 + Math.random() * 0.25,
            rationale: rationales,
            component_scores: {
                total_signals: 8,
                bullish_signals: preset.sentiment.includes('BULLISH') ? 5 + Math.floor(Math.random() * 2) : 2 + Math.floor(Math.random() * 2),
                bearish_signals: preset.sentiment.includes('BEARISH') ? 5 + Math.floor(Math.random() * 2) : 2 + Math.floor(Math.random() * 2),
                neutral_signals: 1 + Math.floor(Math.random() * 2),
            },
        },
        model_contribution: {
            sarima_weight: 0.35 + Math.random() * 0.15,
            lstm_weight: 0.35 + Math.random() * 0.15,
            ensemble_method: 'weighted_average',
        },
        disclaimer: 'This analysis is for informational purposes only and does not constitute financial advice. Past performance is not indicative of future results. Always conduct your own research before making investment decisions.',
    };
}
