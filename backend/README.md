# Stock Predictive Analytics Backend

## Overview

A sophisticated stock prediction API combining traditional statistical methods (SARIMA) with deep learning (LSTM with attention mechanism) for Indonesian Stock Exchange (IDX) analysis.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer (FastAPI)                     │
├─────────────────────────────────────────────────────────────────┤
│  REST Endpoints          │  WebSocket (Real-time)              │
│  - /predict/{ticker}     │  - /ws/{ticker}                     │
│  - /stocks               │  - /ws/market                       │
│  - /indicators           │                                     │
└───────────────────────────┴─────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                      Prediction Engine                          │
├───────────────┬─────────────────────────────┬───────────────────┤
│   SARIMA      │    Adaptive Ensemble        │      LSTM         │
│  (Auto-Fit)   │    (Dynamic Weights)        │   (Attention)     │
├───────────────┴─────────────────────────────┴───────────────────┤
│                    Technical Indicators                         │
│  RSI │ MACD │ Bollinger Bands │ Moving Averages │ ATR │ OBV    │
├─────────────────────────────────────────────────────────────────┤
│                    Data Layer                                   │
│              Yahoo Finance API + Data Validation                │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### 1. Hybrid Prediction Model
- **Auto-SARIMA**: Automatic parameter selection (p,d,q)(P,D,Q,s)
- **LSTM with Attention**: Bidirectional LSTM focusing on important patterns
- **Adaptive Ensemble**: Dynamic weight allocation based on:
  - Model fit quality (AIC, validation loss)
  - Market volatility conditions
  - Recent prediction accuracy

### 2. Technical Analysis
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands with width analysis
- Simple & Exponential Moving Averages (SMA/EMA 20, 50, 200)
- Average True Range (ATR) for volatility
- On-Balance Volume (OBV)

### 3. Sentiment Scoring
Multi-signal sentiment analysis combining:
- Technical indicator signals
- Forecast direction
- Model confidence
- Cross-signal confirmation

### 4. Real-time Features
- WebSocket support for live updates
- Prediction caching (15-minute TTL)
- Background task processing

## Installation

### Requirements
- Python 3.9+
- TensorFlow 2.x
- ~2GB RAM for LSTM training

### Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install optional: pmdarima for auto-ARIMA
pip install pmdarima
```

## Running the Server

### Development Mode (with auto-reload)
```bash
python run.py --dev
```

### Production Mode
```bash
python run.py --host 0.0.0.0 --port 8000 --workers 4
```

### Using Uvicorn Directly
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and endpoints |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/stocks` | List available stocks |
| GET | `/api/v1/stocks/{ticker}/info` | Stock details |
| GET | `/api/v1/stocks/{ticker}/quote` | Real-time quote |
| GET | `/api/v1/stocks/{ticker}/indicators` | Technical indicators |
| POST | `/api/v1/predict` | Generate prediction |
| GET | `/api/v1/predict/{ticker}` | Alternative GET prediction |
| POST | `/api/v1/predict/batch` | Batch predictions (max 10) |
| GET | `/api/v1/market/summary` | Market overview |
| DELETE | `/api/v1/cache` | Clear prediction cache |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/v1/ws/{ticker}` | Real-time updates for a stock |
| `/api/v1/ws/market` | Market-wide updates |

## Example Usage

### cURL - Get Prediction
```bash
curl -X GET "http://localhost:8000/api/v1/predict/BBCA.JK" \
     -H "Accept: application/json"
```

### Python - Using requests
```python
import requests

# Get prediction
response = requests.get("http://localhost:8000/api/v1/predict/BBRI.JK")
data = response.json()

print(f"Ticker: {data['ticker']}")
print(f"Sentiment: {data['sentiment']['label_en']}")
print(f"7-Day Forecast: {data['forecasts'][1]['predicted_change_percent']:.2f}%")
```

### JavaScript - WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/BBCA.JK');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'quote_update') {
        console.log(`Price: ${data.price}`);
    }
};

// Request refresh
ws.send('refresh');
```

## Response Format

### Prediction Response
```json
{
    "success": true,
    "ticker": "BBCA.JK",
    "timestamp": "2024-01-15T10:30:00",
    "processing_time_seconds": 25.4,
    "data_quality": { ... },
    "historical_data": {
        "dates": [...],
        "close": [...],
        "volume": [...]
    },
    "indicators": {
        "rsi": { "value": 55.2, "signal": "Neutral" },
        "macd": { "interpretation": "Bullish", ... },
        "moving_averages": { "trend": "Bullish", ... },
        "volatility": { "atr_percent": 2.3 }
    },
    "forecasts": [
        { "horizon": 1, "predicted_change_percent": 0.5, ... },
        { "horizon": 7, "predicted_change_percent": 2.1, ... },
        { "horizon": 30, "predicted_change_percent": 5.3, ... }
    ],
    "trend": {
        "short_term": "Uptrend",
        "long_term": "Uptrend",
        "overall": "Uptrend"
    },
    "sentiment": {
        "level": "BULLISH",
        "label_en": "Bullish",
        "confidence": 0.78,
        "rationale": [...]
    },
    "model_contribution": {
        "sarima_weight": 0.35,
        "lstm_weight": 0.65
    }
}
```

## Performance Notes

### Prediction Timing
- **First request** for a ticker: 20-60 seconds (model fitting)
- **Cached requests**: < 1 second (within 15-minute TTL)
- **Batch requests**: Scales linearly, ~20-30s per ticker

### Resource Usage
- **Memory**: ~500MB base + ~200MB per concurrent LSTM
- **CPU**: Training is CPU-intensive; GPU recommended for production

### Optimization Tips
1. Use caching (enabled by default)
2. Pre-warm popular stocks on startup
3. Consider reducing LSTM epochs for faster training
4. Use batch endpoint for multiple stocks

## Configuration

Key configuration in `app/core/config.py`:

```python
# Model parameters
LSTM_SEQUENCE_LENGTH = 60
LSTM_EPOCHS = 50
SARIMA_MAX_P = 3

# Cache settings
CACHE_TTL_MINUTES = 15
CACHE_MAX_SIZE = 100

# Ensemble weights (adaptive)
DEFAULT_SARIMA_WEIGHT = 0.35
DEFAULT_LSTM_WEIGHT = 0.65
```

## Troubleshooting

### Common Issues

1. **"TensorFlow not found"**
   ```bash
   pip install tensorflow
   ```

2. **"pmdarima not installed"**
   - SARIMA will use fallback statsmodels (less optimal)
   ```bash
   pip install pmdarima
   ```

3. **"No data available for ticker"**
   - Check ticker format (e.g., `BBCA.JK` not `BBCA`)
   - Verify ticker is in STOCK_UNIVERSE

4. **Slow predictions**
   - First prediction for a stock trains models from scratch
   - Subsequent requests use cache

---

## Disclaimer

⚠️ **This system does NOT provide financial advice.**

All predictions are for informational purposes only. Past performance is not indicative of future results. Always conduct your own research before making investment decisions.

---

## License

MIT License - See LICENSE file for details.
