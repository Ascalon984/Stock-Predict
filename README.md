# Stock Market Predictive Analytics System

## Overview
A hybrid stock prediction system combining classical statistical models (SARIMA) with deep learning (LSTM) to forecast stock prices for the Indonesian market (JK).

## Features
- **Hybrid Modeling**: Ensembles SARIMA (trend/seasonality) and LSTM (volatility/non-linear).
- **Uncertainty Quantification**: Provides confidence intervals that widen with time and volatility.
- **Sentiment Analysis**: Derives market sentiment (Sangat Bearish to Sangat Bullish) from technical indicators.
- **Interactive UI**: Dark-themed dashboard with real-time charts using Plotly.

## Installation

### Prerequisites
- Python 3.9+
- Node.js 18+

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```
API will be running at `http://localhost:8000`.
Docs available at `http://localhost:8000/docs`.

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
App will be running at `http://localhost:3000`.

## Architecture
- **Data Source**: `yfinance` (Real-time OHLCV)
- **Backend**: FastAPI
- **Frontend**: Next.js 14 (App Router) + Tailwind CSS
- **Visualization**: Plotly.js

## Disclaimer
This system is for educational and analytical purposes only. **Do not use for financial trading.**
