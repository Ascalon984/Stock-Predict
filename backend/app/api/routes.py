"""Enhanced API endpoints with WebSocket support for real-time updates."""
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import json
import logging

from ..models.hybrid import create_predictor
from ..services.data_loader import data_loader
from ..services.feature_engine import feature_engine
from ..services.realtime_fetcher import (
    fetch_realtime_quote,
    fetch_intraday_chart_data,
    fetch_batch_quotes,
    get_market_status
)
from ..core.config import STOCK_UNIVERSE, STOCK_NAMES

router = APIRouter()
logger = logging.getLogger(__name__)

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}  # ticker -> connections
    
    async def connect(self, websocket: WebSocket, ticker: str):
        await websocket.accept()
        if ticker not in self.active_connections:
            self.active_connections[ticker] = []
        self.active_connections[ticker].append(websocket)
        logger.info(f"WebSocket connected for {ticker}")
    
    def disconnect(self, websocket: WebSocket, ticker: str):
        if ticker in self.active_connections:
            self.active_connections[ticker].remove(websocket)
            if not self.active_connections[ticker]:
                del self.active_connections[ticker]
        logger.info(f"WebSocket disconnected for {ticker}")
    
    async def broadcast(self, ticker: str, message: dict):
        if ticker in self.active_connections:
            for connection in self.active_connections[ticker]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Broadcast error: {e}")

manager = ConnectionManager()


# Pydantic Models
class StockInfo(BaseModel):
    """Stock information response model."""
    ticker: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    last_price: Optional[float] = None
    change_percent: Optional[float] = None


class PredictionRequest(BaseModel):
    """Prediction request model."""
    ticker: str = Field(..., description="Stock ticker (e.g., BBCA.JK)")
    period: str = Field("1y", description="Historical data period")
    force_refresh: bool = Field(False, description="Bypass cache")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    tickers: List[str] = Field(..., max_length=10)
    period: str = "1y"


class StockListResponse(BaseModel):
    """Response model for stock list."""
    total: int
    stocks: List[StockInfo]


class QuickQuoteResponse(BaseModel):
    """Quick quote response."""
    ticker: str
    name: str
    last_price: float
    change: float
    change_percent: float
    volume: int
    timestamp: str


# ============================================
# REST Endpoints
# ============================================

@router.get("/stocks", response_model=StockListResponse)
async def get_stock_list(
    search: Optional[str] = Query(None, description="Search term"),
    sector: Optional[str] = Query(None, description="Filter by sector"),
    limit: int = Query(50, ge=1, le=300),
    offset: int = Query(0, ge=0)
):
    """
    Get list of available Indonesian stocks.
    
    Supports pagination and filtering by search term or sector.
    """
    stocks = []
    
    for ticker in STOCK_UNIVERSE:
        name = STOCK_NAMES.get(ticker, ticker.replace(".JK", ""))
        
        # Apply search filter
        if search:
            search_lower = search.lower()
            if search_lower not in ticker.lower() and search_lower not in name.lower():
                continue
        
        stocks.append(StockInfo(
            ticker=ticker,
            name=name
        ))
    
    total = len(stocks)
    stocks = stocks[offset:offset + limit]
    
    return StockListResponse(total=total, stocks=stocks)


@router.get("/stocks/{ticker}/info")
async def get_stock_info(ticker: str):
    """Get detailed information about a specific stock."""
    # Ensure .JK suffix
    if not ticker.endswith(".JK"):
        ticker = f"{ticker}.JK"
    
    if ticker not in STOCK_UNIVERSE:
        raise HTTPException(
            status_code=404,
            detail=f"Stock {ticker} not found in universe"
        )
    
    info = data_loader.get_stock_info(ticker)
    return info


@router.get("/stocks/{ticker}/history")
async def get_stock_history(
    ticker: str,
    interval: str = Query("1d", description="Data interval: 1m, 5m, 15m, 1h, 1d, 1wk"),
    period: str = Query(None, description="Data period: 1d, 5d, 1mo, 3mo, 6mo, 1y")
):
    """
    Get historical OHLCV data with flexible interval support for intraday charts.
    
    Supported intervals:
    - 1m: 1 minute (max 7 days of data)
    - 5m: 5 minutes (max 60 days)
    - 15m: 15 minutes (max 60 days)
    - 1h: 1 hour (max 730 days)
    - 1d: 1 day (max 10 years)
    - 1wk: 1 week
    
    Returns OHLCV data formatted for charting with proper timezone handling.
    """
    import yfinance as yf
    from zoneinfo import ZoneInfo
    
    if not ticker.endswith(".JK"):
        ticker = f"{ticker}.JK"
    
    if ticker not in STOCK_UNIVERSE:
        raise HTTPException(status_code=404, detail=f"Stock {ticker} not found")
    
    # Set appropriate period based on interval
    interval_periods = {
        "1m": period or "1d",     # 1 minute max 7 days
        "5m": period or "5d",     # 5 minute max 60 days
        "15m": period or "5d",    # 15 minute max 60 days
        "1h": period or "1mo",    # 1 hour max 730 days
        "1d": period or "6mo",    # daily
        "1wk": period or "1y",    # weekly
    }
    
    data_period = interval_periods.get(interval, period or "6mo")
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=data_period, interval=interval)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Define Jakarta timezone for IDX market
        jakarta_tz = ZoneInfo("Asia/Jakarta")
        
        # Format datetime for intraday (include time with proper timezone)
        is_intraday = interval in ["1m", "5m", "15m", "1h"]
        
        dates = []
        for dt in hist.index:
            try:
                # Convert to Jakarta timezone if timezone-aware
                if dt.tzinfo is not None:
                    dt_jakarta = dt.astimezone(jakarta_tz)
                else:
                    # If naive, assume it's already in correct timezone
                    dt_jakarta = dt
                
                if is_intraday:
                    # Format as ISO datetime for Plotly to parse correctly
                    # Using ISO format ensures proper parsing: "2026-02-05T14:30:00"
                    dates.append(dt_jakarta.strftime("%Y-%m-%dT%H:%M:%S"))
                else:
                    # Just date for daily/weekly
                    dates.append(dt_jakarta.strftime("%Y-%m-%d"))
            except Exception:
                # Fallback to simple formatting
                if is_intraday:
                    dates.append(dt.strftime("%Y-%m-%dT%H:%M:%S"))
                else:
                    dates.append(dt.strftime("%Y-%m-%d"))
        
        # Get last close for calculating change
        current_price = float(hist['Close'].iloc[-1])
        prev_price = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        change = current_price - prev_price
        change_pct = (change / prev_price * 100) if prev_price else 0
        
        # Get the last timestamp for display
        last_timestamp = dates[-1] if dates else datetime.now().isoformat()
        
        return {
            "success": True,
            "ticker": ticker,
            "name": STOCK_NAMES.get(ticker, ticker.replace(".JK", "")),
            "interval": interval,
            "period": data_period,
            "is_intraday": is_intraday,
            "data_points": len(hist),
            "last_updated": datetime.now().isoformat(),
            "last_data_time": last_timestamp,
            "current_price": round(current_price, 2),
            "change": round(change, 2),
            "change_percent": round(change_pct, 2),
            "historical_data": {
                "dates": dates,
                "open": [round(float(x), 2) for x in hist['Open'].tolist()],
                "high": [round(float(x), 2) for x in hist['High'].tolist()],
                "low": [round(float(x), 2) for x in hist['Low'].tolist()],
                "close": [round(float(x), 2) for x in hist['Close'].tolist()],
                "volume": [int(x) for x in hist['Volume'].tolist()]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"History fetch error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stocks/{ticker}/quote")
async def get_quick_quote(ticker: str):
    """
    Get real-time quote for a stock using accurate 1-minute data.
    
    Uses previousClose from ticker.info for accurate change calculation.
    Falls back to daily data if 1m data unavailable.
    
    Returns current price, change, and volume.
    """
    if not ticker.endswith(".JK"):
        ticker = f"{ticker}.JK"
    
    if ticker not in STOCK_UNIVERSE:
        raise HTTPException(status_code=404, detail=f"Stock {ticker} not found")
    
    try:
        # Use realtime_fetcher for accurate data
        result = fetch_realtime_quote(ticker)
        
        if not result.get("success"):
            raise HTTPException(
                status_code=404, 
                detail=result.get("error", "No data available")
            )
        
        data = result["data"]
        
        return {
            "ticker": ticker,
            "name": STOCK_NAMES.get(ticker, ticker.replace(".JK", "")),
            "last_price": data["last_price"],
            "previous_close": data["previous_close"],
            "change": data["change"],
            "change_percent": data["change_percent"],
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "volume": data["volume"],
            "last_update": data["last_update"],
            "timestamp": result["timestamp"],
            "interval_used": result["interval_used"],
            "market_status": result["market_status"],
            "intraday": result.get("intraday")  # Mini chart data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quote error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stocks/{ticker}/indicators")
async def get_technical_indicators(
    ticker: str,
    period: str = Query("6mo", description="Data period")
):
    """
    Get comprehensive technical indicators for a stock.
    
    Includes RSI, MACD, Moving Averages, Bollinger Bands, and ATR.
    """
    if not ticker.endswith(".JK"):
        ticker = f"{ticker}.JK"
    
    if ticker not in STOCK_UNIVERSE:
        raise HTTPException(status_code=404, detail=f"Stock {ticker} not found")
    
    df, quality = data_loader.fetch_stock_data(ticker, period="2y")
    
    if df is None:
        raise HTTPException(status_code=500, detail="Failed to fetch data")
    
    df_with_indicators = feature_engine.compute_all_indicators(df)
    indicators = feature_engine.get_indicator_summary(df_with_indicators)
    
    return {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "data_quality": quality.to_dict(),
        "indicators": indicators
    }


@router.post("/predict")
async def predict_stock(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Generate comprehensive AI predictions for a stock.
    
    This endpoint runs the hybrid SARIMA+LSTM model to produce:
    - Price forecasts for 1, 7, and 30 days
    - Confidence intervals with volatility adjustment
    - Technical indicator analysis
    - Market sentiment scoring
    - Trend classification
    
    **Note**: First-time predictions may take 20-60 seconds due to model fitting.
    Subsequent requests use caching.
    
    **Disclaimer**: This is not financial advice.
    """
    ticker = request.ticker
    
    # Ensure .JK suffix
    if not ticker.endswith(".JK"):
        ticker = f"{ticker}.JK"
    
    if ticker not in STOCK_UNIVERSE:
        raise HTTPException(
            status_code=404,
            detail=f"Stock {ticker} not found. Use a valid IDX ticker."
        )
    
    try:
        predictor = create_predictor(use_cache=True)
        result = predictor.predict(
            ticker,
            period=request.period,
            force_refresh=request.force_refresh
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Prediction failed")
            )
        
        # Broadcast to WebSocket clients
        background_tasks.add_task(
            manager.broadcast,
            ticker,
            {"type": "prediction_update", "data": result}
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predict/{ticker}")
async def predict_stock_get(
    ticker: str,
    period: str = Query("1y", description="Historical data period"),
    force_refresh: bool = Query(False, description="Bypass cache")
):
    """GET endpoint for predictions (convenient for testing)."""
    request = PredictionRequest(ticker=ticker, period=period, force_refresh=force_refresh)
    return await predict_stock(request, BackgroundTasks())


@router.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """
    Generate predictions for multiple stocks.
    
    Maximum 10 stocks per request. Results are returned as they complete.
    """
    if len(request.tickers) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 tickers per batch request"
        )
    
    results = {}
    predictor = create_predictor(use_cache=True)
    
    for ticker in request.tickers:
        if not ticker.endswith(".JK"):
            ticker = f"{ticker}.JK"
        
        if ticker not in STOCK_UNIVERSE:
            results[ticker] = {"success": False, "error": "Not found"}
            continue
        
        try:
            result = predictor.predict(ticker, period=request.period)
            results[ticker] = {
                "success": result.get("success", False),
                "sentiment": result.get("sentiment", {}).get("label_en"),
                "trend": result.get("trend", {}).get("overall"),
                "forecast_7d": result.get("forecasts", [{}])[1].get("predicted_change_percent") if len(result.get("forecasts", [])) > 1 else None
            }
        except Exception as e:
            results[ticker] = {"success": False, "error": str(e)}
    
    return {
        "timestamp": datetime.now().isoformat(),
        "results": results
    }


@router.get("/market/summary")
async def get_market_summary():
    """
    Get market summary with top gainers/losers.
    
    Provides a quick overview of market conditions.
    """
    # Sample of popular stocks for quick summary
    sample_tickers = ["BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK", "BMRI.JK"]
    
    results = []
    for ticker in sample_tickers:
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2d")
            
            if not hist.empty:
                last_price = float(hist['Close'].iloc[-1])
                prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else last_price
                change_pct = ((last_price - prev_close) / prev_close * 100) if prev_close else 0
                
                results.append({
                    "ticker": ticker.replace(".JK", ""),
                    "name": STOCK_NAMES.get(ticker, ticker),
                    "price": round(last_price, 2),
                    "change_percent": round(change_pct, 2)
                })
        except:
            continue
    
    return {
        "timestamp": datetime.now().isoformat(),
        "market": "IDX",
        "stocks": results
    }


@router.delete("/cache")
async def clear_cache(ticker: Optional[str] = None):
    """
    Clear prediction cache.
    
    If ticker is provided, only that ticker's cache is cleared.
    Otherwise, all cache is cleared.
    """
    predictor = create_predictor(use_cache=True)
    if predictor.cache_manager:
        predictor.cache_manager.invalidate(ticker)
    
    return {
        "success": True,
        "message": f"Cache cleared for {ticker}" if ticker else "All cache cleared"
    }


@router.get("/market/status")
async def get_market_status_endpoint():
    """
    Get the current IDX market status.
    
    Returns:
    - is_open: Whether the market is currently open
    - session: Current session (session_1, break, session_2, closed)
    - current_time: Current time in WIB
    - trading_hours: IDX trading schedule
    """
    status = get_market_status()
    
    return {
        **status,
        "market": "IDX",
        "trading_hours": {
            "pre_opening": "08:45 - 09:00 WIB",
            "session_1": "09:00 - 11:30 WIB",
            "break": "11:30 - 13:30 WIB",
            "session_2": "13:30 - 15:30 WIB"
        },
        "notes": "Market closed on weekends and Indonesian public holidays"
    }


@router.post("/stocks/batch-quote")
async def get_batch_quotes(tickers: List[str] = Query(..., max_length=10)):
    """
    Get real-time quotes for multiple stocks in a single request.
    
    Maximum 10 tickers per request.
    """
    if len(tickers) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 tickers per request")
    
    # Ensure .JK suffix
    normalized_tickers = [
        t if t.endswith(".JK") else f"{t}.JK"
        for t in tickers
    ]
    
    result = fetch_batch_quotes(normalized_tickers)
    
    return result


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "components": {
            "data_loader": "ok",
            "feature_engine": "ok",
            "prediction_engine": "ok"
        }
    }


# ============================================
# WebSocket Endpoints
# ============================================

@router.websocket("/ws/{ticker}")
async def websocket_endpoint(websocket: WebSocket, ticker: str):
    """
    WebSocket endpoint for real-time stock updates.
    
    Provides:
    - Real-time price updates (every 30 seconds during market hours)
    - Instant prediction update notifications
    """
    if not ticker.endswith(".JK"):
        ticker = f"{ticker}.JK"
    
    await manager.connect(websocket, ticker)
    
    try:
        # Send initial data
        initial_data = {
            "type": "connected",
            "ticker": ticker,
            "message": f"Connected to real-time updates for {ticker}"
        }
        await websocket.send_json(initial_data)
        
        # Keep connection alive and send periodic updates
        update_interval = 30  # seconds
        
        while True:
            try:
                # Check for client messages (ping/pong)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=update_interval
                )
                
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data == "refresh":
                    # Client requested refresh
                    predictor = create_predictor()
                    result = predictor.predict(ticker, force_refresh=True)
                    await websocket.send_json({
                        "type": "prediction_update",
                        "data": result
                    })
                    
            except asyncio.TimeoutError:
                # Send periodic quote update
                try:
                    import yfinance as yf
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="1d")
                    
                    if not hist.empty:
                        await websocket.send_json({
                            "type": "quote_update",
                            "ticker": ticker,
                            "price": round(float(hist['Close'].iloc[-1]), 2),
                            "volume": int(hist['Volume'].iloc[-1]),
                            "timestamp": datetime.now().isoformat()
                        })
                except Exception as e:
                    logger.error(f"Quote update error: {e}")
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket, ticker)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, ticker)


@router.websocket("/ws/market")
async def market_websocket(websocket: WebSocket):
    """
    WebSocket for market-wide updates.
    
    Broadcasts updates for multiple stocks.
    """
    await websocket.accept()
    
    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=60
                )
                
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
                    
            except asyncio.TimeoutError:
                # Send market summary
                try:
                    summary = await get_market_summary()
                    await websocket.send_json({
                        "type": "market_update",
                        "data": summary
                    })
                except Exception as e:
                    logger.error(f"Market summary error: {e}")
                    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Market WebSocket error: {e}")
