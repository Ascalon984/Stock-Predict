"""
Real-time Stock Data Fetcher for Indonesian Market (IDX)
=========================================================
Fetches accurate, real-time data from Yahoo Finance with proper handling
for Indonesian stocks (.JK suffix) and timezone conversion to WIB.

Features:
- OHLCV data with Previous Close for accurate change calculation
- Automatic fallback from intraday (1m) to daily (1d) data
- Timezone conversion to Asia/Jakarta (WIB)
- JSON output for easy integration with Node.js
"""

import yfinance as yf
import requests
import json
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, List
import logging
from .db_service import db_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Jakarta timezone for IDX
JAKARTA_TZ = ZoneInfo("Asia/Jakarta")

# Configure yfinance with custom session to avoid rate limiting
def get_yf_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })
    return session


def get_market_status() -> Dict[str, Any]:
    """Check if IDX market is currently open."""
    now = datetime.now(JAKARTA_TZ)
    day = now.weekday()  # Monday = 0, Sunday = 6
    hour = now.hour
    minute = now.minute
    
    # IDX trading hours: Mon-Fri, 09:00 - 15:30 WIB
    # Pre-opening: 08:45 - 09:00
    # Session 1: 09:00 - 11:30
    # Break: 11:30 - 13:30
    # Session 2: 13:30 - 15:30
    
    is_weekday = day < 5
    is_trading_hours = (
        (hour == 9 and minute >= 0) or
        (hour > 9 and hour < 11) or
        (hour == 11 and minute <= 30) or
        (hour == 13 and minute >= 30) or
        (hour > 13 and hour < 15) or
        (hour == 15 and minute <= 30)
    )
    
    is_open = is_weekday and is_trading_hours
    
    session = "closed"
    if is_open:
        if hour < 12:
            session = "session_1"
        else:
            session = "session_2"
    elif is_weekday and hour == 11 and minute > 30:
        session = "break"
    elif is_weekday and hour == 12:
        session = "break"
    elif is_weekday and hour == 13 and minute < 30:
        session = "break"
    
    return {
        "is_open": is_open,
        "session": session,
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S WIB"),
        "day_of_week": now.strftime("%A")
    }


def fetch_realtime_quote(ticker: str) -> Dict[str, Any]:
    """
    Fetch real-time quote data for a single ticker.
    
    Uses 1-minute data for accuracy, falls back to daily if unavailable.
    Calculates change based on previousClose from ticker.info.
    
    Args:
        ticker: Stock ticker (e.g., 'BBCA.JK' or 'BBCA')
    
    Returns:
        Dictionary with real-time market data
    """
    # Ensure .JK suffix for Indonesian stocks
    if not ticker.upper().endswith(".JK"):
        ticker = f"{ticker.upper()}.JK"
    
    result = {
        "ticker": ticker,
        "success": False,
        "timestamp": datetime.now(JAKARTA_TZ).isoformat(),
        "market_status": get_market_status()
    }
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get ticker info for previousClose
        info = stock.info
        previous_close = info.get('previousClose') or info.get('regularMarketPreviousClose')
        
        # Try to get 1-minute data first (most accurate for real-time)
        hist = None
        interval_used = "1m"
        
        try:
            # Fetch 2 days of 1-minute data
            hist = stock.history(period="2d", interval="1m")
            
            if hist.empty:
                raise ValueError("No 1m data available")
                
        except Exception as e:
            logger.warning(f"1m data unavailable for {ticker}, falling back to 1d: {e}")
            interval_used = "1d"
            
            # Fallback to daily data
            hist = stock.history(period="5d", interval="1d")
        
        if hist.empty:
            result["error"] = "No data available for this ticker"
            return result
        
        # Get the latest data point
        latest = hist.iloc[-1]
        latest_time = hist.index[-1]
        
        # Convert timestamp to Jakarta timezone
        if latest_time.tzinfo is not None:
            latest_time_jakarta = latest_time.astimezone(JAKARTA_TZ)
        else:
            latest_time_jakarta = latest_time.replace(tzinfo=JAKARTA_TZ)
        
        # Current price data
        current_price = float(latest['Close'])
        open_price = float(latest['Open'])
        high_price = float(latest['High'])
        low_price = float(latest['Low'])
        volume = int(latest['Volume'])
        
        # Calculate change from previousClose (standard method)
        if previous_close:
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100
        else:
            # Fallback: calculate from first record of the day
            today_data = hist[hist.index.date == latest_time.date()] if interval_used == "1m" else hist
            if len(today_data) > 0:
                first_close = float(today_data.iloc[0]['Close'])
                change = current_price - first_close
                change_percent = (change / first_close) * 100 if first_close else 0
            else:
                change = 0
                change_percent = 0
        
        # Get day's OHLC (aggregate from all today's data for intraday)
        if interval_used == "1m":
            today_data = hist[hist.index.date == latest_time.date()]
            if len(today_data) > 0:
                day_open = float(today_data.iloc[0]['Open'])
                day_high = float(today_data['High'].max())
                day_low = float(today_data['Low'].min())
                day_volume = int(today_data['Volume'].sum())
            else:
                day_open = open_price
                day_high = high_price
                day_low = low_price
                day_volume = volume
        else:
            day_open = open_price
            day_high = high_price
            day_low = low_price
            day_volume = volume
        
        result.update({
            "success": True,
            "interval_used": interval_used,
            "data": {
                "last_price": round(current_price, 2),
                "previous_close": round(previous_close, 2) if previous_close else None,
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "open": round(day_open, 2),
                "high": round(day_high, 2),
                "low": round(day_low, 2),
                "close": round(current_price, 2),
                "volume": day_volume,
                "last_update": latest_time_jakarta.strftime("%Y-%m-%d %H:%M:%S"),
                "last_update_iso": latest_time_jakarta.isoformat()
            },
            # Intraday data for charting (last 60 points)
            "intraday": None
        })
        
        # Add intraday data if using 1m interval
        if interval_used == "1m" and len(hist) > 0:
            # Get last 60 data points for mini chart
            recent = hist.tail(60)
            result["intraday"] = {
                "timestamps": [
                    dt.astimezone(JAKARTA_TZ).strftime("%H:%M") 
                    for dt in recent.index
                ],
                "prices": [round(float(p), 2) for p in recent['Close'].tolist()],
                "volumes": [int(v) for v in recent['Volume'].tolist()]
            }
            
        # PERSISTENCE: Save to database
        try:
            db_service.save_market_data(ticker, result["data"], interval_used)
        except Exception as e:
            logger.error(f"Failed to save {ticker} to DB: {e}")
        
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        result["error"] = str(e)
    
    return result


def fetch_batch_quotes(tickers: List[str]) -> Dict[str, Any]:
    """
    Fetch real-time quotes for multiple tickers.
    
    Args:
        tickers: List of stock tickers
    
    Returns:
        Dictionary with results for all tickers
    """
    results = {
        "timestamp": datetime.now(JAKARTA_TZ).isoformat(),
        "market_status": get_market_status(),
        "quotes": {}
    }
    
    for ticker in tickers:
        results["quotes"][ticker] = fetch_realtime_quote(ticker)
    
    return results


def fetch_intraday_chart_data(
    ticker: str,
    interval: str = "5m",
    period: str = "1d"
) -> Dict[str, Any]:
    """
    Fetch intraday chart data with proper time labels.
    
    Args:
        ticker: Stock ticker
        interval: Data interval (1m, 5m, 15m, 1h)
        period: Data period (1d, 5d, etc.)
    
    Returns:
        Chart-ready data with proper WIB time labels
    """
    if not ticker.upper().endswith(".JK"):
        ticker = f"{ticker.upper()}.JK"
    
    result = {
        "ticker": ticker,
        "success": False,
        "interval": interval,
        "period": period,
        "timestamp": datetime.now(JAKARTA_TZ).isoformat()
    }
    
    # Validate interval
    valid_intervals = ["1m", "5m", "15m", "1h", "1d", "1wk"]
    if interval not in valid_intervals:
        result["error"] = f"Invalid interval. Must be one of: {valid_intervals}"
        return result
    
    # Adjust period based on interval limits
    interval_max_periods = {
        "1m": "7d",
        "5m": "60d",
        "15m": "60d",
        "1h": "730d",
        "1d": "10y",
        "1wk": "10y"
    }
    
    try:
        # Use default session which handles headers better in v1.1.0
        stock = yf.Ticker(ticker)
        
        # Get ticker info for additional data
        info = stock.info
        previous_close = info.get('previousClose') or info.get('regularMarketPreviousClose')
        
        # Fetch historical data
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            # Try fallback to daily
            logger.warning(f"No {interval} data for {ticker}, trying 1d")
            hist = stock.history(period="5d", interval="1d")
            interval = "1d"
            
            if hist.empty:
                result["error"] = "No data available"
                return result
        
        is_intraday = interval in ["1m", "5m", "15m", "1h"]
        
        # Convert timestamps to WIB
        dates = []
        for dt in hist.index:
            try:
                if dt.tzinfo is not None:
                    dt_jakarta = dt.astimezone(JAKARTA_TZ)
                else:
                    dt_jakarta = dt
                
                if is_intraday:
                    # ISO format for proper Plotly parsing
                    dates.append(dt_jakarta.strftime("%Y-%m-%dT%H:%M:%S"))
                else:
                    dates.append(dt_jakarta.strftime("%Y-%m-%d"))
            except Exception:
                dates.append(dt.strftime("%Y-%m-%dT%H:%M:%S") if is_intraday else dt.strftime("%Y-%m-%d"))
        
        # Calculate current change
        current_price = float(hist['Close'].iloc[-1])
        if previous_close:
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100
        elif len(hist) > 1:
            prev = float(hist['Close'].iloc[-2])
            change = current_price - prev
            change_percent = (change / prev) * 100 if prev else 0
        else:
            change = 0
            change_percent = 0
        
        result.update({
            "success": True,
            "is_intraday": is_intraday,
            "data_points": len(hist),
            "current_price": round(current_price, 2),
            "previous_close": round(previous_close, 2) if previous_close else None,
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "last_update": dates[-1] if dates else None,
            "historical_data": {
                "dates": dates,
                "open": [round(float(x), 2) for x in hist['Open'].tolist()],
                "high": [round(float(x), 2) for x in hist['High'].tolist()],
                "low": [round(float(x), 2) for x in hist['Low'].tolist()],
                "close": [round(float(x), 2) for x in hist['Close'].tolist()],
                "volume": [int(x) for x in hist['Volume'].tolist()]
            }
        })
        
        # PERSISTENCE: Save historical data to database
        try:
            db_service.save_bulk_history(ticker, result["historical_data"], interval)
        except Exception as e:
            logger.error(f"Failed to save history for {ticker} to DB: {e}")
        
    except Exception as e:
        logger.error(f"Chart data error for {ticker}: {e}")
        result["error"] = str(e)
    
    return result


# CLI interface for Node.js subprocess calls
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch real-time stock data from Yahoo Finance")
    parser.add_argument("command", choices=["quote", "batch", "chart", "status"],
                       help="Command to execute")
    parser.add_argument("--ticker", "-t", help="Stock ticker (e.g., BBCA.JK)")
    parser.add_argument("--tickers", "-T", nargs="+", help="Multiple tickers for batch")
    parser.add_argument("--interval", "-i", default="5m", help="Data interval (1m, 5m, 15m, 1h, 1d)")
    parser.add_argument("--period", "-p", default="1d", help="Data period (1d, 5d, 1mo, etc.)")
    
    args = parser.parse_args()
    
    if args.command == "status":
        result = get_market_status()
    elif args.command == "quote":
        if not args.ticker:
            result = {"error": "Ticker required for quote command"}
        else:
            result = fetch_realtime_quote(args.ticker)
    elif args.command == "batch":
        if not args.tickers:
            result = {"error": "Tickers required for batch command"}
        else:
            result = fetch_batch_quotes(args.tickers)
    elif args.command == "chart":
        if not args.ticker:
            result = {"error": "Ticker required for chart command"}
        else:
            result = fetch_intraday_chart_data(args.ticker, args.interval, args.period)
    else:
        result = {"error": "Unknown command"}
    
    # Output JSON for Node.js to parse
    print(json.dumps(result, ensure_ascii=False))
