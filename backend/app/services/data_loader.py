"""Data loader service for fetching stock data from yfinance."""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataQualityReport:
    """Data quality validation report."""
    
    def __init__(self):
        self.missing_values = 0
        self.total_rows = 0
        self.gaps_detected = []
        self.is_valid = True
        self.warnings = []
        self.data_freshness_hours = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "missing_values": self.missing_values,
            "total_rows": self.total_rows,
            "gaps_detected": len(self.gaps_detected),
            "is_valid": self.is_valid,
            "warnings": self.warnings,
            "data_freshness_hours": self.data_freshness_hours
        }


class DataLoader:
    """Service for loading and validating stock data."""
    
    def __init__(self):
        self.cache = {}
        # Simple in-memory cache: (ticker, period, interval) -> (timestamp, data, report)
        self.data_cache = {}
        # Interval-specific cache TTLs for better consistency
        self.cache_ttl_by_interval = {
            "1m": 30,    # 30 seconds for 1-minute data
            "5m": 60,    # 1 minute for 5-minute data
            "15m": 120,  # 2 minutes for 15-minute data
            "1h": 300,   # 5 minutes for hourly data
            "1d": 300,   # 5 minutes for daily data
            "1wk": 600,  # 10 minutes for weekly data
        }
        self.default_cache_ttl = 60  # 1 minute default
    
    def fetch_stock_data(
        self, 
        ticker: str, 
        period: str = "1y",
        interval: str = "1d"
    ) -> Tuple[Optional[pd.DataFrame], DataQualityReport]:
        """
        Fetch OHLCV data for a stock with caching.
        
        Args:
            ticker: Stock ticker symbol (e.g., "BBCA.JK")
            period: Data period (e.g., "1y", "6mo", "3mo")
            interval: Data interval (e.g., "1d", "1h")
        
        Returns:
            Tuple of (DataFrame with OHLCV data, DataQualityReport)
        """
        cache_key = f"{ticker}_{period}_{interval}"
        now = datetime.now()
        
        # Check cache with interval-specific TTL
        cache_ttl = self.cache_ttl_by_interval.get(interval, self.default_cache_ttl)
        if cache_key in self.data_cache:
            timestamp, cached_df, cached_report = self.data_cache[cache_key]
            if (now - timestamp).total_seconds() < cache_ttl:
                return cached_df, cached_report
        report = DataQualityReport()
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                report.is_valid = False
                report.warnings.append(f"No data available for {ticker}")
                return None, report
            
            # Validate data quality
            report = self._validate_data(df, report)
            
            # Clean data
            df = self._clean_data(df)
            
            # Add ticker column
            df['Ticker'] = ticker
            
            # Update cache
            self.data_cache[cache_key] = (datetime.now(), df, report)
            
            return df, report
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            report.is_valid = False
            report.warnings.append(f"Error fetching data: {str(e)}")
            return None, report
    
    def _validate_data(self, df: pd.DataFrame, report: DataQualityReport) -> DataQualityReport:
        """Validate data quality."""
        report.total_rows = len(df)
        
        # Check for missing values
        report.missing_values = int(df.isnull().sum().sum())
        if report.missing_values > 0:
            missing_pct = (report.missing_values / (len(df) * len(df.columns))) * 100
            report.warnings.append(f"Missing values detected: {report.missing_values} ({missing_pct:.2f}%)")
            if missing_pct > 10:
                report.is_valid = False
        
        # Check for gaps in trading days
        if len(df) > 1:
            df_indexed = df.copy()
            df_indexed.index = pd.to_datetime(df_indexed.index)
            date_diff = df_indexed.index.to_series().diff()
            
            # Flag gaps > 5 business days (accounting for holidays)
            large_gaps = date_diff[date_diff > timedelta(days=7)]
            if len(large_gaps) > 0:
                report.gaps_detected = large_gaps.index.tolist()
                report.warnings.append(f"Large gaps detected in data: {len(large_gaps)} gaps")
        
        # Check data freshness
        if len(df) > 0:
            latest_date = pd.to_datetime(df.index[-1])
            now = datetime.now()
            if latest_date.tzinfo is not None:
                latest_date = latest_date.replace(tzinfo=None)
            report.data_freshness_hours = (now - latest_date).total_seconds() / 3600
            
            # Warn if data is more than 48 hours old (accounting for weekends)
            if report.data_freshness_hours > 96:
                report.warnings.append(f"Data may be stale: {report.data_freshness_hours:.1f} hours old")
        
        return report
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data."""
        # Forward fill missing values
        df = df.ffill()
        
        # Backward fill any remaining NaN at the start
        df = df.bfill()
        
        # Ensure proper column names
        df.columns = [col.title() for col in df.columns]
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Remove timezone information to prevent comparison issues
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Sort by date
        df = df.sort_index()
        
        return df
    
    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """Get basic stock information."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "ticker": ticker,
                "name": info.get("longName", info.get("shortName", ticker)),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap"),
                "currency": info.get("currency", "IDR"),
                "exchange": info.get("exchange", "JK")
            }
        except Exception as e:
            logger.error(f"Error fetching info for {ticker}: {e}")
            return {"ticker": ticker, "name": ticker, "error": str(e)}


# Singleton instance
data_loader = DataLoader()
