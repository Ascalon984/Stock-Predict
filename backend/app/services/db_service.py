"""
Database Service for Stock Data Persistence
===========================================
Handles storage and retrieval of stock market data using SQLite.
Replaces the requirement for PostgreSQL in this local environment while maintaining SQL structure.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import os

# Database file path
DB_PATH = "market_data.db"

class DatabaseService:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initialize the database schema."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create market_data table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            period TEXT,
            interval TEXT,
            source TEXT DEFAULT 'yahoo',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create index for faster lookups
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticker_timestamp 
        ON market_data(ticker, timestamp)
        """)
        
        conn.commit()
        conn.close()

    def save_market_data(self, ticker: str, data: Dict[str, Any], interval: str = "1d"):
        """
        Save a single market data point / quote.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Check if data point already exists to avoid duplicates
            cursor.execute("""
            SELECT id FROM market_data 
            WHERE ticker = ? AND timestamp = ? AND interval = ?
            """, (ticker, data.get('last_update_iso', datetime.now().isoformat()), interval))
            
            existing = cursor.fetchone()
            
            if not existing:
                cursor.execute("""
                INSERT INTO market_data (
                    ticker, timestamp, open, high, low, close, volume, interval
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    data.get('last_update_iso', datetime.now().isoformat()),
                    data.get('open'),
                    data.get('high'),
                    data.get('low'),
                    data.get('last_price', data.get('close')),
                    data.get('volume'),
                    interval
                ))
            else:
                # Update existing record (real-time data might refine previous snapshot)
                cursor.execute("""
                UPDATE market_data SET
                    open = ?, high = ?, low = ?, close = ?, volume = ?, created_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """, (
                    data.get('open'),
                    data.get('high'),
                    data.get('low'),
                    data.get('last_price', data.get('close')),
                    data.get('volume'),
                    existing['id']
                ))
                
            conn.commit()
            return True
        except Exception as e:
            print(f"Database error: {e}")
            return False
        finally:
            conn.close()

    def save_bulk_history(self, ticker: str, history_data: Dict[str, List[Any]], interval: str):
        """
        Save bulk historical data (e.g. from chart fetch).
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            dates = history_data.get('dates', [])
            opens = history_data.get('open', [])
            highs = history_data.get('high', [])
            lows = history_data.get('low', [])
            closes = history_data.get('close', [])
            volumes = history_data.get('volume', [])
            
            for i in range(len(dates)):
                timestamp = dates[i]
                
                cursor.execute("""
                INSERT OR REPLACE INTO market_data (
                    ticker, timestamp, open, high, low, close, volume, interval
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker, timestamp, opens[i], highs[i], lows[i], closes[i], volumes[i], interval
                ))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Bulk save error: {e}")
            return False
        finally:
            conn.close()

    def get_latest_data(self, ticker: str, interval: str = "1m") -> Optional[Dict[str, Any]]:
        """Get the most recent data point from DB."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT * FROM market_data 
        WHERE ticker = ? AND interval = ? 
        ORDER BY timestamp DESC LIMIT 1
        """, (ticker, interval))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None

db_service = DatabaseService()
