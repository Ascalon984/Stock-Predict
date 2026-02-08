"""Core Package - Configuration and utilities."""
from .config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    STOCK_UNIVERSE,
    STOCK_NAMES,
    SECTORS,
    POPULAR_STOCKS,
    get_stock_name,
    get_sector,
    is_valid_ticker
)

__all__ = [
    "API_TITLE",
    "API_VERSION", 
    "API_DESCRIPTION",
    "STOCK_UNIVERSE",
    "STOCK_NAMES",
    "SECTORS",
    "POPULAR_STOCKS",
    "get_stock_name",
    "get_sector",
    "is_valid_ticker"
]
