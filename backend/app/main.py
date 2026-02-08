"""
FastAPI main application for Stock Predictive Analytics.

This is the entry point for the hybrid SARIMA-LSTM prediction API.
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
import time
import os

from .api.routes import router
from .core.config import API_TITLE, API_DESCRIPTION, API_VERSION

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("=" * 50)
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    logger.info("=" * 50)
    
    # Pre-warm TensorFlow (optional)
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow version: {tf.__version__}")
        # Disable GPU if not available
        if not tf.config.list_physical_devices('GPU'):
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            logger.info("Running on CPU (no GPU detected)")
    except ImportError:
        logger.warning("TensorFlow not available - LSTM features limited")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# ============================================
# Middleware
# ============================================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "*"  # Allow all in development - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"]
)

# Gzip compression for large responses
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.3f}s"
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response


# ============================================
# Exception Handlers
# ============================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with clear messages."""
    errors = []
    for error in exc.errors():
        field = ".".join(str(x) for x in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Validation Error",
            "details": errors
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal Server Error",
            "message": str(exc) if os.getenv("DEBUG", "false") == "true" else "An unexpected error occurred"
        }
    )


# ============================================
# Root Routes
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """
    API root endpoint.
    
    Returns basic API information and available endpoints.
    """
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": "Stock Market Predictive Analytics using Hybrid SARIMA+LSTM",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/api/v1/health",
            "stocks": "/api/v1/stocks",
            "predict": "/api/v1/predict/{ticker}",
            "websocket": "ws://localhost:8000/api/v1/ws/{ticker}"
        },
        "features": [
            "Real-time data from Yahoo Finance",
            "Auto-SARIMA with optimal parameter selection",
            "LSTM with attention mechanism",
            "Adaptive ensemble weights",
            "Technical indicator analysis",
            "Sentiment scoring",
            "WebSocket for real-time updates"
        ],
        "disclaimer": "This system does NOT provide financial advice. Use at your own risk."
    }


@app.get("/status", tags=["Root"])
async def status_check():
    """Quick status check for load balancers."""
    return {"status": "ok"}


# ============================================
# Include API Routes
# ============================================

app.include_router(
    router,
    prefix="/api/v1",
    tags=["Stock Prediction"]
)


# ============================================
# Development Server
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
