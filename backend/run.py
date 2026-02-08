#!/usr/bin/env python
"""
Run script for Stock Predictive Analytics Backend.
"""
import argparse
import os
import sys

# Ensure backend root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def main():
    parser = argparse.ArgumentParser(description="Stock Prediction API Server")
    parser.add_argument("--dev", action="store_true", help="Run in development mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    # Set environment
    os.environ["ENVIRONMENT"] = "development" if args.dev else "production"
    os.environ["DEBUG"] = "true" if args.dev else "false"
    
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)
    
    print("=" * 60)
    print("  Stock Predictive Analytics API")
    print("  Hybrid SARIMA-LSTM Engine v2.0")
    print("=" * 60)
    print(f"  Mode:    {'Development' if args.dev else 'Production'}")
    print(f"  Host:    {args.host}")
    print(f"  Port:    {args.port}")
    print(f"  Workers: {args.workers}")
    print("=" * 60)
    
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.dev,
        workers=1 if args.dev else args.workers,
        log_level="info" if args.dev else "warning",
        access_log=args.dev
    )


if __name__ == "__main__":
    main()
