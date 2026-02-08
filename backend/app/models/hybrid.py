"""Enhanced hybrid model orchestrator with caching and optimized ensemble."""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import hashlib
import json
from functools import lru_cache

from .sarima import SARIMAModel
from .lstm import LSTMModel
from ..services.data_loader import data_loader
from ..services.feature_engine import feature_engine
from ..services.sentiment import sentiment_engine

logger = logging.getLogger(__name__)


@dataclass
class PredictionCache:
    """Cache for storing prediction results."""
    data: Dict[str, Any]
    timestamp: datetime
    ttl_minutes: int = 15
    
    def is_valid(self) -> bool:
        """Check if cache is still valid."""
        return datetime.now() - self.timestamp < timedelta(minutes=self.ttl_minutes)


class PredictionCacheManager:
    """Manages prediction caching for performance."""
    
    def __init__(self, max_size: int = 100, ttl_minutes: int = 15):
        self.cache: Dict[str, PredictionCache] = {}
        self.max_size = max_size
        self.ttl_minutes = ttl_minutes
    
    def _generate_key(self, ticker: str, period: str) -> str:
        """Generate cache key."""
        return hashlib.md5(f"{ticker}:{period}".encode()).hexdigest()
    
    def get(self, ticker: str, period: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction if valid."""
        key = self._generate_key(ticker, period)
        if key in self.cache:
            cached = self.cache[key]
            if cached.is_valid():
                logger.info(f"Cache hit for {ticker}")
                return cached.data
            else:
                del self.cache[key]
        return None
    
    def set(self, ticker: str, period: str, data: Dict[str, Any]) -> None:
        """Cache prediction result."""
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_key]
        
        key = self._generate_key(ticker, period)
        self.cache[key] = PredictionCache(
            data=data,
            timestamp=datetime.now(),
            ttl_minutes=self.ttl_minutes
        )
    
    def invalidate(self, ticker: str = None) -> None:
        """Invalidate cache entries."""
        if ticker:
            keys_to_delete = [k for k in self.cache.keys() if ticker in str(self.cache[k].data.get('ticker', ''))]
            for k in keys_to_delete:
                del self.cache[k]
        else:
            self.cache.clear()


class HybridPredictor:
    """
    Enhanced hybrid prediction system combining SARIMA and LSTM models.
    
    Improvements:
    - Adaptive weight allocation based on model performance
    - Volatility-adjusted confidence intervals
    - Multi-horizon ensemble optimization
    - Prediction caching for performance
    - Diagnostic metrics for model quality assessment
    
    Architecture:
    1. Fetch and validate OHLCV data with quality checks
    2. Apply auto-SARIMA to capture trend and seasonality
    3. Compute SARIMA residuals (detrended series)
    4. Generate comprehensive technical indicators
    5. Feed residuals + indicators into LSTM with attention
    6. Dynamic ensemble based on recent model accuracy
    7. Volatility-adjusted uncertainty bands
    8. Sentiment scoring from multiple signals
    """
    
    def __init__(
        self,
        sarima_weight: float = 0.35,
        lstm_weight: float = 0.65,
        sequence_length: int = 60,
        prediction_horizons: List[int] = None,
        use_cache: bool = True,
        cache_ttl_minutes: int = 15
    ):
        self.base_sarima_weight = sarima_weight
        self.base_lstm_weight = lstm_weight
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons or [1, 7, 30]
        self.use_cache = use_cache
        
        self.sarima_model: Optional[SARIMAModel] = None
        self.lstm_model: Optional[LSTMModel] = None
        self.cache_manager = PredictionCacheManager(ttl_minutes=cache_ttl_minutes) if use_cache else None
        
        # Dynamic weights (adjusted based on performance)
        self.sarima_weight = sarima_weight
        self.lstm_weight = lstm_weight
    
    def predict(
        self,
        ticker: str,
        period: str = "1y",
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Generate comprehensive predictions for a stock.
        
        Args:
            ticker: Stock ticker symbol
            period: Historical data period
            force_refresh: Bypass cache
            
        Returns:
            Complete prediction response with forecasts, indicators, and sentiment
        """
        # Check cache first
        if self.use_cache and not force_refresh:
            cached = self.cache_manager.get(ticker, period)
            if cached:
                cached['from_cache'] = True
                return cached
        
        start_time = datetime.now()
        
        # Step 1: Fetch and validate data
        logger.info(f"Fetching data for {ticker}...")
        df, quality_report = data_loader.fetch_stock_data(ticker, period)
        
        if df is None or not quality_report.is_valid:
            return {
                "success": False,
                "error": "Failed to fetch valid data",
                "data_quality": quality_report.to_dict(),
                "ticker": ticker
            }
        
        logger.info(f"Fetched {len(df)} data points for {ticker}")
        
        # Step 2: Compute technical indicators
        logger.info("Computing technical indicators...")
        df_with_indicators = feature_engine.compute_all_indicators(df)
        indicator_summary = feature_engine.get_indicator_summary(df_with_indicators)
        
        # Step 3: Fit SARIMA model with auto-parameter selection
        logger.info("Fitting SARIMA model...")
        close_prices = df['Close']
        self.sarima_model = SARIMAModel(auto_select=True)
        sarima_fit_result = self.sarima_model.fit(close_prices)
        
        # Step 4: Get SARIMA residuals for LSTM
        sarima_residuals = self.sarima_model.get_residuals()
        sarima_diagnostics = self.sarima_model.get_diagnostics()
        
        # Step 5: Prepare features for LSTM
        logger.info("Training LSTM model...")
        
        # Add SARIMA residuals to features if available
        if sarima_residuals is not None:
            # Ensure alignment
            if len(sarima_residuals) == len(df_with_indicators):
                df_with_indicators['SARIMA_Resid'] = sarima_residuals
            else:
                # Pad or trim if necessary (though usually they match)
                diff = len(df_with_indicators) - len(sarima_residuals)
                if diff > 0:
                    df_with_indicators['SARIMA_Resid'] = np.concatenate([np.zeros(diff), sarima_residuals])
                else:
                    df_with_indicators['SARIMA_Resid'] = sarima_residuals[-len(df_with_indicators):]

        feature_columns = [
            'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal',
            'SMA_20', 'SMA_50', 'EMA_20', 'BB_Width', 'ATR', 'OBV',
            'SARIMA_Resid'  # Added residuals
        ]
        available_cols = [c for c in feature_columns if c in df_with_indicators.columns]
        
        # Drop NaN and prepare data
        feature_data = df_with_indicators[available_cols].dropna()
        
        # Adjust sequence length based on available data
        effective_seq_length = min(self.sequence_length, len(feature_data) - 20)
        
        # Fit LSTM
        self.lstm_model = LSTMModel(
            sequence_length=effective_seq_length,
            lstm_units=[128, 64],
            use_attention=True,
            use_bidirectional=True
        )
        lstm_fit_result = self.lstm_model.fit(
            feature_data.values,
            epochs=50,
            batch_size=min(32, len(feature_data) // 4),
            patience=10
        )
        
        # Step 6: Generate forecasts for each horizon
        logger.info("Generating forecasts...")
        max_horizon = max(self.prediction_horizons)
        
        # SARIMA forecast
        sarima_forecast = self.sarima_model.forecast(steps=max_horizon)
        
        # LSTM forecast
        # Generate deterministic seed to ensure consistent MC Dropout results
        # for the same ticker/period combination across requests
        seed_str = f"{ticker}_{period}"
        prediction_seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
        
        lstm_forecast = self.lstm_model.predict(
            feature_data.values, 
            steps=max_horizon,
            seed=prediction_seed
        )
        
        # Step 7: Adjust weights based on model fit quality
        self._adjust_weights(sarima_fit_result, lstm_fit_result, indicator_summary)
        
        # Step 8: Ensemble forecasts with dynamic weights
        ensemble_result = self._ensemble_forecasts(
            sarima_forecast,
            lstm_forecast,
            float(close_prices.iloc[-1]),
            indicator_summary
        )
        
        # Step 9: Compute comprehensive sentiment
        forecast_direction = 0
        for f in ensemble_result["forecasts"]:
            if f["horizon"] == 7:
                forecast_direction = f["predicted_change_percent"]
                break
        
        if forecast_direction == 0 and ensemble_result["forecasts"]:
            forecast_direction = ensemble_result["forecasts"][-1]["predicted_change_percent"]
        
        sentiment = sentiment_engine.compute_sentiment(
            indicator_summary,
            forecast_direction,
            ensemble_result.get("confidence", 0.7)
        )
        
        # Step 10: Classify trend
        trend = self._classify_trend(
            ensemble_result["forecasts"],
            float(close_prices.iloc[-1]),
            indicator_summary
        )
        
        # Build final response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            "success": True,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": round(processing_time, 2),
            "from_cache": False,
            "data_quality": quality_report.to_dict(),
            "historical_data": {
                "dates": df.index.strftime('%Y-%m-%d').tolist()[-90:],
                "open": [round(x, 2) for x in df['Open'].values.tolist()[-90:]],
                "high": [round(x, 2) for x in df['High'].values.tolist()[-90:]],
                "low": [round(x, 2) for x in df['Low'].values.tolist()[-90:]],
                "close": [round(x, 2) for x in df['Close'].values.tolist()[-90:]],
                "volume": [int(x) for x in df['Volume'].values.tolist()[-90:]]
            },
            "indicators": indicator_summary,
            "forecasts": ensemble_result["forecasts"],
            "model_contribution": {
                "sarima_weight": round(self.sarima_weight, 3),
                "lstm_weight": round(self.lstm_weight, 3),
                "ensemble_method": "adaptive_weighted_average",
                "sarima_diagnostics": sarima_diagnostics,
                "sarima_fit": sarima_fit_result,
                "lstm_fit": lstm_fit_result
            },
            "trend": trend,
            "sentiment": sentiment,
            "disclaimer": "This analysis is for informational purposes only and does not constitute financial advice. Past performance is not indicative of future results. Always conduct your own research before making investment decisions."
        }
        
        # Cache the result
        if self.use_cache:
            self.cache_manager.set(ticker, period, result)
        
        logger.info(f"Prediction completed for {ticker} in {processing_time:.2f}s")
        
        return result
    
    def _adjust_weights(
        self,
        sarima_fit: Dict[str, Any],
        lstm_fit: Dict[str, Any],
        indicators: Dict[str, Any]
    ) -> None:
        """
        Dynamically adjust model weights based on accuracy metrics and market conditions.
        
        Uses MAPE/RMSE from cross-validation when available for data-driven weighting.
        """
        sarima_score = 1.0
        lstm_score = 1.0
        
        # Score based on success
        if not sarima_fit.get("success"):
            sarima_score *= 0.2
        else:
            # Use MAPE if available (lower is better)
            sarima_mape = sarima_fit.get("mape")
            if sarima_mape is not None:
                if sarima_mape < 2:
                    sarima_score *= 1.5  # Excellent
                elif sarima_mape < 5:
                    sarima_score *= 1.2  # Good
                elif sarima_mape > 10:
                    sarima_score *= 0.6  # Poor
            
            # Boost if residuals are white noise (good model fit)
            residual_diagnostics = sarima_fit.get("residual_diagnostics", {})
            if residual_diagnostics.get("residuals_are_white_noise"):
                sarima_score *= 1.15
            
            # Stationary series are better for SARIMA
            if sarima_fit.get("is_stationary"):
                sarima_score *= 1.1
        
        # LSTM scoring
        if not lstm_fit.get("success"):
            lstm_score *= 0.2
        else:
            # Use MAPE if available
            lstm_mape = lstm_fit.get("mape")
            if lstm_mape is not None:
                if lstm_mape < 2:
                    lstm_score *= 1.5
                elif lstm_mape < 5:
                    lstm_score *= 1.2
                elif lstm_mape > 10:
                    lstm_score *= 0.6
            
            # Use R² if available (higher is better)
            r2 = lstm_fit.get("r2")
            if r2 is not None:
                if r2 > 0.9:
                    lstm_score *= 1.3
                elif r2 > 0.7:
                    lstm_score *= 1.1
                elif r2 < 0.3:
                    lstm_score *= 0.7
            
            # Lower validation loss = better fit
            val_loss = lstm_fit.get("final_val_loss", 1)
            if val_loss < 0.005:
                lstm_score *= 1.2
            elif val_loss > 0.1:
                lstm_score *= 0.8
        
        # Adjust based on volatility regime
        volatility = indicators.get("volatility", {}).get("atr_percent", 2)
        
        # High volatility: LSTM better at capturing non-linear patterns
        if volatility > 5:
            lstm_score *= 1.3
            sarima_score *= 0.85
        elif volatility > 3:
            lstm_score *= 1.15
            sarima_score *= 0.95
        # Low volatility: SARIMA better at linear trends
        elif volatility < 1.5:
            sarima_score *= 1.25
            lstm_score *= 0.9
        
        # Adjust based on trend strength
        ma_trend = indicators.get("moving_averages", {}).get("trend")
        if ma_trend in ["Bullish", "Bearish"]:
            # Clear trend: SARIMA can capture it well
            sarima_score *= 1.1
        
        # Normalize weights
        total = sarima_score + lstm_score
        if total > 0:
            self.sarima_weight = self.base_sarima_weight * (sarima_score / total) * 2
            self.lstm_weight = self.base_lstm_weight * (lstm_score / total) * 2
        
        # Ensure weights sum to 1
        weight_sum = self.sarima_weight + self.lstm_weight
        if weight_sum > 0:
            self.sarima_weight /= weight_sum
            self.lstm_weight /= weight_sum
        
        logger.info(
            f"Adjusted weights: SARIMA={self.sarima_weight:.3f} (score={sarima_score:.2f}), "
            f"LSTM={self.lstm_weight:.3f} (score={lstm_score:.2f})"
        )
    
    def _ensemble_forecasts(
        self,
        sarima_forecast: Dict[str, Any],
        lstm_forecast: Dict[str, Any],
        last_price: float,
        indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine SARIMA and LSTM forecasts with volatility-adjusted confidence intervals."""
        forecasts = []
        overall_confidence = 0.75
        
        # Get volatility for uncertainty scaling
        volatility_pct = indicators.get("volatility", {}).get("atr_percent", 2)
        volatility_factor = 1 + (volatility_pct / 100)
        
        for horizon in self.prediction_horizons:
            horizon_forecast = {
                "horizon": horizon,
                "horizon_label": self._get_horizon_label(horizon)
            }
            
            # Extract SARIMA predictions
            if sarima_forecast.get("success") and len(sarima_forecast.get("forecast", [])) >= horizon:
                sarima_pred = sarima_forecast["forecast"][:horizon]
                sarima_lower = sarima_forecast["confidence_lower"][:horizon]
                sarima_upper = sarima_forecast["confidence_upper"][:horizon]
            else:
                # Fallback: simple random walk
                sarima_pred = self._simple_forecast(last_price, horizon, volatility_pct * 0.5)
                sarima_lower = [p * (1 - volatility_pct/100) for p in sarima_pred]
                sarima_upper = [p * (1 + volatility_pct/100) for p in sarima_pred]
            
            # Extract LSTM predictions
            if lstm_forecast.get("success") and len(lstm_forecast.get("predictions", [])) >= horizon:
                lstm_pred = lstm_forecast["predictions"][:horizon]
                lstm_intervals = lstm_forecast.get("intervals", [])
            else:
                lstm_pred = self._simple_forecast(last_price, horizon, volatility_pct * 0.8)
                lstm_intervals = []
            
            # Ensemble with dynamic weights
            ensemble_pred = []
            ensemble_lower = []
            ensemble_upper = []
            
            for i in range(horizon):
                s_pred = sarima_pred[i] if i < len(sarima_pred) else sarima_pred[-1]
                l_pred = lstm_pred[i] if i < len(lstm_pred) else lstm_pred[-1]
                
                # Weighted combination
                combined = (s_pred * self.sarima_weight) + (l_pred * self.lstm_weight)
                ensemble_pred.append(round(combined, 2))
                
                # Confidence intervals - widen with horizon and volatility
                day_uncertainty = volatility_factor * (1 + i * 0.03)  # 3% per day
                
                s_lower = sarima_lower[i] if i < len(sarima_lower) else s_pred * 0.95
                s_upper = sarima_upper[i] if i < len(sarima_upper) else s_pred * 1.05
                
                l_lower = lstm_intervals[i]["lower"] if i < len(lstm_intervals) else l_pred * 0.95
                l_upper = lstm_intervals[i]["upper"] if i < len(lstm_intervals) else l_pred * 1.05
                
                # Combine intervals
                lower = min(s_lower, l_lower) * day_uncertainty
                upper = max(s_upper, l_upper) * day_uncertainty
                
                # Ensure reasonable bounds
                lower = max(combined * 0.85, lower)
                upper = min(combined * 1.15, upper)
                
                ensemble_lower.append(round(lower, 2))
                ensemble_upper.append(round(upper, 2))
            
            horizon_forecast["forecast"] = ensemble_pred
            horizon_forecast["confidence_lower"] = ensemble_lower
            horizon_forecast["confidence_upper"] = ensemble_upper
            horizon_forecast["last_price"] = round(last_price, 2)
            horizon_forecast["predicted_change_percent"] = round(
                (ensemble_pred[-1] - last_price) / last_price * 100, 2
            )
            
            forecasts.append(horizon_forecast)
        
        # Adjust confidence based on volatility
        if volatility_pct > 5:
            overall_confidence *= 0.6
        elif volatility_pct > 3:
            overall_confidence *= 0.8
        
        return {
            "forecasts": forecasts,
            "confidence": round(overall_confidence, 3)
        }
    
    def _drift_forecast(self, last_price: float, horizon: int, drift_pct: float, historical_returns: Optional[np.ndarray] = None) -> List[float]:
        """
        Generate drift-based forecast as fallback.
        
        Uses historical mean return if available, otherwise uses a small positive drift
        with volatility-based noise.
        """
        forecasts = []
        price = last_price
        
        # Estimate drift from historical returns if available
        if historical_returns is not None and len(historical_returns) > 0:
            mean_return = np.mean(historical_returns)
            std_return = np.std(historical_returns)
        else:
            # Default: slight positive drift with given volatility
            mean_return = 0.0001  # ~0.01% daily
            std_return = drift_pct / 100
        
        for i in range(horizon):
            # Drift + noise, with noise increasing over horizon
            horizon_factor = 1 + (i * 0.1)  # Increase uncertainty over time
            noise = np.random.normal(0, std_return * horizon_factor)
            daily_return = mean_return + noise
            
            price = price * (1 + daily_return)
            forecasts.append(max(0, price))
        
        return forecasts
    
    def _simple_forecast(self, last_price: float, horizon: int, drift_pct: float) -> List[float]:
        """Legacy method - calls drift forecast for backward compatibility."""
        return self._drift_forecast(last_price, horizon, drift_pct)
    
    def _get_horizon_label(self, horizon: int) -> str:
        """Get human-readable horizon label."""
        if horizon == 1:
            return "1 Day"
        elif horizon == 7:
            return "7 Days"
        elif horizon == 30:
            return "1 Month"
        else:
            return f"{horizon} Days"
    
    def _classify_trend(
        self,
        forecasts: List[Dict[str, Any]],
        last_price: float,
        indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhanced trend classification with indicator confirmation."""
        weekly = next((f for f in forecasts if f["horizon"] == 7), forecasts[0] if forecasts else None)
        monthly = next((f for f in forecasts if f["horizon"] == 30), None)
        
        # Determine directions
        short_term_direction = "Sideways"
        long_term_direction = "Sideways"
        
        if weekly:
            weekly_change = weekly["predicted_change_percent"]
            if weekly_change > 2:
                short_term_direction = "Uptrend"
            elif weekly_change < -2:
                short_term_direction = "Downtrend"
        
        if monthly:
            monthly_change = monthly["predicted_change_percent"]
            if monthly_change > 5:
                long_term_direction = "Uptrend"
            elif monthly_change < -5:
                long_term_direction = "Downtrend"
        
        # Confirm with technical indicators
        ma_trend = indicators.get("moving_averages", {}).get("trend", "Neutral")
        rsi_signal = indicators.get("rsi", {}).get("signal", "Neutral")
        macd_signal = indicators.get("macd", {}).get("interpretation", "Neutral")
        
        # Count confirmations
        bullish_signals = sum([
            ma_trend == "Bullish",
            macd_signal == "Bullish",
            short_term_direction == "Uptrend"
        ])
        bearish_signals = sum([
            ma_trend == "Bearish",
            macd_signal == "Bearish",
            short_term_direction == "Downtrend"
        ])
        
        # Check for inconsistency
        inconsistent = (
            short_term_direction != long_term_direction and
            short_term_direction != "Sideways" and
            long_term_direction != "Sideways"
        )
        
        # Overall trend with confidence
        if bullish_signals >= 2:
            overall = "Uptrend"
        elif bearish_signals >= 2:
            overall = "Downtrend"
        else:
            overall = "Sideways"
        
        return {
            "short_term": short_term_direction,
            "long_term": long_term_direction,
            "overall": overall if not inconsistent else "Mixed",
            "inconsistency_warning": inconsistent,
            "interpretation": self._get_trend_interpretation(
                short_term_direction,
                long_term_direction,
                bullish_signals,
                bearish_signals
            ),
            "signal_confirmation": {
                "ma_trend": ma_trend,
                "rsi_signal": rsi_signal,
                "macd_signal": macd_signal,
                "bullish_count": bullish_signals,
                "bearish_count": bearish_signals
            }
        }
    
    def _get_trend_interpretation(
        self,
        short: str,
        long: str,
        bullish: int,
        bearish: int
    ) -> str:
        """Generate detailed trend interpretation."""
        if short == long:
            if short == "Uptrend":
                strength = "strong" if bullish >= 3 else "moderate"
                return f"Consistent bullish momentum with {strength} technical confirmation"
            elif short == "Downtrend":
                strength = "strong" if bearish >= 3 else "moderate"
                return f"Consistent bearish pressure with {strength} technical confirmation"
            else:
                return "Market consolidating with no clear directional bias"
        else:
            return f"Near-term {short.lower()} conflicting with longer-term {long.lower()}. Consider waiting for clearer signals."


# Global cache manager instance to persist across requests
GLOBAL_CACHE_MANAGER = PredictionCacheManager(max_size=200, ttl_minutes=60)

# Factory function
def create_predictor(
    sarima_weight: float = 0.35,
    lstm_weight: float = 0.65,
    use_cache: bool = True
) -> HybridPredictor:
    """
    Create a new hybrid predictor instance.
    
    Uses a global singleton cache manager to ensure persistence across requests.
    """
    predictor = HybridPredictor(
        sarima_weight=sarima_weight,
        lstm_weight=lstm_weight,
        use_cache=use_cache,
        cache_ttl_minutes=60
    )
    
    # Inject the global cache manager if caching is enabled
    if use_cache:
        predictor.cache_manager = GLOBAL_CACHE_MANAGER
        
    return predictor
