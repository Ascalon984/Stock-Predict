"""Sentiment scoring service."""
from typing import Dict, Any, Tuple


class SentimentEngine:
    """Determine market sentiment based on indicators and forecast."""
    
    SENTIMENT_LEVELS = {
        "SANGAT_BEARISH": {"id": "Sangat Bearish", "en": "Very Bearish", "score": -2},
        "BEARISH": {"id": "Bearish", "en": "Bearish", "score": -1},
        "NEUTRAL": {"id": "Netral", "en": "Neutral", "score": 0},
        "BULLISH": {"id": "Bullish", "en": "Bullish", "score": 1},
        "SANGAT_BULLISH": {"id": "Sangat Bullish", "en": "Very Bullish", "score": 2}
    }
    
    def compute_sentiment(
        self,
        indicator_summary: Dict[str, Any],
        forecast_direction: float,  # Positive = up, Negative = down
        forecast_confidence: float  # 0 to 1
    ) -> Dict[str, Any]:
        """
        Compute overall sentiment based on indicators and forecast.
        
        Args:
            indicator_summary: Output from FeatureEngine.get_indicator_summary()
            forecast_direction: Direction of price forecast (% change)
            forecast_confidence: Confidence level (0-1)
        
        Returns:
            Sentiment analysis with level, score, rationale, and impact analysis
        """
        scores = []
        rationale = []
        
        # Track individual component contribution magnitude (absolute value)
        impact_weights = {
            "RSI": 0.0,
            "MACD": 0.0,
            "Moving Avg": 0.0,
            "Bollinger": 0.0,
            "Forecast": 0.0
        }
        
        # --- 1. RSI Score ---
        rsi = indicator_summary.get("rsi", {})
        rsi_val = rsi.get("value", 50)
        rsi_score = 0
        if rsi_val > 70:
            rsi_score = -1
            rationale.append(f"RSI overbought ({rsi_val:.1f})")
        elif rsi_val < 30:
            rsi_score = 1
            rationale.append(f"RSI oversold ({rsi_val:.1f})")
        
        scores.append(rsi_score)
        impact_weights["RSI"] = abs(rsi_score)
        
        # --- 2. MACD Score ---
        macd = indicator_summary.get("macd", {})
        macd_score = 0
        if macd.get("interpretation") == "Bullish":
            macd_score = 1
            if macd.get("crossover_detected"):
                macd_score += 1  # Extra point for fresh crossover
                rationale.append("MACD bullish crossover detected")
            else:
                rationale.append("MACD above signal line")
        else:
            macd_score = -1
            if macd.get("crossover_detected"):
                macd_score -= 1
                rationale.append("MACD bearish crossover detected")
            else:
                rationale.append("MACD below signal line")
                
        scores.append(macd_score)
        impact_weights["MACD"] = abs(macd_score)
        
        # --- 3. Moving Average Score ---
        ma = indicator_summary.get("moving_averages", {})
        ma_score = 0
        if ma.get("trend") == "Bullish":
            ma_score = 1
            rationale.append("Price above SMA50 (bullish trend)")
        else:
            ma_score = -1
            rationale.append("Price below SMA50 (bearish trend)")
            
        scores.append(ma_score)
        impact_weights["Moving Avg"] = abs(ma_score)
        
        # --- 4. Bollinger Band Score ---
        bb = indicator_summary.get("bollinger_bands", {})
        bb_pos = bb.get("position", "Middle")
        bb_score = 0
        if bb_pos == "Lower":
            bb_score = 1  # Potential bounce
            rationale.append("Price at lower Bollinger Band (potential bounce)")
        elif bb_pos == "Upper":
            bb_score = -1  # Potential pullback
            rationale.append("Price at upper Bollinger Band (potential pullback)")
            
        scores.append(bb_score)
        impact_weights["Bollinger"] = abs(bb_score)
        
        # --- 5. Forecast Direction Score (weighted by confidence) ---
        forecast_score = 0
        abs_change = abs(forecast_direction)
        
        # Use continuous ramp instead of hard threshold to prevent jumping
        # 0% - 0.3%: Neutral (0)
        # 0.3% - 1.5%: Linear ramp to max score
        # > 1.5%: Max score (2 or -2)
        if abs_change > 0.3:
            # Calculate intensity (0.0 to 1.0)
            intensity = min((abs_change - 0.3) / 1.2, 1.0)
            
            base_score = 2 if forecast_direction > 0 else -2
            forecast_score = base_score * intensity * forecast_confidence
            
            # Only add rationale if significant enough to matter
            if abs_change > 0.5:
                direction = "upward" if forecast_direction > 0 else "downward"
                rationale.append(f"Model forecasts {abs_change:.1f}% {direction} move")
        
        scores.append(forecast_score)
        impact_weights["Forecast"] = abs(forecast_score)
        
        # --- Final Scoring ---
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Adjust score by forecast confidence to penalize low-confidence predictions
        # This naturally pushes low-confidence signals towards Neutral
        adjusted_score = avg_score * (0.5 + 0.5 * forecast_confidence)
        
        # Determine sentiment level
        sentiment_level = self._score_to_level(adjusted_score)
        
        # --- Calculate Impact Percentages ---
        total_impact = sum(impact_weights.values())
        indicator_impact = {}
        
        if total_impact < 0.1:
            indicator_impact = {k: 20 for k in impact_weights.keys()}
        else:
            current_sum = 0
            keys = list(impact_weights.keys())
            for i, k in enumerate(keys):
                if i == len(keys) - 1:
                    # Last item gets the remainder to ensure sum is 100
                    indicator_impact[k] = 100 - current_sum
                else:
                    val = round((impact_weights[k] / total_impact) * 100)
                    indicator_impact[k] = val
                    current_sum += val
        
        level_data = self.SENTIMENT_LEVELS[sentiment_level]
        
        return {
            "level": sentiment_level,
            "label_id": level_data["id"],
            "label_en": level_data["en"],
            "score": level_data["score"],
            "raw_score": round(avg_score, 2),
            "adjusted_score": round(adjusted_score, 2),
            "confidence": round(forecast_confidence, 2),
            "rationale": rationale,
            "component_scores": {
                "total_signals": len(scores),
                "bullish_signals": sum(1 for s in scores if s > 0),
                "bearish_signals": sum(1 for s in scores if s < 0),
                "neutral_signals": sum(1 for s in scores if s == 0)
            },
            "indicator_impact": indicator_impact
        }
    
    def _score_to_level(self, score: float) -> str:
        """
        Convert numeric score to sentiment level.
        Max possible raw score avg is ~1.4.
        """
        if score >= 1.0:
            return "SANGAT_BULLISH"
        elif score >= 0.3:
            return "BULLISH"
        elif score <= -1.0:
            return "SANGAT_BEARISH"
        elif score <= -0.3:
            return "BEARISH"
        else:
            return "NEUTRAL"


# Singleton instance
sentiment_engine = SentimentEngine()
