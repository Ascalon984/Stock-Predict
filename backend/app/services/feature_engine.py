"""Feature engineering service for technical indicators."""
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.preprocessing import MinMaxScaler


class FeatureEngine:
    """Compute technical indicators for stock analysis."""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def compute_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        df = df.copy()
        
        # Ensure we have required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Compute indicators
        df = self._compute_rsi(df)
        df = self._compute_macd(df)
        df = self._compute_moving_averages(df)
        df = self._compute_bollinger_bands(df)
        df = self._compute_atr(df)
        df = self._compute_obv(df)
        
        return df
    
    def _compute_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Relative Strength Index using Wilder's Smoothing."""
        delta = df['Close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        # Wilder's Smoothing (alpha = 1/n)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _compute_macd(
        self, 
        df: pd.DataFrame, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> pd.DataFrame:
        """Compute MACD and Signal Line."""
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        return df
    
    def _compute_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Simple and Exponential Moving Averages."""
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        df['SMA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
        
        # Exponential Moving Averages
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        return df
    
    def _compute_bollinger_bands(
        self, 
        df: pd.DataFrame, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """Compute Bollinger Bands."""
        sma = df['Close'].rolling(window=period, min_periods=1).mean()
        std = df['Close'].rolling(window=period, min_periods=1).std()
        
        df['BB_Upper'] = sma + (std * std_dev)
        df['BB_Middle'] = sma
        df['BB_Lower'] = sma - (std * std_dev)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        return df
    
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Average True Range using Wilder's Smoothing."""
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        # Wilder's Smoothing (alpha = 1/n)
        df['ATR'] = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        return df
    
    def _compute_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute On-Balance Volume (Vectorized)."""
        df['OBV'] = (
            np.sign(df['Close'].diff())
            .fillna(0)
            .astype(int)
        ) * df['Volume']
        df['OBV'] = df['OBV'].cumsum()
        return df
    
    def get_indicator_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get current indicator values summary with comprehensive signals."""
        if len(df) == 0:
            return {}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Determine trend based on MAs
        ma_trend = "Bullish" if latest['Close'] > latest['SMA_50'] else "Bearish"
        
        # RSI interpretation with gradient zones
        rsi_val = latest['RSI']
        if rsi_val >= 80:
            rsi_signal = "Extreme Overbought"
        elif rsi_val >= 70:
            rsi_signal = "Overbought"
        elif rsi_val >= 60:
            rsi_signal = "Mildly Bullish"
        elif rsi_val <= 20:
            rsi_signal = "Extreme Oversold"
        elif rsi_val <= 30:
            rsi_signal = "Oversold"
        elif rsi_val <= 40:
            rsi_signal = "Mildly Bearish"
        else:
            rsi_signal = "Neutral"
        
        # MACD interpretation with strength
        macd_val = latest['MACD']
        macd_signal_val = latest['MACD_Signal']
        histogram = latest['MACD_Histogram']
        macd_interp = "Bullish" if macd_val > macd_signal_val else "Bearish"
        
        # Detect crossover
        macd_crossover = False
        crossover_type = None
        if len(df) > 1:
            prev_above = prev['MACD'] > prev['MACD_Signal']
            curr_above = macd_val > macd_signal_val
            macd_crossover = bool(prev_above != curr_above)
            if macd_crossover:
                crossover_type = "bullish" if curr_above else "bearish"
        
        # Bollinger Band position with percentage
        bb_upper = latest['BB_Upper']
        bb_lower = latest['BB_Lower']
        bb_middle = latest['BB_Middle']
        bb_range = bb_upper - bb_lower
        
        bb_position = "Middle"
        bb_percent = 50.0
        if bb_range > 0:
            bb_percent = ((latest['Close'] - bb_lower) / bb_range) * 100
            if latest['Close'] >= bb_upper:
                bb_position = "Upper"
            elif latest['Close'] <= bb_lower:
                bb_position = "Lower"
            elif bb_percent > 70:
                bb_position = "Upper-Mid"
            elif bb_percent < 30:
                bb_position = "Lower-Mid"
        
        # OBV trend analysis (if enough data)
        obv_trend = "Neutral"
        obv_change_pct = 0.0
        if 'OBV' in df.columns and len(df) >= 10:
            recent_obv = df['OBV'].tail(10)
            obv_slope = (recent_obv.iloc[-1] - recent_obv.iloc[0]) / max(abs(recent_obv.iloc[0]), 1)
            obv_change_pct = obv_slope * 100
            if obv_slope > 0.05:
                obv_trend = "Accumulation"
            elif obv_slope < -0.05:
                obv_trend = "Distribution"
        
        # Volatility regime classification
        atr_pct = float(latest['ATR'] / latest['Close'] * 100)
        if atr_pct >= 5:
            volatility_regime = "Extreme"
        elif atr_pct >= 3:
            volatility_regime = "High"
        elif atr_pct >= 2:
            volatility_regime = "Moderate"
        elif atr_pct >= 1:
            volatility_regime = "Low"
        else:
            volatility_regime = "Very Low"
        
        return {
            "close": float(latest['Close']),
            "rsi": {
                "value": float(rsi_val),
                "signal": rsi_signal,
                "zone": "overbought" if rsi_val > 70 else "oversold" if rsi_val < 30 else "neutral"
            },
            "macd": {
                "value": float(macd_val),
                "signal_line": float(macd_signal_val),
                "histogram": float(histogram),
                "interpretation": macd_interp,
                "crossover_detected": macd_crossover,
                "crossover_type": crossover_type,
                "strength": abs(histogram) / max(abs(macd_val), 0.01) if macd_val != 0 else 0
            },
            "moving_averages": {
                "sma_20": float(latest['SMA_20']),
                "sma_50": float(latest['SMA_50']),
                "sma_200": float(latest['SMA_200']),
                "ema_20": float(latest['EMA_20']),
                "trend": ma_trend,
                "price_vs_sma50_pct": float((latest['Close'] - latest['SMA_50']) / latest['SMA_50'] * 100),
                "price_vs_sma200_pct": float((latest['Close'] - latest['SMA_200']) / latest['SMA_200'] * 100)
            },
            "bollinger_bands": {
                "upper": float(bb_upper),
                "middle": float(bb_middle),
                "lower": float(bb_lower),
                "width": float(latest['BB_Width']),
                "position": bb_position,
                "percent_b": float(bb_percent)
            },
            "volatility": {
                "atr": float(latest['ATR']),
                "atr_percent": atr_pct,
                "regime": volatility_regime
            },
            "obv": {
                "value": float(latest['OBV']) if 'OBV' in df.columns else 0,
                "trend": obv_trend,
                "change_percent": float(obv_change_pct)
            }
        }
    
    def prepare_features_for_model(
        self, 
        df: pd.DataFrame, 
        feature_columns: list = None
    ) -> tuple:
        """
        Prepare features for ML model input.
        
        Returns:
            Tuple of (scaled_features, scaler)
        """
        if feature_columns is None:
            feature_columns = [
                'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal',
                'SMA_20', 'EMA_20', 'BB_Width', 'ATR'
            ]
        
        # Filter to available columns
        available_cols = [c for c in feature_columns if c in df.columns]
        features = df[available_cols].copy()
        
        # Drop NaN rows
        features = features.dropna()
        
        # Scale features
        scaled = self.scaler.fit_transform(features.values)
        
        return scaled, self.scaler, features.index


# Singleton instance
feature_engine = FeatureEngine()
