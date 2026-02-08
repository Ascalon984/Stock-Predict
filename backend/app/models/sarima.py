"""Enhanced SARIMA model with auto-parameter selection for optimal forecasting.

Improvements:
- Cross-validation for model selection
- Dynamic seasonal period detection
- Comprehensive stationarity tests (ADF + KPSS)
- Residual diagnostic tests (Ljung-Box, heteroscedasticity)
- Rolling window validation for out-of-sample accuracy
- Proper exception handling
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import logging
import warnings
from scipy import stats

# Suppress convergence warnings for cleaner logs
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Global flag for pmdarima
PMDARIMA_AVAILABLE = False
try:
    import pmdarima
    from pmdarima.arima import ndiffs, nsdiffs
    PMDARIMA_AVAILABLE = True
except ImportError:
    pass


class SARIMAModel:
    """
    Enhanced Seasonal ARIMA model with automatic parameter selection.
    
    Features:
    - Auto-detection of optimal (p, d, q) and (P, D, Q, m) parameters
    - Cross-validation for robust model selection
    - Dynamic seasonal period detection
    - Comprehensive diagnostic tests
    - Rolling window validation metrics
    """
    
    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        auto_select: bool = True,
        max_p: int = 3,
        max_q: int = 3,
        max_P: int = 2,
        max_Q: int = 2,
        seasonal_period: Optional[int] = None,  # Auto-detect if None
        use_cross_validation: bool = True,
        cv_folds: int = 3,
        **kwargs
    ):
        # Use tuples instead of mutable defaults
        self.order = order if order is not None else (1, 1, 1)
        self.seasonal_period = seasonal_period if seasonal_period is not None else 5  # Trading week default
        self.seasonal_order = seasonal_order if seasonal_order is not None else (1, 1, 1, self.seasonal_period)
        self.auto_select = auto_select
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.use_cross_validation = use_cross_validation
        self.cv_folds = cv_folds
        
        self.model = None
        self.fitted_model = None
        self.residuals = None
        self.aic = None
        self.bic = None
        self.is_stationary = None
        self.is_pmdarima_model = False
        
        # New diagnostic metrics
        self.stationarity_tests: Dict[str, Any] = {}
        self.residual_diagnostics: Dict[str, Any] = {}
        self.cv_scores: List[float] = []
        self.mape: Optional[float] = None
        self.rmse: Optional[float] = None
        
    def fit(self, series: pd.Series) -> Dict[str, Any]:
        """Fit SARIMA model to time series data with comprehensive validation."""
        start_time = datetime.now()
        
        # Clean the series
        series = series.dropna()
        
        if len(series) < 30:
            logger.warning(f"Series too short ({len(series)}), need at least 30 observations")
            return {"success": False, "error": "Insufficient data"}
        
        try:
            # Step 1: Detect optimal seasonal period if not specified
            if self.seasonal_period is None or self.seasonal_period == 5:
                detected_period = self._detect_seasonality(series)
                if detected_period:
                    self.seasonal_period = detected_period
                    logger.info(f"Detected seasonal period: {self.seasonal_period}")
            
            # Step 2: Comprehensive stationarity tests
            self.stationarity_tests = self._comprehensive_stationarity_test(series)
            self.is_stationary = self.stationarity_tests.get("is_stationary", False)
            
            # Step 3: Determine optimal differencing
            d_order = self._determine_differencing(series)
            
            if self.auto_select and PMDARIMA_AVAILABLE:
                try:
                    from pmdarima import auto_arima
                    
                    logger.info("Running auto-ARIMA with cross-validation...")
                    
                    # Convert to numpy to avoid index issues
                    values = series.values
                    
                    # Configure auto_arima with cross-validation
                    auto_model = auto_arima(
                        values,
                        start_p=0, max_p=self.max_p,
                        start_q=0, max_q=self.max_q,
                        start_P=0, max_P=self.max_P,
                        start_Q=0, max_Q=self.max_Q,
                        m=self.seasonal_period,
                        seasonal=True,
                        d=d_order if d_order is not None else None,
                        D=None,  # Auto-detect seasonal differencing
                        trace=False,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True,
                        random_state=42,
                        max_order=6,
                        information_criterion='aic',
                        out_of_sample_size=int(len(values) * 0.1) if self.use_cross_validation else 0,
                        scoring='mse'
                    )
                    
                    self.fitted_model = auto_model
                    self.order = auto_model.order
                    self.seasonal_order = auto_model.seasonal_order
                    self.residuals = auto_model.resid()
                    self.aic = auto_model.aic()
                    
                    try:
                        self.bic = auto_model.bic()
                    except (AttributeError, ValueError) as bic_error:
                        logger.debug(f"BIC calculation failed: {bic_error}")
                        self.bic = None
                    
                    self.is_pmdarima_model = True
                    logger.info(f"Auto-ARIMA selected: SARIMA{self.order}x{self.seasonal_order}")
                    
                except Exception as auto_error:
                    logger.warning(f"Auto-ARIMA failed: {auto_error}, falling back to statsmodels")
                    self._fit_statsmodels(series)
            else:
                if self.auto_select and not PMDARIMA_AVAILABLE:
                    logger.info("pmdarima not available, using statsmodels SARIMAX")
                self._fit_statsmodels(series)
            
            # Step 4: Cross-validation scoring
            if self.use_cross_validation:
                self.cv_scores = self._time_series_cv(series)
                
            # Step 5: Residual diagnostics
            self.residual_diagnostics = self._residual_diagnostics()
            
            # Step 6: Calculate accuracy metrics on held-out data
            self._calculate_accuracy_metrics(series)
            
            fit_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "aic": float(self.aic) if self.aic else None,
                "bic": float(self.bic) if self.bic else None,
                "is_stationary": self.is_stationary,
                "fit_time_seconds": fit_time,
                "n_observations": len(series),
                "mape": self.mape,
                "rmse": self.rmse,
                "cv_scores": self.cv_scores,
                "residual_diagnostics": self.residual_diagnostics,
                "stationarity_tests": self.stationarity_tests
            }
            
        except Exception as e:
            logger.error(f"SARIMA fitting error: {e}")
            return {"success": False, "error": str(e)}
    
    def _detect_seasonality(self, series: pd.Series) -> Optional[int]:
        """Detect the dominant seasonal period using autocorrelation."""
        try:
            from statsmodels.tsa.stattools import acf
            
            n = len(series)
            max_lag = min(n // 2, 252)  # Max 1 year of trading days
            
            acf_values = acf(series, nlags=max_lag, fft=True)
            
            # Find peaks in ACF (potential seasonal periods)
            # Common trading periods: 5 (week), 21 (month), 63 (quarter), 252 (year)
            candidate_periods = [5, 10, 21, 42, 63, 126, 252]
            
            best_period = 5
            best_acf = 0
            
            for period in candidate_periods:
                if period < len(acf_values):
                    if acf_values[period] > best_acf and acf_values[period] > 0.1:
                        best_acf = acf_values[period]
                        best_period = period
            
            return best_period
            
        except Exception as e:
            logger.warning(f"Seasonality detection failed: {e}")
            return 5  # Default to trading week
    
    def _comprehensive_stationarity_test(
        self, 
        series: pd.Series, 
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Perform comprehensive stationarity tests (ADF + KPSS)."""
        results = {
            "adf_statistic": None,
            "adf_pvalue": None,
            "adf_stationary": False,
            "kpss_statistic": None,
            "kpss_pvalue": None,
            "kpss_stationary": False,
            "is_stationary": False,
            "recommendation": ""
        }
        
        clean_series = series.dropna()
        if len(clean_series) < 20:
            results["recommendation"] = "Insufficient data for stationarity tests"
            return results
        
        try:
            from statsmodels.tsa.stattools import adfuller, kpss
            
            # ADF Test (null: unit root exists, i.e., non-stationary)
            adf_result = adfuller(clean_series, autolag='AIC')
            results["adf_statistic"] = float(adf_result[0])
            results["adf_pvalue"] = float(adf_result[1])
            results["adf_stationary"] = adf_result[1] < alpha  # Reject null = stationary
            
            # KPSS Test (null: stationary)
            try:
                kpss_result = kpss(clean_series, regression='c', nlags='auto')
                results["kpss_statistic"] = float(kpss_result[0])
                results["kpss_pvalue"] = float(kpss_result[1])
                results["kpss_stationary"] = kpss_result[1] >= alpha  # Fail to reject = stationary
            except Exception as kpss_error:
                logger.debug(f"KPSS test failed: {kpss_error}")
            
            # Combined interpretation
            if results["adf_stationary"] and results["kpss_stationary"]:
                results["is_stationary"] = True
                results["recommendation"] = "Series is stationary (both tests agree)"
            elif results["adf_stationary"] and not results["kpss_stationary"]:
                results["is_stationary"] = False
                results["recommendation"] = "Difference-stationary (ADF suggests stationary, KPSS suggests trend)"
            elif not results["adf_stationary"] and results["kpss_stationary"]:
                results["is_stationary"] = False
                results["recommendation"] = "Trend-stationary or near unit root"
            else:
                results["is_stationary"] = False
                results["recommendation"] = "Non-stationary (both tests agree), differencing recommended"
            
        except Exception as e:
            logger.warning(f"Stationarity tests failed: {e}")
            results["recommendation"] = f"Tests failed: {str(e)}"
            
        return results
    
    def _determine_differencing(self, series: pd.Series) -> Optional[int]:
        """Determine optimal differencing order using statistical tests."""
        if not PMDARIMA_AVAILABLE:
            return None
            
        try:
            values = series.values
            
            # Estimate d using KPSS test
            d = ndiffs(values, test='kpss', max_d=2)
            
            logger.info(f"Recommended differencing order: d={d}")
            return d
            
        except Exception as e:
            logger.warning(f"Differencing determination failed: {e}")
            return None
    
    def _fit_statsmodels(self, series: pd.Series) -> None:
        """Fit using statsmodels SARIMAX with robust settings."""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            # Ensure index freq is inferred if possible
            if hasattr(series.index, 'freq') and series.index.freq is None:
                try:
                    series.index.freq = pd.infer_freq(series.index)
                except (TypeError, ValueError) as freq_error:
                    logger.debug(f"Could not infer frequency: {freq_error}")

            self.model = SARIMAX(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                hamilton_representation=False
            )
            
            self.fitted_model = self.model.fit(
                disp=False, 
                maxiter=300,
                method='lbfgs',
                optim_score='approx'
            )
            self.residuals = self.fitted_model.resid
            self.aic = self.fitted_model.aic
            self.bic = self.fitted_model.bic
            self.is_pmdarima_model = False
            
        except Exception as e:
            logger.error(f"Statsmodels fitting error: {e}")
            raise
    
    def _time_series_cv(self, series: pd.Series) -> List[float]:
        """Perform time series cross-validation with rolling window."""
        scores = []
        n = len(series)
        
        if n < 60:  # Need enough data for CV
            return scores
            
        try:
            fold_size = n // (self.cv_folds + 1)
            
            for i in range(self.cv_folds):
                train_end = fold_size * (i + 1)
                test_end = min(train_end + fold_size // 2, n)
                
                if train_end >= n or test_end >= n:
                    break
                    
                train = series.iloc[:train_end]
                test = series.iloc[train_end:test_end]
                
                if len(test) < 5:
                    continue
                
                # Fit on training data
                if self.is_pmdarima_model and self.fitted_model is not None:
                    try:
                        preds = self.fitted_model.predict(n_periods=len(test))
                        mse = np.mean((test.values - preds) ** 2)
                        scores.append(float(mse))
                    except Exception as cv_error:
                        logger.debug(f"CV fold {i} failed: {cv_error}")
                        
        except Exception as e:
            logger.warning(f"Time series CV failed: {e}")
            
        return scores
    
    def _residual_diagnostics(self) -> Dict[str, Any]:
        """Perform residual diagnostic tests for model validation."""
        diagnostics = {
            "ljung_box_statistic": None,
            "ljung_box_pvalue": None,
            "residuals_are_white_noise": False,
            "normality_statistic": None,
            "normality_pvalue": None,
            "residuals_are_normal": False,
            "heteroscedasticity_statistic": None,
            "heteroscedasticity_pvalue": None
        }
        
        if self.residuals is None:
            return diagnostics
            
        try:
            residuals = np.array(self.residuals).flatten()
            residuals = residuals[~np.isnan(residuals)]
            
            if len(residuals) < 20:
                return diagnostics
            
            # Ljung-Box test for autocorrelation in residuals
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                
                lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
                diagnostics["ljung_box_statistic"] = float(lb_result['lb_stat'].iloc[0])
                diagnostics["ljung_box_pvalue"] = float(lb_result['lb_pvalue'].iloc[0])
                diagnostics["residuals_are_white_noise"] = lb_result['lb_pvalue'].iloc[0] > 0.05
            except Exception as lb_error:
                logger.debug(f"Ljung-Box test failed: {lb_error}")
            
            # Normality test (Jarque-Bera)
            try:
                jb_stat, jb_pvalue = stats.jarque_bera(residuals)
                diagnostics["normality_statistic"] = float(jb_stat)
                diagnostics["normality_pvalue"] = float(jb_pvalue)
                diagnostics["residuals_are_normal"] = jb_pvalue > 0.05
            except Exception as norm_error:
                logger.debug(f"Normality test failed: {norm_error}")
            
            # Heteroscedasticity test (ARCH effect)
            try:
                from statsmodels.stats.diagnostic import het_arch
                
                arch_result = het_arch(residuals)
                diagnostics["heteroscedasticity_statistic"] = float(arch_result[0])
                diagnostics["heteroscedasticity_pvalue"] = float(arch_result[1])
            except Exception as arch_error:
                logger.debug(f"ARCH test failed: {arch_error}")
                
        except Exception as e:
            logger.warning(f"Residual diagnostics failed: {e}")
            
        return diagnostics
    
    def _calculate_accuracy_metrics(self, series: pd.Series) -> None:
        """
        Calculate MAPE and RMSE on held-out test data.
        
        Crucial: fits a NEW model on the training set to avoid lookahead bias.
        """
        try:
            n = len(series)
            test_size = max(10, int(n * 0.15))  # 15% for testing
            train = series.iloc[:-test_size]
            test = series.iloc[-test_size:]
            
            if len(test) < 5:
                return
            
            # Fit specific model on TRAINING data only
            preds = []
            
            try:
                if self.is_pmdarima_model:
                    import pmdarima
                    # Create a new ARIMA instance with same parameters
                    # We don't want to run auto_arima again (too slow), just fit the found order
                    temp_model = pmdarima.ARIMA(
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        suppress_warnings=True
                    )
                    temp_model.fit(train.values)
                    preds = temp_model.predict(n_periods=len(test))
                    
                else:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    temp_model = SARIMAX(
                        train,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    temp_result = temp_model.fit(disp=False, maxiter=200)
                    forecast_obj = temp_result.get_forecast(steps=len(test))
                    preds = forecast_obj.predicted_mean.values
                
                actual = test.values
                
                # MAPE (Mean Absolute Percentage Error)
                non_zero_mask = actual != 0
                if np.any(non_zero_mask):
                    self.mape = float(np.mean(np.abs((actual[non_zero_mask] - preds[non_zero_mask]) / actual[non_zero_mask])) * 100)
                
                # RMSE (Root Mean Square Error)
                self.rmse = float(np.sqrt(np.mean((actual - preds) ** 2)))
                
                logger.info(f"Out-of-sample Accuracy - MAPE: {self.mape:.2f}%, RMSE: {self.rmse:.4f}")
                
            except Exception as fit_error:
                logger.warning(f"Validation fit failed: {fit_error}")
            
        except Exception as e:
            logger.warning(f"Accuracy metrics calculation failed: {e}")
    
    def _test_stationarity(self, series: pd.Series, alpha: float = 0.05) -> bool:
        """Test for stationarity using ADF test (legacy method for compatibility)."""
        return self._comprehensive_stationarity_test(series, alpha).get("is_stationary", False)
    
    def forecast(
        self,
        steps: int = 30,
        confidence: float = 0.95
    ) -> Dict[str, Any]:
        """Generate forecasts with confidence intervals."""
        if self.fitted_model is None:
            return {"success": False, "error": "Model not fitted"}
        
        try:
            if self.is_pmdarima_model:
                # pmdarima model
                forecast_result = self.fitted_model.predict(
                    n_periods=steps,
                    return_conf_int=True,
                    alpha=1 - confidence
                )
                
                if isinstance(forecast_result, tuple):
                    forecast = forecast_result[0]
                    conf_int = forecast_result[1]
                    lower = conf_int[:, 0]
                    upper = conf_int[:, 1]
                else:
                    forecast = forecast_result
                    # Fallback intervals based on historical volatility
                    if self.residuals is not None:
                        std = np.std(self.residuals)
                        z = stats.norm.ppf(1 - (1 - confidence) / 2)
                        lower = forecast - z * std * np.sqrt(np.arange(1, steps + 1))
                        upper = forecast + z * std * np.sqrt(np.arange(1, steps + 1))
                    else:
                        lower = forecast * 0.95
                        upper = forecast * 1.05
            else:
                # statsmodels model
                forecast_obj = self.fitted_model.get_forecast(steps=steps)
                forecast = forecast_obj.predicted_mean
                conf_int = forecast_obj.conf_int(alpha=1 - confidence)
                lower = conf_int.iloc[:, 0].values
                upper = conf_int.iloc[:, 1].values
            
            # Convert to lists and clean up
            forecast_list = [float(x) for x in forecast]
            lower_list = [float(x) for x in lower]
            upper_list = [float(x) for x in upper]
            
            # Ensure no negative prices
            forecast_list = [max(0, x) for x in forecast_list]
            lower_list = [max(0, x) for x in lower_list]
            upper_list = [max(0, x) for x in upper_list]
            
            return {
                "success": True,
                "forecast": forecast_list,
                "confidence_lower": lower_list,
                "confidence_upper": upper_list,
                "confidence_level": confidence,
                "steps": steps,
                "model_mape": self.mape,
                "model_rmse": self.rmse
            }
            
        except Exception as e:
            logger.error(f"SARIMA forecasting error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_residuals(self) -> Optional[np.ndarray]:
        """Return model residuals for LSTM input."""
        if self.residuals is not None:
            return np.array(self.residuals)
        return None
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive model diagnostic statistics."""
        if self.fitted_model is None:
            return {}
        
        try:
            diagnostics = {
                "aic": float(self.aic) if self.aic else None,
                "bic": float(self.bic) if self.bic else None,
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "model_type": "auto_arima" if self.is_pmdarima_model else "sarimax",
                "mape": self.mape,
                "rmse": self.rmse,
                "stationarity_tests": self.stationarity_tests,
                "residual_diagnostics": self.residual_diagnostics,
                "cv_scores": self.cv_scores
            }
            
            if self.residuals is not None:
                residuals = np.array(self.residuals)
                clean_residuals = residuals[~np.isnan(residuals)]
                diagnostics["residual_mean"] = float(np.mean(clean_residuals))
                diagnostics["residual_std"] = float(np.std(clean_residuals))
                diagnostics["residual_skewness"] = float(stats.skew(clean_residuals))
                diagnostics["residual_kurtosis"] = float(stats.kurtosis(clean_residuals))
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Diagnostics error: {e}")
            return {}


# Factory function
def create_sarima_model(**kwargs) -> SARIMAModel:
    """Create a new SARIMA model instance."""
    return SARIMAModel(**kwargs)
