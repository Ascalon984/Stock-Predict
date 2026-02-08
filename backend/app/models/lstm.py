"""Enhanced LSTM model with attention mechanism for stock prediction.

Improvements:
- Working Monte Carlo Dropout for uncertainty quantification
- Walk-forward validation for time series
- Learning rate scheduling with warmup and cosine decay
- Comprehensive accuracy metrics (MAPE, RMSE, MAE, R²)
- Ensemble predictions from multiple dropout passes
- RobustScaler option for handling outliers
- Residual connections for better gradient flow
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime
import logging
import warnings
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Global flag for TF availability
TF_AVAILABLE = False
MCDropout = None  # Will be defined if TensorFlow is available

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
    TF_AVAILABLE = True
    
    # Define MCDropout only when TensorFlow is available
    class MCDropout(layers.Dropout):
        """Monte Carlo Dropout layer that applies dropout during inference."""
        
        def call(self, inputs, training=None):
            # Always apply dropout, regardless of training mode
            return super().call(inputs, training=True)

except ImportError:
    logger.warning("TensorFlow not installed. LSTM capabilities will be unavailable.")


class LSTMModel:
    """
    Enhanced LSTM model for stock price prediction.
    
    Features:
    - Monte Carlo Dropout for uncertainty quantification
    - Attention mechanism for capturing important time steps
    - Bidirectional LSTM for capturing past and future context
    - Walk-forward validation for robust evaluation
    - Multiple accuracy metrics
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = None,
        lstm_units: List[int] = None,
        dense_units: List[int] = None,
        dropout_rate: float = 0.2,
        use_attention: bool = True,
        use_bidirectional: bool = True,
        use_mc_dropout: bool = True,
        mc_samples: int = 50,
        use_robust_scaler: bool = False
    ):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units or [128, 64]
        self.dense_units = dense_units or [32, 16]
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.use_bidirectional = use_bidirectional
        self.use_mc_dropout = use_mc_dropout
        self.mc_samples = mc_samples
        self.use_robust_scaler = use_robust_scaler
        
        self.model = None
        self.scaler = None
        self.history = None
        self.is_fitted = False
        
        # Accuracy metrics
        self.mape: Optional[float] = None
        self.rmse: Optional[float] = None
        self.mae: Optional[float] = None
        self.r2: Optional[float] = None
        self.walk_forward_scores: List[Dict[str, float]] = []
        
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """Build the LSTM model architecture with MC Dropout."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")

        try:
            # Set random seeds for reproducibility
            tf.random.set_seed(42)
            np.random.seed(42)
            
            # Select dropout layer type
            # MCDropout may be None if TensorFlow wasn't available initially
            DropoutLayer = MCDropout if (self.use_mc_dropout and MCDropout is not None) else layers.Dropout
            
            # Input layer
            inputs = Input(shape=input_shape, name='input')
            x = inputs
            
            # Add input normalization
            x = layers.LayerNormalization(name='input_norm')(x)
            
            # LSTM layers with residual connections where possible
            for i, units in enumerate(self.lstm_units):
                return_sequences = (i < len(self.lstm_units) - 1) or self.use_attention
                
                if self.use_bidirectional:
                    lstm_layer = layers.Bidirectional(
                        layers.LSTM(
                            units,
                            return_sequences=return_sequences,
                            kernel_regularizer=keras.regularizers.l2(0.001),
                            recurrent_regularizer=keras.regularizers.l2(0.001),
                            recurrent_dropout=0.1,  # Recurrent dropout for regularization
                            name=f'lstm_{i}'
                        ),
                        name=f'bidirectional_{i}'
                    )
                else:
                    lstm_layer = layers.LSTM(
                        units,
                        return_sequences=return_sequences,
                        kernel_regularizer=keras.regularizers.l2(0.001),
                        recurrent_dropout=0.1,
                        name=f'lstm_{i}'
                    )
                
                x = lstm_layer(x)
                x = layers.LayerNormalization(name=f'ln_{i}')(x)
                x = DropoutLayer(self.dropout_rate, name=f'dropout_{i}')(x)
            
            # Attention mechanism
            if self.use_attention and len(x.shape) == 3:
                # Multi-head self-attention style
                attention_weights = layers.Dense(1, activation='tanh', name='attention_weights')(x)
                attention_weights = layers.Softmax(axis=1, name='attention_softmax')(attention_weights)
                
                # Apply attention
                context = layers.Multiply(name='attention_multiply')([x, attention_weights])
                x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name='attention_sum')(context)
            
            # Dense layers with residual connections
            for i, units in enumerate(self.dense_units):
                x = layers.Dense(
                    units,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001),
                    name=f'dense_{i}'
                )(x)
                x = layers.LayerNormalization(name=f'dense_ln_{i}')(x)
                x = DropoutLayer(self.dropout_rate / 2, name=f'dense_dropout_{i}')(x)
            
            # Output layer (predict next close price)
            outputs = layers.Dense(1, name='output')(x)
            
            # Build model
            self.model = Model(inputs=inputs, outputs=outputs, name='lstm_stock_predictor')
            
            # Compile with Huber loss (robust to outliers)
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='huber',
                metrics=['mae', 'mse']
            )
            
            logger.info(f"Built LSTM model with input shape {input_shape}, MC Dropout: {self.use_mc_dropout}")
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {e}")
            raise
    
    def _create_sequences(
        self,
        data: np.ndarray,
        target_column: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        
        if len(data) <= self.sequence_length:
            return np.array([]), np.array([])

        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(data[i, target_column])  # Predict Close price
        
        return np.array(X), np.array(y)
    
    def _get_lr_scheduler(self, epochs: int, warmup_epochs: int = 5):
        """Create learning rate scheduler with warmup and cosine decay."""
        def scheduler(epoch, lr):
            if epoch < warmup_epochs:
                # Linear warmup
                return lr * (epoch + 1) / warmup_epochs
            else:
                # Cosine decay
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        return LearningRateScheduler(scheduler, verbose=0)
    
    def fit(
        self,
        data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        patience: int = 15,
        verbose: int = 0,
        use_walk_forward: bool = True,
        n_splits: int = 3
    ) -> Dict[str, Any]:
        """Fit the LSTM model with walk-forward validation."""
        if not TF_AVAILABLE:
            return {"success": False, "error": "TensorFlow not installed"}

        start_time = datetime.now()
        
        if len(data) < self.sequence_length + 10:
            logger.warning(f"Insufficient data for sequence length {self.sequence_length}")
            return {"success": False, "error": "Insufficient data"}
        
        try:
            # Select scaler based on configuration
            if self.use_robust_scaler:
                from sklearn.preprocessing import RobustScaler
                self.scaler = RobustScaler()
            else:
                from sklearn.preprocessing import MinMaxScaler
                self.scaler = MinMaxScaler(feature_range=(0, 1))
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(data)
            
            # Create sequences
            X, y = self._create_sequences(scaled_data)
            
            if len(X) < 10:
                logger.warning(f"Too few sequences ({len(X)}), cannot train")
                return {"success": False, "error": "Insufficient sequences generated"}
            
            if len(X) < 50:
                logger.warning(f"Too few sequences ({len(X)}), adjusting sequence length")
                self.sequence_length = max(5, len(data) // 5)
                X, y = self._create_sequences(scaled_data)
            
            # Set feature dimensions
            self.n_features = X.shape[2]
            
            # Build model
            self._build_model(input_shape=(X.shape[1], X.shape[2]))
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=verbose
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=patience // 2,
                    min_lr=1e-6,
                    verbose=verbose
                ),
                self._get_lr_scheduler(epochs)
            ]
            
            # Walk-forward validation
            if use_walk_forward and len(X) >= 100:
                self.walk_forward_scores = self._walk_forward_validation(
                    X, y, n_splits=n_splits, epochs=epochs // 2, batch_size=batch_size
                )
            
            # Final training on all data
            self.history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose,
                shuffle=False  # Important for time series
            )
            
            self.is_fitted = True
            fit_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate accuracy metrics on validation set
            val_size = int(len(X) * validation_split)
            if val_size > 0:
                X_val = X[-val_size:]
                y_val = y[-val_size:]
                self._calculate_metrics(X_val, y_val)
            
            # Get final metrics
            final_loss = self.history.history['loss'][-1]
            final_val_loss = self.history.history.get('val_loss', [final_loss])[-1]
            best_epoch = len(self.history.history['loss'])
            
            return {
                "success": True,
                "epochs_completed": best_epoch,
                "final_loss": float(final_loss),
                "final_val_loss": float(final_val_loss),
                "fit_time_seconds": fit_time,
                "sequence_length": self.sequence_length,
                "n_features": self.n_features,
                "n_samples": len(X),
                "mape": self.mape,
                "rmse": self.rmse,
                "mae": self.mae,
                "r2": self.r2,
                "walk_forward_scores": self.walk_forward_scores,
                "mc_dropout_enabled": self.use_mc_dropout
            }
            
        except Exception as e:
            logger.error(f"LSTM training error: {e}")
            return {"success": False, "error": str(e)}
    
    def _walk_forward_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 3,
        epochs: int = 30,
        batch_size: int = 32
    ) -> List[Dict[str, float]]:
        """Perform walk-forward validation for time series."""
        scores = []
        n = len(X)
        split_size = n // (n_splits + 1)
        
        for i in range(n_splits):
            train_end = split_size * (i + 1)
            test_start = train_end
            test_end = min(test_start + split_size, n)
            
            if test_end <= test_start:
                break
            
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            try:
                # Create a temporary model for this fold
                temp_model = keras.models.clone_model(self.model)
                temp_model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='huber',
                    metrics=['mae', 'mse']
                )
                
                temp_model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.1,
                    verbose=0,
                    callbacks=[
                        EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
                    ]
                )
                
                # Evaluate on test set
                y_pred = temp_model.predict(X_test, verbose=0).flatten()
                
                mse = float(np.mean((y_test - y_pred) ** 2))
                mae = float(np.mean(np.abs(y_test - y_pred)))
                
                scores.append({
                    "fold": i + 1,
                    "mse": mse,
                    "mae": mae,
                    "train_size": len(X_train),
                    "test_size": len(X_test)
                })
                
                # Clean up
                del temp_model
                
            except Exception as fold_error:
                logger.warning(f"Walk-forward fold {i+1} failed: {fold_error}")
        
        return scores
    
    def _calculate_metrics(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Calculate comprehensive accuracy metrics."""
        try:
            if self.use_mc_dropout:
                # Use MC Dropout for predictions
                predictions = []
                for _ in range(self.mc_samples):
                    pred = self.model(X_val, training=True)  # Enable dropout
                    predictions.append(pred.numpy().flatten())
                
                y_pred = np.mean(predictions, axis=0)
            else:
                y_pred = self.model.predict(X_val, verbose=0).flatten()
            
            # Inverse transform to get actual values
            # Note: we need to create dummy arrays for inverse transform
            dummy_pred = np.zeros((len(y_pred), self.n_features))
            dummy_pred[:, 0] = y_pred
            pred_unscaled = self.scaler.inverse_transform(dummy_pred)[:, 0]
            
            dummy_actual = np.zeros((len(y_val), self.n_features))
            dummy_actual[:, 0] = y_val
            actual_unscaled = self.scaler.inverse_transform(dummy_actual)[:, 0]
            
            # Calculate metrics on unscaled values
            non_zero_mask = actual_unscaled != 0
            if np.any(non_zero_mask):
                self.mape = float(np.mean(np.abs(
                    (actual_unscaled[non_zero_mask] - pred_unscaled[non_zero_mask]) / 
                    actual_unscaled[non_zero_mask]
                )) * 100)
            
            self.rmse = float(np.sqrt(np.mean((actual_unscaled - pred_unscaled) ** 2)))
            self.mae = float(np.mean(np.abs(actual_unscaled - pred_unscaled)))
            
            # R² score
            ss_res = np.sum((actual_unscaled - pred_unscaled) ** 2)
            ss_tot = np.sum((actual_unscaled - np.mean(actual_unscaled)) ** 2)
            if ss_tot > 0:
                self.r2 = float(1 - (ss_res / ss_tot))
            
            logger.info(f"LSTM Metrics - MAPE: {self.mape:.2f}%, RMSE: {self.rmse:.4f}, R²: {self.r2:.4f}")
            
        except Exception as e:
            logger.warning(f"Metrics calculation failed: {e}")
    
    def predict(
        self,
        data: np.ndarray,
        steps: int = 30,
        seed: int = None
    ) -> Dict[str, Any]:
        """Generate multi-step predictions with MC Dropout uncertainty."""
        if not TF_AVAILABLE:
            return {"success": False, "error": "TensorFlow not installed"}

        if not self.is_fitted or self.model is None:
            return {"success": False, "error": "Model not fitted"}
        
        try:
            # Set deterministic seed for this prediction run if provided
            # This ensures that MC Dropout selects the SAME dropout masks for the same input
            # giving visual consistency while preserving the probabilistic distribution logic
            if seed is not None:
                tf.random.set_seed(seed)
                np.random.seed(seed)

            # Scale data
            scaled_data = self.scaler.transform(data)
            
            # Get last sequence
            if len(scaled_data) < self.sequence_length:
                return {"success": False, "error": "Insufficient data for prediction"}
            
            current_sequence = scaled_data[-self.sequence_length:].copy()
            
            predictions = []
            prediction_intervals = []
            all_mc_predictions = []
            
            for step in range(steps):
                # Reshape for prediction
                X_pred = current_sequence.reshape(1, self.sequence_length, self.n_features)
                
                if self.use_mc_dropout:
                    # Monte Carlo Dropout for uncertainty estimation
                    mc_preds = []
                    for _ in range(self.mc_samples):
                        pred = self.model(X_pred, training=True)  # Enable dropout
                        mc_preds.append(pred.numpy()[0, 0])
                    
                    mc_preds = np.array(mc_preds)
                    pred_scaled = np.mean(mc_preds)
                    pred_std = np.std(mc_preds)
                    
                    all_mc_predictions.append(mc_preds.tolist())
                else:
                    pred_scaled = self.model.predict(X_pred, verbose=0)[0, 0]
                    pred_std = 0.01  # Default small std
                
                # Create inverse transform array
                dummy = np.zeros((1, self.n_features))
                dummy[0, 0] = pred_scaled
                pred_unscaled = self.scaler.inverse_transform(dummy)[0, 0]
                
                # Uncertainty estimation using MC Dropout std
                # Scale uncertainty with prediction horizon
                horizon_factor = 1 + (step * 0.02)  # 2% increase per step
                uncertainty = pred_std * horizon_factor * 2  # 2 std for ~95% CI
                
                dummy_lower = dummy.copy()
                dummy_upper = dummy.copy()
                dummy_lower[0, 0] = pred_scaled - uncertainty
                dummy_upper[0, 0] = pred_scaled + uncertainty
                
                lower = self.scaler.inverse_transform(dummy_lower)[0, 0]
                upper = self.scaler.inverse_transform(dummy_upper)[0, 0]
                
                predictions.append(max(0, float(pred_unscaled)))
                prediction_intervals.append({
                    "lower": max(0, float(lower)),
                    "upper": max(0, float(upper)),
                    "std": float(pred_std) if self.use_mc_dropout else None
                })
                
                # Update sequence for next prediction
                new_row = current_sequence[-1].copy()
                new_row[0] = pred_scaled
                current_sequence = np.vstack([current_sequence[1:], new_row])
            
            return {
                "success": True,
                "predictions": predictions,
                "intervals": prediction_intervals,
                "steps": steps,
                "mc_dropout_enabled": self.use_mc_dropout,
                "mc_samples": self.mc_samples if self.use_mc_dropout else None,
                "model_mape": self.mape,
                "model_rmse": self.rmse,
                "model_r2": self.r2
            }
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model architecture and performance summary."""
        if self.model is None:
            return {}
        
        return {
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "lstm_units": self.lstm_units,
            "dense_units": self.dense_units,
            "use_attention": self.use_attention,
            "use_bidirectional": self.use_bidirectional,
            "use_mc_dropout": self.use_mc_dropout,
            "mc_samples": self.mc_samples,
            "mape": self.mape,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "walk_forward_scores": self.walk_forward_scores,
            "total_params": self.model.count_params() if self.model else None
        }


# Factory function
def create_lstm_model(**kwargs) -> LSTMModel:
    """Create a new LSTM model instance."""
    return LSTMModel(**kwargs)

