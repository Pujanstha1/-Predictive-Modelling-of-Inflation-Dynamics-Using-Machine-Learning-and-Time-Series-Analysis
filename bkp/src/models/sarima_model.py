import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Tuple, Dict
from .base_model import BaseModel
import traceback

class SARIMAModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.exog_scaler = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train SARIMA model"""
        print("Training SARIMA model...")
        
        model = SARIMAX(
            y_train,
            exog=X_train,
            **self.config.sarima_params
        )
        
        self.model = model.fit(disp=False)
        print("SARIMA training completed")
        
        # Store feature importance (coefficients)
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.params[:len(X_train.columns)]
        }).sort_values('importance', ascending=False)
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate predictions with SARIMA model"""
        try:
            # Clean input data
            X_test = X_test.copy()
            numeric_data = X_test.select_dtypes(include=[np.number])
            numeric_data = numeric_data.ffill().bfill()
            
            # Generate forecast
            forecast = self.model.get_forecast(
                steps=len(X_test),
                exog=numeric_data
            )
            
            # Get predictions
            predictions = forecast.predicted_mean
            
            # Convert to numpy array and ensure right shape
            predictions = np.array(predictions)
            
            # Store predictions
            self.predictions = predictions
            
            print(f"Generated SARIMA predictions: {predictions[:5]}")
            
            return predictions
            
        except Exception as e:
            print(f"Error in SARIMA predict: {str(e)}")
            print(traceback.format_exc())
            return None
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get SARIMA coefficients as feature importance"""
        return self.feature_importance
    
    def calculate_prediction_intervals(self, X_test: pd.DataFrame, 
                                    confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate prediction intervals for SARIMA"""
        forecast = self.model.get_forecast(
            len(X_test),
            exog=X_test
        )
        conf_int = forecast.conf_int(alpha=1-confidence)
        
        self.prediction_intervals = {
            'lower': conf_int.iloc[:, 0],
            'upper': conf_int.iloc[:, 1]
        }
        
        return conf_int.iloc[:, 0], conf_int.iloc[:, 1]