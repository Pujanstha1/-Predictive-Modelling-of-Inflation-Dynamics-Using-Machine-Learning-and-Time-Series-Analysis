import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Tuple, Dict
from .base_model import BaseModel

class SARIMAModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.exog_scaler = None
        
    def preprocess_data(self, data):
        """Preprocess data for SARIMA model"""
        # Convert to numeric and handle missing values
        numeric_data = data.apply(pd.to_numeric, errors='coerce')
        
        # Replace infinite values with NaN
        numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill NaN values
        numeric_data = numeric_data.fillna(method='ffill').fillna(method='bfill')
        
        return numeric_data
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train SARIMA model"""
        print("Training SARIMA model...")
        
        # Preprocess exogenous variables
        X_train_processed = self.preprocess_data(X_train)
        y_train_processed = pd.to_numeric(y_train, errors='coerce')
        y_train_processed = y_train_processed.fillna(method='ffill').fillna(method='bfill')
        
        try:
            model = SARIMAX(
                y_train_processed,
                exog=X_train_processed if not X_train_processed.empty else None,
                **self.config.sarima_params
            )
            
            self.model = model.fit(disp=False)
            print("SARIMA training completed")
            
            # Store feature importance (coefficients)
            if not X_train_processed.empty:
                self.feature_importance = pd.DataFrame({
                    'feature': X_train_processed.columns,
                    'importance': np.abs(self.model.params[1:len(X_train_processed.columns) + 1])
                }).sort_values('importance', ascending=False)
            
        except Exception as e:
            print(f"Error in SARIMA training: {str(e)}")
            raise
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate predictions with SARIMA model"""
        # Preprocess exogenous variables
        X_test_processed = self.preprocess_data(X_test)
        
        forecast = self.model.get_forecast(
            steps=len(X_test),
            exog=X_test_processed if not X_test_processed.empty else None
        )
        self.predictions = forecast.predicted_mean
        return self.predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get SARIMA coefficients as feature importance"""
        return self.feature_importance
    
    def calculate_prediction_intervals(self, X_test: pd.DataFrame, 
                                    confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate prediction intervals for SARIMA"""
        X_test_processed = self.preprocess_data(X_test)
        
        forecast = self.model.get_forecast(
            steps=len(X_test),
            exog=X_test_processed if not X_test_processed.empty else None
        )
        conf_int = forecast.conf_int(alpha=1-confidence)
        
        self.prediction_intervals = {
            'lower': conf_int.iloc[:, 0],
            'upper': conf_int.iloc[:, 1]
        }
        
        return conf_int.iloc[:, 0], conf_int.iloc[:, 1]
    
# import pandas as pd
# import numpy as np
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from typing import Tuple, Dict
# from .base_model import BaseModel

# class SARIMAModel(BaseModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.exog_scaler = None
        
#     def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
#         """Train SARIMA model"""
#         print("Training SARIMA model...")
        
#         model = SARIMAX(
#             y_train,
#             exog=X_train,
#             **self.config.sarima_params
#         )
        
#         self.model = model.fit(disp=False)
#         print("SARIMA training completed")
        
#         # Store feature importance (coefficients)
#         self.feature_importance = pd.DataFrame({
#             'feature': X_train.columns,
#             'importance': self.model.params[:len(X_train.columns)]
#         }).sort_values('importance', ascending=False)
    
#     def predict(self, X_test: pd.DataFrame) -> np.ndarray:
#         """Generate predictions with SARIMA model"""
#         forecast = self.model.get_forecast(
#             len(X_test),
#             exog=X_test
#         )
#         self.predictions = forecast.predicted_mean
#         return self.predictions
    
#     def get_feature_importance(self) -> pd.DataFrame:
#         """Get SARIMA coefficients as feature importance"""
#         return self.feature_importance
    
#     def calculate_prediction_intervals(self, X_test: pd.DataFrame, 
#                                     confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
#         """Calculate prediction intervals for SARIMA"""
#         forecast = self.model.get_forecast(
#             len(X_test),
#             exog=X_test
#         )
#         conf_int = forecast.conf_int(alpha=1-confidence)
        
#         self.prediction_intervals = {
#             'lower': conf_int.iloc[:, 0],
#             'upper': conf_int.iloc[:, 1]
#         }
        
#         return conf_int.iloc[:, 0], conf_int.iloc[:, 1]