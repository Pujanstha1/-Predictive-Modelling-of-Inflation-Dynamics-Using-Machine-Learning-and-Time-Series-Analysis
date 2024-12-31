from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple
from ..utils.config import Config

class BaseModel(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.predictions = None
        self.prediction_intervals = None
        self.feature_importance = None
        self.metrics = {}
        
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        pass
    
    # def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
    #     """Calculate model performance metrics"""
    #     metrics = {
    #         'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
    #         'mae': mean_absolute_error(y_true, y_pred),
    #         'r2': r2_score(y_true, y_pred),
    #         'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    #         'directional_accuracy': np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
    #     }
    #     self.metrics = metrics
    #     return metrics
    
    # def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
    #     """Calculate model performance metrics with proper index handling"""
    #     try:
    #         # Convert predictions to Series if necessary
    #         if isinstance(y_pred, np.ndarray):
    #             y_pred = pd.Series(y_pred, index=y_true.index)
            
    #         # Ensure both are Series with same index
    #         if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
    #             y_true = y_true.reindex(y_pred.index)
            
    #         # Calculate metrics
    #         metrics = {
    #             'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
    #             'mae': mean_absolute_error(y_true, y_pred),
    #             'r2': r2_score(y_true, y_pred),
    #             'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    #             'directional_accuracy': np.mean(
    #                 np.sign(y_true.values) == np.sign(y_pred.values)
    #             ) * 100
    #         }
            
    #         return metrics
            
    #     except Exception as e:
    #         print(f"Error calculating metrics: {str(e)}")
    #         return {
    #             'rmse': np.nan,
    #             'mae': np.nan,
    #             'r2': np.nan,
    #             'mape': np.nan,
    #             'directional_accuracy': np.nan
    #         }
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate model performance metrics"""
        # Convert numpy array to pandas Series if needed
        if isinstance(y_pred, np.ndarray):
            y_pred = pd.Series(y_pred, index=y_true.index[-len(y_pred):])

        # Align the series
        y_true = y_true[-len(y_pred):]
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'directional_accuracy': np.mean(np.sign(y_true.values) == np.sign(y_pred.values)) * 100
        }
        
        return metrics
    
    def calculate_prediction_intervals(self, X_test: pd.DataFrame, 
                                    confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate prediction intervals"""
        pass  # Implemented by child classes if applicable
    
    def save_metrics(self, model_name: str) -> None:
        """Save model metrics to file"""
        self.config.save_model_metrics(self.metrics, model_name)