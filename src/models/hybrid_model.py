import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .base_model import BaseModel
from .sarima_model import SARIMAModel
from .ml_models import XGBoostModel, RandomForestModel

class HybridModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.sarima = SARIMAModel(config)
        self.xgboost = XGBoostModel(config)
        self.random_forest = RandomForestModel(config)
        self.dynamic_weights = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train all component models"""
        # Train SARIMA
        sarima_features = [col for col in X_train.columns 
                         if col in self.config.price_features + ['month_sin', 'month_cos']]
        self.sarima.train(X_train[sarima_features], y_train)
        
        # Train ML models
        self.xgboost.train(X_train, y_train)
        self.random_forest.train(X_train, y_train)
        
        # Initialize weights
        self.dynamic_weights = self.config.initial_weights
        
    def calculate_dynamic_weights(self, y_true: pd.Series, 
                                predictions: Dict[str, np.ndarray],
                                window_size: int = 3) -> Dict[str, float]:
        """Calculate dynamic weights based on recent performance"""
        recent_errors = {}
        
        for name, preds in predictions.items():
            # Calculate weighted recent errors
            errors = np.abs(y_true - preds)
            weights = np.linspace(0.5, 1.0, len(errors))
            weighted_errors = errors * weights
            recent_errors[name] = np.mean(weighted_errors[-window_size:])
        
        # Calculate weights based on inverse error
        total_inv_error = sum(1/e for e in recent_errors.values())
        weights = {name: (1/error)/total_inv_error 
                  for name, error in recent_errors.items()}
        
        self.dynamic_weights = weights
        return weights
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions"""
        # Generate predictions from each model
        sarima_features = [col for col in X_test.columns 
                         if col in self.config.price_features + ['month_sin', 'month_cos']]
        
        predictions = {
            'sarima': self.sarima.predict(X_test[sarima_features]),
            'xgboost': self.xgboost.predict(X_test),
            'random_forest': self.random_forest.predict(X_test)
        }
        
        # Calculate weighted ensemble predictions
        self.predictions = np.zeros_like(predictions['sarima'])
        for name, preds in predictions.items():
            self.predictions += self.dynamic_weights[name] * preds
        
        return self.predictions
    
    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """Get feature importance from all models"""
        return {
            'sarima': self.sarima.get_feature_importance(),
            'xgboost': self.xgboost.get_feature_importance(),
            'random_forest': self.random_forest.get_feature_importance()
        }
    
    def calculate_prediction_intervals(self, X_test: pd.DataFrame, 
                                    confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate prediction intervals for hybrid model"""
        # Get intervals from each model
        sarima_features = [col for col in X_test.columns 
                         if col in self.config.price_features + ['month_sin', 'month_cos']]
        
        _, sarima_upper = self.sarima.calculate_prediction_intervals(X_test[sarima_features], confidence)
        _, xgb_upper = self.xgboost.calculate_prediction_intervals(X_test, confidence)
        _, rf_upper = self.random_forest.calculate_prediction_intervals(X_test, confidence)
        
        # Combine intervals using weighted average
        lower_bounds = np.zeros_like(self.predictions)
        upper_bounds = np.zeros_like(self.predictions)
        
        # Weight the intervals from each model
        intervals = {
            'sarima': self.sarima.prediction_intervals,
            'xgboost': self.xgboost.prediction_intervals,
            'random_forest': self.random_forest.prediction_intervals
        }
        
        for name, interval in intervals.items():
            lower_bounds += self.dynamic_weights[name] * interval['lower']
            upper_bounds += self.dynamic_weights[name] * interval['upper']
        
        self.prediction_intervals = {'lower': lower_bounds, 'upper': upper_bounds}
        return lower_bounds, upper_bounds
    
    def evaluate_models(self, y_true: pd.Series) -> Dict[str, Dict]:
        """Evaluate all component models and hybrid model"""
        metrics = {}
        
        # Calculate metrics for individual models
        models = {
            'sarima': self.sarima,
            'xgboost': self.xgboost,
            'random_forest': self.random_forest
        }
        
        for name, model in models.items():
            metrics[name] = model.calculate_metrics(y_true, model.predictions)
        
        # Calculate metrics for hybrid model
        metrics['hybrid'] = self.calculate_metrics(y_true, self.predictions)
        
        return metrics