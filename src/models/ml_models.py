import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, Dict, List
from .base_model import BaseModel

# class XGBoostModel(BaseModel):
#     def __init__(self, config):
#         super().__init__(config)
        
#     def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
#         """Train XGBoost model with time series cross-validation"""
#         print("Training XGBoost model...")
        
#         self.model = xgb.XGBRegressor(**self.config.xgboost_params)
        
#         # Time series cross-validation
#         tscv = TimeSeriesSplit(n_splits=5)
#         cv_scores = []
        
#         for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
#             X_fold_train = X_train.iloc[train_idx]
#             y_fold_train = y_train.iloc[train_idx]
#             X_fold_val = X_train.iloc[val_idx]
#             y_fold_val = y_train.iloc[val_idx]
            
#             # Sample weights giving more importance to recent data
#             train_weights = np.linspace(0.5, 1.0, len(X_fold_train))
            
#             self.model.fit(
#                 X_fold_train, y_fold_train,
#                 sample_weight=train_weights,
#                 eval_set=[(X_fold_val, y_fold_val)],
#                 verbose=100
#             )
            
#             fold_pred = self.model.predict(X_fold_val)
#             fold_score = self.calculate_metrics(y_fold_val, fold_pred)['r2']
#             cv_scores.append(fold_score)
#             print(f"Fold {fold+1} R² Score: {fold_score:.3f}")
        
#         print(f"Mean CV R² Score: {np.mean(cv_scores):.3f}")
        
#         # Final training on all data
#         sample_weights = np.linspace(0.5, 1.0, len(X_train))
#         self.model.fit(
#             X_train, y_train,
#             sample_weight=sample_weights,
#             eval_set=[(X_train, y_train)],
#             verbose=100
#         )
        
#         # Store feature importance
#         self.feature_importance = pd.DataFrame({
#             'feature': X_train.columns,
#             'importance': self.model.feature_importances_
#         }).sort_values('importance', ascending=False)
    
#     def predict(self, X_test: pd.DataFrame) -> np.ndarray:
#         """Generate predictions with XGBoost"""
#         self.predictions = self.model.predict(X_test)
#         return self.predictions
    
#     def get_feature_importance(self) -> pd.DataFrame:
#         """Get XGBoost feature importance"""
#         return self.feature_importance
    
#     def calculate_prediction_intervals(self, X_test: pd.DataFrame, 
#                                     confidence: float = 0.95,
#                                     n_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
#         """Calculate prediction intervals using bootstrap"""
#         predictions = []
        
#         for _ in range(n_iterations):
#             # Random sample with replacement
#             idx = np.random.choice(len(X_test), size=len(X_test), replace=True)
#             X_boot = X_test.iloc[idx]
            
#             pred = self.model.predict(X_boot)
#             predictions.append(pred)
        
#         predictions = np.array(predictions)
#         lower = np.percentile(predictions, (1-confidence)*100/2, axis=0)
#         upper = np.percentile(predictions, 100-(1-confidence)*100/2, axis=0)
        
#         self.prediction_intervals = {'lower': lower, 'upper': upper}
#         return lower, upper

class XGBoostModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.feature_names = None
        self.feature_importance = None
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train XGBoost model"""
        print("Training XGBoost model...")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Create sample weights giving more importance to recent data
        sample_weights = np.linspace(0.5, 1.0, len(X_train))
        dtrain.set_weight(sample_weights)
        
        # Train model
        self.model = xgb.train(
            self.config.xgboost_params,
            dtrain,
            num_boost_round=1000
        )
        
        # Calculate feature importance
        importance_scores = self.model.get_score(importance_type='gain')
        self.feature_importance = pd.DataFrame({
            'feature': list(importance_scores.keys()),
            'importance': list(importance_scores.values())
        }).sort_values('importance', ascending=False)
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        dtest = xgb.DMatrix(X_test)
        self.predictions = self.model.predict(dtest)
        return self.predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if self.feature_importance is None:
            return pd.DataFrame(columns=['feature', 'importance'])
        return self.feature_importance
    
    def calculate_prediction_intervals(self, X_test: pd.DataFrame, 
                                    confidence: float = 0.95,
                                    n_iterations: int = 100) -> tuple:
        """Calculate prediction intervals using bootstrapped predictions"""
        predictions = []
        dtest = xgb.DMatrix(X_test)
        
        for _ in range(n_iterations):
            # Add random noise based on historical volatility
            base_pred = self.model.predict(dtest)
            noise = np.random.normal(0, np.std(base_pred) * 0.1, size=len(base_pred))
            predictions.append(base_pred + noise)
        
        predictions = np.array(predictions)
        lower = np.percentile(predictions, (1-confidence)*100/2, axis=0)
        upper = np.percentile(predictions, 100-(1-confidence)*100/2, axis=0)
        
        self.prediction_intervals = {'lower': lower, 'upper': upper}
        return lower, upper
    

class RandomForestModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train Random Forest model"""
        print("Training Random Forest model...")
        
        # Convert to numpy arrays to avoid feature names warning
        X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y = y_train.values if isinstance(y_train, pd.Series) else y_train
        
        self.feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else None
        
        self.model = RandomForestRegressor(**self.config.random_forest_params)
        self.model.fit(X, y)
        
        # Store feature importance
        if self.feature_names is not None:
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        X = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        self.predictions = self.model.predict(X)
        return self.predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        return self.feature_importance
    
    def calculate_prediction_intervals(self, X_test: pd.DataFrame, 
                                    confidence: float = 0.95) -> tuple:
        """Calculate prediction intervals using tree variance"""
        X = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        predictions = []
        
        for estimator in self.model.estimators_:
            pred = estimator.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        lower = np.percentile(predictions, (1-confidence)*100/2, axis=0)
        upper = np.percentile(predictions, 100-(1-confidence)*100/2, axis=0)
        
        self.prediction_intervals = {'lower': lower, 'upper': upper}
        return lower, upper
    
# class RandomForestModel(BaseModel):
#     def __init__(self, config):
#         super().__init__(config)
        
#     def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
#         """Train Random Forest model with time series cross-validation"""
#         print("Training Random Forest model...")
        
#         self.model = RandomForestRegressor(**self.config.random_forest_params)
        
#         # Time series cross-validation
#         tscv = TimeSeriesSplit(n_splits=5)
#         cv_scores = []
        
#         for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
#             X_fold_train = X_train.iloc[train_idx]
#             y_fold_train = y_train.iloc[train_idx]
#             X_fold_val = X_train.iloc[val_idx]
#             y_fold_val = y_train.iloc[val_idx]
            
#             # Sample weights giving more importance to recent data
#             train_weights = np.linspace(0.5, 1.0, len(X_fold_train))
            
#             self.model.fit(X_fold_train, y_fold_train, sample_weight=train_weights)
            
#             fold_pred = self.model.predict(X_fold_val)
#             fold_score = self.calculate_metrics(y_fold_val, fold_pred)['r2']
#             cv_scores.append(fold_score)
#             print(f"Fold {fold+1} R² Score: {fold_score:.3f}")
        
#         print(f"Mean CV R² Score: {np.mean(cv_scores):.3f}")
        
#         # Final training on all data
#         sample_weights = np.linspace(0.5, 1.0, len(X_train))
#         self.model.fit(X_train, y_train, sample_weight=sample_weights)
        
#         # Store feature importance
#         self.feature_importance = pd.DataFrame({
#             'feature': X_train.columns,
#             'importance': self.model.feature_importances_
#         }).sort_values('importance', ascending=False)
    
#     def predict(self, X_test: pd.DataFrame) -> np.ndarray:
#         """Generate predictions with Random Forest"""
#         self.predictions = self.model.predict(X_test)
#         return self.predictions
    
#     def get_feature_importance(self) -> pd.DataFrame:
#         """Get Random Forest feature importance"""
#         return self.feature_importance
    
#     def calculate_prediction_intervals(self, X_test: pd.DataFrame, 
#                                     confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
#         """Calculate prediction intervals using tree variance"""
#         predictions = []
        
#         # Get predictions from all trees
#         for estimator in self.model.estimators_:
#             pred = estimator.predict(X_test)
#             predictions.append(pred)
        
#         predictions = np.array(predictions)
#         lower = np.percentile(predictions, (1-confidence)*100/2, axis=0)
#         upper = np.percentile(predictions, 100-(1-confidence)*100/2, axis=0)
        
#         self.prediction_intervals = {'lower': lower, 'upper': upper}
#         return lower, upper