import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, Dict, List
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.early_stopping_rounds = 50
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train XGBoost model with time series cross-validation"""
        print("Training XGBoost model...")
        
        # Convert parameters for XGBoost
        params = {
            'objective': 'reg:squarederror',
            'max_depth': self.config.xgboost_params.get('max_depth', 4),
            'learning_rate': self.config.xgboost_params.get('learning_rate', 0.01),
            'subsample': self.config.xgboost_params.get('subsample', 0.8),
            'colsample_bytree': self.config.xgboost_params.get('colsample_bytree', 0.8),
            'min_child_weight': self.config.xgboost_params.get('min_child_weight', 3),
            'gamma': self.config.xgboost_params.get('gamma', 0.1),
            'reg_alpha': self.config.xgboost_params.get('reg_alpha', 0.1),
            'reg_lambda': self.config.xgboost_params.get('reg_lambda', 1),
            'random_state': self.config.xgboost_params.get('random_state', 42)
        }
        
        # Initialize model
        self.model = xgb.XGBRegressor(**params)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Sample weights giving more importance to recent data
            train_weights = np.linspace(0.5, 1.0, len(X_fold_train))
            
            # Create DMatrix objects for XGBoost
            dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train, weight=train_weights)
            dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
            
            # Train the model
            model_fold = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtrain, 'train'), (dval, 'eval')],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False
            )
            
            # Make predictions
            fold_pred = model_fold.predict(dval)
            fold_score = self.calculate_metrics(y_fold_val, fold_pred)['r2']
            cv_scores.append(fold_score)
            print(f"Fold {fold+1} R² Score: {fold_score:.3f}")
        
        print(f"Mean CV R² Score: {np.mean(cv_scores):.3f}")
        
        # Final training on all data
        sample_weights = np.linspace(0.5, 1.0, len(X_train))
        dtrain_full = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
        
        # Train final model
        self.model = xgb.train(
            params,
            dtrain_full,
            num_boost_round=1000,
            verbose_eval=False
        )
        
        # Store feature importance
        importance_scores = self.model.get_score(importance_type='gain')
        self.feature_importance = pd.DataFrame(
            [(feat, score) for feat, score in importance_scores.items()],
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate predictions with XGBoost"""
        dtest = xgb.DMatrix(X_test)
        self.predictions = self.model.predict(dtest)
        return self.predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get XGBoost feature importance"""
        return self.feature_importance
    
    def calculate_prediction_intervals(self, X_test: pd.DataFrame, 
                                    confidence: float = 0.95,
                                    n_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate prediction intervals using bootstrap"""
        dtest = xgb.DMatrix(X_test)
        predictions = []
        
        for _ in range(n_iterations):
            # Random sample with replacement
            idx = np.random.choice(len(X_test), size=len(X_test), replace=True)
            X_boot = X_test.iloc[idx]
            dboot = xgb.DMatrix(X_boot)
            
            pred = self.model.predict(dboot)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        lower = np.percentile(predictions, (1-confidence)*100/2, axis=0)
        upper = np.percentile(predictions, 100-(1-confidence)*100/2, axis=0)
        
        self.prediction_intervals = {'lower': lower, 'upper': upper}
        return lower, upper

    
class RandomForestModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train Random Forest model with time series cross-validation"""
        print("Training Random Forest model...")
        
        self.model = RandomForestRegressor(**self.config.random_forest_params)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Sample weights giving more importance to recent data
            train_weights = np.linspace(0.5, 1.0, len(X_fold_train))
            
            self.model.fit(X_fold_train, y_fold_train, sample_weight=train_weights)
            
            fold_pred = self.model.predict(X_fold_val)
            fold_score = self.calculate_metrics(y_fold_val, fold_pred)['r2']
            cv_scores.append(fold_score)
            print(f"Fold {fold+1} R² Score: {fold_score:.3f}")
        
        print(f"Mean CV R² Score: {np.mean(cv_scores):.3f}")
        
        # Final training on all data
        sample_weights = np.linspace(0.5, 1.0, len(X_train))
        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate predictions with Random Forest"""
        self.predictions = self.model.predict(X_test)
        return self.predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get Random Forest feature importance"""
        return self.feature_importance
    
    def calculate_prediction_intervals(self, X_test: pd.DataFrame, 
                                    confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate prediction intervals using tree variance"""
        predictions = []
        
        # Get predictions from all trees
        for estimator in self.model.estimators_:
            pred = estimator.predict(X_test)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        lower = np.percentile(predictions, (1-confidence)*100/2, axis=0)
        upper = np.percentile(predictions, 100-(1-confidence)*100/2, axis=0)
        
        self.prediction_intervals = {'lower': lower, 'upper': upper}
        return lower, upper