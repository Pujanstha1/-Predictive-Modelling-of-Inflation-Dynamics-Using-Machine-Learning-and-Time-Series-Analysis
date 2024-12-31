import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Tuple, List
from ..utils.config import Config

class FeatureEngineer:
    def __init__(self, config: Config):
        self.config = config
        self.scaler = RobustScaler()
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from date index"""
        df = df.copy()
        
        # Cyclical encoding of month and quarter
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter']/4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter']/4)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """Create lag features for specified columns"""
        df = df.copy()
        
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str], 
                              windows: List[int]) -> pd.DataFrame:
        """Create rolling mean and std features"""
        df = df.copy()
        
        for col in columns:
            for window in windows:
                # Rolling mean
                df[f'{col}_ma{window}'] = df[col].rolling(window=window).mean()
                # Rolling std (volatility)
                df[f'{col}_vol{window}'] = df[col].rolling(window=window).std()
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables"""
        df = df.copy()
        
        # Price interactions
        df['food_non_food_interaction'] = (
            df['food_and_beverage_mom'] * df['non_food_and_services_mom']
        )
        
        # Monetary policy interactions
        df['money_credit_interaction'] = (
            df['broad_money_m2_mom'] * df['domestic_credit_mom']
        )
        
        df['price_policy_interaction'] = (
            df['overall_index_mom'] * df['policy_rate']
        )
        
        return df
    
    def create_technical_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Create technical indicators"""
        df = df.copy()
        
        for col in columns:
            # Momentum
            df[f'{col}_momentum'] = df[col].diff()
            # Rate of change
            df[f'{col}_roc'] = df[col].pct_change()
            # Acceleration
            df[f'{col}_acceleration'] = df[f'{col}_momentum'].diff()
        
        return df
    
    def scale_features(self, train_df: pd.DataFrame, 
                      test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale features using RobustScaler"""
        # Fit scaler on training data
        train_scaled = pd.DataFrame(
            self.scaler.fit_transform(train_df),
            columns=train_df.columns,
            index=train_df.index
        )
        
        # Transform test data
        test_scaled = pd.DataFrame(
            self.scaler.transform(test_df),
            columns=test_df.columns,
            index=test_df.index
        )
        
        return train_scaled, test_scaled
    
    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        print("Engineering features...")
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Create lag features for key indicators
        lag_columns = (
            self.config.price_features + 
            self.config.monetary_features
        )
        df = self.create_lag_features(df, lag_columns, [1, 2, 3, 6, 12])
        
        # Create rolling features
        roll_columns = ['overall_index_mom'] + self.config.price_features
        df = self.create_rolling_features(df, roll_columns, [3, 6])
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Create technical features for price indicators
        df = self.create_technical_features(df, self.config.price_features)
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df