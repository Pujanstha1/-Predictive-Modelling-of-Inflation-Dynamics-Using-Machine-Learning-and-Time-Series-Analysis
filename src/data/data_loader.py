import pandas as pd
from typing import Tuple, Optional
from ..utils.config import Config

class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.data = None
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from CSV file"""
        print("Loading raw data...")
        data = pd.read_csv(self.config.get_data_path('raw'))
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        self.data = data
        return data
    
    def load_processed_data(self) -> pd.DataFrame:
        """Load processed data from CSV file"""
        print("Loading processed data...")
        data = pd.read_csv(self.config.get_data_path('processed'))
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        self.data = data
        return data
    
    def split_data(self, data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets"""
        if data is None:
            data = self.data
        
        train = data[data.index < self.config.test_split_date]
        test = data[data.index >= self.config.test_split_date]
        
        return train, test
    
    def get_features_and_target(self, 
                              data: pd.DataFrame,
                              feature_list: list,
                              target: str = 'overall_index_mom') -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and target from dataset"""
        X = data[feature_list]
        y = data[target]
        return X, y