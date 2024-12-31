from pathlib import Path
import yaml

class Config:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.base_dir / 'data'
        self.raw_data_dir = self.data_dir / 'raw'
        self.processed_data_dir = self.data_dir / 'processed'
        self.reports_dir = self.base_dir / 'reports'
        self.figures_dir = self.reports_dir / 'figures'
        
        # Create directories if they don't exist
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Data configuration
        self.raw_data_file = 'nepal_economic_data.csv'
        self.processed_data_file = 'modified_dataset.csv'
        self.test_split_date = '2023-06-01'
        
        # Feature groups
        self.price_features = [
            'food_and_beverage_mom', 'non_food_and_services_mom',
            'wholesale_price_index_mom', 'indian_cpi_mom'
        ]
        
        self.monetary_features = [
            'broad_money_m2_mom', 'domestic_credit_mom',
            'private_sector_credit_mom', 'policy_rate'
        ]
        
        self.external_features = [
            'usd', 'inr', 'crude_oil_prices',
            'global_food_price_index'
        ]
        
        # Model parameters
        self.sarima_params = {
            'order': (0, 1, 2),
            'seasonal_order': (0, 1, 0, 12),
            'enforce_stationarity': False,
            'enforce_invertibility': False
        }
        
        self.xgboost_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,                # Increased from 4
            'learning_rate': 0.05,         # Reduced from 0.01
            'n_estimators': 500,           # Reduced from 1000
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42,
            'eval_metric': ['rmse', 'mae'],
            'early_stopping_rounds': 10    # Added early stopping
        }
        
        self.random_forest_params = {
            'n_estimators': 500,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        # Hybrid model weights
        self.initial_weights = {
            'sarima': 0.5,
            'xgboost': 0.25,
            'random_forest': 0.25
        }
        
    def save_model_metrics(self, metrics, model_name):
        """Save model metrics to reports directory"""
        metrics_file = self.reports_dir / f'{model_name}_metrics.yaml'
        with open(metrics_file, 'w') as f:
            yaml.dump(metrics, f)
    
    def get_data_path(self, data_type='raw'):
        """Get path for data file"""
        if data_type == 'raw':
            return self.raw_data_dir / self.raw_data_file
        return self.processed_data_dir / self.processed_data_file