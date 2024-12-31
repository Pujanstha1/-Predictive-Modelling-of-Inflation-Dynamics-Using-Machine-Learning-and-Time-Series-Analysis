import pandas as pd
import numpy as np
from src.utils.config import Config
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.sarima_model import SARIMAModel
from src.models.ml_models import XGBoostModel, RandomForestModel
from src.models.hybrid_model import HybridModel
import warnings
warnings.filterwarnings('ignore')

class InflationForecastPipeline:
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.models = {
            'sarima': SARIMAModel(self.config),
            'xgboost': XGBoostModel(self.config),
            'random_forest': RandomForestModel(self.config),
            'hybrid': HybridModel(self.config)
        }
        self.results = {}
        
    def run_pipeline(self):
        """Execute the complete modeling pipeline"""
        try:
            # 1. Load and preprocess data
            print("Loading and preparing data...")
            df = self.data_loader.load_raw_data()
            processed_df = self.feature_engineer.process_features(df)
            
            # Save processed dataset
            processed_path = self.config.data_dir / 'processed' / 'modified_dataset.csv'
            processed_df.to_csv(processed_path)
            
            self.data_loader.data = processed_df
            
            # 2. Split data
            train_df, test_df = self.data_loader.split_data()
            print(f"Training data shape: {train_df.shape}")
            print(f"Test data shape: {test_df.shape}\n")
            
            # 3. Prepare features for each model type
            sarima_features = [col for col in train_df.columns 
                             if col in self.config.price_features + ['month_sin', 'month_cos']]
            
            X_train_sarima = train_df[sarima_features]
            X_test_sarima = test_df[sarima_features]
            
            # ML features
            ml_features = (self.config.price_features + 
                         self.config.monetary_features + 
                         self.config.external_features)
            X_train_ml = train_df[ml_features]
            X_test_ml = test_df[ml_features]
            
            # Target variable
            y_train = train_df['overall_index_mom']
            y_test = test_df['overall_index_mom']
            
            # 4. Scale features
            X_train_scaled, X_test_scaled = self.feature_engineer.scale_features(
                X_train_ml, X_test_ml
            )
            
            # 5. Train and evaluate models
            print("Training models...")
            all_metrics = {}
            
            # Train SARIMA
            print("Training SARIMA model...")
            self.models['sarima'].train(X_train_sarima, y_train)
            sarima_preds = self.models['sarima'].predict(X_test_sarima)
            all_metrics['sarima'] = self.models['sarima'].calculate_metrics(y_test, sarima_preds)
            
            # Train XGBoost
            print("\nTraining XGBoost...")
            self.models['xgboost'].train(X_train_scaled, y_train)
            xgb_preds = self.models['xgboost'].predict(X_test_scaled)
            all_metrics['xgboost'] = self.models['xgboost'].calculate_metrics(y_test, xgb_preds)
            
            # Train Random Forest
            print("\nTraining Random Forest...")
            self.models['random_forest'].train(X_train_scaled, y_train)
            rf_preds = self.models['random_forest'].predict(X_test_scaled)
            all_metrics['random_forest'] = self.models['random_forest'].calculate_metrics(y_test, rf_preds)
            
            # Train Hybrid Model
            print("\nGenerating predictions...")
            print("Calculating dynamic weights...")
            print("Calculating prediction intervals...")
            
            self.models['hybrid'].train(X_train_scaled, y_train)
            hybrid_preds = self.models['hybrid'].predict(X_test_scaled)
            all_metrics['hybrid'] = self.models['hybrid'].calculate_metrics(y_test, hybrid_preds)
            
            # Print performance summary
            print("\nRobust Hybrid Model Performance Summary:")
            print("=====================================")
            
            print("\nPerformance Metrics by Model:")
            print("===========================")
            metrics_df = pd.DataFrame(all_metrics)
            print(metrics_df.round(4))
            
            # Print dynamic weights
            print("\nDynamic Model Weights:")
            print("====================")
            weights = self.models['hybrid'].dynamic_weights
            for model, weight in weights.items():
                print(f"{model.upper()}: {weight:.4f}")
            
            # Print feature importance
            print("\nTop 5 Important Features by Model:")
            print("==============================")
            for name in ['xgboost', 'random_forest']:
                print(f"{name.upper()}:")
                importance_df = self.models[name].get_feature_importance()
                print(importance_df.head().to_string())
                print()
            
            # Print error statistics
            print("Error Statistics:")
            print("===============")
            for name, metrics in all_metrics.items():
                print(f"{name.upper()}:")
                print(f"RMSE: {metrics['rmse']:.4f}")
                print(f"MAPE: {metrics['mape']:.2f}%")
                print(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%\n")
            
            # Save results
            self.save_results()
            
        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            raise
    
    def save_results(self):
        """Save model results and metrics"""
        # Save metrics for each model
        for name, result in self.results.items():
            metrics_file = self.config.reports_dir / f'{name}_metrics.csv'
            pd.DataFrame(result['metrics'], index=[0]).to_csv(metrics_file)
            
            # Save feature importance if available
            if result['feature_importance'] is not None:
                importance_file = self.config.reports_dir / f'{name}_feature_importance.csv'
                if isinstance(result['feature_importance'], dict):
                    # For hybrid model
                    for model_name, importance in result['feature_importance'].items():
                        importance.to_csv(
                            self.config.reports_dir / f'{name}_{model_name}_feature_importance.csv'
                        )
                else:
                    result['feature_importance'].to_csv(importance_file)

if __name__ == "__main__":
    # Run the pipeline
    pipeline = InflationForecastPipeline()
    pipeline.run_pipeline()