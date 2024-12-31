import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.data.data_loader import DataLoader
from src.utils.config import Config
from src.models.sarima_model import SARIMAModel
from src.models.ml_models import XGBoostModel, RandomForestModel
from src.models.hybrid_model import HybridModel
import traceback

class ModelTrainer:
    def __init__(self, data):
        self.data = data
        self.config = Config()
        self.models = {}
        self.results = {}
    
    def preprocess_data(self, data, selected_features=None):
        """Preprocess data by handling missing values and infinite values"""
        # Make a copy of the data
        processed_data = data.copy()
        
        if selected_features is None:
            # Default features for SARIMA
            selected_features = [
                'month_sin', 'month_cos',
                'quarter_sin', 'quarter_cos',
                'food_and_beverage_mom',
                'non_food_and_services_mom'
            ]
        
        # Select only required features
        features_exist = [feat for feat in selected_features if feat in processed_data.columns]
        processed_data = processed_data[features_exist]
        
        # Replace infinite values with NaN
        processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill NaN values
        processed_data = processed_data.fillna(method='ffill').fillna(method='bfill')
        
        return processed_data
    
    
    def prepare_train_test_data(self, test_size, target_variable):
        """Prepare training and testing datasets"""
        # Calculate split point
        split_index = len(self.data) - test_size
        
        # Split data
        train_data = self.data.iloc[:split_index].copy()
        test_data = self.data.iloc[split_index:].copy()
        
        # Ensure target variable is numeric and handle missing values
        train_data[target_variable] = pd.to_numeric(train_data[target_variable], errors='coerce')
        test_data[target_variable] = pd.to_numeric(test_data[target_variable], errors='coerce')
        
        # Handle missing values in target
        train_data[target_variable] = train_data[target_variable].fillna(method='ffill').fillna(method='bfill')
        test_data[target_variable] = test_data[target_variable].fillna(method='ffill').fillna(method='bfill')
        
        return train_data, test_data
    
    def train_sarima(self, train_data, test_data, params):
        """Train SARIMA model"""
        try:
            # Preprocess data for SARIMA
            sarima_features = ['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']
            train_processed = self.preprocess_data(train_data, sarima_features)
            test_processed = self.preprocess_data(test_data, sarima_features)
            
            # Initialize and train model
            model = SARIMAModel(self.config)
            model.train(train_processed, train_data[params['target']])
            predictions = model.predict(test_processed)
            self.models['SARIMA'] = model
            return predictions
            
        except Exception as e:
            st.error(f"Error in SARIMA training: {str(e)}")
            return None
    
    def train_xgboost(self, train_data, test_data, params):
        """Train XGBoost model"""
        try:
            # Preprocess data for XGBoost
            xgb_features = [col for col in train_data.columns if col != params['target']]
            train_processed = self.preprocess_data(train_data, xgb_features)
            test_processed = self.preprocess_data(test_data, xgb_features)
            
            model = XGBoostModel(self.config)
            model.train(train_processed, train_data[params['target']])
            predictions = model.predict(test_processed)
            self.models['XGBoost'] = model
            return predictions
            
        except Exception as e:
            st.error(f"Error in XGBoost training: {str(e)}")
            return None
    
    def train_random_forest(self, train_data, test_data, params):
        """Train Random Forest model"""
        try:
            # Preprocess data for Random Forest
            rf_features = [col for col in train_data.columns if col != params['target']]
            train_processed = self.preprocess_data(train_data, rf_features)
            test_processed = self.preprocess_data(test_data, rf_features)
            
            model = RandomForestModel(self.config)
            model.train(train_processed, train_data[params['target']])
            predictions = model.predict(test_processed)
            self.models['RandomForest'] = model
            return predictions
            
        except Exception as e:
            st.error(f"Error in Random Forest training: {str(e)}")
            return None
    
    def train_hybrid(self, train_data, test_data, params):
        """Train Hybrid model"""
        try:
            # Preprocess data for Hybrid model
            train_processed = self.preprocess_data(train_data)
            test_processed = self.preprocess_data(test_data)
            
            model = HybridModel(self.config)
            model.train(train_processed, train_data[params['target']])
            predictions = model.predict(test_processed)
            self.models['Hybrid'] = model
            return predictions
            
        except Exception as e:
            st.error(f"Error in Hybrid model training: {str(e)}")
            return None

def plot_feature_importance(model, model_name):
    """Plot feature importance for ML models"""
    if hasattr(model, 'get_feature_importance'):
        importance_df = model.get_feature_importance()
        
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=importance_df['importance'][:10],
                y=importance_df['feature'][:10],
                orientation='h'
            )
        )
        
        fig.update_layout(
            title=f"{model_name} Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Features",
            height=400
        )
        
        return fig
    return None

def plot_training_results(train_data, test_data, predictions_dict, target):
    """Plot training results"""
    fig = go.Figure()
    
    # Plot training data
    fig.add_trace(
        go.Scatter(
            x=train_data.index,
            y=train_data[target],
            name='Training Data',
            line=dict(color='blue')
        )
    )
    
    # Plot test data
    fig.add_trace(
        go.Scatter(
            x=test_data.index,
            y=test_data[target],
            name='Test Data',
            line=dict(color='green')
        )
    )
    
    # Plot predictions for each model
    colors = ['red', 'purple', 'orange', 'brown']
    for (model_name, predictions), color in zip(predictions_dict.items(), colors):
        fig.add_trace(
            go.Scatter(
                x=test_data.index,
                y=predictions,
                name=f'{model_name} Predictions',
                line=dict(color=color)
            )
        )
    
    fig.update_layout(
        title="Model Training Results",
        xaxis_title="Date",
        yaxis_title="Value",
        height=600
    )
    
    return fig

def show_model_training():
    st.title("Model Training")
    
    try:
        # Load data
        config = Config()
        loader = DataLoader(config)
        data = loader.load_processed_data()
        
        # Sidebar settings
        st.sidebar.subheader("Training Parameters")
        
        # Select target variable
        target_variable = st.sidebar.selectbox(
            "Select Target Variable",
            ['overall_index_mom', 'food_and_beverage_mom', 'non_food_and_services_mom']
        )
        
        # Test size selection
        test_size = st.sidebar.slider(
            "Test Set Size (months)",
            min_value=6,
            max_value=24,
            value=12
        )
        
        # Model selection
        models_to_train = st.sidebar.multiselect(
            "Select Models to Train",
            ['SARIMA', 'XGBoost', 'RandomForest', 'Hybrid'],
            default=['SARIMA', 'XGBoost']
        )
        
        # Training button
        if st.sidebar.button("Train Models"):
            with st.spinner("Training models..."):
                trainer = ModelTrainer(data)
                train_data, test_data = trainer.prepare_train_test_data(
                    test_size,
                    target_variable
                )
                
                # Dictionary to store predictions
                predictions_dict = {}
                
                # Train selected models
                params = {'target': target_variable}
                
                if 'SARIMA' in models_to_train:
                    predictions_dict['SARIMA'] = trainer.train_sarima(
                        train_data, test_data, params
                    )
                
                if 'XGBoost' in models_to_train:
                    predictions_dict['XGBoost'] = trainer.train_xgboost(
                        train_data, test_data, params
                    )
                
                if 'RandomForest' in models_to_train:
                    predictions_dict['RandomForest'] = trainer.train_random_forest(
                        train_data, test_data, params
                    )
                
                if 'Hybrid' in models_to_train:
                    predictions_dict['Hybrid'] = trainer.train_hybrid(
                        train_data, test_data, params
                    )
                
                # Plot results
                st.plotly_chart(
                    plot_training_results(
                        train_data, test_data, predictions_dict, target_variable
                    ),
                    use_container_width=True
                )
                
                # Display metrics
                st.subheader("Model Performance Metrics")
                metrics_df = pd.DataFrame()
                
                for model_name, predictions in predictions_dict.items():
                    model = trainer.models[model_name]
                    metrics = model.calculate_metrics(
                        test_data[target_variable],
                        predictions
                    )
                    metrics_df[model_name] = pd.Series(metrics)
                
                st.dataframe(metrics_df.style.highlight_min(axis=1))
                
                # Plot feature importance for ML models
                st.subheader("Feature Importance Analysis")
                for model_name in ['XGBoost', 'RandomForest']:
                    if model_name in trainer.models:
                        fig = plot_feature_importance(
                            trainer.models[model_name],
                            model_name
                        )
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
        
        # Display instructions if no training has been done
        if not st.session_state.get('models_trained', False):
            st.info(
                "Select training parameters in the sidebar and click 'Train Models' "
                "to start the training process."
            )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Stack trace: " + traceback.format_exc())
        st.info("Please check your data and configuration settings.")