# app/pages/model_performance.py
# Standard libraries
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import traceback

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Metrics and evaluation
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    mean_absolute_percentage_error
)

# Project modules
from src.data.data_loader import DataLoader
from src.utils.config import Config
from src.models.sarima_model import SARIMAModel
from src.models.ml_models import XGBoostModel, RandomForestModel
from src.models.hybrid_model import HybridModel

# Forecasting module
from .forecasting import ForecastGenerator

# File handling and system
import os
from pathlib import Path
import sys

# Add project root to system path if needed
root_path = Path(__file__).parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

def show_model_performance():
    """Display detailed model performance analysis"""
    # [Rest of the code remains the same]

def show_model_performance():
    """Display detailed model performance analysis"""
    st.title("Model Performance Analysis")
    
    try:
        # Load data
        config = Config()
        loader = DataLoader(config)
        data = loader.load_processed_data()
        
        # Sidebar controls
        st.sidebar.subheader("Analysis Parameters")
        
        evaluation_period = st.sidebar.slider(
            "Evaluation Period (months)",
            min_value=6,
            max_value=24,
            value=12
        )
        
        models_to_evaluate = st.sidebar.multiselect(
            "Select Models to Evaluate",
            ['SARIMA', 'XGBoost', 'RandomForest', 'Hybrid'],
            default=['SARIMA', 'XGBoost', 'RandomForest', 'Hybrid']
        )
        
        if st.sidebar.button("Run Performance Analysis"):
            with st.spinner("Analyzing model performance..."):
                # Initialize forecast generator
                forecast_generator = ForecastGenerator(data, config)
                
                # Generate predictions for evaluation period
                forecasts = {}
                intervals = {}
                performance_metrics = {}
                
                # Generate predictions and calculate metrics
                for model in models_to_evaluate:
                    if model == 'SARIMA':
                        pred, lower, upper = forecast_generator.generate_sarima_forecast(evaluation_period)
                    elif model == 'XGBoost':
                        pred, lower, upper = forecast_generator.generate_ml_forecast('XGBoost', evaluation_period)
                    elif model == 'RandomForest':
                        pred, lower, upper = forecast_generator.generate_ml_forecast('RandomForest', evaluation_period)
                    elif model == 'Hybrid':
                        pred, lower, upper = forecast_generator.generate_hybrid_forecast(evaluation_period)
                    
                    if pred is not None:
                        forecasts[model] = pred
                        intervals[model] = {'lower': lower, 'upper': upper}
                
                # 1. Overall Performance Metrics
                st.subheader("Overall Performance Metrics")
                metrics_df = pd.DataFrame()
                recent_actual = data['overall_index_mom'].iloc[-evaluation_period:]
                
                for model_name, pred in forecasts.items():
                    metrics = {
                        'RMSE': np.sqrt(mean_squared_error(recent_actual, pred)),
                        'MAE': mean_absolute_error(recent_actual, pred),
                        'R²': r2_score(recent_actual, pred),
                        'MAPE': np.mean(np.abs((recent_actual - pred) / recent_actual)) * 100,
                        'Directional Accuracy': np.mean(np.sign(recent_actual) == np.sign(pred)) * 100
                    }
                    metrics_df[model_name] = pd.Series(metrics)
                
                # Display metrics with conditional formatting
                st.dataframe(
                    metrics_df.style.background_gradient(cmap='RdYlGn', axis=1)
                             .format("{:.3f}")
                )
                
                # 2. Prediction vs Actual Plot
                st.subheader("Prediction vs Actual Values")
                fig = go.Figure()
                
                # Plot actual values
                fig.add_trace(
                    go.Scatter(
                        x=data.index[-evaluation_period:],
                        y=recent_actual,
                        name='Actual',
                        line=dict(color='blue', width=2),
                        mode='lines'
                    )
                )
                
                # Plot predictions for each model
                colors = ['red', 'green', 'purple', 'orange']
                for (model_name, pred), color in zip(forecasts.items(), colors):
                    fig.add_trace(
                        go.Scatter(
                            x=data.index[-evaluation_period:],
                            y=pred,
                            name=f'{model_name} Predicted',
                            line=dict(color=color, width=1),
                            mode='lines'
                        )
                    )
                
                fig.update_layout(
                    title="Model Predictions vs Actual Values",
                    xaxis_title="Date",
                    yaxis_title="Inflation Rate (%)",
                    height=500,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 3. Error Analysis
                st.subheader("Error Analysis")
                error_dfs = []
                
                for model_name, pred in forecasts.items():
                    errors = recent_actual - pred
                    error_df = pd.DataFrame({
                        'Model': model_name,
                        'Error': errors,
                        'Absolute Error': np.abs(errors),
                        'Percentage Error': np.abs(errors / recent_actual) * 100
                    })
                    error_dfs.append(error_df)
                
                error_analysis = pd.concat(error_dfs)
                
                # Error Distribution Plot
                fig = go.Figure()
                for model_name in forecasts.keys():
                    model_errors = error_analysis[error_analysis['Model'] == model_name]['Error']
                    fig.add_trace(
                        go.Box(
                            y=model_errors,
                            name=model_name,
                            boxpoints='outliers'
                        )
                    )
                
                fig.update_layout(
                    title="Error Distribution by Model",
                    yaxis_title="Error",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 4. Interval Coverage Analysis
                st.subheader("Prediction Interval Coverage")
                coverage_metrics = {}
                
                for model_name in forecasts.keys():
                    if model_name in intervals:
                        lower = intervals[model_name]['lower']
                        upper = intervals[model_name]['upper']
                        actual = recent_actual.values
                        
                        coverage = np.mean((actual >= lower) & (actual <= upper)) * 100
                        avg_interval_width = np.mean(upper - lower)
                        
                        coverage_metrics[model_name] = {
                            'Coverage (%)': coverage,
                            'Avg Interval Width': avg_interval_width
                        }
                
                coverage_df = pd.DataFrame(coverage_metrics).T
                st.dataframe(coverage_df.round(3))
                
                # 5. Download Detailed Results
                st.subheader("Download Detailed Results")
                detailed_results = pd.DataFrame({
                    'Date': data.index[-evaluation_period:],
                    'Actual': recent_actual
                })
                
                for model_name, pred in forecasts.items():
                    detailed_results[f'{model_name}_Predicted'] = pred
                    detailed_results[f'{model_name}_Error'] = recent_actual - pred
                    if model_name in intervals:
                        detailed_results[f'{model_name}_Lower'] = intervals[model_name]['lower']
                        detailed_results[f'{model_name}_Upper'] = intervals[model_name]['upper']
                
                csv = detailed_results.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Detailed Results",
                    data=csv,
                    file_name="model_performance_analysis.csv",
                    mime="text/csv"
                )
                
        else:
            st.info(
                "Select analysis parameters in the sidebar and click 'Run Performance Analysis' "
                "to start the evaluation."
            )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Stack trace: " + traceback.format_exc())
        st.info("Please check your data and configuration settings.")