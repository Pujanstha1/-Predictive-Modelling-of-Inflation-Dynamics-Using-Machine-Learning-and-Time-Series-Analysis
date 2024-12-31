import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from src.data.data_loader import DataLoader
from src.utils.config import Config
from src.models.sarima_model import SARIMAModel
from src.models.ml_models import XGBoostModel, RandomForestModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from src.models.hybrid_model import HybridModel
import xgboost as xgb
from typing import List, Dict, Tuple
import traceback

class ForecastGenerator:
    def __init__(self, data: pd.DataFrame, config: Config):
        """Initialize ForecastGenerator with data and config"""
        self.data = self.preprocess_input_data(data)
        self.config = config
        self.models = {}
        self.forecasts = {}
        self.prediction_intervals = {}
    
    def validate_data(self, data: pd.DataFrame, required_features: List[str]) -> bool:
        """Validate data and features"""
        try:
            # Check for NaN values
            if data.isna().any().any():
                print("Warning: Dataset contains NaN values")
                print("NaN counts by column:")
                print(data.isna().sum())
            
            # Check for required features
            missing_features = [feat for feat in required_features if feat not in data.columns]
            if missing_features:
                print(f"Warning: Missing required features: {missing_features}")
                return False
            
            # Check data types
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                print("Warning: No numeric columns found in dataset")
                return False
            
            # Check for infinite values
            inf_cols = data.columns[np.isinf(data).any()]
            if len(inf_cols) > 0:
                print(f"Warning: Infinite values found in columns: {inf_cols}")
            
            # Check data range
            for col in numeric_cols:
                col_range = data[col].max() - data[col].min()
                if col_range == 0:
                    print(f"Warning: Zero variance in column: {col}")
            
            return True
            
        except Exception as e:
            print(f"Error in data validation: {str(e)}")
            return False
    
    def preprocess_input_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data with updated cleaning methods"""
        processed_data = data.copy()
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Replace infinite values with NaN
            processed_data[col] = processed_data[col].replace([np.inf, -np.inf], np.nan)
            # Forward fill then backward fill
            processed_data[col] = processed_data[col].ffill().bfill()
        
        return processed_data
    
    def prepare_forecast_data(self, horizon):
        """Prepare data for forecasting with improved performance and error handling"""
        # Get the last date in the dataset
        last_date = self.data.index.max()
        
        # Create future dates using 'ME' (month end) frequency
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq='ME'  # Changed from 'M' to 'ME'
        )
        
        # Initialize all features at once
        feature_dict = {
            'month': future_dates.month,
            'quarter': future_dates.quarter,
            'month_sin': np.sin(2 * np.pi * future_dates.month/12),
            'month_cos': np.cos(2 * np.pi * future_dates.month/12),
            'quarter_sin': np.sin(2 * np.pi * future_dates.quarter/4),
            'quarter_cos': np.cos(2 * np.pi * future_dates.quarter/4)
        }
        
        # Calculate trends and projections for all numeric columns at once
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        last_values = self.data[numeric_cols].iloc[-1]
        trends = (self.data[numeric_cols].iloc[-1] - self.data[numeric_cols].iloc[-12]) / 12
        
        for col in numeric_cols:
            if col not in feature_dict:
                base_values = np.array([last_values[col] + (i+1) * trends[col] for i in range(horizon)])
                feature_dict[col] = base_values
                
                # Calculate MoM changes directly if needed
                if '_mom' in col:
                    # Safe percentage calculation with error handling
                    pct_changes = np.zeros(len(base_values))
                    for i in range(1, len(base_values)):
                        if base_values[i-1] != 0:
                            pct_changes[i] = ((base_values[i] - base_values[i-1]) / abs(base_values[i-1])) * 100
                        else:
                            pct_changes[i] = 0  # Default value for undefined percent change
                    
                    # Handle first value
                    if last_values[col] != 0:
                        pct_changes[0] = ((base_values[0] - last_values[col]) / abs(last_values[col])) * 100
                    else:
                        pct_changes[0] = 0
                    
                    feature_dict[col] = pct_changes
        
        # Create DataFrame at once
        forecast_data = pd.DataFrame(feature_dict, index=future_dates).copy()
        
        # Handle any remaining NaN values
        for col in forecast_data.columns:
            if forecast_data[col].isna().any():
                forecast_data[col] = forecast_data[col].ffill().bfill()
        
        return forecast_data
    
    def prepare_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models with trend awareness"""
        features = data.copy()
        
        # Add trend indicators
        features['trend_3m'] = features['overall_index_mom'].rolling(window=3).mean()
        features['trend_6m'] = features['overall_index_mom'].rolling(window=6).mean()
        features['trend_12m'] = features['overall_index_mom'].rolling(window=12).mean()
        
        # Add momentum indicators
        features['momentum_3m'] = features['overall_index_mom'] - features['trend_3m']
        features['momentum_6m'] = features['overall_index_mom'] - features['trend_6m']
        
        # Add volatility indicators
        features['volatility_3m'] = features['overall_index_mom'].rolling(window=3).std()
        features['volatility_6m'] = features['overall_index_mom'].rolling(window=6).std()
        
        # Add economic indicators momentum
        for col in ['broad_money_m2_mom', 'policy_rate', 'usd']:
            if col in features.columns:
                features[f'{col}_momentum'] = features[col].diff()
                features[f'{col}_trend'] = features[col].rolling(window=3).mean()
        
        # Fill NaN values created by rolling windows
        features = features.ffill().bfill()
        
        return features
    
    def generate_sarima_forecast(self, horizon):
        """Generate SARIMA forecast"""
        try:
            # Get seasonal features
            seasonal_features = ['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']
            X = self.data[seasonal_features].copy()
            y = self.data['overall_index_mom'].copy()

            # Train SARIMA model
            print("Training SARIMA model...")
            sarima_model = SARIMAModel(self.config)
            sarima_model.train(X, y)

            # Create future features for forecasting
            future_dates = pd.date_range(
                start=self.data.index[-1] + pd.DateOffset(months=1),
                periods=horizon,
                freq='ME'
            )
            
            # Generate seasonal features for future dates
            future_features = pd.DataFrame(index=future_dates)
            future_features['month'] = future_dates.month
            future_features['month_sin'] = np.sin(2 * np.pi * future_features['month']/12)
            future_features['month_cos'] = np.cos(2 * np.pi * future_features['month']/12)
            future_features['quarter'] = future_dates.quarter
            future_features['quarter_sin'] = np.sin(2 * np.pi * future_features['quarter']/4)
            future_features['quarter_cos'] = np.cos(2 * np.pi * future_features['quarter']/4)

            # Select only required features
            X_future = future_features[seasonal_features]

            # Generate predictions
            predictions = sarima_model.predict(X_future)
            lower_bound, upper_bound = sarima_model.calculate_prediction_intervals(X_future)

            # Convert to numpy arrays
            predictions = np.array(predictions)
            lower_bound = np.array(lower_bound)
            upper_bound = np.array(upper_bound)

            # Store results
            self.models['SARIMA'] = sarima_model
            self.forecasts['SARIMA'] = predictions
            self.prediction_intervals['SARIMA'] = {
                'lower': lower_bound,
                'upper': upper_bound
            }

            print(f"SARIMA predictions shape: {predictions.shape}")
            print(f"Sample predictions: {predictions[:5]}")

            return predictions, lower_bound, upper_bound
            
        except Exception as e:
            st.error(f"Error in SARIMA forecasting: {str(e)}")
            print(f"Detailed error in SARIMA forecasting:", traceback.format_exc())
            return None, None, None
    
    def generate_ml_forecast(self, model_type, horizon):
        """Generate forecast with economic constraints and ensemble approach"""
        try:
            # Initialize models
            if model_type == 'XGBoost':
                model = XGBoostModel(self.config)
                model_history = XGBoostModel(self.config)  # For historical patterns
                model_trend = XGBoostModel(self.config)    # For trend
            else:
                model = RandomForestModel(self.config)
                model_history = RandomForestModel(self.config)
                model_trend = RandomForestModel(self.config)
            
            data = self.data.copy()
            
            # 1. Historical Pattern Features
            def create_historical_features(df):
                features = pd.DataFrame(index=df.index)
                
                # Seasonal patterns
                features['month'] = df.index.month
                features['quarter'] = df.index.quarter
                features['month_sin'] = np.sin(2 * np.pi * features['month']/12)
                features['month_cos'] = np.cos(2 * np.pi * features['month']/12)
                
                # Historical patterns (last 5 years)
                for m in range(1, 13):
                    month_data = df[df.index.month == m]['overall_index_mom']
                    features.loc[features['month'] == m, 'hist_mean'] = month_data.mean()
                    features.loc[features['month'] == m, 'hist_std'] = month_data.std()
                
                return features
            
            # 2. Trend Features
            def create_trend_features(df):
                features = pd.DataFrame(index=df.index)
                
                # Economic indicators
                key_indicators = ['broad_money_m2_mom', 'policy_rate', 'food_and_beverage_mom']
                for col in key_indicators:
                    if col in df.columns:
                        features[col] = df[col]
                        features[f'{col}_ma3'] = df[col].rolling(3).mean()
                
                # Add target lags
                for i in [1, 3, 6, 12]:
                    features[f'target_lag_{i}'] = df['overall_index_mom'].shift(i)
                
                return features
            
            # 3. Prepare Training Data
            X_hist = create_historical_features(data)
            X_trend = create_trend_features(data)
            y = data['overall_index_mom']
            
            # Remove NaN rows
            valid_idx = y.notna() & X_hist.notna().all(1) & X_trend.notna().all(1)
            X_hist = X_hist[valid_idx]
            X_trend = X_trend[valid_idx]
            y = y[valid_idx]
            
            # Train-test split
            train_size = int(len(y) * 0.8)
            train_idx = y.index[:train_size]
            test_idx = y.index[train_size:]
            
            # 4. Train Models
            print("\nTraining models...")
            
            # Historical pattern model
            model_history.train(X_hist.loc[train_idx], y.loc[train_idx])
            hist_pred_test = model_history.predict(X_hist.loc[test_idx])
            
            # Trend model
            model_trend.train(X_trend.loc[train_idx], y.loc[train_idx])
            trend_pred_test = model_trend.predict(X_trend.loc[test_idx])
            
            # Combined features
            X_combined = pd.concat([X_hist, X_trend], axis=1)
            model.train(X_combined.loc[train_idx], y.loc[train_idx])
            combined_pred_test = model.predict(X_combined.loc[test_idx])
            
            # 5. Validate Models
            val_metrics = model.calculate_metrics(y.loc[test_idx], combined_pred_test)
            print("\nValidation Metrics:")
            for metric, value in val_metrics.items():
                print(f"{metric}: {value:.4f}")
            
            # 6. Generate Forecasts
            future_dates = pd.date_range(
                start=data.index[-1] + pd.DateOffset(months=1),
                periods=horizon,
                freq='ME'
            )
            
            # Initialize forecast data
            forecast_hist = pd.DataFrame(index=future_dates)
            forecast_trend = pd.DataFrame(index=future_dates)
            
            # Create historical features
            forecast_hist['month'] = forecast_hist.index.month
            forecast_hist['quarter'] = forecast_hist.index.quarter
            forecast_hist['month_sin'] = np.sin(2 * np.pi * forecast_hist['month']/12)
            forecast_hist['month_cos'] = np.cos(2 * np.pi * forecast_hist['month']/12)
            
            # Add historical patterns
            for m in range(1, 13):
                month_data = data[data.index.month == m]['overall_index_mom']
                forecast_hist.loc[forecast_hist['month'] == m, 'hist_mean'] = month_data.mean()
                forecast_hist.loc[forecast_hist['month'] == m, 'hist_std'] = month_data.std()
            
            # Initialize predictions
            hist_pred = model_history.predict(forecast_hist)
            
            # Generate trend predictions iteratively
            trend_predictions = []
            current_features = X_trend.iloc[-1:].copy()
            
            for i in range(horizon):
                # Update trend features
                if i > 0:
                    current_features.index = [future_dates[i]]
                    # Update lags with previous predictions
                    for lag in [1, 3, 6, 12]:
                        if i >= lag:
                            current_features[f'target_lag_{lag}'] = trend_predictions[i-lag]
                        else:
                            current_features[f'target_lag_{lag}'] = data['overall_index_mom'].iloc[-(lag-i)]
                
                # Predict
                pred = model_trend.predict(current_features)[0]
                trend_predictions.append(pred)
            
            trend_pred = np.array(trend_predictions)
            
            # 7. Combine Predictions with Economic Constraints
            # Weight historical and trend components
            weights = np.array([0.4, 0.6])  # Historical vs Trend
            predictions = weights[0] * hist_pred + weights[1] * trend_pred
            
            # Apply economic constraints
            max_monthly_change = 2.0  # Maximum 2% change per month
            min_monthly_change = -1.0  # Minimum -1% change per month
            
            for i in range(1, len(predictions)):
                max_change = max_monthly_change
                min_change = min_monthly_change
                
                # Adjust based on recent history
                if i > 1:
                    recent_volatility = np.std(predictions[max(0, i-3):i])
                    max_change = min(max_monthly_change, 2 * recent_volatility)
                    min_change = max(min_monthly_change, -2 * recent_volatility)
                
                # Constrain the change
                current_change = predictions[i] - predictions[i-1]
                if current_change > max_change:
                    predictions[i] = predictions[i-1] + max_change
                elif current_change < min_change:
                    predictions[i] = predictions[i-1] + min_change
            
            # 8. Calculate Prediction Intervals
            std_dev = np.std(y.loc[test_idx] - combined_pred_test)
            lower_bound = predictions - 1.96 * std_dev
            upper_bound = predictions + 1.96 * std_dev
            
            # Store results
            self.models[model_type] = model
            self.forecasts[model_type] = predictions
            self.prediction_intervals[model_type] = {
                'lower': lower_bound,
                'upper': upper_bound
            }
            
            return predictions, lower_bound, upper_bound
            
        except Exception as e:
            st.error(f"Error in {model_type} forecasting: {str(e)}")
            print(f"Detailed error in {model_type}:", traceback.format_exc())
            return None, None, None
        
    def generate_hybrid_forecast(self, horizon):
        """Generate forecast using hybrid approach"""
        try:
            # Generate predictions using individual models first
            sarima_pred, sarima_lower, sarima_upper = self.generate_sarima_forecast(horizon)
            xgb_pred, xgb_lower, xgb_upper = self.generate_ml_forecast('XGBoost', horizon)
            rf_pred, rf_lower, rf_upper = self.generate_ml_forecast('RandomForest', horizon)
            
            # Use weighted average for final predictions
            weights = {
                'sarima': 0.4,
                'xgboost': 0.3,
                'random_forest': 0.3
            }
            
            predictions = (
                weights['sarima'] * sarima_pred +
                weights['xgboost'] * xgb_pred +
                weights['random_forest'] * rf_pred
            )
            
            # Calculate prediction intervals
            lower_bound = (
                weights['sarima'] * sarima_lower +
                weights['xgboost'] * xgb_lower +
                weights['random_forest'] * rf_lower
            )
            
            upper_bound = (
                weights['sarima'] * sarima_upper +
                weights['xgboost'] * xgb_upper +
                weights['random_forest'] * rf_upper
            )
            
            # Store results
            self.forecasts['Hybrid'] = predictions
            self.prediction_intervals['Hybrid'] = {
                'lower': lower_bound,
                'upper': upper_bound
            }
            
            # Store composite model results
            self.models['Hybrid'] = {
                'SARIMA': self.models['SARIMA'],
                'XGBoost': self.models['XGBoost'],
                'RandomForest': self.models['RandomForest'],
                'predictions': predictions,
                'weights': weights
            }
            
            return predictions, lower_bound, upper_bound
            
        except Exception as e:
            st.error(f"Error in hybrid forecasting: {str(e)}")
            print(f"Detailed error in hybrid forecasting:", traceback.format_exc())
            return None, None, None
        
def plot_forecasts(historical_data, forecast_dates, forecasts, intervals):
    """Plot forecasting results"""
    fig = go.Figure()
    
    # Plot historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data['overall_index_mom'],
            name='Historical',
            line=dict(color='blue')
        )
    )
    
    # Plot forecasts and intervals for each model
    colors = {'SARIMA': 'red', 'XGBoost': 'green', 'RandomForest': 'purple', 'Hybrid': 'orange'}
    
    for model_name, forecast in forecasts.items():
        if forecast is not None:
            # Add forecast line
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=forecast,
                    name=f'{model_name} Forecast',
                    line=dict(color=colors.get(model_name, 'gray'))
                )
            )
            
            # Add prediction intervals
            if model_name in intervals:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=intervals[model_name]['upper'],
                        fill=None,
                        mode='lines',
                        line=dict(color=colors.get(model_name, 'gray'), width=0),
                        showlegend=False
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=intervals[model_name]['lower'],
                        fill='tonexty',
                        mode='lines',
                        line=dict(color=colors.get(model_name, 'gray'), width=0),
                        name=f'{model_name} Interval'
                    )
                )
    
    fig.update_layout(
        height=600,
        title="Inflation Forecast",
        xaxis_title="Date",
        yaxis_title="Month-over-Month Inflation (%)",
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def show_forecasting():
    """Main forecasting page"""
    st.title("Inflation Forecasting")
    
    try:
        # Load data
        config = Config()
        loader = DataLoader(config)
        data = loader.load_processed_data()
        
        # Sidebar controls
        st.sidebar.subheader("Forecasting Parameters")
        
        horizon = st.sidebar.slider(
            "Forecast Horizon (months)",
            min_value=1,
            max_value=24,
            value=12
        )
        
        models_to_use = st.sidebar.multiselect(
            "Select Models for Forecasting",
            ['SARIMA', 'XGBoost', 'RandomForest', 'Hybrid'],
            default=['SARIMA', 'XGBoost', 'RandomForest', 'Hybrid']
        )
        
        def calculate_metrics(y_true, y_pred):
            """Calculate performance metrics for any model"""
            if isinstance(y_pred, np.ndarray):
                y_pred = pd.Series(y_pred, index=y_true.index[-len(y_pred):])
            
            y_true = y_true[-len(y_pred):]
            
            return {
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
                'directional_accuracy': np.mean(np.sign(y_true.values) == np.sign(y_pred.values)) * 100
            }
        
        # Generate forecasts button
        if st.sidebar.button("Generate Forecasts"):
            with st.spinner("Generating forecasts..."):
                # Initialize forecast generator
                forecast_generator = ForecastGenerator(data, config)
                
                # Store forecasts and intervals
                forecasts = {}
                intervals = {}
                
                # Generate forecasts for selected models
                for model in models_to_use:
                    if model == 'SARIMA':
                        pred, lower, upper = forecast_generator.generate_sarima_forecast(horizon)
                    elif model == 'XGBoost':
                        pred, lower, upper = forecast_generator.generate_ml_forecast('XGBoost', horizon)
                    elif model == 'RandomForest':
                        pred, lower, upper = forecast_generator.generate_ml_forecast('RandomForest', horizon)
                    elif model == 'Hybrid':
                        pred, lower, upper = forecast_generator.generate_hybrid_forecast(horizon)
                    
                    if pred is not None:
                        forecasts[model] = pred
                        intervals[model] = {'lower': lower, 'upper': upper}
                
                # Plot results
                if forecasts:
                    # Future dates for visualization
                    forecast_dates = pd.date_range(
                        start=data.index[-1] + pd.DateOffset(months=1),
                        periods=horizon,
                        freq='ME'
                    )
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Plot historical data first
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['overall_index_mom'],
                            name='Historical',
                            line=dict(color='blue'),
                            mode='lines'  # Only show lines, no markers
                        )
                    )
                    
                    # Plot forecasts for each model
                    for model_name, forecast in forecasts.items():
                        # Define color based on model
                        if model_name == 'SARIMA':
                            line_color = 'red'
                            fill_color = 'rgba(255,0,0,0.2)'
                        elif model_name == 'XGBoost':
                            line_color = 'green'
                            fill_color = 'rgba(0,255,0,0.2)'
                        elif model_name == 'RandomForest':
                            line_color = 'purple'
                            fill_color = 'rgba(128,0,128,0.2)'
                        elif model_name == 'Hybrid':
                            line_color = 'orange'
                            fill_color = 'rgba(255,165,0,0.2)'
                        else:
                            line_color = 'gray'
                            fill_color = 'rgba(128,128,128,0.2)'
                        
                        # Add forecast line
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_dates,
                                y=forecast,
                                name=f'{model_name} Forecast',
                                line=dict(color=line_color),
                                mode='lines'  # Only show lines, no markers
                            )
                        )
                        
                        # Add prediction intervals
                        if model_name in intervals:
                            fig.add_trace(
                                go.Scatter(
                                    x=forecast_dates,
                                    y=intervals[model_name]['upper'],
                                    name=f'{model_name} Upper Bound',
                                    line=dict(width=0),
                                    showlegend=False,
                                    mode='lines'  # Only show lines, no markers
                                )
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=forecast_dates,
                                    y=intervals[model_name]['lower'],
                                    name=f'{model_name} Interval',
                                    fill='tonexty',
                                    fillcolor=fill_color,
                                    line=dict(width=0),
                                    showlegend=False,
                                    mode='lines'  # Only show lines, no markers
                                )
                            )
                    
                    # Update layout
                    fig.update_layout(
                        title="Inflation Forecast",
                        xaxis_title="Date",
                        yaxis_title="Month-over-Month Inflation (%)",
                        height=600,
                        hovermode='x unified',
                        template='plotly_white'
                    )
                    
                    # Show plot
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display forecast values
                    st.subheader("Forecast Values")
                    forecast_df = pd.DataFrame(
                        {model: values for model, values in forecasts.items()},
                        index=forecast_dates
                    )
                    st.dataframe(forecast_df.round(3))
                    
                    # Display metrics
                    st.subheader("Model Performance Comparison")
                    metrics_df = pd.DataFrame()
                    
                    recent_actual = data['overall_index_mom'].iloc[-horizon:]
                    for model_name, pred in forecasts.items():
                        metrics = calculate_metrics(recent_actual, pred)
                        metrics_df[model_name] = pd.Series(metrics)
                    
                    # Display metrics with highlighting
                    st.dataframe(
                        metrics_df.style.highlight_min(axis=1)
                                    .format("{:.3f}")
                    )
                    
                    # Add download button for forecasts
                    csv = forecast_df.to_csv()
                    st.download_button(
                        label="Download Forecasts",
                        data=csv,
                        file_name="inflation_forecasts.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.warning("No forecasts were generated. Please check the error messages above.")
        
        else:
            st.info(
                "Select forecasting parameters in the sidebar and click 'Generate Forecasts' "
                "to start the forecasting process."
            )
            
            # Display current data summary
            st.subheader("Current Data Summary")
            recent_data = data['overall_index_mom'].tail(6)
            st.write("Recent inflation rates:")
            st.line_chart(recent_data)
            
            # Display summary statistics
            st.write("\nSummary Statistics:")
            st.dataframe(data['overall_index_mom'].describe().round(3))
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Stack trace: " + traceback.format_exc())
        st.info("Please check your data and configuration settings.")