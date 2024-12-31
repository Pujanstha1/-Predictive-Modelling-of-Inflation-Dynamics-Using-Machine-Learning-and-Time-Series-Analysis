import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import io
from src.data.data_loader import DataLoader
from src.utils.config import Config
import traceback
from statsmodels.tsa.stattools import adfuller

def plot_correlation_matrix(data, variables):
    """Create correlation matrix heatmap"""
    corr_matrix = data[variables].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=variables,
        y=variables,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Correlation Matrix",
        height=600,
        template='plotly_white'
    )
    
    return fig

def plot_seasonal_patterns(data):
    """Analyze and plot seasonal patterns"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Monthly Pattern', 
            'Quarterly Pattern',
            'Yearly Average Inflation',
            'Monthly Volatility'
        ]
    )
    
    # Monthly Pattern
    monthly_avg = data.groupby(data.index.month)['overall_index_mom'].mean()
    monthly_std = data.groupby(data.index.month)['overall_index_mom'].std()
    
    fig.add_trace(
        go.Bar(
            x=monthly_avg.index,
            y=monthly_avg,
            error_y=dict(type='data', array=monthly_std),
            name='Monthly Average'
        ),
        row=1, col=1
    )
    
    # Quarterly Pattern
    quarterly_avg = data.groupby(data.index.quarter)['overall_index_mom'].mean()
    quarterly_std = data.groupby(data.index.quarter)['overall_index_mom'].std()
    
    fig.add_trace(
        go.Bar(
            x=quarterly_avg.index,
            y=quarterly_avg,
            error_y=dict(type='data', array=quarterly_std),
            name='Quarterly Average'
        ),
        row=1, col=2
    )
    
    # Yearly Pattern
    yearly_avg = data.groupby(data.index.year)['overall_index_mom'].mean()
    
    fig.add_trace(
        go.Scatter(
            x=yearly_avg.index,
            y=yearly_avg,
            mode='lines+markers',
            name='Yearly Average'
        ),
        row=2, col=1
    )
    
    # Monthly Volatility
    monthly_vol = data.groupby(data.index.month)['overall_index_mom'].std()
    
    fig.add_trace(
        go.Bar(
            x=monthly_vol.index,
            y=monthly_vol,
            name='Monthly Volatility'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        template='plotly_white',
        title_text="Seasonal Patterns in Inflation"
    )
    
    return fig

def plot_distribution_analysis(data):
    """Analyze and plot distribution of inflation rates"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Overall Inflation Distribution',
            'Food Inflation Distribution',
            'Core Inflation Distribution',
            'Distribution Comparison'
        ]
    )
    
    # Overall Inflation Distribution
    fig.add_trace(
        go.Histogram(
            x=data['overall_index_mom'],
            name='Overall',
            nbinsx=30,
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Food Inflation Distribution
    fig.add_trace(
        go.Histogram(
            x=data['food_and_beverage_mom'],
            name='Food',
            nbinsx=30,
            opacity=0.7
        ),
        row=1, col=2
    )
    
    # Core Inflation Distribution
    fig.add_trace(
        go.Histogram(
            x=data['non_food_and_services_mom'],
            name='Core',
            nbinsx=30,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Distribution Comparison (Box Plot)
    fig.add_trace(
        go.Box(
            y=data['overall_index_mom'],
            name='Overall',
            boxpoints='outliers'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Box(
            y=data['food_and_beverage_mom'],
            name='Food',
            boxpoints='outliers'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Box(
            y=data['non_food_and_services_mom'],
            name='Core',
            boxpoints='outliers'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        template='plotly_white',
        title_text="Distribution Analysis of Inflation Rates"
    )
    
    return fig

def show_summary_statistics(data):
    """Display summary statistics"""
    st.subheader("Summary Statistics")
    
    # Select relevant columns
    cols = ['overall_index_mom', 'food_and_beverage_mom', 'non_food_and_services_mom']
    
    # Calculate statistics
    stats = data[cols].agg(['mean', 'std', 'min', 'max', 'skew']).round(2)
    stats.columns = ['Overall', 'Food', 'Core']
    stats.index = ['Mean', 'Std Dev', 'Min', 'Max', 'Skewness']
    
    # Display as a styled dataframe
    st.dataframe(stats.style.highlight_max(axis=1, color='lightgreen'))

def show_stationarity_analysis(data):
    """Perform and display stationarity analysis"""
    from statsmodels.tsa.stattools import adfuller
    
    st.subheader("Stationarity Analysis (ADF Test)")
    
    # Debug info
    st.write("Data Types of Columns:")
    st.write(data.dtypes)
    
    cols = ['overall_index_mom', 'food_and_beverage_mom', 'non_food_and_services_mom']
    results = []
    
    for col in cols:
        try:
            # Explicitly convert column to float
            series = data[col].astype(float)
            
            # Debug info
            st.write(f"Processing column: {col}")
            st.write(f"First few values: {series.head()}")
            
            # Perform ADF test
            if len(series.dropna()) > 0:
                adf_test = adfuller(series.dropna())
                results.append({
                    'Variable': col,
                    'ADF Statistic': round(float(adf_test[0]), 4),
                    'p-value': round(float(adf_test[1]), 4),
                    'Is Stationary': 'Yes' if float(adf_test[1]) < 0.05 else 'No'
                })
            else:
                st.warning(f"No valid data for column {col} after cleaning")
        except Exception as e:
            st.warning(f"Error processing {col}: {str(e)}")
            results.append({
                'Variable': col,
                'ADF Statistic': 'Error',
                'p-value': 'Error',
                'Is Stationary': 'Error'
            })
    
    # Create and display results DataFrame
    if results:
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        
        
def show_data_analysis():
    """Main data analysis display function"""
    st.title("Data Analysis")
    
    try:
        # Initialize config and load data
        config = Config()
        loader = DataLoader(config)
        data = loader.load_processed_data()
        
        # Ensure index is datetime
        data.index = pd.to_datetime(data.index)
        
        # Convert relevant columns to float
        columns_to_convert = [
            'overall_index_mom', 
            'food_and_beverage_mom', 
            'non_food_and_services_mom',
            'wholesale_price_index_mom',
            'broad_money_m2_mom',
            'indian_cpi_mom'
        ]
        
        for col in columns_to_convert:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Add date filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=data.index.min()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=data.index.max()
            )
        
        # Filter data based on date range
        mask = (data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))
        filtered_data = data[mask].copy()
        
        # Display data overview
        st.subheader("Data Overview")
        st.write(f"Total observations: {len(filtered_data)}")
        st.write(f"Time period: {filtered_data.index.min()} to {filtered_data.index.max()}")
        
        # Summary Statistics
        st.subheader("Summary Statistics")
        summary_cols = ['overall_index_mom', 'food_and_beverage_mom', 'non_food_and_services_mom']
        summary_stats = filtered_data[summary_cols].describe()
        st.dataframe(summary_stats)
        
        # Correlation Analysis
        st.subheader("Correlation Analysis")
        selected_features = st.multiselect(
            "Select features for correlation analysis",
            options=[col for col in filtered_data.columns if '_mom' in col],
            default=['overall_index_mom', 'food_and_beverage_mom', 'non_food_and_services_mom']
        )
        
        if selected_features:
            corr_matrix = filtered_data[selected_features].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=selected_features,
                y=selected_features,
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            ))
            fig.update_layout(title="Correlation Matrix", height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal Analysis
        st.subheader("Seasonal Analysis")
        seasonal_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Monthly Pattern",
                "Quarterly Pattern",
                "Yearly Trend",
                "Monthly Volatility"
            ]
        )
        
        # Monthly Pattern
        monthly_avg = filtered_data.groupby(filtered_data.index.month)['overall_index_mom'].mean()
        seasonal_fig.add_trace(
            go.Bar(x=monthly_avg.index, y=monthly_avg, name="Monthly Average"),
            row=1, col=1
        )
        
        # Quarterly Pattern
        quarterly_avg = filtered_data.groupby(filtered_data.index.quarter)['overall_index_mom'].mean()
        seasonal_fig.add_trace(
            go.Bar(x=quarterly_avg.index, y=quarterly_avg, name="Quarterly Average"),
            row=1, col=2
        )
        
        # Yearly Trend
        yearly_avg = filtered_data.groupby(filtered_data.index.year)['overall_index_mom'].mean()
        seasonal_fig.add_trace(
            go.Scatter(x=yearly_avg.index, y=yearly_avg, name="Yearly Average"),
            row=2, col=1
        )
        
        # Monthly Volatility
        monthly_std = filtered_data.groupby(filtered_data.index.month)['overall_index_mom'].std()
        seasonal_fig.add_trace(
            go.Bar(x=monthly_std.index, y=monthly_std, name="Monthly Volatility"),
            row=2, col=2
        )
        
        seasonal_fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(seasonal_fig, use_container_width=True)
        
        # Distribution Analysis
        st.subheader("Distribution Analysis")
        dist_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Overall Inflation Distribution",
                "Food Inflation Distribution",
                "Core Inflation Distribution",
                "Box Plot Comparison"
            ]
        )
        
        # Histograms
        dist_fig.add_trace(
            go.Histogram(x=filtered_data['overall_index_mom'], name="Overall"),
            row=1, col=1
        )
        dist_fig.add_trace(
            go.Histogram(x=filtered_data['food_and_beverage_mom'], name="Food"),
            row=1, col=2
        )
        dist_fig.add_trace(
            go.Histogram(x=filtered_data['non_food_and_services_mom'], name="Core"),
            row=2, col=1
        )
        
        # Box Plot
        box_data = [
            filtered_data['overall_index_mom'],
            filtered_data['food_and_beverage_mom'],
            filtered_data['non_food_and_services_mom']
        ]
        dist_fig.add_trace(
            go.Box(y=box_data[0], name="Overall", boxpoints='outliers'),
            row=2, col=2
        )
        dist_fig.add_trace(
            go.Box(y=box_data[1], name="Food", boxpoints='outliers'),
            row=2, col=2
        )
        dist_fig.add_trace(
            go.Box(y=box_data[2], name="Core", boxpoints='outliers'),
            row=2, col=2
        )
        
        dist_fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(dist_fig, use_container_width=True)
        
        # Stationarity Analysis
        st.subheader("Stationarity Analysis (ADF Test)")
        stationarity_results = []
        
        for col in ['overall_index_mom', 'food_and_beverage_mom', 'non_food_and_services_mom']:
            series = filtered_data[col].dropna()
            adf_result = adfuller(series)
            stationarity_results.append({
                'Variable': col,
                'ADF Statistic': adf_result[0],
                'p-value': adf_result[1],
                'Critical Values': adf_result[4]
            })
        
        stationarity_df = pd.DataFrame(stationarity_results)
        st.dataframe(stationarity_df)
        
        # Display critical values
        for result in stationarity_results:
            st.write(f"\nCritical Values for {result['Variable']}:")
            for key, value in result['Critical Values'].items():
                st.write(f"{key}: {value}")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Stack trace: " + traceback.format_exc())
        st.info("Please check your data path and configuration settings.")