import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from src.data.data_loader import DataLoader
from src.utils.config import Config

def create_metrics(data):
    """Create and display key metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_inflation = data['overall_index_mom'].iloc[-1]
        prev_inflation = data['overall_index_mom'].iloc[-2]
        delta = current_inflation - prev_inflation
        st.metric(
            "Current Inflation (MoM)", 
            f"{current_inflation:.2f}%",
            f"{delta:+.2f}%"
        )
    
    with col2:
        yoy_inflation = data['overall_index_yoy'].iloc[-1]
        prev_yoy = data['overall_index_yoy'].iloc[-2]
        delta_yoy = yoy_inflation - prev_yoy
        st.metric(
            "YoY Inflation",
            f"{yoy_inflation:.2f}%",
            f"{delta_yoy:+.2f}%"
        )
    
    with col3:
        food_inflation = data['food_and_beverage_mom'].iloc[-1]
        prev_food = data['food_and_beverage_mom'].iloc[-2]
        delta_food = food_inflation - prev_food
        st.metric(
            "Food Inflation (MoM)",
            f"{food_inflation:.2f}%",
            f"{delta_food:+.2f}%"
        )
    
    with col4:
        core_inflation = data['non_food_and_services_mom'].iloc[-1]
        prev_core = data['non_food_and_services_mom'].iloc[-2]
        delta_core = core_inflation - prev_core
        st.metric(
            "Core Inflation (MoM)",
            f"{core_inflation:.2f}%",
            f"{delta_core:+.2f}%"
        )

def plot_inflation_trends(data):
    """Plot main inflation trends"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Indices', 'Month-over-Month Changes'),
        vertical_spacing=0.12
    )
    
    # Price Indices
    components = ['overall_index', 'food_and_beverage', 'non_food_and_services']
    names = ['Overall', 'Food & Beverage', 'Non-food & Services']
    colors = ['blue', 'green', 'red']
    
    for comp, name, color in zip(components, names, colors):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[comp],
                name=name,
                line=dict(color=color)
            ),
            row=1, col=1
        )
    
    # MoM Changes
    for comp, name, color in zip(components, names, colors):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[f"{comp}_mom"],
                name=f"{name} MoM",
                line=dict(color=color, dash='dot')
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=700,
        showlegend=True,
        template='plotly_white',
        title_text="Nepal Inflation Trends"
    )
    
    return fig

def plot_monetary_indicators(data):
    """Plot monetary policy indicators"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Monetary Aggregates Growth', 'Interest Rates'),
        specs=[[{"secondary_y": True}, {"secondary_y": True}]]
    )
    
    # Monetary Aggregates
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['broad_money_m2_mom'],
            name='M2 Growth',
            line=dict(color='purple')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['overall_index_mom'],
            name='Inflation',
            line=dict(color='red')
        ),
        row=1, col=1, secondary_y=True
    )
    
    # Interest Rates
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['policy_rate'],
            name='Policy Rate',
            line=dict(color='blue')
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['commercial_bank_lending_rates'],
            name='Lending Rate',
            line=dict(color='orange')
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        title_text="Monetary Policy Indicators"
    )
    
    return fig

def plot_external_sector(data):
    """Plot external sector indicators"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Exchange Rates', 'International Prices'),
        specs=[[{"secondary_y": True}, {"secondary_y": True}]]
    )
    
    # Exchange Rates
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['usd'],
            name='USD/NPR',
            line=dict(color='green')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['inr'],
            name='INR/NPR',
            line=dict(color='orange')
        ),
        row=1, col=1, secondary_y=True
    )
    
    # International Prices
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['crude_oil_prices'],
            name='Oil Price',
            line=dict(color='brown')
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['global_food_price_index'],
            name='Global Food Price',
            line=dict(color='red')
        ),
        row=1, col=2, secondary_y=True
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        title_text="External Sector Indicators"
    )
    
    return fig

def plot_international_comparison(data):
    """Plot international inflation comparison"""
    fig = go.Figure()
    
    countries = ['nepal', 'indian', 'china', 'usa']
    names = ['Nepal', 'India', 'China', 'USA']
    colors = ['blue', 'orange', 'red', 'green']
    
    for country, name, color in zip(countries, names, colors):
        if country == 'nepal':
            col = 'overall_index_mom'
        else:
            col = f'{country}_cpi_mom'
            
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[col],
                name=name,
                line=dict(color=color)
            )
        )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        title_text="International Inflation Comparison (MoM)",
        yaxis_title="Inflation Rate (%)"
    )
    
    return fig

def show_dashboard():
    """Main dashboard display function"""
    st.title("Nepal Inflation Dashboard")
    
    try:
        # Initialize config and load data
        config = Config()
        loader = DataLoader(config)
        data = loader.load_processed_data()
        
        # Add date filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.to_datetime(data.index.min())
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=pd.to_datetime(data.index.max())
            )
        
        # Filter data based on date range
        mask = (data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))
        filtered_data = data[mask]
        
        # Display metrics
        create_metrics(filtered_data)
        
        # Main inflation trends
        st.plotly_chart(
            plot_inflation_trends(filtered_data),
            use_container_width=True
        )
        
        # Monetary indicators
        st.plotly_chart(
            plot_monetary_indicators(filtered_data),
            use_container_width=True
        )
        
        # External sector
        st.plotly_chart(
            plot_external_sector(filtered_data),
            use_container_width=True
        )
        
        # International comparison
        st.plotly_chart(
            plot_international_comparison(filtered_data),
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your data path and configuration settings.")