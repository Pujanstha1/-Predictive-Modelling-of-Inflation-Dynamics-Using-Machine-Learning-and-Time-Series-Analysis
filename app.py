import streamlit as st
import sys
from pathlib import Path
root_path = Path(__file__).parent  # Add the project root to Python path
sys.path.append(str(root_path))

# Page configuration
st.set_page_config(
    page_title="Nepal Inflation Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize container for main content
main_container = st.container()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page",
    ["Dashboard", "Data Analysis", "Model Training", "Forecasting", "Model Performance"]
)

# Import pages after Streamlit configuration
from app.pages.dashboard import show_dashboard
from app.pages.data_analysis import show_data_analysis
from app.pages.model_training import show_model_training
from app.pages.forecasting import show_forecasting
from app.pages.model_performance import show_model_performance

with main_container:
    if page == "Dashboard":
        show_dashboard()
    elif page == "Data Analysis":
        show_data_analysis()
    elif page == "Model Training":
        show_model_training()
    elif page == "Forecasting":
        show_forecasting()
    elif page == "Model Performance":
        show_model_performance()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application provides comprehensive analysis and forecasting "
    "of inflation dynamics in Nepal using machine learning and time series analysis."
)