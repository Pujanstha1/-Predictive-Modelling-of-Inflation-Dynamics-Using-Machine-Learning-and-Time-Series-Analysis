# Nepal Inflation Forecasting Application

## Overview
This application provides comprehensive analysis and forecasting of inflation dynamics in Nepal using machine learning and time series analysis techniques. It combines traditional statistical methods (SARIMA) with modern machine learning approaches (XGBoost, Random Forest) to deliver accurate inflation forecasts.

## Features
- **Interactive Dashboard**: Real-time visualization of inflation trends and components
- **Data Analysis**: Comprehensive analysis of economic indicators and their relationships
- **Model Training**: Multiple forecasting models with customizable parameters
- **Forecasting**: Advanced hybrid forecasting system combining multiple models
- **Performance Analysis**: Detailed model evaluation and comparison metrics

## Project Structure
```
nepal_inflation_forecast/
├── app/                    # Streamlit web application
│   ├── __init__.py
│   └── pages/
│       ├── dashboard.py
│       ├── data_analysis.py
│       ├── model_training.py
│       ├── forecasting.py
│       └── model_performance.py
├── data/                   # Data directory
│   ├── raw/               # Original data files
│   └── processed/         # Processed datasets
├── src/                   # Source code
│   ├── data/             # Data processing modules
│   ├── features/         # Feature engineering
│   ├── models/           # Model implementations
│   └── utils/            # Utility functions
├── notebooks/            # Jupyter notebooks
├── reports/              # Generated reports
├── requirements.txt      # Project dependencies
├── app.py               # Main application file
└── README.md            # Project documentation
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/nepal_inflation_forecast.git
cd nepal_inflation_forecast
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## Models
### 1. SARIMA Model
- Captures seasonal patterns and trends
- Handles autocorrelation in time series data
- Parameters optimized for Nepal's inflation dynamics

### 2. XGBoost Model
- Non-linear relationships modeling
- Feature importance analysis
- Robust to outliers

### 3. Random Forest Model
- Ensemble learning approach
- Handles multicollinearity
- Provides feature importance rankings

### 4. Hybrid Model
- Combines predictions from multiple models
- Weighted averaging based on model performance
- Adaptive to changing economic conditions

## Data Sources
- Nepal Rastra Bank
- Central Bureau of Statistics
- Ministry of Finance
- International data sources (for global indicators)

## Usage Guide
1. **Dashboard**
   - View current inflation trends
   - Analyze component-wise breakdown
   - Monitor key economic indicators

2. **Data Analysis**
   - Explore relationships between variables
   - Analyze seasonal patterns
   - Study correlation matrices

3. **Model Training**
   - Select models to train
   - Customize training parameters
   - View training metrics

4. **Forecasting**
   - Generate future predictions
   - View confidence intervals
   - Compare model performances

5. **Model Performance**
   - Evaluate model accuracy
   - Compare different models
   - Analyze prediction errors

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Authors
- Pujan Shrestha

## Acknowledgments
- Data collected from Nepal Rastra Bank
- Manoj Shrestha Sir and Softwarica College for guidance and support
- Faculty members and advisors

## Contact
For any queries or suggestions, please contact:
- Email: pujanshrestha240@gmail.com
- LinkedIn: pujan-shrestha-9368b22b2
