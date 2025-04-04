# Bitcoin Price Forecast Application

This repository contains a Streamlit application that predicts future Bitcoin prices based on historical data. You can visualize Bitcoin price trends using multiple forecasting models.

## Features

- **Multiple Prediction Models**: Choose from ARIMA model, Simple Linear Forecast, and Moving Average-based prediction
- **Adjustable Forecast Period**: Select prediction timeframes from 4 to 52 weeks (1 year)
- **Data Visualization**: Display historical price data and forecast results in graphs
- **Forecast Data Export**: View prediction results in tabular format

## Usage

### Running Locally

1. Clone the repository
   ```
   git clone https://github.com/yourusername/bitcoin-price-forecast.git
   cd bitcoin-price-forecast
   ```

2. Install required packages
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app
   ```
   streamlit run bitcoin_forecast_app.py
   ```

4. Your browser will automatically open the application (typically at http://localhost:8501)

### How to Use the Features

1. Select a prediction model from the sidebar (ARIMA, Simple Forecast, Moving Average)
2. Adjust the forecast period in weeks
3. The graph and prediction table will update automatically

## About the Data

The application uses the following data:

- `bitcoin_weekly_data_2020_2025.csv`: Weekly Bitcoin price data from April 2020 to April 2025
- If the CSV file is not found, sample data will be used automatically

## Technologies Used

- **Python 3.8+**
- **Streamlit**: For creating interactive web applications
- **Pandas & NumPy**: For data processing and numerical calculations
- **Matplotlib**: For data visualization
- **statsmodels**: For time series forecasting with ARIMA models

## Installation Requirements

```
streamlit>=1.24.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.7.0
statsmodels>=0.13.0
```

## Disclaimer

This application is intended for educational purposes only. The forecast results are based solely on historical data patterns and do not guarantee future prices. Always consult with professionals for investment decisions.

## License

MIT License

## Contributions

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Future Improvements

- Add advanced prediction models like Prophet and LSTM
- Support for multiple cryptocurrencies
- Allow users to upload their own data
- Add confidence intervals and prediction accuracy metrics

---
