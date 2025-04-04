import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import io

# ページ設定
st.set_page_config(page_title="Bitcoin Price Forecast", layout="wide")

# タイトル
st.title("Bitcoin Price Forecast")

# データの読み込み
@st.cache_data
def load_data():
    try:
        # CSVファイルの読み込みを試行
        df = pd.read_csv("bitcoin_weekly_data_2020_2025.csv")
    except Exception as e:
        st.warning(f"Failed to read CSV file: {str(e)}")
        st.info("Use sample data")
        
        # サンプルデータ
        data_str = """Date,Open,High,Low,Close,Volume,Market_Cap
2020-04-05,7246.65,7622.79,6746.06,7180.02,10836056,132830337813
2020-04-12,7141.32,7645.73,6870.51,7269.83,13781058,134491851948
2020-04-19,7217.64,7382.47,6793.94,7310.68,12920378,135247555997
2020-04-26,7485.23,7712.67,7207.0,7505.66,11240817,138854682226
2020-05-03,7466.22,7951.24,6973.28,7479.36,13163410,138368214969
2020-05-10,7401.96,7500.05,7124.6,7374.7,11204880,136432017831
2020-05-17,7450.52,7754.59,6918.13,7396.02,12918066,136826384868
2020-05-24,7325.9,7865.47,6872.53,7464.93,13912879,138101198844
2020-05-31,7626.43,7801.38,7533.93,7612.21,14787419,140825830647
2020-06-07,7549.44,8097.82,7261.45,7419.02,15174204,137251813694"""
        
        df = pd.read_csv(io.StringIO(data_str))
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# データ読み込み
df = load_data()

# 予測期間（週数）
weeks_to_forecast = st.sidebar.slider("Forecast period (weeks)", min_value=4, max_value=52, value=12, step=4)

# 予測モデルの選択
selected_model = st.sidebar.selectbox(
    "Select Prediction Model",
    ["ARIMA", "Simple Prediction", "Moving Average"]
)

# データの表示
st.subheader("Bitcoin Price Data")
st.dataframe(df.head())

# ARIMA モデル
def run_arima(df, weeks_to_forecast):
    st.subheader("ARIMA Forecast")
    st.write("Forecasting with an autoregressive summed moving average model. This is a traditional forecasting method that takes advantage of the statistical properties of time series data.")
    
    try:
        model = ARIMA(df['Close'], order=(2,1,0))
        model_fit = model.fit()
        
        last_date = df.index[-1]
        forecast_dates = [last_date + timedelta(days=7*i) for i in range(1, weeks_to_forecast+1)]
        
        forecast = model_fit.forecast(steps=weeks_to_forecast)
        forecast_df = pd.DataFrame({
            'Predicted_Close': forecast.round(2)
        }, index=forecast_dates)
        
        # グラフ作成
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, df['Close'], label='Historical Price')
        ax.plot(forecast_df.index, forecast_df['Predicted_Close'], 'r--', label='ARIMA Forecast')
        ax.set_title('Bitcoin Price Forecast (ARIMA Model)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)
        st.subheader("Estimated Price")
        st.dataframe(forecast_df)
    except Exception as e:
        st.error(f"An error has occurred: {str(e)}")

# シンプル予測
def run_simple_forecast(df, weeks_to_forecast):
    st.subheader("Simple Prediction")
    st.write("It is a simple linear forecast based on the most recent trend.")
    
    try:
        # 直近のトレンドに基づいた単純な線形予測
        recent_df = df.tail(4)  # 直近の4週間
        x = np.arange(len(recent_df))
        y = recent_df['Close'].values
        
        # 線形回帰で傾きと切片を計算
        z = np.polyfit(x, y, 1)
        slope = z[0]
        intercept = z[1]
        
        # 予測期間
        last_date = df.index[-1]
        forecast_dates = [last_date + timedelta(days=7*i) for i in range(1, weeks_to_forecast+1)]
        
        # 予測値を計算
        predictions = [slope * (len(recent_df) + i) + intercept for i in range(1, weeks_to_forecast+1)]
        forecast_df = pd.DataFrame({
            'Predicted_Close': np.round(predictions, 2)
        }, index=forecast_dates)
        
        # グラフ作成
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, df['Close'], label='Historical Price')
        ax.plot(forecast_df.index, forecast_df['Predicted_Close'], 'g--', label='Simple Linear Forecast')
        ax.set_title('Bitcoin Price Forecast (Simple Linear Model)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)
        st.subheader("Estimated Price")
        st.dataframe(forecast_df)
    except Exception as e:
        st.error(f"An error has occurred: {str(e)}")

# 移動平均予測
def run_moving_average(df, weeks_to_forecast):
    st.subheader("Moving Average Forecast")
    st.write("This is a forecasting model using moving averages. It allows us to see trends with smoothed fluctuations.")
    
    try:
        # 移動平均を計算
        df['MA4'] = df['Close'].rolling(window=4).mean()
        
        # 過去4週間のMAの変化率を計算
        recent_ma = df['MA4'].dropna().tail(4)
        avg_change = recent_ma.pct_change().dropna().mean()
        
        # 予測期間
        last_date = df.index[-1]
        forecast_dates = [last_date + timedelta(days=7*i) for i in range(1, weeks_to_forecast+1)]
        
        # 最新の移動平均から予測
        last_ma = recent_ma.iloc[-1]
        predictions = []
        current_val = last_ma
        
        for i in range(weeks_to_forecast):
            current_val = current_val * (1 + avg_change)
            predictions.append(current_val)
        
        forecast_df = pd.DataFrame({
            'Predicted_Close': np.round(predictions, 2)
        }, index=forecast_dates)
        
        # グラフ作成
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, df['Close'], label='Historical Price')
        ax.plot(df.index, df['MA4'], label='4-Week Moving Average')
        ax.plot(forecast_df.index, forecast_df['Predicted_Close'], 'b--', label='Moving Avg Based Forecast')
        ax.set_title('Bitcoin Price Forecast (Moving Average Model)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)
        st.subheader("Estimated Price")
        st.dataframe(forecast_df)
    except Exception as e:
        st.error(f"An error has occurred: {str(e)}")

# 選択したモデルを実行
if selected_model == "ARIMA":
    run_arima(df, weeks_to_forecast)
elif selected_model == "Simple Prediction":
    run_simple_forecast(df, weeks_to_forecast)
else:
    run_moving_average(df, weeks_to_forecast)

# フッター
st.markdown("---")
st.caption("Note: This forecast is based on historical data and is not a guarantee of future prices. Please consult a professional to make an investment decision.")
