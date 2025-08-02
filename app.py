import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import pickle

# === FEATURES ===
PM25_FEATURES = [
    'AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3',
    'PM10', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC',
    'WIND_SPEED', 'WS_HR', 'pm2.5_lag1', 'pm2.5_lag2', 'pm2.5_lag3',
    'pm2.5_roll3', 'hour', 'dayofweek'
]

PM10_FEATURES = [
    'AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3',
    'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC',
    'WIND_SPEED', 'WS_HR', 'pm10_lag1', 'pm10_lag2', 'pm10_lag3',
    'pm10_roll3', 'hour', 'dayofweek'
]

# === LOAD DATA ===
@st.cache_data
def load_data():
    df = pd.read_csv("data/merged_sorted.csv", parse_dates=["datetime"])
    df.columns = df.columns.str.strip()
    return df

# === LOAD MODELS ===
@st.cache_resource
def load_models():
    with open("model/final_model_20250802_141012.pkl", "rb") as f1, \
         open("model/final_model_pm10_20250802_164747.pkl", "rb") as f2:
        return pickle.load(f1), pickle.load(f2)

data = load_data()
model_pm25, model_pm10 = load_models()
sitename_list = data["sitename"].unique()

# === SIDEBAR ===
st.sidebar.title("🗓️ Select Prediction Time")
sitename = st.sidebar.selectbox("Monitoring Site", sitename_list)
date_input = st.sidebar.date_input("Target Date", value=datetime.now().date())
hour = st.sidebar.slider("Target Hour", 0, 23, value=datetime.now().hour)
target_time = datetime.combine(date_input, datetime.min.time()) + timedelta(hours=hour)

# === MAIN LOGIC ===
if st.sidebar.button("🔮 Start Prediction"):
    st.title("🌫️ Taipei Air Quality Forecast")
    st.subheader(f"{sitename} — Forecast up to {target_time.strftime('%Y-%m-%d %H:00')}")

    site_data = data[data["sitename"] == sitename].sort_values("datetime").copy()
    site_data = site_data.set_index("datetime")
    last_obs_time = site_data.index.max()
    full_series = site_data.copy()

    if last_obs_time >= target_time:
        st.warning("⚠️ Target time is not after the latest observation.")
        st.stop()

    current_time = last_obs_time + timedelta(hours=1)
    while current_time <= target_time:
        past = full_series.loc[:current_time - timedelta(hours=1)].tail(3)
        if len(past) < 3:
            break
        last_row = past.iloc[-1].copy()

        # Predict PM2.5
        row_pm25 = last_row.copy()
        row_pm25["pm2.5_lag1"] = past.iloc[-1]["PM2.5"]
        row_pm25["pm2.5_lag2"] = past.iloc[-2]["PM2.5"]
        row_pm25["pm2.5_lag3"] = past.iloc[-3]["PM2.5"]
        row_pm25["pm2.5_roll3"] = past["PM2.5"].mean()
        row_pm25["hour"] = current_time.hour
        row_pm25["dayofweek"] = current_time.weekday()
        y_pm25 = model_pm25.predict(pd.DataFrame([row_pm25])[PM25_FEATURES])[0]
        full_series.loc[current_time, "PM2.5"] = y_pm25

        # Predict PM10
        row_pm10 = last_row.copy()
        row_pm10["pm10_lag1"] = past.iloc[-1]["PM10"]
        row_pm10["pm10_lag2"] = past.iloc[-2]["PM10"]
        row_pm10["pm10_lag3"] = past.iloc[-3]["PM10"]
        row_pm10["pm10_roll3"] = past["PM10"].mean()
        row_pm10["hour"] = current_time.hour
        row_pm10["dayofweek"] = current_time.weekday()
        y_pm10 = model_pm10.predict(pd.DataFrame([row_pm10])[PM10_FEATURES])[0]
        full_series.loc[current_time, "PM10"] = y_pm10

        current_time += timedelta(hours=1)

    forecast_pm25 = full_series.loc[last_obs_time + timedelta(hours=1):]["PM2.5"]
    forecast_pm10 = full_series.loc[last_obs_time + timedelta(hours=1):]["PM10"]

    # === PLOTS ===
    st.markdown("## 📈 Forecast Plots")

    # PM2.5
    st.markdown("### PM2.5 Forecast (Historical + Predicted with CI)")
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(site_data.index, site_data["PM2.5"], label="Observed", color='black')
    ax1.plot(forecast_pm25.index, forecast_pm25.values, label="Forecast", color='royalblue')

    # Confidence Interval ±5%
    ci_pm25_lower = forecast_pm25 * 0.95
    ci_pm25_upper = forecast_pm25 * 1.05
    ax1.fill_between(forecast_pm25.index, ci_pm25_lower, ci_pm25_upper, color='royalblue', alpha=0.2, label="±5% CI")

    ax1.axvline(last_obs_time, linestyle="--", color="gray", label="Last Obs")
    ax1.legend()
    ax1.set_title("PM2.5 Levels with Confidence Interval")
    st.pyplot(fig1)

    # PM10
    st.markdown("### PM10 Forecast (Historical + Predicted with CI)")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(site_data.index, site_data["PM10"], label="Observed", color='black')
    ax2.plot(forecast_pm10.index, forecast_pm10.values, label="Forecast", color='darkorange')

    # Confidence Interval ±5%
    ci_pm10_lower = forecast_pm10 * 0.95
    ci_pm10_upper = forecast_pm10 * 1.05
    ax2.fill_between(forecast_pm10.index, ci_pm10_lower, ci_pm10_upper, color='darkorange', alpha=0.2, label="±5% CI")

    ax2.axvline(last_obs_time, linestyle="--", color="gray", label="Last Obs")
    ax2.legend()
    ax2.set_title("PM10 Levels with Confidence Interval")
    st.pyplot(fig2)


    # === HEATMAPS ===
    st.markdown("## 🔥 Heatmaps")
    col1, col2 = st.columns(2)

    with col1:
        df25 = forecast_pm25.reset_index().rename(columns={"datetime": "Datetime", "PM2.5": "PM2.5"})
        df25["date"] = df25["Datetime"].dt.date
        df25["hour"] = df25["Datetime"].dt.hour
        pivot25 = df25.pivot(index="hour", columns="date", values="PM2.5")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.heatmap(pivot25, cmap="Blues", ax=ax3)
        ax3.set_title("PM2.5 Forecast Heatmap")
        st.pyplot(fig3)

    with col2:
        df10 = forecast_pm10.reset_index().rename(columns={"datetime": "Datetime", "PM10": "PM10"})
        df10["date"] = df10["Datetime"].dt.date
        df10["hour"] = df10["Datetime"].dt.hour
        pivot10 = df10.pivot(index="hour", columns="date", values="PM10")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.heatmap(pivot10, cmap="YlOrRd", ax=ax4)
        ax4.set_title("PM10 Forecast Heatmap")
        st.pyplot(fig4)

    # === PM2.5 HEALTH BAR ===
    st.markdown("## 🫁 PM2.5 Health Info")
    latest_time = forecast_pm25.index[-1]
    pm25_value = forecast_pm25[-1]

    def pm25_level_info(val):
        if val <= 15:
            return "🟢 Good", "green", "Air quality is good. It's safe to be outside."
        elif val <= 35:
            return "🟡 Moderate", "yellow", "Air quality is acceptable."
        else:
            return "🔴 Unhealthy", "red", "Air quality is poor. Avoid outdoor activity."

    status, color, advice = pm25_level_info(pm25_value)

    if color == "red":
        st.error(f"⚠️ {status}: PM2.5 = {pm25_value:.1f} μg/m³\n\n{advice}")
    elif color == "yellow":
        st.warning(f"{status}: PM2.5 = {pm25_value:.1f} μg/m³\n\n{advice}")
    else:
        st.success(f"{status}: PM2.5 = {pm25_value:.1f} μg/m³\n\n{advice}")

    fig5, ax5 = plt.subplots(figsize=(6, 1.5))
    ax5.barh(["PM2.5"], [pm25_value], color=color)
    ax5.set_xlim(0, max(50, pm25_value + 10))
    ax5.set_xlabel("μg/m³")
    ax5.set_title(f"PM2.5 at {latest_time.strftime('%Y-%m-%d %H:%M')}")
    ax5.text(pm25_value + 1, 0, f"{pm25_value:.1f}", va='center')
    ax5.grid(axis='x', linestyle='--', alpha=0.3)
    st.pyplot(fig5)

    # === FORECAST TABLES ===
    st.markdown("## 📊 Forecast Tables")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### PM2.5")
        st.dataframe(forecast_pm25.reset_index().rename(columns={"datetime": "Datetime", "PM2.5": "Predicted_PM2.5"}), use_container_width=True, hide_index=True)
    with col4:
        st.markdown("### PM10")
        st.dataframe(forecast_pm10.reset_index().rename(columns={"datetime": "Datetime", "PM10": "Predicted_PM10"}), use_container_width=True, hide_index=True)

else:
    st.info("👈 Select a site and time to start predicting future PM2.5 and PM10 levels.")
