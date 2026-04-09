import sys
import os

# 🔥 FIX: allow Streamlit Cloud to find src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from datetime import datetime

# 🔥 IMPORT YOUR LOGIC DIRECTLY (NO API CALLS)
from src.weather import get_weather_by_city
from src.predict import predict_kaggle
from src.irradiance import estimate_irradiation

st.set_page_config(page_title="Solar AI Dashboard", layout="wide")

# 🔥 Header
st.title("⚡ Solar Power AI Dashboard")
st.caption("Real-time Solar Energy Prediction System")

# 🔹 Sidebar input
st.sidebar.header("Controls")
city = st.sidebar.text_input("Enter City", "Shimoga")

predict = st.sidebar.button("Run Prediction")

if predict:
    try:
        # 🔥 GET DATA DIRECTLY (NO REQUESTS)
        hour = datetime.now().hour

        temp, clouds = get_weather_by_city(city)
        irradiation = estimate_irradiation(hour, clouds)
        module_temp = temp + 5

        power = predict_kaggle(irradiation, temp, module_temp, hour)

        data = {
            "temperature": temp,
            "clouds": clouds,
            "irradiation": irradiation,
            "predicted_power": power
        }

        st.success(f"Data loaded for {city}")

        # 🔥 Top Metrics
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("🌡️ Temperature", f"{data['temperature']} °C")
        col2.metric("☁️ Clouds", f"{data['clouds']} %")
        col3.metric("☀️ Irradiation", data["irradiation"])
        col4.metric("⚡ Power", round(data["predicted_power"], 2))

        st.divider()

        # 🔥 Dashboard Layout
        left, right = st.columns([2, 1])

        with left:
            st.subheader("📊 Power Analysis")

            # Simulated graph
            hours = list(range(6, 19))
            power_values = [data["predicted_power"] * (i/12) for i in range(len(hours))]

            st.line_chart(power_values)

        with right:
            st.subheader("⚡ Status")

            if data["predicted_power"] == 0:
                st.error("No Generation (Night / Low Sunlight)")
            else:
                st.success("Generating Power")

            st.progress(min(int(data["predicted_power"]), 100))

        st.divider()

        st.subheader("📋 Raw Data")
        st.json(data)

    except Exception as e:
        st.error(f"Error: {e}")