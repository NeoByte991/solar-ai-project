import time
import sys
import os
import math
import pandas as pd

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

predict = st.sidebar.button("⚡ Run Prediction")

if predict:
    with st.spinner("Fetching weather & predicting power... ⚡"):
        try:
            time.sleep(1)

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
            col3.metric("☀️ Irradiation", round(data["irradiation"], 2))
            col4.metric("⚡ Power", round(data["predicted_power"], 2))

            st.divider()

            # 🔥 Dashboard Layout
            left, right = st.columns([2, 1])

            with left:
                st.subheader("📊 Power Analysis")

                hours = list(range(6, 19))  # 6 AM to 6 PM
                power_values = []

                for h in hours:
                    angle = (h - 6) / 12 * math.pi
                    base_power = math.sin(angle)

                    cloud_factor = (100 - data["clouds"]) / 100

                    value = base_power * data["predicted_power"] * cloud_factor
                    power_values.append(max(value, 0))

                df = pd.DataFrame({
                    "Hour": hours,
                    "Power": power_values
                }).set_index("Hour")

                st.line_chart(df)

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

            st.toast("Prediction complete ⚡")

        except Exception as e:
            st.error(f"Error: {e}")