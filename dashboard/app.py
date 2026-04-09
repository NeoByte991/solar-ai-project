import time
import sys
import os
import math
import pandas as pd

# 🔥 FIX: allow Streamlit Cloud to find src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from datetime import datetime

# 🔥 IMPORT YOUR LOGIC DIRECTLY
from src.weather import get_weather_by_city
from src.predict import predict_kaggle
from src.irradiance import estimate_irradiation

st.set_page_config(page_title="Helios", layout="wide")

# 🔥 Header
st.title("⚡ Helios")
st.caption("AI-Based Solar Power Prediction System")

# 🔹 Sidebar
st.sidebar.header("Controls")
city = st.sidebar.text_input("Enter City", "Shimoga").strip().title()

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

            st.success(f"Live data for {city}")

            # 🔥 Metrics
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("🌡️ Temperature", f"{data['temperature']} °C")
            col2.metric("☁️ Clouds", f"{data['clouds']} %")
            col3.metric("☀️ Irradiation", round(data["irradiation"], 2))
            col4.metric("⚡ Power", round(data["predicted_power"], 2))

            st.divider()

            # 🔥 Layout
            left, right = st.columns([2, 1])

            with left:
                st.subheader("📊 Power Analysis")

                sunrise = 6
                sunset = 18

                hours = list(range(sunrise, sunset + 1))
                power_values = []

                for h in hours:
                    angle = (h - sunrise) / (sunset - sunrise) * math.pi
                    base_power = math.sin(angle)

                    cloud_factor = (100 - data["clouds"]) / 100
                    adjusted_cloud = 0.5 + (cloud_factor / 2)

                    value = base_power * data["predicted_power"] * adjusted_cloud
                    power_values.append(max(value, 0))

                df = pd.DataFrame({
                    "Hour": hours,
                    "Power": power_values
                }).set_index("Hour")

                st.line_chart(df)

                st.caption(f"🌅 Sunrise: {sunrise}:00   |   🌇 Sunset: {sunset}:00")

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

# 🔥 Footer (clean finish)
st.markdown("---")
st.caption("Developed by Batch 6 • Helios AI System")