import streamlit as st
import requests

st.set_page_config(page_title="Solar AI", layout="centered")

st.title("☀️ AI Solar Power Prediction")
st.markdown("Predict solar irradiance using AI based on temperature and time.")

st.divider()

# Inputs
temp = st.slider("🌡 Temperature (°C)", 0, 50, 25)
hour = st.slider("⏰ Hour of Day", 0, 23, 12)

st.divider()

if st.button("⚡ Predict Solar Power"):
    try:
        res = requests.get(
            f"https://solar-ai-project-v08o.onrender.com/predict?temp={temp}&hour={hour}"
        )
        result = res.json()

        st.success(
            f"🌞 Predicted Solar Irradiance: {result['predicted_irradiance']:.2f} W/m²"
        )

    except:
        st.error("⚠️ API not running. Start backend first.")