import streamlit as st
import requests

st.set_page_config(page_title="Solar AI Dashboard", layout="wide")

# 🔥 Header
st.title("⚡ Solar Power AI Dashboard")
st.caption("Real-time Solar Energy Prediction System")

# 🔹 Sidebar input (THIS makes it feel like real app)
st.sidebar.header("Controls")
city = st.sidebar.text_input("Enter City", "Shimoga")

predict = st.sidebar.button("Run Prediction")

if predict:

    url = f"http://127.0.0.1:8000/predict_city?city={city}"
    res = requests.get(url)
    data = res.json()

    if "error" in data:
        st.error(data["error"])
    else:
        st.success(f"Data loaded for {city}")

        # 🔥 Top Metrics Row
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("🌡️ Temperature", f"{data['temperature']} °C")
        col2.metric("☁️ Clouds", f"{data['clouds']} %")
        col3.metric("☀️ Irradiation", data["irradiation"])
        col4.metric("⚡ Power", data["predicted_power"])

        st.divider()

        # 🔥 Layout split (THIS is real dashboard feel)
        left, right = st.columns([2, 1])

        with left:
            st.subheader("📊 Power Analysis")

            # fake data for demo (replace later)
            hours = list(range(6, 19))
            power_values = [data["predicted_power"] * (i/12) for i in range(len(hours))]

            st.line_chart(power_values)

        with right:
            st.subheader("⚡ Status")

            if data["predicted_power"] == 0:
                st.error("No Generation")
            else:
                st.success("Generating Power")

            st.progress(min(int(data["predicted_power"]), 100))

        st.divider()

        st.subheader("📋 Raw Data")
        st.json(data)