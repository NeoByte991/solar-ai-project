import sys
import os
import math
import pandas as pd
import requests
import altair as alt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from datetime import datetime
from astral import LocationInfo
from astral.sun import sun

from src.weather import get_weather_by_city
from src.predict import predict_kaggle, get_feature_importance
from src.era5 import get_wind_speed

st.set_page_config(page_title="Helios", layout="wide")

# -------------------- LOCATION --------------------

@st.cache_data
def get_lat_lon(city):
    url = f"https://nominatim.openstreetmap.org/search?city={city}&format=json"
    response = requests.get(url, headers={"User-Agent": "Helios"})
    data = response.json()

    if not data:
        raise Exception("City not found")

    return float(data[0]["lat"]), float(data[0]["lon"])


def get_sun_times(lat, lon, city):
    location = LocationInfo(name=city, region="Custom", latitude=lat, longitude=lon)
    s = sun(location.observer, date=datetime.now())
    return s["sunrise"], s["sunset"]


def get_nasa_irradiation(lat, lon):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        "?parameters=ALLSKY_SFC_SW_DWN&community=RE"
        f"&latitude={lat}&longitude={lon}&start=20240101&end=20241231&format=JSON"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    values = list(data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"].values())
    return sum(values) / len(values)


# -------------------- UI --------------------

st.title("Helios")
st.caption("AI-based solar power prediction system")

st.sidebar.header("Controls")
city = st.sidebar.text_input("City", "Shimoga").strip().title()
max_capacity = st.sidebar.number_input("Plant Capacity (W)", value=3500, step=500)

predict = st.sidebar.button("Run Prediction")

# -------------------- MAIN --------------------

if predict:
    try:
        hour = datetime.now().hour

        lat, lon = get_lat_lon(city)
        temp, clouds = get_weather_by_city(city)
        wind = get_wind_speed()

        try:
            irradiation = get_nasa_irradiation(lat, lon)
        except:
            st.warning("NASA data unavailable, using estimated value")
            irradiation = 4.5

        adjusted_irr = irradiation * (1 - (clouds / 100) * 0.75)
        module_temp = temp + 5

        # 🌅 Get sun times BEFORE prediction
        sunrise, sunset = get_sun_times(lat, lon, city)

        sunrise_hour = sunrise.hour
        sunset_hour = sunset.hour

        # ✅ FIXED prediction call
        power = predict_kaggle(
            adjusted_irr,
            temp,
            module_temp,
            hour,
            sunrise_hour,
            sunset_hour
        )

        st.success(f"Live data for {city}")
        st.caption(f"{lat:.2f}, {lon:.2f}")

        # -------------------- METRICS --------------------

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Temperature", temp)
        c2.metric("Clouds (%)", clouds)
        c3.metric("Wind (m/s)", round(wind, 2))
        c4.metric("Irradiation", round(adjusted_irr, 2))
        c5.metric("Power (W)", round(power, 2))

        st.divider()

        # -------------------- GRAPH --------------------

        st.subheader("Power Analysis")

        sunrise_hour_graph = sunrise.hour + sunrise.minute / 60
        sunset_hour_graph = sunset.hour + sunset.minute / 60

        hours = list(range(int(sunrise_hour_graph), int(sunset_hour_graph) + 1))
        values = []

        for h in hours:
            angle = (h - sunrise_hour_graph) / (sunset_hour_graph - sunrise_hour_graph) * math.pi
            base = math.sin(angle)

            cloud_factor = (100 - clouds) / 100
            adjusted_cloud = 0.5 + (cloud_factor / 2)

            values.append(max(base * power * adjusted_cloud, 0))

        df = pd.DataFrame({"Hour": hours, "Power": values})

        line = alt.Chart(df).mark_line(interpolate="monotone").encode(
            x="Hour:Q",
            y="Power:Q",
            tooltip=["Hour", "Power"]
        )

        area = alt.Chart(df).mark_area(opacity=0.3).encode(
            x="Hour:Q",
            y="Power:Q"
        )

        sun_df = pd.DataFrame({
            "Hour": [sunrise_hour_graph, sunset_hour_graph],
            "Event": ["Sunrise", "Sunset"]
        })

        rules = alt.Chart(sun_df).mark_rule(strokeDash=[5, 5]).encode(
            x="Hour:Q",
            color="Event:N",
            tooltip=["Event", "Hour"]
        )

        st.altair_chart(area + line + rules, use_container_width=True)

        st.caption(
            f"Sunrise: {sunrise.strftime('%H:%M')} | Sunset: {sunset.strftime('%H:%M')}"
        )

        st.divider()

        # -------------------- NASA DEBUG --------------------

        st.subheader("Irradiation Details")
        st.write(f"Raw: {irradiation:.2f}")
        st.write(f"Adjusted: {adjusted_irr:.2f}")

        st.divider()

        # -------------------- DOWNLOAD --------------------

        st.subheader("Download Data")

        download_df = pd.DataFrame({
            "City": [city],
            "Latitude": [lat],
            "Longitude": [lon],
            "Temperature": [temp],
            "Clouds": [clouds],
            "Wind": [wind],
            "Raw_Irradiation": [irradiation],
            "Adjusted_Irradiation": [adjusted_irr],
            "Predicted_Power": [power],
            "Time": [datetime.now()]
        })

        csv = download_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download CSV",
            data=csv,
            file_name=f"{city}_solar_data.csv",
            mime="text/csv"
        )

        st.divider()

        # -------------------- FEATURE IMPORTANCE --------------------

        st.subheader("Model Feature Importance")

        try:
            features, importance = get_feature_importance()

            fi_df = pd.DataFrame({
                "Feature": features,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(fi_df.set_index("Feature"))

            st.caption("Relative influence of each input feature on prediction.")

        except Exception:
            st.warning("Feature importance not available")

        st.divider()

        # -------------------- STATUS --------------------

        if power == 0:
            st.warning("No solar generation (night or low sunlight)")
        else:
            st.success("Generating power")

        progress = int((power / max_capacity) * 100)
        st.progress(max(0, min(progress, 100)))

    except Exception as e:
        st.error(f"Error: {e}")

# -------------------- PROJECT DETAILS --------------------

if predict:
    st.divider()

    st.subheader("About the System")

    st.markdown("""
Helios is a solar power prediction system that combines real-time data with machine learning.

Data sources:
- NASA POWER (irradiance)
- Weather API (temperature and clouds)
- ERA5 (wind data)

Workflow:
1. City is converted to coordinates
2. Weather and irradiance are fetched
3. Irradiance is adjusted using cloud cover
4. A Random Forest model predicts power output

Model inputs:
- Irradiance
- Temperature
- Module temperature
- Hour

Objective:
Provide reliable real-time estimates of solar power generation.
""")

st.markdown("---")
st.caption("Helios • Batch 6")