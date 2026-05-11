import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import altair as alt
import pandas as pd
import requests
import streamlit as st

from src.config import DEFAULT_CITY, NOMINATIM_USER_AGENT, CACHE_TTL_SECONDS
from src.forecast import get_open_meteo_forecast
from src.predict import (
    get_forecast_feature_importance,
    get_forecast_model_metrics,
    predict_forecast_frame,
)
from src.training_data import build_training_frame


st.set_page_config(page_title="Helios Solar Forecast", layout="wide")


@st.cache_data(ttl=24 * 60 * 60)
def get_lat_lon(city):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"city": city, "format": "json", "limit": 1}
    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": NOMINATIM_USER_AGENT},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    if not data:
        raise ValueError("City not found")

    return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name", city)


def safe_get_lat_lon(city):
    try:
        return get_lat_lon(city)
    except Exception as exc:
        st.error(f"Unable to find location for '{city}': {exc}")
        st.error("Please check the city name and try again.")
        return None, None, city


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_forecast(lat, lon, days):
    forecast = get_open_meteo_forecast(lat, lon, days=days)
    return predict_forecast_frame(forecast)


def safe_load_forecast(lat, lon, days):
    try:
        return load_forecast(lat, lon, days)
    except Exception as exc:
        st.error("Unable to load forecast data. Please refresh or try again later.")
        st.error(str(exc))
        return pd.DataFrame(columns=[
            "timestamp",
            "predicted_power",
            "IRRADIATION",
            "AMBIENT_TEMPERATURE",
            "HUMIDITY",
            "CLOUD_COVER",
            "WIND_SPEED",
        ])


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_recent_historical():
    try:
        history = build_training_frame().copy()
    except Exception:
        return pd.DataFrame(columns=["timestamp", "AC_POWER"])

    # Get last 24 hours
    end_time = pd.Timestamp.now()
    start_time = end_time - pd.Timedelta(hours=24)
    recent = history[(history["DATE_TIME"] >= start_time) & (history["DATE_TIME"] <= end_time)].copy()
    recent = recent.rename(columns={"DATE_TIME": "timestamp", "AC_POWER": "actual_power"})
    recent = recent[["timestamp", "actual_power"]]
    return recent


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_historical_comparison():
    try:
        history = build_training_frame().copy()
    except Exception:
        return pd.DataFrame(columns=["timestamp", "Series", "Power"])

    history["date"] = history["DATE_TIME"].dt.date

    daylight_dates = (
        history.groupby("date")["AC_POWER"]
        .max()
        .loc[lambda series: series > 0]
        .index
    )
    if len(daylight_dates):
        history = history[history["date"] == daylight_dates[-1]].copy()
    else:
        history = history.tail(96).copy()

    comparison_input = history.rename(columns={"DATE_TIME": "timestamp"})
    predicted = predict_forecast_frame(comparison_input)
    comparison = pd.DataFrame(
        {
            "timestamp": comparison_input["timestamp"],
            "Actual": history["AC_POWER"],
            "Predicted": predicted["predicted_power"],
        }
    )
    comparison = comparison.groupby("timestamp", as_index=False).mean()
    return comparison.melt("timestamp", var_name="Series", value_name="Power")


def metric_value(df, hours):
    return df.head(hours)["predicted_power"].sum() / 1000


def line_chart(df, y, color=None):
    enc = {
        "x": alt.X("timestamp:T", title="Time"),
        "y": alt.Y(f"{y}:Q", title="Power (W)"),
        "tooltip": ["timestamp:T", f"{y}:Q"],
    }
    if color:
        enc["color"] = alt.Color(f"{color}:N", title="")
    return alt.Chart(df).mark_line(interpolate="monotone").encode(**enc)


def format_forecast_table(df):
    columns = {
        "timestamp": "Time",
        "predicted_power": "Predicted power (W)",
        "IRRADIATION": "Irradiation (kWh/m2)",
        "AMBIENT_TEMPERATURE": "Temp (C)",
        "HUMIDITY": "Humidity (%)",
        "CLOUD_COVER": "Cloud (%)",
        "WIND_SPEED": "Wind (km/h)",
    }
    formatted = df[list(columns)].rename(columns=columns).copy()
    formatted["Time"] = formatted["Time"].dt.strftime("%Y-%m-%d %H:%M")
    formatted["Predicted power (W)"] = formatted["Predicted power (W)"].round(0).astype(int)
    formatted["Irradiation (kWh/m2)"] = formatted["Irradiation (kWh/m2)"].round(3)
    formatted["Temp (C)"] = formatted["Temp (C)"].round(1)
    formatted["Humidity (%)"] = formatted["Humidity (%)"].round(0).astype(int)
    formatted["Cloud (%)"] = formatted["Cloud (%)"].round(0).astype(int)
    formatted["Wind (km/h)"] = formatted["Wind (km/h)"].round(1)
    return formatted


def forecast_condition(row):
    if row["predicted_power"] <= 0:
        return "No generation"
    if row["IRRADIATION"] >= 0.7 and row["CLOUD_COVER"] <= 50:
        return "Strong"
    if row["IRRADIATION"] >= 0.35:
        return "Moderate"
    return "Low"


def format_simple_forecast_table(df):
    simple = pd.DataFrame(
        {
            "Time": df["timestamp"].dt.strftime("%a %d, %I %p"),
            "Expected output": df["predicted_power"].round(0).astype(int).astype(str) + " W",
            "Condition": df.apply(forecast_condition, axis=1),
        }
    )
    return simple


def format_daily_summary(df):
    columns = {
        "date": "Date",
        "energy_kwh": "Predicted energy (kWh)",
        "peak_power": "Peak power (W)",
        "avg_cloud": "Avg cloud (%)",
    }
    formatted = df.rename(columns=columns).copy()
    formatted["Predicted energy (kWh)"] = formatted["Predicted energy (kWh)"].round(2)
    formatted["Peak power (W)"] = formatted["Peak power (W)"].round(0).astype(int)
    formatted["Avg cloud (%)"] = formatted["Avg cloud (%)"].round(0).astype(int)
    return formatted


def display_feature_name(feature):
    names = {
        "IRRADIATION": "Irradiation",
        "AMBIENT_TEMPERATURE": "Ambient temperature",
        "MODULE_TEMPERATURE": "Module temperature",
        "HUMIDITY": "Humidity",
        "CLOUD_COVER": "Cloud cover",
        "WIND_SPEED": "Wind speed",
        "hour": "Hour",
        "day_of_year": "Day of year",
        "month": "Month",
        "hour_sin": "Hour cycle sin",
        "hour_cos": "Hour cycle cos",
        "daylight_flag": "Daylight flag",
        "solar_position": "Solar position",
        "clear_sky_index": "Clear-sky index",
        "cloud_loss_proxy": "Cloud loss proxy",
        "effective_irradiation": "Effective irradiation",
        "module_temp_delta": "Module temp delta",
        "irradiation_rolling_3": "Rolling irradiation",
    }
    return names.get(feature, feature.replace("_", " ").title())


def format_model_timestamp(value):
    if not value or value == "unknown":
        return "unknown"
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return value


st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: #151922;
        border: 1px solid #2d3340;
        border-radius: 8px;
        padding: 14px 16px;
    }
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f8fafc;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 650;
    }
    h1, h2, h3 {letter-spacing: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("Helios Solar Forecast")
st.caption("AI-based solar generation forecasting from weather and solar irradiance forecasts")

with st.sidebar:
    st.header("Forecast")
    city = st.text_input("Location", DEFAULT_CITY).strip()
    forecast_days = st.slider("Forecast days", min_value=1, max_value=7, value=2)
    refresh = st.button("Refresh forecast", width="stretch")

if refresh:
    st.cache_data.clear()

st.caption(f"Data is cached for {CACHE_TTL_SECONDS // 60} minutes. Use 'Refresh forecast' to update immediately.")

try:
    lat, lon, display_name = safe_get_lat_lon(city)
    if lat is None:
        st.stop()

    forecast_df = safe_load_forecast(lat, lon, forecast_days)
    if forecast_df.empty:
        st.stop()

    try:
        metrics = get_forecast_model_metrics()
    except Exception as exc:
        st.error(f"Unable to load model metrics: {exc}")
        metrics = {"rows": 0, "mae": 0, "r2": 0, "daylight_mae": None, "peak_mae": None, "trained_at": "unknown", "training_start": "unknown", "training_end": "unknown", "model_type": "Unknown"}

    current = forecast_df.iloc[0]
    daylight = forecast_df[forecast_df["IRRADIATION"] > 0.01]

    st.caption(f"{display_name} | {lat:.3f}, {lon:.3f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Now", f"{current['predicted_power']:.0f} W")
    c2.metric("Next 1 hour", f"{metric_value(forecast_df, 1):.2f} kWh")
    c3.metric("Next 6 hours", f"{metric_value(forecast_df, 6):.2f} kWh")
    c4.metric("Next 24 hours", f"{metric_value(forecast_df, 24):.2f} kWh")

    with st.expander("Forecast inputs"):
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Peak forecast", f"{forecast_df['predicted_power'].max():.0f} W")
        e2.metric("Sky cover", f"{current['CLOUD_COVER']:.0f}%")
        e3.metric("Irradiation", f"{current['IRRADIATION']:.2f} kWh/m2")
        e4.metric("Wind speed", f"{current['WIND_SPEED']:.1f} km/h")

    st.divider()

    tab_overview, tab_forecast, tab_history, tab_weather, tab_model = st.tabs(
        ["Overview", "Forecast", "Historical", "Weather impact", "Model"]
    )

    with tab_overview:
        st.subheader("Power trend")
        chart = line_chart(forecast_df, "predicted_power")
        st.altair_chart(chart, width="stretch")

        daily = (
            forecast_df.assign(date=forecast_df["timestamp"].dt.date)
            .groupby("date", as_index=False)
            .agg(
                energy_kwh=("predicted_power", lambda x: x.sum() / 1000),
                peak_power=("predicted_power", "max"),
                avg_cloud=("CLOUD_COVER", "mean"),
            )
        )
        st.dataframe(format_daily_summary(daily), width="stretch", hide_index=True)

    with tab_forecast:
        st.subheader("Hourly prediction")
        st.caption("Simple hourly view of expected solar power output.")

        # Load recent historical data
        recent_history = load_recent_historical()
        if not recent_history.empty:
            # Combine historical and forecast
            forecast_with_past = pd.concat([
                pd.DataFrame({
                    "timestamp": recent_history["timestamp"],
                    "power": recent_history["actual_power"],
                    "type": "Actual"
                }),
                pd.DataFrame({
                    "timestamp": forecast_df["timestamp"],
                    "power": forecast_df["predicted_power"],
                    "type": "Predicted"
                })
            ], ignore_index=True)
            
            # Sort by timestamp
            forecast_with_past = forecast_with_past.sort_values("timestamp")
            
            # Chart
            chart = line_chart(forecast_with_past, "power", color="type")
            st.altair_chart(chart, width="stretch")

        simple_forecast = format_simple_forecast_table(forecast_df)
        st.dataframe(simple_forecast, width="stretch", hide_index=True)

        detailed_forecast = format_forecast_table(forecast_df)
        with st.expander("Detailed forecast data"):
            st.dataframe(detailed_forecast, width="stretch", hide_index=True)

        csv = detailed_forecast.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download forecast CSV",
            data=csv,
            file_name=f"{city.lower().replace(' ', '_')}_solar_forecast.csv",
            mime="text/csv",
        )

    with tab_history:
        st.subheader("Historical vs predicted")
        comparison = load_historical_comparison()
        if comparison.empty:
            st.info("Historical validation data is not available in this deployment.")
        else:
            history_chart = line_chart(comparison, "Power", color="Series")
            st.altair_chart(history_chart, width="stretch")

    with tab_weather:
        st.subheader("Weather impact")
        impact = forecast_df[forecast_df["IRRADIATION"] > 0.01]
        scatter = (
            alt.Chart(impact)
            .mark_circle(size=70, opacity=0.75)
            .encode(
                x=alt.X("IRRADIATION:Q", title="Irradiation (kWh/m2)"),
                y=alt.Y("predicted_power:Q", title="Predicted power (W)"),
                color=alt.Color("CLOUD_COVER:Q", title="Cloud cover (%)"),
                tooltip=[
                    "timestamp:T",
                    "predicted_power:Q",
                    "IRRADIATION:Q",
                    "CLOUD_COVER:Q",
                    "WIND_SPEED:Q",
                ],
            )
        )
        st.altair_chart(scatter, width="stretch")

        if daylight.empty:
            st.warning("No daylight forecast available in the selected range.")
        else:
            st.dataframe(
                format_forecast_table(
                    daylight.sort_values("predicted_power", ascending=False).head(10)
                ),
                width="stretch",
                hide_index=True,
            )

    with tab_model:
        st.subheader("Model quality")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Training rows", f"{metrics.get('rows', 0):,}")
        m2.metric("Average error", f"{metrics.get('mae', 0):.2f} W")
        daylight_mae = metrics.get("daylight_mae")
        m3.metric(
            "Daylight error",
            f"{daylight_mae:.2f} W" if daylight_mae is not None else "N/A",
        )
        m4.metric("R2 score", f"{metrics.get('r2', 0):.3f}")

        peak_mae = metrics.get("peak_mae")
        trained_at = format_model_timestamp(metrics.get("trained_at", "unknown"))
        model_type = metrics.get("model_type", "Model")
        peak_text = f"{peak_mae:.2f} W" if peak_mae is not None else "N/A"
        training_start = format_model_timestamp(metrics.get("training_start", "unknown"))
        training_end = format_model_timestamp(metrics.get("training_end", "unknown"))

        st.caption(f"{model_type} | Peak-hour error: {peak_text}")
        st.caption(
            f"Training window: {training_start} to {training_end} | Trained: {trained_at}"
        )

        try:
            features, importance = get_forecast_feature_importance()
        except Exception:
            features, importance = [], []

        if len(features):
            fi_df = pd.DataFrame(
                {
                    "Feature": [display_feature_name(feature) for feature in features],
                    "Importance": importance,
                }
            )
            top_features = fi_df.sort_values("Importance", ascending=False).head(5)
            st.write(
                "Main prediction driver: "
                f"**{top_features.iloc[0]['Feature']}**"
            )

            with st.expander("Technical feature importance"):
                bars = (
                    alt.Chart(top_features)
                    .mark_bar()
                    .encode(
                        x=alt.X("Importance:Q", title="Importance"),
                        y=alt.Y("Feature:N", sort="-x", title=None),
                        tooltip=[
                            "Feature",
                            alt.Tooltip("Importance:Q", format=".3f"),
                        ],
                    )
                )
                st.altair_chart(bars, width="stretch")

except Exception as exc:
    st.error(f"Unable to build forecast: {exc}")

st.markdown("---")
st.caption("Helios | Historical plant data + Open-Meteo forecast + machine learning prediction")
