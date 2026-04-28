import pandas as pd
import requests

def fetch_data(lat, lon, start="20240101", end="20240105"):
    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point?"
        f"parameters=ALLSKY_SFC_SW_DWN,T2M&community=RE"
        f"&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()

    if "properties" not in data:
        raise ValueError(f"Unexpected API response: {list(data.keys())}")

    records = data["properties"]["parameter"]

    df = pd.DataFrame({
        "irradiance": records["ALLSKY_SFC_SW_DWN"],
        "temp": records["T2M"]
    })

    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
    df["hour"] = df.index.hour

    return df.dropna()