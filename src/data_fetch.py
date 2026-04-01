import pandas as pd
import requests

def fetch_data(lat, lon):
    url = f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=ALLSKY_SFC_SW_DWN,T2M&community=RE&longitude={lon}&latitude={lat}&start=20240101&end=20240105&format=JSON"

    response = requests.get(url)
    data = response.json()

    # Debug check (important)
    if "properties" not in data:
        raise Exception("API response error")

    records = data["properties"]["parameter"]

    irradiance = records["ALLSKY_SFC_SW_DWN"]
    temp = records["T2M"]

    df = pd.DataFrame({
        "IRRADIANCE": irradiance,
        "T2M": temp
    })

    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
    df["hour"] = df.index.hour

    return df.dropna()