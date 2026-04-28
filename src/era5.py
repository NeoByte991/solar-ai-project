import xarray as xr
import numpy as np

def get_wind_speed(filepath="data/era5.nc"):
    ds = xr.open_dataset(filepath)

    u = ds["u10"].values
    v = ds["v10"].values

    wind_speed = np.sqrt(u**2 + v**2)

    return float(wind_speed.mean())