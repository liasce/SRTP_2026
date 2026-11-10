"""Validate ERA5 monthly files without contacting CDS."""
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from download_samples import validate_era5_file


def make_file(path, month="2018-01", region="west", missing=False):
    longitude = np.arange(120.0, 180.0 + 0.25, 0.25) if region == "west" else np.arange(-180.0, -80.0 + 0.25, 0.25)
    latitude = np.arange(20.25, -20.25 - 0.25, -0.25)
    values = np.ones((1, latitude.size, longitude.size), dtype=np.float32)
    if missing:
        values[:] = np.nan
    ds = xr.Dataset(
        {"u10": (("valid_time", "latitude", "longitude"), values),
         "v10": (("valid_time", "latitude", "longitude"), values * 2)},
        coords={"valid_time": [np.datetime64(month + "-01")],
                "latitude": latitude, "longitude": longitude},
    )
    ds.to_netcdf(path)


def require_rejected(path, month, region):
    try:
        validate_era5_file(path, month, region)
    except ValueError:
        return
    raise AssertionError(f"Invalid ERA5 file accepted: {path.name}")


def main():
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
        folder = Path(folder)
        west = folder / "west.nc"
        east = folder / "east.nc"
        make_file(west, region="west")
        make_file(east, region="east")
        assert validate_era5_file(west, "201801", "west")["longitude_range"] == [120.0, 180.0]
        assert validate_era5_file(east, "201801", "east")["longitude_range"] == [-180.0, -80.0]

        wrong_month = folder / "wrong_month.nc"
        make_file(wrong_month, month="2018-02", region="west")
        require_rejected(wrong_month, "201801", "west")

        missing = folder / "missing.nc"
        make_file(missing, region="west", missing=True)
        require_rejected(missing, "201801", "west")

        require_rejected(west, "201801", "east")
    print("ERA5 month, domain, variables, and finite-value checks passed")


if __name__ == "__main__":
    main()
