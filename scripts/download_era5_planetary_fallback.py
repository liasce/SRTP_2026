"""Build ERA5 monthly 10 m wind from the Planetary Computer ERA5 mirror.

The mirror contains the original hourly ERA5 analysis in monthly, spatially
chunked Zarr assets.  Monthly means are computed from every hourly field.  An
existing CDS monthly file can be compared first; missing targets are never
overwritten silently.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import dask
import numpy as np
import planetary_computer
import pystac_client
import xarray as xr

from download_samples import ROOT, validate_era5_file


RAW = ROOT / "data" / "raw" / "era5"
CACHE = ROOT / "data" / "download_cache" / "era5" / "planetary_monthly_means"
AUDIT = ROOT / "research" / "audits" / "era5_planetary_fallback_audit.json"
CATALOGUE = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "era5-pds"
ASSETS = {
    "u10": "eastward_wind_at_10_metres",
    "v10": "northward_wind_at_10_metres",
}


def open_item(catalogue, month: str):
    items = list(catalogue.search(
        collections=[COLLECTION], datetime=f"{month[:4]}-{month[4:]}",
        query={"era5:kind": {"eq": "an"}},
    ).items())
    if len(items) != 1:
        raise ValueError(f"Expected one analysis item for {month}, found {len(items)}")
    return items[0]


def monthly_mean(item, month: str, variable: str) -> tuple[xr.DataArray, int]:
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{month}_{variable}.nc"
    if cached.exists():
        with xr.open_dataset(cached) as dataset:
            result = dataset[variable].load()
            hourly_count = int(dataset.attrs["hourly_count"])
        if result.sizes != {"latitude": 163, "longitude": 641} or not bool(np.isfinite(result).all()):
            raise ValueError(f"Invalid cached monthly mean: {cached}")
        print("CACHED", cached.name, flush=True)
        return result, hourly_count

    asset_name = ASSETS[variable]
    for attempt in range(1, 4):
        try:
            signed = planetary_computer.sign(item)
            asset = signed.assets[asset_name]
            kwargs = dict(asset.extra_fields["xarray:open_kwargs"])
            kwargs["chunks"] = {}
            storage_options = dict(kwargs["storage_options"])
            storage_options.update(max_concurrency=2, connection_timeout=15,
                                   read_timeout=60, timeout=60,
                                   skip_instance_cache=True)
            kwargs["storage_options"] = storage_options
            with xr.open_dataset(asset.href, **kwargs) as dataset:
                data = dataset[asset_name]
                hourly_count = int(data.sizes.get("time", 0))
                if hourly_count not in {672, 696, 720, 744}:
                    raise ValueError(f"Unexpected hourly count: {hourly_count}")
                latitude_slice = (slice(20.25, -20.25)
                                  if float(data.lat[0]) > float(data.lat[-1])
                                  else slice(-20.25, 20.25))
                subset = data.sel(lat=latitude_slice, lon=slice(120.0, 280.0))
                if subset.sizes.get("lat") != 163 or subset.sizes.get("lon") != 641:
                    raise ValueError(f"Unexpected regional shape: {dict(subset.sizes)}")
                with dask.config.set(scheduler="synchronous"):
                    mean = subset.mean("time", dtype=np.float64).astype(np.float32).compute()
            result = mean.rename({"lat": "latitude", "lon": "longitude"}).rename(variable)
            part = cached.with_suffix(".part.nc")
            part.unlink(missing_ok=True)
            result.to_dataset().assign_attrs(hourly_count=hourly_count).to_netcdf(part)
            part.replace(cached)
            return result, hourly_count
        except Exception as error:
            print("MIRROR_RETRY", month, variable, attempt, type(error).__name__, flush=True)
            if attempt == 3:
                raise
            time.sleep(attempt * 5)
    raise AssertionError("unreachable")


def region_dataset(u10: xr.DataArray, v10: xr.DataArray, month: str, region: str) -> xr.Dataset:
    longitude = u10.longitude
    if region == "west":
        selector = (longitude >= 120) & (longitude <= 180)
        u_region = u10.where(selector, drop=True)
        v_region = v10.where(selector, drop=True)
    elif region == "east":
        selector = (longitude >= 180) & (longitude <= 280)
        u_region = u10.where(selector, drop=True)
        v_region = v10.where(selector, drop=True)
        converted = ((u_region.longitude + 180) % 360) - 180
        u_region = u_region.assign_coords(longitude=converted).sortby("longitude")
        v_region = v_region.assign_coords(longitude=converted).sortby("longitude")
    else:
        raise ValueError(region)
    timestamp = np.array([np.datetime64(f"{month[:4]}-{month[4:]}-01")])
    return xr.Dataset({"u10": u_region, "v10": v_region}).expand_dims(valid_time=timestamp)


def compare(existing: Path, candidate: xr.Dataset) -> dict:
    with xr.open_dataset(existing) as reference:
        time_name = "valid_time" if "valid_time" in reference else "time"
        reference = reference[["u10", "v10"]].isel({time_name: 0}, drop=True)
        candidate = candidate.isel(valid_time=0, drop=True)
        reference = reference.sortby("latitude").sortby("longitude")
        candidate = candidate.sortby("latitude").sortby("longitude")
        reference, candidate = xr.align(reference, candidate, join="exact")
        result = {}
        for variable in ("u10", "v10"):
            difference = (candidate[variable].astype(np.float64) -
                          reference[variable].astype(np.float64)).values
            result[variable] = {
                "rmse": float(np.sqrt(np.mean(difference ** 2))),
                "max_abs": float(np.max(np.abs(difference))),
                "mean_abs": float(np.mean(np.abs(difference))),
                "points": int(difference.size),
            }
        return result


def load_audit() -> dict:
    if AUDIT.exists():
        return json.loads(AUDIT.read_text(encoding="utf-8"))
    return {
        "source": "Microsoft Planetary Computer era5-pds mirror of ECMWF ERA5",
        "method": "arithmetic mean of all hourly instantaneous u10/v10 analyses in each natural month",
        "credentials_recorded": False,
        "months": {},
    }


def save_audit(audit: dict) -> None:
    audit["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    part = AUDIT.with_suffix(".part.json")
    part.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    part.replace(AUDIT)


def run(months: tuple[str, ...], compare_only: bool) -> None:
    catalogue = pystac_client.Client.open(CATALOGUE)
    audit = load_audit()
    RAW.mkdir(parents=True, exist_ok=True)
    for month in months:
        original = open_item(catalogue, month)
        print("READING", month, "u10", flush=True)
        u10, u_hours = monthly_mean(original, month, "u10")
        print("READING", month, "v10", flush=True)
        v10, v_hours = monthly_mean(original, month, "v10")
        if u_hours != v_hours:
            raise ValueError(f"u10/v10 hourly count mismatch: {u_hours} vs {v_hours}")
        record = {
            "item_id": original.id,
            "hourly_count": u_hours,
            "regions": {},
        }
        for region in ("west", "east"):
            candidate = region_dataset(u10, v10, month, region)
            target = RAW / f"era5_monthly_{month}_{region}_uv.nc"
            if target.exists():
                validation = validate_era5_file(target, month, region)
                metrics = compare(target, candidate)
                # The CDS precomputed monthly product and a float32 hourly-field
                # arithmetic mean differ slightly through accumulation/encoding.
                # January 2018 showed ~6.8e-4 m/s RMSE and <3.2e-3 m/s max.
                if any(metrics[var]["rmse"] > 2e-3 or metrics[var]["max_abs"] > 1e-2
                       for var in ("u10", "v10")):
                    raise ValueError(f"Planetary mirror comparison failed for {month} {region}: {metrics}")
                record["regions"][region] = {
                    "status": "compared_equivalent", "comparison": metrics,
                    "existing_validation": validation,
                }
                print("EQUIVALENT", month, region, json.dumps(metrics), flush=True)
                continue
            if compare_only:
                record["regions"][region] = {"status": "missing_not_written"}
                continue
            part = target.with_suffix(".part.nc")
            part.unlink(missing_ok=True)
            candidate.to_netcdf(part)
            validation = validate_era5_file(part, month, region)
            part.replace(target)
            record["regions"][region] = {
                "status": "written_from_hourly_mirror", "bytes": target.stat().st_size,
                "validation": validation,
            }
            print("SAVED", target.name, target.stat().st_size, flush=True)
        audit["months"][month] = record
        save_audit(audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", required=True, metavar="YYYYMM")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    months = tuple(dict.fromkeys(args.months))
    if any(len(month) != 6 or not month.isdigit() or not 1 <= int(month[4:]) <= 12
           for month in months):
        parser.error("months must be YYYYMM")
    run(months, args.compare_only)
