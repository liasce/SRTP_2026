"""Resume and split the persisted ERA5 MARS monthly-mean request."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from ecmwf.datastores import Client

from download_samples import ROOT, secrets_for, validate_era5_file


AUDIT_PATH = ROOT / "research" / "audits" / "era5_2018_mars_job.json"
CACHE = ROOT / "data" / "download_cache" / "era5"
RAW = ROOT / "data" / "raw" / "era5"


def save(record: dict) -> None:
    record["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    part = AUDIT_PATH.with_suffix(".part.json")
    part.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    part.replace(AUDIT_PATH)


def coordinate_name(dataset: xr.Dataset, choices: tuple[str, ...]) -> str:
    for name in choices:
        if name in dataset.coords or name in dataset.dims:
            return name
    raise ValueError(f"Missing coordinate; expected one of {choices}")


def split(source: Path, record: dict) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(source) as dataset:
        rename = {}
        if "10u" in dataset and "u10" not in dataset:
            rename["10u"] = "u10"
        if "10v" in dataset and "v10" not in dataset:
            rename["10v"] = "v10"
        dataset = dataset.rename(rename)
        if not {"u10", "v10"}.issubset(dataset.data_vars):
            raise ValueError(f"MARS output variables are {sorted(dataset.data_vars)}")
        time_name = coordinate_name(dataset, ("valid_time", "time"))
        lon_name = coordinate_name(dataset, ("longitude", "lon"))
        lat_name = coordinate_name(dataset, ("latitude", "lat"))
        if lon_name != "longitude" or lat_name != "latitude":
            dataset = dataset.rename({lon_name: "longitude", lat_name: "latitude"})
        month_index = {
            str(np.datetime_as_string(value, unit="M")).replace("-", ""): index
            for index, value in enumerate(dataset[time_name].values)
        }
        if set(month_index) != set(record["months"]):
            raise ValueError(f"MARS output months mismatch: {sorted(month_index)}")
        longitude = dataset.longitude
        for month in record["months"]:
            monthly = dataset.isel({time_name: [month_index[month]]})
            for region in ("west", "east"):
                target = RAW / f"era5_monthly_{month}_{region}_uv.nc"
                if target.exists():
                    validate_era5_file(target, month, region)
                    print("VALID", target.name, flush=True)
                    continue
                if region == "west":
                    selected = monthly.where((longitude >= 120) & (longitude <= 180), drop=True)
                else:
                    selected = monthly.where((longitude >= 180) & (longitude <= 280), drop=True)
                    selected = selected.assign_coords(
                        longitude=((selected.longitude + 180) % 360) - 180
                    ).sortby("longitude")
                selected = selected.load()
                part = target.with_suffix(".part.nc")
                part.unlink(missing_ok=True)
                selected.to_netcdf(part)
                validate_era5_file(part, month, region)
                part.replace(target)
                print("SAVED", target.name, target.stat().st_size, flush=True)


def run(wait_minutes: int, poll_seconds: int) -> None:
    record = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    secrets = secrets_for("CDSAPI_URL", "CDSAPI_KEY")
    client = Client(url=secrets["CDSAPI_URL"], key=secrets["CDSAPI_KEY"],
                    progress=False, cleanup=False, timeout=60, sleep_max=30,
                    retry_after=10, maximum_tries=12)
    remote = client.get_remote(record["request_id"])
    deadline = time.monotonic() + wait_minutes * 60
    while True:
        status = remote.status
        record["status"] = status
        save(record)
        print("MARS_STATUS", status, flush=True)
        if status == "successful":
            break
        if status in {"failed", "rejected", "dismissed", "deleted"}:
            raise RuntimeError(f"MARS job ended with status {status}")
        if time.monotonic() >= deadline:
            return
        time.sleep(poll_seconds)

    CACHE.mkdir(parents=True, exist_ok=True)
    source = CACHE / "era5_mars_moda_201803_201812_120E_280E.nc"
    if not source.exists():
        part = source.with_suffix(".part.nc")
        remote.download(str(part))
        with xr.open_dataset(part):
            pass
        part.replace(source)
    record.update(downloaded_bytes=source.stat().st_size, downloaded_at_utc=datetime.now(timezone.utc).isoformat())
    save(record)
    split(source, record)
    record["status"] = "split_complete"
    save(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-minutes", type=int, default=180)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.wait_minutes < 1 or args.poll_seconds < 10:
        parser.error("wait must be >=1 minute and polling >=10 seconds")
    run(args.wait_minutes, args.poll_seconds)
