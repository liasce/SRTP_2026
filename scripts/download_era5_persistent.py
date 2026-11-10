"""Resume-safe ERA5 2018-03..12 download using persisted CDS request IDs.

Two regional multi-month requests are submitted.  Their public request IDs and
statuses are stored locally so a later invocation can resume polling/download
without submitting duplicate CDS jobs.  Credentials are read by the existing
download helper and are never written to the job manifest.
"""
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


DATASET = "reanalysis-era5-single-levels-monthly-means"
MONTHS = tuple(f"2018{month:02d}" for month in range(3, 13))
REGIONS = {"west": (120, 180), "east": (-180, -80)}
RAW = ROOT / "data" / "raw" / "era5"
CACHE = ROOT / "data" / "download_cache" / "era5"
AUDIT = ROOT / "research" / "audits"
MANIFEST = AUDIT / "era5_2018_persistent_jobs.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"created_at_utc": now_utc(), "credentials_recorded": False, "jobs": {}}


def save_manifest(manifest: dict) -> None:
    manifest["updated_at_utc"] = now_utc()
    AUDIT.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".part.json")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST)


def request_for(region: str) -> dict:
    west, east = REGIONS[region]
    return {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "year": ["2018"],
        "month": [month[4:] for month in MONTHS],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [20.25, west, -20.25, east],
    }


def complete_targets(region: str) -> bool:
    complete = True
    for month in MONTHS:
        target = RAW / f"era5_monthly_{month}_{region}_uv.nc"
        if not target.exists():
            complete = False
            continue
        validate_era5_file(target, month, region)
    return complete


def time_name(dataset: xr.Dataset) -> str:
    for candidate in ("valid_time", "time"):
        if candidate in dataset.dims or candidate in dataset.coords:
            return candidate
    raise ValueError("ERA5 batch lacks valid_time/time coordinate")


def split_batch(batch: Path, region: str) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(batch) as dataset:
        if not {"u10", "v10"}.issubset(dataset.data_vars):
            raise ValueError("ERA5 batch lacks u10/v10")
        coordinate = time_name(dataset)
        decoded = dataset[coordinate].values
        month_to_index = {
            str(np.datetime_as_string(value, unit="M")).replace("-", ""): index
            for index, value in enumerate(decoded)
        }
        if set(month_to_index) != set(MONTHS):
            raise ValueError(f"ERA5 batch months mismatch: {sorted(month_to_index)}")
        for month in MONTHS:
            target = RAW / f"era5_monthly_{month}_{region}_uv.nc"
            if target.exists():
                validate_era5_file(target, month, region)
                print("VALID", target.name, flush=True)
                continue
            part = target.with_suffix(".part.nc")
            part.unlink(missing_ok=True)
            selected = dataset.isel({coordinate: [month_to_index[month]]}).load()
            selected.to_netcdf(part)
            validate_era5_file(part, month, region)
            part.replace(target)
            print("SAVED", target.name, target.stat().st_size, flush=True)


def run(regions: tuple[str, ...], poll_seconds: int, wait_minutes: int) -> None:
    secrets = secrets_for("CDSAPI_URL", "CDSAPI_KEY")
    client = Client(
        url=secrets["CDSAPI_URL"], key=secrets["CDSAPI_KEY"], progress=False,
        cleanup=False, timeout=60, sleep_max=30, retry_after=10, maximum_tries=12,
    )
    manifest = load_manifest()
    CACHE.mkdir(parents=True, exist_ok=True)

    for region in regions:
        if complete_targets(region):
            print("COMPLETE", region, flush=True)
            continue
        record = manifest["jobs"].setdefault(region, {"months": list(MONTHS)})
        batch = CACHE / f"era5_monthly_201803_201812_{region}_uv.nc"
        if batch.exists():
            split_batch(batch, region)
            record.update(status="split_complete", batch_bytes=batch.stat().st_size)
            save_manifest(manifest)
            continue

        request_id = record.get("request_id")
        if request_id:
            remote = client.get_remote(request_id)
            print("RESUMED", region, request_id, flush=True)
        else:
            remote = client.submit(DATASET, request_for(region))
            record.update(request_id=remote.request_id, submitted_at_utc=now_utc(), status="submitted")
            save_manifest(manifest)
            print("SUBMITTED", region, remote.request_id, flush=True)

        deadline = time.monotonic() + wait_minutes * 60
        while True:
            status = remote.status
            record.update(status=status, last_checked_at_utc=now_utc())
            save_manifest(manifest)
            print("STATUS", region, status, flush=True)
            if status == "successful":
                break
            if status in {"failed", "rejected", "dismissed", "deleted"}:
                raise RuntimeError(f"CDS job {region} ended with status {status}")
            if time.monotonic() >= deadline:
                print("PENDING", region, remote.request_id, flush=True)
                return
            time.sleep(poll_seconds)

        part = batch.with_suffix(".part.nc")
        part.unlink(missing_ok=True)
        remote.download(str(part))
        with xr.open_dataset(part):
            pass
        part.replace(batch)
        record.update(status="downloaded", downloaded_at_utc=now_utc(), batch_bytes=batch.stat().st_size)
        save_manifest(manifest)
        split_batch(batch, region)
        record.update(status="split_complete", split_at_utc=now_utc())
        save_manifest(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", nargs="+", choices=tuple(REGIONS), default=list(REGIONS))
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--wait-minutes", type=int, default=30)
    arguments = parser.parse_args()
    if arguments.poll_seconds < 5 or arguments.wait_minutes < 1:
        parser.error("poll interval must be >=5 seconds and wait must be >=1 minute")
    run(tuple(dict.fromkeys(arguments.regions)), arguments.poll_seconds, arguments.wait_minutes)
