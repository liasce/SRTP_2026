"""Recover existing CDS jobs and sequentially complete ERA5 2018-03..12.

The CDS currently limits queued requests for this dataset.  This entry point
first discovers already accepted/running/successful monthly jobs, downloads
their results, and only submits a new month-region request when no desired job
is active.  Request IDs and non-sensitive statuses are persisted for restart.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ecmwf.datastores import Client

from download_samples import ROOT, secrets_for, validate_era5_file


DATASET = "reanalysis-era5-single-levels-monthly-means"
MONTHS = tuple(f"2018{month:02d}" for month in range(3, 13))
REGIONS = {"west": (120, 180), "east": (-180, -80)}
RAW = ROOT / "data" / "raw" / "era5"
AUDIT = ROOT / "research" / "audits"
MANIFEST = AUDIT / "era5_2018_recovered_jobs.json"
ACTIVE = {"accepted", "running"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def target_path(month: str, region: str) -> Path:
    return RAW / f"era5_monthly_{month}_{region}_uv.nc"


def request_for(month: str, region: str) -> dict:
    west, east = REGIONS[region]
    return {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "year": [month[:4]], "month": [month[4:]], "time": ["00:00"],
        "data_format": "netcdf", "download_format": "unarchived",
        "area": [20.25, west, -20.25, east],
    }


def identify(request: dict) -> tuple[str, str] | None:
    years = request.get("year") or []
    months = request.get("month") or []
    area = request.get("area")
    variables = set(request.get("variable") or [])
    if len(years) != 1 or len(months) != 1:
        return None
    month = f"{years[0]}{months[0]}"
    if month not in MONTHS or variables != {
        "10m_u_component_of_wind", "10m_v_component_of_wind"
    }:
        return None
    for region, (west, east) in REGIONS.items():
        if area == [20.25, west, -20.25, east]:
            return month, region
    return None


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"created_at_utc": now_utc(), "credentials_recorded": False, "targets": {}}


def save_manifest(manifest: dict) -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    manifest["updated_at_utc"] = now_utc()
    part = MANIFEST.with_suffix(".part.json")
    part.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    part.replace(MANIFEST)


def existing_complete() -> set[tuple[str, str]]:
    complete = set()
    for month in MONTHS:
        for region in REGIONS:
            target = target_path(month, region)
            if target.exists():
                validate_era5_file(target, month, region)
                complete.add((month, region))
    return complete


def discover(client: Client, manifest: dict) -> dict[tuple[str, str], tuple[object, str]]:
    result = {}
    jobs = client.get_jobs(limit=100, sortby="-created")._json_dict.get("jobs", [])
    for job in jobs:
        if job.get("processID") != DATASET or job.get("status") not in ACTIVE | {"successful"}:
            continue
        remote = client.get_remote(job["jobID"])
        key = identify(remote.request)
        if key is None or key in result:
            continue
        status = remote.status
        result[key] = (remote, status)
        name = f"{key[0]}_{key[1]}"
        manifest["targets"][name] = {
            "request_id": remote.request_id, "status": status,
            "last_checked_at_utc": now_utc(),
        }
    save_manifest(manifest)
    return result


def download_result(remote, month: str, region: str, manifest: dict) -> None:
    target = target_path(month, region)
    part = target.with_suffix(".part.nc")
    part.unlink(missing_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    remote.download(str(part))
    validation = validate_era5_file(part, month, region)
    part.replace(target)
    manifest["targets"][f"{month}_{region}"].update(
        status="downloaded", downloaded_at_utc=now_utc(), bytes=target.stat().st_size,
        validation=validation,
    )
    save_manifest(manifest)
    print("SAVED", target.name, target.stat().st_size, flush=True)


def run(poll_seconds: int, wait_minutes: int) -> None:
    secrets = secrets_for("CDSAPI_URL", "CDSAPI_KEY")
    client = Client(
        url=secrets["CDSAPI_URL"], key=secrets["CDSAPI_KEY"], progress=False,
        cleanup=False, timeout=60, sleep_max=30, retry_after=10, maximum_tries=12,
    )
    manifest = load_manifest()
    deadline = time.monotonic() + wait_minutes * 60

    while True:
        complete = existing_complete()
        if len(complete) == len(MONTHS) * len(REGIONS):
            print("ERA5_COMPLETE", len(complete), flush=True)
            return

        jobs = discover(client, manifest)
        progressed = False
        for key, (remote, status) in jobs.items():
            if key not in complete and status == "successful":
                download_result(remote, *key, manifest)
                progressed = True
        if progressed:
            continue

        active = [(key, remote.request_id, status) for key, (remote, status) in jobs.items()
                  if key not in complete and status in ACTIVE]
        if active:
            print("ACTIVE", json.dumps(active), flush=True)
        else:
            missing = [(month, region) for month in MONTHS for region in REGIONS
                       if (month, region) not in complete]
            month, region = missing[0]
            remote = client.submit(DATASET, request_for(month, region))
            manifest["targets"][f"{month}_{region}"] = {
                "request_id": remote.request_id, "status": "submitted",
                "submitted_at_utc": now_utc(),
            }
            save_manifest(manifest)
            print("SUBMITTED", month, region, remote.request_id, flush=True)

        if time.monotonic() >= deadline:
            print("WAIT_LIMIT_REACHED", len(complete), flush=True)
            return
        time.sleep(poll_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--wait-minutes", type=int, default=180)
    arguments = parser.parse_args()
    if arguments.poll_seconds < 10 or arguments.wait_minutes < 1:
        parser.error("poll interval must be >=10 seconds and wait must be >=1 minute")
    run(arguments.poll_seconds, arguments.wait_minutes)
