"""Download requested months with restart-safe validation.

The ``argo`` command keeps the historical 50-file-per-month audit sample.
Use ``argo-index`` followed by ``argo-full`` for the complete regional profile
inventory in the requested months.
"""
from pathlib import Path
import argparse
import calendar
import concurrent.futures
import csv
import gzip
import hashlib
import json
import multiprocessing
import os
import re
import shlex
import threading
import time
from datetime import datetime, timezone

import requests
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
AUDIT = ROOT / "research/audits"
DEFAULT_MONTHS = ("201801", "201807")
MONTHS = DEFAULT_MONTHS
REGIONS = ("west", "east")
ERA5_ATTEMPT_TIMEOUT = 1800
ERA5_MAX_ATTEMPTS = 3
NETCDF_LOCK = threading.Lock()  # NetCDF C-library validation is serialized; HTTP transfers remain parallel.


def secrets_for(*keys):
    result = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() in keys:
            parts = shlex.split(value, comments=True)
            if len(parts) != 1:
                raise ValueError("Credential must be one nonempty quoted or unquoted value")
            result[key.strip()] = parts[0]
    if set(result) != set(keys):
        raise ValueError("Required credentials missing from local .env")
    return result


def fetch(url, path, session=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.suffix == ".nc":
            with NETCDF_LOCK, xr.open_dataset(path, decode_times=False):
                pass
        print("EXISTS", path.relative_to(ROOT), flush=True)
        return path
    part = path.with_suffix(path.suffix + ".part")
    client = session or requests
    for attempt in range(3):
        try:
            with client.get(url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                size = 0
                with part.open("wb") as out:
                    for chunk in response.iter_content(1024 * 1024):
                        size += len(chunk)
                        if size > 1024**3:
                            raise ValueError("Sample file exceeds 1 GiB safety limit")
                        out.write(chunk)
            if path.suffix == ".nc":
                with NETCDF_LOCK, xr.open_dataset(part, decode_times=False):
                    pass
            elif path.suffix == ".json":
                json.loads(part.read_text())
            part.rename(path)
            print("SAVED", path.relative_to(ROOT), size, flush=True)
            return path
        except (requests.RequestException, OSError, ValueError):
            part.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def argo_list():
    """Fixed spatially stratified index sample, without selecting on observed QC."""
    source = RAW / "profiles/ar_index_global_prof.txt.gz"
    candidates = {m: [] for m in MONTHS}
    with gzip.open(source, "rt") as stream:
        rows = csv.DictReader(line for line in stream if not line.startswith("#"))
        for row in rows:
            month = row["date"][:6]
            if month not in candidates or row["ocean"] != "P":
                continue
            try:
                lat, lon = float(row["latitude"]), float(row["longitude"]) % 360
            except ValueError:
                continue
            if -20 <= lat <= 20 and 120 <= lon <= 280:
                candidates[month].append(row)
    selected = []
    for month, rows in candidates.items():
        bins = {}
        for row in sorted(rows, key=lambda r: hashlib.sha256(r["file"].encode()).hexdigest()):
            key = (int((float(row["latitude"]) + 20) // 10),
                   int(((float(row["longitude"]) % 360) - 120) // 10))
            bins.setdefault(key, []).append(row)
        used, sample = set(), []
        while len(sample) < 50 and any(bins.values()):
            for key in sorted(bins):
                if not bins[key] or len(sample) == 50:
                    continue
                row = bins[key].pop()
                wmo = row["file"].split("/")[1]
                if wmo not in used:
                    used.add(wmo)
                    sample.append(row)
        assert len(sample) == 50
        selected.extend(sample)
        print("Argo index candidates", month, len(rows), "selected", len(sample), flush=True)
    assert len({r["file"] for r in selected}) == 50 * len(MONTHS)
    AUDIT.mkdir(parents=True, exist_ok=True)
    suffix = "" if MONTHS == DEFAULT_MONTHS else "_" + "_".join(MONTHS)
    target = AUDIT / f"argo_sample_candidates{suffix}.csv"
    if target.exists():
        with target.open() as stream:
            assert list(csv.DictReader(stream)) == selected, "Do not silently change fixed sample"
    else:
        with target.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
    return selected


def argo_index():
    """Download the official GDAC profile index used to build full inventories."""
    fetch(
        "https://data-argo.ifremer.fr/ar_index_global_prof.txt.gz",
        RAW / "profiles/ar_index_global_prof.txt.gz",
    )


def argo_full_list():
    """Return every unique Pacific profile file in the requested region/months."""
    source = RAW / "profiles/ar_index_global_prof.txt.gz"
    if not source.exists():
        raise FileNotFoundError("Run argo-index before argo-full")
    selected = {}
    with gzip.open(source, "rt") as stream:
        rows = csv.DictReader(line for line in stream if not line.startswith("#"))
        for row in rows:
            month = row["date"][:6]
            if month not in MONTHS or row["ocean"] != "P":
                continue
            try:
                lat, lon = float(row["latitude"]), float(row["longitude"]) % 360
            except ValueError:
                continue
            if -20 <= lat <= 20 and 120 <= lon <= 280:
                selected[row["file"]] = row
    counts = {month: 0 for month in MONTHS}
    for row in selected.values():
        counts[row["date"][:6]] += 1
    missing = [month for month, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"No Argo index candidates for months: {missing}")
    print("Argo full index", counts, "unique files", len(selected), flush=True)
    return list(selected.values())


def argo():
    # Official GDAC daily mirror; keep the same frozen profile paths.
    jobs = [("https://argo-gdac-sandbox.s3.eu-west-3.amazonaws.com/pub/dac/" + row["file"],
             RAW / "argo" / row["file"]) for row in argo_list()]
    run_downloads(jobs)


def argo_full():
    jobs = [("https://argo-gdac-sandbox.s3.eu-west-3.amazonaws.com/pub/dac/" + row["file"],
             RAW / "argo" / row["file"]) for row in argo_full_list()]
    run_downloads(jobs, max_workers=64)


def run_downloads(jobs, max_workers=4):
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending = {pool.submit(fetch, url, path): path for url, path in jobs}
        for future in concurrent.futures.as_completed(pending):
            try:
                future.result()
            except Exception as error:
                path = str(pending[future].relative_to(ROOT))
                failures.append({"path": path, "error": type(error).__name__})
                print("FAILED", path, type(error).__name__, flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} sample downloads failed; rerun skips valid files")


def oisst():
    jobs = []
    for month in MONTHS:
        for day in range(1, calendar.monthrange(int(month[:4]), int(month[4:]))[1] + 1):
            filename = f"oisst-avhrr-v02r01.{month}{day:02d}.nc"
            jobs.append((f"https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/{month}/{filename}",
                         RAW / "oisst" / month / filename))
    run_downloads(jobs)


def tao():
    for dataset, variables in [
        ("pmelTaoDyT", "array,station,wmo_platform_code,time,latitude,longitude,depth,T_20,QT_5020,ST_6020"),
        ("pmelTaoDyS", "array,station,wmo_platform_code,time,latitude,longitude,depth,S_41,QS_5041,SS_6041,SIC_8041"),
    ]:
        base = "https://data.pmel.noaa.gov/generic/erddap"
        fetch(f"{base}/info/{dataset}/index.json", RAW / "tao_daily" / f"{dataset}_metadata.json")
        for month in MONTHS:
            start = f"{month[:4]}-{month[4:]}-01T00:00:00Z"
            year, number = int(month[:4]), int(month[4:])
            next_year, next_month = (year + 1, 1) if number == 12 else (year, number + 1)
            end = f"{next_year:04d}-{next_month:02d}-01T00:00:00Z"
            # Keep original values and QC flags; filtering by QC happens in the audited pipeline.
            query = (f'{variables}&array="TAO/TRITON"&time>={start}&time<{end}'
                     '&longitude>=120&longitude<=280&latitude>=-20&latitude<=20&depth<=510')
            fetch(f"{base}/tabledap/{dataset}.nc?{requests.utils.requote_uri(query)}",
                  RAW / "tao_daily" / f"{dataset}_{month}.nc")


def cora():
    base = ("https://s3.waw3-1.cloudferro.com/mdl-native-03/native/"
            "INSITU_GLO_PHY_TS_OA_MY_013_052/"
            "cmems_obs-ins_glo_phy-temp-sal_my_cora-oa_P1M_202603")
    jobs = []
    for month in MONTHS:
        year = month[:4]
        for folder, kind in [("field", "fld"), ("data", "dat")]:
            for variable in ["TEMP", "PSAL"]:
                name = f"OA_CORA5.2_{month}15_{kind}_{variable}.nc"
                jobs.append((f"{base}/{folder}/{year}/{name}",
                             RAW / "cora/original_202603" / name))
    # The CORA object store throttles multiple long-lived field/data transfers.
    run_downloads(jobs, max_workers=1)


def adt():
    import copernicusmarine
    secrets = secrets_for("COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD")
    for month in MONTHS:
        for region, west, east in [("west", 120, 180), ("east", -180, -80)]:
            directory = RAW / "adt"
            filename = f"duacs_202411_{month}_{region}_adt.nc"
            if (directory / filename).exists():
                with xr.open_dataset(directory / filename):
                    pass
                continue
            directory.mkdir(parents=True, exist_ok=True)
            # Keep incomplete SDK output separate from validated files.
            part = filename + ".part.nc"
            copernicusmarine.subset(
                dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
                dataset_version="202411", variables=["adt"],
                username=secrets["COPERNICUSMARINE_SERVICE_USERNAME"],
                password=secrets["COPERNICUSMARINE_SERVICE_PASSWORD"],
                minimum_longitude=west, maximum_longitude=east,
                minimum_latitude=-20.125, maximum_latitude=20.125,
                start_datetime=f"{month[:4]}-{month[4:]}-01T00:00:00",
                end_datetime=f"{month[:4]}-{month[4:]}-{calendar.monthrange(int(month[:4]), int(month[4:]))[1]:02d}T23:59:59",
                output_directory=directory, output_filename=part,
                overwrite=True, disable_progress_bar=True,
                netcdf_compression_level=1)
            with xr.open_dataset(directory / part) as ds:
                expected_days = calendar.monthrange(int(month[:4]), int(month[4:]))[1]
                assert ds.sizes["time"] == expected_days and "adt" in ds
            (directory / part).rename(directory / filename)
            print("SAVED", filename, flush=True)


def smap():
    import earthaccess
    os.environ.update(secrets_for("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"))
    earthaccess.login(strategy="environment", persist=False)
    granules = []
    for month in MONTHS:
        found = earthaccess.search_data(
            short_name="SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V6", count=100,
            temporal=(f"{month[:4]}-{month[4:]}-01", f"{month[:4]}-{month[4:]}-{calendar.monthrange(int(month[:4]), int(month[4:]))[1]:02d}T23:59:59"))
        selected = []
        for granule in found:
            links = granule.data_links()
            match = re.search(r"running_(\d{4}_\d{3})_FNL", " ".join(links))
            if match and datetime.strptime(match[1], "%Y_%j").strftime("%Y%m") == month:
                selected.append(granule)
        expected_days = calendar.monthrange(int(month[:4]), int(month[4:]))[1]
        assert len(selected) == expected_days, "Unexpected SMAP nominal-date coverage"
        granules.extend(selected)
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "smap_sample_granules.json").write_text(json.dumps(
        [dict(g) for g in granules], ensure_ascii=False, indent=2))
    directory = RAW / "smap"
    directory.mkdir(parents=True, exist_ok=True)
    files = earthaccess.download(granules, local_path=directory, threads=4, show_progress=False)
    for path in files:
        with xr.open_dataset(path, decode_times=False):
            pass
    print("SMAP verified", len(files), flush=True)


def validate_era5_file(path, month, region):
    """Reject incomplete, wrong-month, or wrong-domain ERA5 monthly files."""
    expected = {
        "west": {"latitude": (-20.25, 20.25), "longitude": (120.0, 180.0)},
        "east": {"latitude": (-20.25, 20.25), "longitude": (-180.0, -80.0)},
    }
    if region not in expected:
        raise ValueError(f"Unknown ERA5 region: {region}")
    with xr.open_dataset(path) as ds:
        if not {"u10", "v10"}.issubset(ds.data_vars):
            raise ValueError("ERA5 file lacks u10 or v10")
        time_name = "valid_time" if "valid_time" in ds else "time" if "time" in ds else None
        if time_name is None or ds[time_name].size != 1:
            raise ValueError("ERA5 file must contain exactly one monthly timestamp")
        timestamp = np.asarray(ds[time_name].values).reshape(-1)[0]
        if str(timestamp)[:7].replace("-", "") != month:
            raise ValueError("ERA5 timestamp does not match requested month")
        for coordinate in ["latitude", "longitude"]:
            if coordinate not in ds.coords or ds[coordinate].size < 2:
                raise ValueError(f"ERA5 file lacks usable {coordinate}")
            actual = (float(ds[coordinate].min()), float(ds[coordinate].max()))
            if not np.allclose(actual, expected[region][coordinate], atol=1e-6, rtol=0):
                raise ValueError(f"ERA5 {coordinate} range does not match requested region")
        for variable in ["u10", "v10"]:
            if not bool(np.isfinite(ds[variable].values).any()):
                raise ValueError(f"ERA5 {variable} is entirely missing")
        return {
            "variables": ["u10", "v10"],
            "sizes": {name: int(size) for name, size in ds.sizes.items()},
            "time": str(timestamp),
            "latitude_range": [float(ds.latitude.min()), float(ds.latitude.max())],
            "longitude_range": [float(ds.longitude.min()), float(ds.longitude.max())],
        }


def write_era5_download_audit(records):
    AUDIT.mkdir(parents=True, exist_ok=True)
    target = AUDIT / "era5_2018_download_attempts.json"
    existing = []
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8")).get("records", [])
        except (OSError, json.JSONDecodeError):
            existing = []
    merged = {(record["month"], record["region"]): record for record in existing}
    merged.update({(record["month"], record["region"]): record for record in records})
    target.write_text(json.dumps({
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "credential_values_recorded": False,
        "records": [merged[key] for key in sorted(merged)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def classify_cds_error(response):
    if response is None:
        return None
    try:
        text = response.text.lower()
    except Exception:
        return "unreadable_response"
    if any(word in text for word in ("licence", "license", "terms of use", "terms-and-conditions", "agree")):
        return "licence_not_accepted"
    if any(word in text for word in ("too many requests", "rate limit", "quota", "cost limit")):
        return "rate_or_cost_limit"
    if any(word in text for word in ("invalid request", "invalid field", "invalid value", "validation error")):
        return "request_validation"
    if getattr(response, "status_code", None) in (401, 403):
        return "authentication_or_authorization"
    if getattr(response, "status_code", 0) >= 500:
        return "server_error"
    return "other_http_error"


class Era5WorkerError(RuntimeError):
    def __init__(self, remote_type, http_status=None, error_category=None):
        super().__init__(remote_type)
        self.remote_type = remote_type
        self.http_status = http_status
        self.error_category = error_category


def _era5_retrieve_worker(url, key, request, part, status_path):
    import cdsapi
    try:
        client = cdsapi.Client(url=url, key=key, timeout=60, retry_max=0,
                               sleep_max=10, quiet=True)
        client.retrieve("reanalysis-era5-single-levels-monthly-means", request, str(part))
        status = {"status": "ok"}
    except Exception as error:
        response = getattr(error, "response", None)
        status = {"status": "error", "exception_type": type(error).__name__,
                  "http_status": getattr(response, "status_code", None),
                  "error_category": classify_cds_error(response)}
    Path(status_path).write_text(json.dumps(status), encoding="utf-8")


def retrieve_era5_once(url, key, request, part, timeout_seconds):
    status_path = Path(str(part) + ".attempt.json")
    status_path.unlink(missing_ok=True)
    process = multiprocessing.get_context("spawn").Process(
        target=_era5_retrieve_worker,
        args=(url, key, request, str(part), str(status_path)))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        status_path.unlink(missing_ok=True)
        raise TimeoutError(f"CDS retrieval exceeded {timeout_seconds} seconds")
    if not status_path.exists():
        raise RuntimeError("CDS retrieval worker exited without status")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status_path.unlink(missing_ok=True)
    if status.get("status") != "ok":
        raise Era5WorkerError(status.get("exception_type", "UnknownError"),
                              status.get("http_status"), status.get("error_category"))


def era5():
    secrets = secrets_for("CDSAPI_URL", "CDSAPI_KEY")
    if secrets["CDSAPI_URL"].rstrip("/") != "https://cds.climate.copernicus.eu/api":
        raise ValueError("CDS URL must be the official service")
    directory = RAW / "era5"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    region_bounds = {"west": (120, 180), "east": (-180, -80)}
    for month in MONTHS:
        for region in REGIONS:
            west, east = region_bounds[region]
            target = directory / f"era5_monthly_{month}_{region}_uv.nc"
            if target.exists():
                validation = validate_era5_file(target, month, region)
                records.append({"month": month, "region": region, "status": "skipped_valid",
                                "attempts": 0, "bytes": target.stat().st_size,
                                "validation": validation})
                write_era5_download_audit(records)
                print("VALID", target.name, flush=True)
                continue
            request = {"product_type": ["monthly_averaged_reanalysis"],
                       "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                       "year": [month[:4]], "month": [month[4:]], "time": ["00:00"],
                       "data_format": "netcdf", "download_format": "unarchived",
                       "area": [20.25, west, -20.25, east]}
            AUDIT.mkdir(parents=True, exist_ok=True)
            (AUDIT / f"era5_sample_request_{month}_{region}.json").write_text(
                json.dumps(request, indent=2), encoding="utf-8")
            part = target.with_suffix(".part.nc")
            failures = []
            for attempt in range(1, ERA5_MAX_ATTEMPTS + 1):
                try:
                    retrieve_era5_once(
                        secrets["CDSAPI_URL"], secrets["CDSAPI_KEY"], request, part,
                        ERA5_ATTEMPT_TIMEOUT)
                    validation = validate_era5_file(part, month, region)
                    part.replace(target)
                    records.append({"month": month, "region": region, "status": "downloaded",
                                    "attempts": attempt, "bytes": target.stat().st_size,
                                    "validation": validation, "failures": failures})
                    write_era5_download_audit(records)
                    print("SAVED", target.name, flush=True)
                    break
                except Exception as error:
                    exception_type = getattr(error, "remote_type", type(error).__name__)
                    failure = {"attempt": attempt, "exception_type": exception_type,
                               "http_status": getattr(error, "http_status", None),
                               "error_category": getattr(error, "error_category", None),
                               "partial_file_present": part.exists(),
                               "partial_bytes": part.stat().st_size if part.exists() else 0}
                    failures.append(failure)
                    print("RETRY", month, region, attempt, exception_type,
                          failure["http_status"], failure["error_category"], flush=True)
                    if attempt < ERA5_MAX_ATTEMPTS:
                        time.sleep(attempt * 2)
            else:
                records.append({"month": month, "region": region, "status": "failed",
                                "attempts": ERA5_MAX_ATTEMPTS, "failures": failures})
                write_era5_download_audit(records)
                print("FAILED_TARGET", month, region, failures[-1]["exception_type"], flush=True)
    failed = [record for record in records if record["status"] == "failed"]
    if failed:
        raise RuntimeError(f"{len(failed)} ERA5 targets failed after three attempts")


def audit():
    import numpy as np
    records = []
    for directory in ["argo", "oisst", "tao_daily", "cora", "adt", "smap", "era5"]:
        for path in sorted((RAW / directory).rglob("*.nc")):
            if ".part." in path.name:
                continue
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            record = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                      "sha256": digest}
            if directory == "argo":
                record["source"] = "https://argo-gdac-sandbox.s3.eu-west-3.amazonaws.com/pub/dac/" + path.relative_to(RAW / "argo").as_posix()
            elif directory == "oisst":
                record["source"] = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/" + path.parent.name + "/" + path.name
            elif directory == "cora" and path.parent.name == "original_202603":
                folder = "field" if "_fld_" in path.name else "data"
                record["source"] = "https://s3.waw3-1.cloudferro.com/mdl-native-03/native/INSITU_GLO_PHY_TS_OA_MY_013_052/cmems_obs-ins_glo_phy-temp-sal_my_cora-oa_P1M_202603/" + folder + "/" + path.name.split("_")[2][:4] + "/" + path.name
            elif directory == "adt":
                record["source"] = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D; version=202411; variable=adt; selection in download_samples.py"
            elif directory == "smap":
                record["source"] = "https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V6/" + path.name
            with xr.open_dataset(path, decode_times=False) as ds:
                record["sizes"] = dict(ds.sizes)
                record["variables"] = {}
                for name, var in ds.variables.items():
                    info = {"dims": list(var.dims), "units": var.attrs.get("units"),
                            "long_name": var.attrs.get("long_name"), "dtype": str(var.dtype)}
                    measured = ds[name]
                    if "_fld_" in path.name and name in ds.data_vars:
                        measured = measured.sel(depth=slice(1, 510), latitude=slice(-20.5, 20.5))
                        measured = measured.sel(longitude=(measured.longitude >= 119.5) | (measured.longitude <= -79.5))
                        info["statistics_scope"] = "tropical Pacific rectangle with buffer, 1-510 m; original full-file dimensions retained above"
                    if np.issubdtype(var.dtype, np.number) and var.size:
                        values = measured.values
                        finite = np.isfinite(values)
                        info["statistics_count"] = int(values.size)
                        info["finite_fraction"] = float(finite.mean())
                        if finite.any():
                            info["min"] = float(values[finite].min())
                            info["max"] = float(values[finite].max())
                    elif np.issubdtype(var.dtype, np.datetime64) and var.size:
                        info.update(min=str(var.min().values), max=str(var.max().values))
                    record["variables"][name] = info
                record["global_attributes"] = {k: str(v) for k, v in ds.attrs.items()}
            records.append(record)
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "sample_file_audit.json").write_text(json.dumps(
        {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "files": records},
        ensure_ascii=False, indent=2))
    print("AUDITED", len(records), "files; bytes", sum(r["bytes"] for r in records), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["argo-index", "argo-list", "argo", "argo-full", "oisst", "tao", "cora", "adt", "smap", "era5", "audit"])
    parser.add_argument("--months", nargs="+", default=list(DEFAULT_MONTHS), metavar="YYYYMM")
    parser.add_argument("--regions", nargs="+", choices=["west", "east"],
                        default=["west", "east"], help="ERA5 regions; ignored by other sources")
    parser.add_argument("--era5-attempt-timeout", type=int, default=1800, metavar="SECONDS",
                        help="Hard wall-clock limit for each ERA5 CDS attempt")
    parser.add_argument("--era5-max-attempts", type=int, default=3, choices=[1, 2, 3],
                        help="Maximum attempts for each ERA5 month-region target")
    args = parser.parse_args()
    if len(args.months) > 12 or any(not re.fullmatch(r"\d{6}", month) or not 1 <= int(month[4:]) <= 12 for month in args.months):
        parser.error("--months requires at most twelve valid YYYYMM months")
    if args.era5_attempt_timeout < 60:
        parser.error("--era5-attempt-timeout must be at least 60 seconds")
    MONTHS = tuple(dict.fromkeys(args.months))
    REGIONS = tuple(dict.fromkeys(args.regions))
    ERA5_ATTEMPT_TIMEOUT = args.era5_attempt_timeout
    ERA5_MAX_ATTEMPTS = args.era5_max_attempts
    try:
        {"argo-index": argo_index, "argo-list": argo_list, "argo": argo, "argo-full": argo_full,
         "oisst": oisst, "tao": tao,
         "cora": cora, "adt": adt, "smap": smap, "era5": era5, "audit": audit}[args.source]()
    except Exception as error:
        # Authenticated service exceptions can include signed URLs: never echo them.
        if args.source not in ["adt", "smap", "era5"]:
            raise
        print("FAILED", args.source, type(error).__name__, flush=True)
        raise SystemExit(1)
