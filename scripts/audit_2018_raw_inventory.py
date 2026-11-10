#!/usr/bin/env python3
"""Build a bounded, machine-readable inventory for 2018 common-modality raw data."""
from __future__ import annotations

import calendar
import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
AUDIT = ROOT / "research/audits"
MONTHS = [f"2018{month:02d}" for month in range(1, 13)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_range(variable):
    values = np.ma.filled(variable[:], np.nan)
    if not np.issubdtype(values.dtype, np.number):
        return None
    finite = np.isfinite(values)
    if not finite.any():
        return None
    return [float(values[finite].min()), float(values[finite].max())]


def time_coverage(variable):
    values = np.ma.compressed(variable[:])
    if not len(values):
        return None
    units = getattr(variable, "units", None)
    calendar_name = getattr(variable, "calendar", "standard")
    if units:
        try:
            decoded = num2date(values, units, calendar=calendar_name,
                               only_use_cftime_datetimes=False,
                               only_use_python_datetimes=False)
            text = sorted(str(value) for value in np.asarray(decoded).reshape(-1))
            return {"start": text[0], "end": text[-1], "count": len(text), "units": units}
        except Exception as error:
            return {"raw_range": [float(np.min(values)), float(np.max(values))],
                    "count": int(len(values)), "units": units,
                    "decode_error": type(error).__name__}
    return {"raw_range": [float(np.min(values)), float(np.max(values))], "count": int(len(values))}


def inspect_netcdf(path: Path, source: str):
    names = {
        "cora": {"time": ["time", "JULD"], "lat": ["latitude", "LATITUDE"], "lon": ["longitude", "LONGITUDE"]},
        "oisst": {"time": ["time"], "lat": ["lat"], "lon": ["lon"]},
        "adt": {"time": ["time"], "lat": ["latitude"], "lon": ["longitude"]},
        "era5": {"time": ["valid_time", "time"], "lat": ["latitude"], "lon": ["longitude"]},
        "argo": {"time": ["JULD"], "lat": ["LATITUDE"], "lon": ["LONGITUDE"]},
    }[source]
    with Dataset(path) as ds:
        variables = sorted(ds.variables)
        dimensions = {name: len(dimension) for name, dimension in ds.dimensions.items()}
        selected = {}
        for role, candidates in names.items():
            selected[role] = next((name for name in candidates if name in ds.variables), None)
        return {
            "variables": variables,
            "dimensions": dimensions,
            "time_coverage": time_coverage(ds[selected["time"]]) if selected["time"] else None,
            "latitude_range": finite_range(ds[selected["lat"]]) if selected["lat"] else None,
            "longitude_range": finite_range(ds[selected["lon"]]) if selected["lon"] else None,
        }


def expected_argo():
    selected = {}
    source = RAW / "profiles/ar_index_global_prof.txt.gz"
    with gzip.open(source, "rt") as stream:
        rows = csv.DictReader(line for line in stream if not line.startswith("#"))
        for row in rows:
            month = row["date"][:6]
            if month not in MONTHS or row["ocean"] != "P":
                continue
            try:
                latitude = float(row["latitude"])
                longitude = float(row["longitude"]) % 360
            except ValueError:
                continue
            if -20 <= latitude <= 20 and 120 <= longitude <= 280:
                selected[row["file"]] = month
    return selected


def discover(expected_argo_paths):
    records = []
    cora_pattern = re.compile(r"OA_CORA5\.2_(2018\d{2})15_(fld|dat)_(TEMP|PSAL)\.nc$")
    for path in sorted((RAW / "cora/original_202603").glob("OA_CORA5.2_2018*.nc")):
        match = cora_pattern.match(path.name)
        if match:
            records.append(("cora", match.group(1), path, {"kind": match.group(2), "variable": match.group(3)}))
    for path in sorted((RAW / "oisst").glob("2018??/oisst-avhrr-v02r01.2018????.nc")):
        date = re.search(r"(2018\d{4})\.nc$", path.name).group(1)
        records.append(("oisst", date[:6], path, {"date": date}))
    for path in sorted((RAW / "adt").glob("duacs_202411_2018??_*_adt.nc")):
        match = re.match(r"duacs_202411_(2018\d{2})_(west|east)_adt\.nc$", path.name)
        if match:
            records.append(("adt", match.group(1), path, {"region": match.group(2)}))
    for path in sorted((RAW / "era5").glob("era5_monthly_2018??_*_uv.nc")):
        match = re.match(r"era5_monthly_(2018\d{2})_(west|east)_uv\.nc$", path.name)
        if match:
            records.append(("era5", match.group(1), path, {"region": match.group(2)}))
    for relative, month in sorted(expected_argo_paths.items()):
        path = RAW / "argo" / Path(relative)
        if path.exists():
            records.append(("argo", month, path, {"index_relative_path": relative}))
    return records


def expected_checks(discovered, argo_expected):
    by_source = defaultdict(list)
    for source, month, path, details in discovered:
        by_source[source].append((month, path, details))

    expected_dates = {
        f"2018{month:02d}{day:02d}"
        for month in range(1, 13)
        for day in range(1, calendar.monthrange(2018, month)[1] + 1)
    }
    oisst_dates = [details["date"] for _, _, details in by_source["oisst"]]
    cora_keys = [(month, details["kind"], details["variable"]) for month, _, details in by_source["cora"]]
    adt_keys = [(month, details["region"]) for month, _, details in by_source["adt"]]
    era5_keys = [(month, details["region"]) for month, _, details in by_source["era5"]]
    expected_cora = {(month, kind, variable) for month in MONTHS for kind in ("fld", "dat") for variable in ("TEMP", "PSAL")}
    expected_regions = {(month, region) for month in MONTHS for region in ("west", "east")}
    actual_argo = {details["index_relative_path"] for _, _, details in by_source["argo"]}
    return {
        "cora": {"expected": 48, "actual": len(cora_keys),
                 "missing": sorted(expected_cora - set(cora_keys)),
                 "duplicates": [list(key) for key, count in Counter(cora_keys).items() if count > 1]},
        "oisst": {"expected": 365, "actual": len(oisst_dates),
                  "missing_dates": sorted(expected_dates - set(oisst_dates)),
                  "duplicate_dates": sorted(date for date, count in Counter(oisst_dates).items() if count > 1)},
        "adt": {"expected": 24, "actual": len(adt_keys),
                "missing": sorted(expected_regions - set(adt_keys)),
                "duplicates": [list(key) for key, count in Counter(adt_keys).items() if count > 1]},
        "era5": {"expected": 24, "actual": len(era5_keys),
                 "missing": sorted(expected_regions - set(era5_keys)),
                 "duplicates": [list(key) for key, count in Counter(era5_keys).items() if count > 1]},
        "argo": {"expected_unique_relative_paths": len(argo_expected), "actual": len(actual_argo),
                 "missing_relative_paths": sorted(set(argo_expected) - actual_argo),
                 "unexpected_relative_paths": sorted(
                     path.relative_to(RAW / "argo").as_posix()
                     for path in (RAW / "argo").rglob("*.nc")
                     if path.relative_to(RAW / "argo").as_posix() not in argo_expected)},
    }


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    argo_expected = expected_argo()
    discovered = discover(argo_expected)
    checks = expected_checks(discovered, argo_expected)
    sizes = defaultdict(list)
    unreadable = []
    output = AUDIT / "raw_2018_common_inventory.jsonl"
    cache = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            try:
                cached = json.loads(line)
                cache[cached["relative_path"]] = cached
            except (json.JSONDecodeError, KeyError):
                continue
    reused = 0
    temporary_output = output.with_suffix(".tmp.jsonl")
    with temporary_output.open("w", encoding="utf-8") as stream:
        for index, (source, month, path, details) in enumerate(discovered, 1):
            relative_path = path.relative_to(ROOT).as_posix()
            stat = path.stat()
            cached = cache.get(relative_path)
            if (cached and cached.get("bytes") == stat.st_size
                    and cached.get("mtime_ns") == stat.st_mtime_ns
                    and cached.get("readable") is True):
                record = cached
                reused += 1
            else:
                record = {"relative_path": relative_path, "source": source,
                          "month": month, "bytes": stat.st_size,
                          "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path), **details}
                try:
                    record.update(inspect_netcdf(path, source))
                    record["readable"] = True
                except Exception as error:
                    record.update(readable=False, error_type=type(error).__name__)
            if not record.get("readable"):
                unreadable.append(record["relative_path"])
            sizes[source].append(record["bytes"])
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index % 1000 == 0:
                print("AUDITED", index, "REUSED", reused, flush=True)
    temporary_output.replace(output)

    all_raw_files = [path for directory in (RAW / name for name in ("cora", "oisst", "argo", "adt", "era5"))
                     for path in directory.rglob("*") if path.is_file()]
    temporary = [path.relative_to(ROOT).as_posix() for path in all_raw_files if ".part" in path.name]
    zero = [path.relative_to(ROOT).as_posix() for path in all_raw_files if path.stat().st_size == 0]
    small = {}
    for source, values in sizes.items():
        threshold = max(1024, int(np.median(values) * 0.1))
        small[source] = {"rule": "bytes < max(1024, 10% of source median)", "threshold_bytes": threshold,
                         "paths": [path.relative_to(ROOT).as_posix() for current_source, _, path, _ in discovered
                                   if current_source == source and path.stat().st_size < threshold]}
    argo_monthly = Counter(argo_expected.values())
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "2018 common-modality raw files and the directly related Argo GDAC index",
        "inventory": output.relative_to(ROOT).as_posix(),
        "files": len(discovered),
        "reused_cached_records": reused,
        "bytes": sum(path.stat().st_size for _, _, path, _ in discovered),
        "by_source": {source: {"files": len(values), "bytes": sum(values)} for source, values in sizes.items()},
        "coverage_checks": checks,
        "argo_monthly_expected_counts": dict(sorted(argo_monthly.items())),
        "unreadable_count": len(unreadable), "unreadable_paths": unreadable,
        "temporary_file_count": len(temporary), "temporary_paths": temporary,
        "zero_byte_count": len(zero), "zero_byte_paths": zero,
        "abnormally_small": small,
    }
    summary["status"] = "passed" if (
        not unreadable and not temporary and not zero
        and not checks["cora"]["missing"] and not checks["cora"]["duplicates"]
        and not checks["oisst"]["missing_dates"] and not checks["oisst"]["duplicate_dates"]
        and not checks["adt"]["missing"] and not checks["adt"]["duplicates"]
        and not checks["era5"]["missing"] and not checks["era5"]["duplicates"]
        and not checks["argo"]["missing_relative_paths"] and not checks["argo"]["unexpected_relative_paths"]
    ) else "failed"
    summary_path = AUDIT / "raw_2018_common_inventory_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "files": summary["files"],
                      "bytes": summary["bytes"], "by_source": summary["by_source"]}, indent=2))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
