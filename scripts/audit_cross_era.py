#!/usr/bin/env python3
"""Audit 2005/2010/2015 samples against the existing 2018 reference files."""
import calendar
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, chartostring, num2date

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "research/audits/cross_era_consistency.json"
MONTHS = ("200501", "201001", "201501", "201801")
AUDIT_MONTHS = MONTHS[:3]


def schema(ds):
    return {name: {"dimensions": list(v.dimensions), "dtype": str(v.dtype),
                   "units": getattr(v, "units", None), "standard_name": getattr(v, "standard_name", None)}
            for name, v in ds.variables.items()}


def finite(values):
    a = np.ma.filled(values, np.nan).astype(float)
    good = np.isfinite(a)
    return {"count": int(a.size), "valid_fraction": float(good.mean()),
            "min": float(a[good].min()), "max": float(a[good].max()), "mean": float(a[good].mean())}


def same(items):
    return all(np.array_equal(items[0], item) for item in items[1:])


def cora():
    result, grids, schemas = {}, [], []
    for month in MONTHS:
        entry = {"fields": {}, "input_profiles": {}}
        for name in ("TEMP", "PSAL"):
            path = RAW / f"cora/original_202603/OA_CORA5.2_{month}15_fld_{name}.nc"
            with Dataset(path) as ds:
                grids.append(tuple(ds[v][:] for v in ("depth", "latitude", "longitude")))
                schemas.append(schema(ds))
                z, y, x = ds["depth"][:], ds["latitude"][:], ds["longitude"][:]
                selectors = ((z >= 1) & (z <= 510), (y >= -20) & (y <= 20),
                             (np.mod(x, 360) >= 120) & (np.mod(x, 360) <= 280))
                diagnostics = {}
                for variable in (name, name + "_ERR", name + "_PCTVAR"):
                    values = ds[variable][0, selectors[0], selectors[1], :][..., selectors[2]]
                    diagnostics[variable] = finite(values)
                entry["fields"][name] = {"time": str(num2date(ds["time"][:], ds["time"].units)[0]),
                    "statistics": diagnostics, "variables": list(ds.variables),
                    "units": ds[name].units, "standard_name": ds[name].standard_name,
                    "source": ds.source, "grid_shape": [len(z), len(y), len(x)]}
            path = RAW / f"cora/original_202603/OA_CORA5.2_{month}15_dat_{name}.nc"
            with Dataset(path) as ds:
                reference = np.ma.filled(ds["REFERENCE_DATE_TIME"][:], b"").tobytes().decode().strip()
                origin = datetime.strptime(reference[:8], "%Y%m%d")
                juld = np.ma.compressed(ds["JULD"][:])
                entry["input_profiles"][name] = {"profiles": len(ds.dimensions["N_PROF"]),
                    "levels": len(ds.dimensions["N_LEVELS"]),
                    "first_time": str(origin + timedelta(days=float(juld.min()))),
                    "last_time": str(origin + timedelta(days=float(juld.max()))),
                    "variables": list(ds.variables), "software_version": ds.software_version}
        result[month] = entry
    grid_equal = all(same([grid[i] for grid in grids]) for i in range(3))
    by_variable = {name: [schema_ for schema_, (_, field) in zip(schemas, [(m, n) for m in MONTHS for n in ("TEMP", "PSAL")]) if field == name]
                   for name in ("TEMP", "PSAL")}
    return {"status": "consistent" if grid_equal and all(all(x == values[0] for x in values[1:]) for values in by_variable.values()) else "review",
            "grid_identical": grid_equal, "field_schema_identical_by_variable": {k: all(x == v[0] for x in v[1:]) for k, v in by_variable.items()},
            "months": result}


def oisst():
    result, schemas, lats, lons, time_dtypes = {}, [], [], [], {}
    for month in MONTHS:
        paths = sorted((RAW / "oisst" / month).glob("*.nc"))
        dates, valid, versions = [], [], set()
        for path in paths:
            with Dataset(path) as ds:
                schemas.append(schema(ds)); lats.append(ds["lat"][:]); lons.append(ds["lon"][:])
                time_dtypes[month] = str(ds["time"].dtype)
                dates.append(num2date(ds["time"][:], ds["time"].units)[0].strftime("%Y%m%d"))
                y, x = ds["lat"][:], ds["lon"][:]
                valid.append(finite(ds["sst"][0, 0, (y >= -20) & (y <= 20), :][..., (x >= 120) & (x <= 280)]))
                versions.add(ds.product_version)
        expected = [f"{month}{d:02d}" for d in range(1, calendar.monthrange(int(month[:4]), int(month[4:]))[1] + 1)]
        result[month] = {"files": len(paths), "calendar_complete": dates == expected,
                         "product_versions": sorted(versions),
                         "sst_valid_fraction_mean": float(np.mean([x["valid_fraction"] for x in valid]))}
    data_schemas = [{k: v for k, v in item.items() if k != "time"} for item in schemas]
    ok = same(lats) and same(lons) and all(x == data_schemas[0] for x in data_schemas[1:]) and all(v["calendar_complete"] for v in result.values())
    return {"status": "consistent_with_time_encoding_change" if ok and len(set(time_dtypes.values())) > 1 else "consistent" if ok else "review",
            "grid_identical": same(lats) and same(lons), "data_schema_identical": all(x == data_schemas[0] for x in data_schemas[1:]),
            "time_coordinate_dtype": time_dtypes, "months": result}


def regional_product(source, pattern, variable_names, regions, months=MONTHS):
    result, schemas, coords = {}, [], {region: {"latitude": [], "longitude": []} for region in regions}
    for month in months:
        result[month] = {"regions": {}}
        for region in regions:
            path = RAW / source / pattern.format(month=month, region=region)
            with Dataset(path) as ds:
                schemas.append(schema(ds))
                coords[region]["latitude"].append(ds["latitude"][:]); coords[region]["longitude"].append(ds["longitude"][:])
                time_name = "valid_time" if "valid_time" in ds.variables else "time"
                result[month]["regions"][region] = {"shape": {k: len(v) for k, v in ds.dimensions.items()},
                    "time": [str(x) for x in num2date(ds[time_name][:], ds[time_name].units)],
                    "variables": {name: finite(ds[name][:]) | {"units": ds[name].units} for name in variable_names},
                    "product_version": getattr(ds, "product_version", None)}
    coordinate_equal = all(same(values) for region in coords.values() for values in region.values())
    schema_equal = all(x == schemas[0] for x in schemas[1:])
    time_complete = all(len(region["time"]) == (calendar.monthrange(int(month[:4]), int(month[4:]))[1] if source == "adt" else 1)
                        and all(value.startswith(month[:4] + "-" + month[4:]) for value in region["time"])
                        for month, entry in result.items() for region in entry["regions"].values())
    return {"status": "consistent" if coordinate_equal and schema_equal and time_complete else "review",
            "coordinate_identical_within_region": coordinate_equal, "schema_identical": schema_equal,
            "time_complete": time_complete, "months": result}


def argo():
    candidates = list(csv.DictReader((ROOT / "research/audits/argo_sample_candidates_200501_201001_201501.csv").open()))
    manifest = {Path(row["source"]).resolve(): row for row in json.loads((ROOT / "data/processed/argo/manifest.json").read_text())}
    result = {}
    required = {"PRES_ADJUSTED", "PRES_ADJUSTED_QC", "TEMP_ADJUSTED", "TEMP_ADJUSTED_QC",
                "PSAL_ADJUSTED", "PSAL_ADJUSTED_QC", "DATA_MODE", "JULD", "LATITUDE", "LONGITUDE"}
    for month in AUDIT_MONTHS:
        rows = [row for row in candidates if row["date"].startswith(month)]
        totals = {key: 0 for key in ("profiles", "eligible_profiles", "temperature_points", "salinity_points", "joint_points",
                                      "target_temperature_points", "target_joint_points")}
        missing_required, missing_files, temp500, joint500, processed_profiles = 0, [], 0, 0, 0
        time_qc, position_qc = {}, {}
        modes = {}
        for row in rows:
            source = (RAW / "argo" / row["file"]).resolve()
            with Dataset(source) as ds:
                missing = sorted(required - set(ds.variables))
                missing_required += int(bool(missing))
                if missing: missing_files.append({"file": row["file"], "missing": missing})
                values = chartostring(ds["DATA_MODE"][:]).astype(str) if ds["DATA_MODE"].ndim > 1 else ds["DATA_MODE"][:].astype(str)
                for value in values:
                    mode = str(value).strip(); modes[mode] = modes.get(mode, 0) + 1
                for variable, counts in (("JULD_QC", time_qc), ("POSITION_QC", position_qc)):
                    for value in np.ma.filled(ds[variable][:], b" ").astype(str):
                        counts[str(value)] = counts.get(str(value), 0) + 1
            record = manifest[source]
            for key in totals: totals[key] += record[key]
            processed = Path(record["output"])
            with Dataset(processed) as ds:
                temp500 += int(np.isfinite(np.ma.filled(ds["temperature_on_target_depth"][:, -1], np.nan)).sum())
                joint500 += int((np.isfinite(np.ma.filled(ds["temperature_on_target_depth"][:, -1], np.nan)) &
                                 np.isfinite(np.ma.filled(ds["practical_salinity_on_target_depth"][:, -1], np.nan))).sum())
                processed_profiles += len(ds.dimensions["profile"])
        result[month] = {"files": len(rows), "files_missing_required_adjusted_fields": missing_required,
                         "missing_adjusted_field_files": missing_files, "time_qc_profiles": time_qc,
                         "position_qc_profiles": position_qc,
                         **totals, "data_mode_profiles": modes, "temperature_profiles_at_500m": temp500,
                         "joint_TS_profiles_at_500m": joint500, "processed_profiles": processed_profiles}
    return {"status": "format_consistent_with_early_coverage_warning" if all(v["files"] == 50 for v in result.values()) else "review",
            "caveat": "Schema consistency does not imply equal adjusted-data availability or vertical coverage.", "months": result}


def main():
    adt = regional_product("adt", "duacs_202411_{month}_{region}_adt.nc", ("adt",), ("west", "east"))
    era5 = regional_product("era5", "era5_monthly_{month}_{region}_uv.nc", ("u10", "v10"), ("west", "east"), AUDIT_MONTHS)
    report = {"audit_months": list(AUDIT_MONTHS), "reference_month": MONTHS[-1],
              "interpretation": "Format/coverage audit only; three January samples cannot prove full-period homogeneity.",
              "products": {"CORA": cora(), "OISST": oisst(), "ADT": adt, "ERA5": era5, "Argo": argo()}}
    report["status"] = "passed_with_documented_coverage_changes" if all(x["status"] != "review" for x in report["products"].values()) else "review_required"
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    assert report["status"] == "passed_with_documented_coverage_changes"
    print(json.dumps({"status": report["status"], "products": {k: v["status"] for k, v in report["products"].items()}}, indent=2))


if __name__ == "__main__":
    main()
