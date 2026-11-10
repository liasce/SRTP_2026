#!/usr/bin/env python3
"""Build leave-one-month-out seasonal backgrounds for the single-year 2018 development set.

These outputs are engineering backgrounds, not multi-year monthly climatologies. The
entire target month is excluded before fitting a mean plus annual sine/cosine cycle.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
VARIABLES = {
    "conservative_temperature": "degree_Celsius",
    "absolute_salinity": "g kg-1",
}


def harmonic_design(month_numbers):
    phase = 2 * np.pi * (np.asarray(month_numbers, dtype=float) - 1) / 12
    return np.column_stack([np.ones(len(phase)), np.sin(phase), np.cos(phase)])


def fit_lomo(values, fit_month_numbers, target_month_number, minimum_support=6):
    """Fit harmonic where all fit months exist; otherwise traceable fit-month mean."""
    values = np.asarray(values, dtype=float)
    if values.shape[0] != len(fit_month_numbers):
        raise ValueError("Time dimension and fit months differ")
    finite = np.isfinite(values)
    support = finite.sum(axis=0).astype(np.uint8)
    output = np.full(values.shape[1:], np.nan, dtype=np.float32)
    variance = np.full(values.shape[1:], np.nan, dtype=np.float32)
    fallback = np.full(values.shape[1:], 2, dtype=np.uint8)
    complete = support == values.shape[0]
    if complete.any():
        design = harmonic_design(fit_month_numbers)
        target_design = harmonic_design([target_month_number])[0]
        selected = values[:, complete]
        coefficients = np.linalg.pinv(design) @ selected
        output[complete] = (target_design @ coefficients).astype(np.float32)
        residual = selected - design @ coefficients
        variance[complete] = np.mean(residual ** 2, axis=0).astype(np.float32)
        fallback[complete] = 0
    mean_supported = (support >= minimum_support) & ~complete
    if mean_supported.any():
        selected = values[:, mean_supported]
        output[mean_supported] = np.nanmean(selected, axis=0).astype(np.float32)
        variance[mean_supported] = np.nanvar(selected, axis=0).astype(np.float32)
        fallback[mean_supported] = 1
    return output, variance, support, fallback


def load_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = manifest["months"]
    months = [record["month"] for record in records]
    expected = [f"2018{month:02d}" for month in range(1, 13)]
    if months != expected:
        raise ValueError(f"Expected ordered 2018 months, got {months}")
    if any(record["split"] not in ("pretrain", "finetune") for record in records):
        raise ValueError("LOMO development background accepts training-side months only")
    return records


def write_background(target_record, fit_records, output, minimum_support):
    target_month = target_record["month"]
    target_number = int(target_month[4:])
    fit_numbers = [int(record["month"][4:]) for record in fit_records]
    if target_month in {record["month"] for record in fit_records}:
        raise ValueError("Target month leaked into background fit")
    reference_path = ROOT / target_record["files"]["cora"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.nc")
    with Dataset(reference_path) as reference, Dataset(temporary, "w") as out:
        depth = np.asarray(reference["depth"][:])
        latitude = np.asarray(reference["latitude"][:])
        longitude = np.asarray(reference["longitude"][:])
        for name, values, units in [("depth", depth, "m"), ("latitude", latitude, "degrees_north"),
                                    ("longitude", longitude, "degrees_east")]:
            out.createDimension(name, len(values))
            variable = out.createVariable(name, "f4", (name,))
            variable[:] = values
            variable.units = units
        shape = (len(depth), len(latitude), len(longitude))
        for name, units in VARIABLES.items():
            background = out.createVariable(name + "_background", "f4",
                                            ("depth", "latitude", "longitude"),
                                            zlib=True, fill_value=np.nan)
            error = out.createVariable(name + "_background_error_variance", "f4",
                                       ("depth", "latitude", "longitude"),
                                       zlib=True, fill_value=np.nan)
            support_var = out.createVariable(name + "_support_count", "u1",
                                             ("depth", "latitude", "longitude"), zlib=True)
            fallback_var = out.createVariable(name + "_fallback_code", "u1",
                                              ("depth", "latitude", "longitude"), zlib=True)
            background.units = units
            error.units = units + " squared"
            for depth_index in range(shape[0]):
                layers = []
                for record in fit_records:
                    with Dataset(ROOT / record["files"]["cora"]) as source:
                        layers.append(np.ma.filled(source[name][0, depth_index], np.nan).astype(float))
                fitted, variance, support, fallback = fit_lomo(
                    np.stack(layers), fit_numbers, target_number, minimum_support)
                background[depth_index] = fitted
                error[depth_index] = variance
                support_var[depth_index] = support
                fallback_var[depth_index] = fallback
        out.target_month = target_month
        out.fit_months = ",".join(record["month"] for record in fit_records)
        out.target_month_excluded = "true"
        out.method = "mean + annual sine/cosine when all 11 fit months finite; fit-month mean fallback"
        out.fallback_codes = "0=harmonic; 1=fit-month mean; 2=insufficient/no background"
        out.minimum_support_for_mean_fallback = minimum_support
        out.scope_warning = "Single-year leave-one-month-out engineering seasonal background; not a multi-year monthly climatology"
        out.created_at_utc = datetime.now(timezone.utc).isoformat()
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/processed/training_manifest_common.json")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/processed/baselines")
    parser.add_argument("--minimum-support", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute existing backgrounds instead of validating and reusing them")
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    if not 3 <= args.minimum_support <= 11:
        raise ValueError("--minimum-support must be between 3 and 11")
    records = load_manifest(args.manifest)
    outputs = []
    for target in records:
        fit_records = [record for record in records if record["month"] != target["month"]]
        output = args.output_dir / f"seasonal_background_lomo_2018_{target['month']}.nc"
        if output.exists() and not args.overwrite:
            with Dataset(output) as existing:
                required = {name + suffix for name in VARIABLES
                            for suffix in ('_background', '_background_error_variance',
                                           '_support_count', '_fallback_code')}
                if (existing.target_month != target['month']
                        or existing.target_month_excluded != 'true'
                        or not required.issubset(existing.variables)):
                    raise ValueError(f'{output}: existing background validation failed')
            print('VALID', output.relative_to(ROOT), flush=True)
        else:
            write_background(target, fit_records, output, args.minimum_support)
            print(output.relative_to(ROOT), flush=True)
        outputs.append(str(output.relative_to(ROOT)))
    audit = ROOT / "research/audits/seasonal_background_lomo_2018.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps({
        "status": "completed",
        "scientific_label": "single-year leave-one-month-out engineering seasonal background",
        "not_claimed": "multi-year monthly climatology",
        "source_manifest": str(args.manifest.relative_to(ROOT)),
        "target_month_excluded": True,
        "minimum_support": args.minimum_support,
        "outputs": outputs,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
