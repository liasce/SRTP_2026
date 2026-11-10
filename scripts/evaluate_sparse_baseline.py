#!/usr/bin/env python3
"""Evaluate C-profile Gaussian interpolation on separate Q and fixed-platform E targets."""
import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
VARIABLES = {
    "temperature": "conservative_temperature_on_target_depth",
    "salinity": "absolute_salinity_on_target_depth",
}
DEPTH_BINS = {"5-100m": (5, 100), "110-300m": (110, 300), "310-500m": (310, 500)}
DISTANCE_BINS = {"0-500km": (0, 500), "500-1000km": (500, 1000), "1000-2000km": (1000, 2000), "2000km+": (2000, np.inf)}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distance_km(lat, lon, lats, lons):
    lat1, lat2 = np.radians(lat), np.radians(lats)
    dlat = lat2 - lat1
    dlon = np.radians((lons - lon + 180) % 360 - 180)
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def pooled_metrics(observed, predicted):
    valid_observed = np.isfinite(observed)
    valid = valid_observed & np.isfinite(predicted)
    error = predicted[valid] - observed[valid]
    return {
        "n_points": int(valid.sum()),
        "observed_points": int(valid_observed.sum()),
        "prediction_coverage": float(valid.sum() / valid_observed.sum()) if valid_observed.any() else None,
        "rmse": float(np.sqrt(np.mean(error ** 2))) if error.size else None,
        "mae": float(np.mean(np.abs(error))) if error.size else None,
        "bias": float(np.mean(error)) if error.size else None,
    }


def summarize(records, variable, selector=None):
    selected = []
    profile_metrics = []
    platforms = set()
    for record in records:
        observed = record[variable]
        predicted = record["prediction"][variable]
        keep = np.ones(observed.shape, dtype=bool) if selector is None else selector(record)
        observed = observed[keep]
        predicted = predicted[keep]
        selected.append((observed, predicted))
        valid = np.isfinite(observed) & np.isfinite(predicted)
        if valid.any():
            error = predicted[valid] - observed[valid]
            profile_metrics.append((float(np.mean(error ** 2)), float(np.mean(np.abs(error)))))
            platforms.add(record["platform"])
    if selected:
        observed = np.concatenate([item[0] for item in selected])
        predicted = np.concatenate([item[1] for item in selected])
    else:
        observed = predicted = np.array([], dtype=float)
    result = pooled_metrics(observed, predicted)
    result.update(
        n_profiles=len(profile_metrics), n_platforms=len(platforms),
        profile_equal_rmse=(float(np.sqrt(np.mean([item[0] for item in profile_metrics])))
                            if profile_metrics else None),
        profile_equal_mae=(float(np.mean([item[1] for item in profile_metrics]))
                           if profile_metrics else None),
    )
    return result


def load_profiles(roles_path):
    grouped = defaultdict(list)
    for line in roles_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        grouped[record["source"]].append(record)
    profiles = []
    for source, records in sorted(grouped.items()):
        with Dataset(ROOT / Path(source)) as ds:
            depth = np.asarray(ds["target_depth"][:], dtype=float)
            for record in records:
                index = record["source_profile_index"]
                record["depth"] = depth
                for key, variable in VARIABLES.items():
                    record[key] = np.ma.filled(ds[variable][index], np.nan).astype(float)
                profiles.append(record)
    return profiles


def interpolate(target, inputs, bandwidth_km, localization_factor, minimum_support):
    if not inputs:
        return {key: np.full(target["depth"].shape, np.nan) for key in VARIABLES}, None, "no_context_profiles"
    distances = distance_km(
        target["latitude"], target["longitude"],
        np.array([profile["latitude"] for profile in inputs]),
        np.array([profile["longitude"] for profile in inputs]))
    within = distances <= bandwidth_km * localization_factor
    if not within.any():
        return {key: np.full(target["depth"].shape, np.nan) for key in VARIABLES}, float(distances.min()), "no_local_support"
    local_inputs = [profile for profile, keep in zip(inputs, within) if keep]
    local_distances = distances[within]
    weights = np.exp(-0.5 * (local_distances / bandwidth_km) ** 2)
    predictions = {}
    for key in VARIABLES:
        values = np.stack([profile[key] for profile in local_inputs])
        valid = np.isfinite(values)
        numerator = np.sum(np.where(valid, values * weights[:, None], 0), axis=0)
        denominator = np.sum(valid * weights[:, None], axis=0)
        support = valid.sum(axis=0)
        predictions[key] = np.divide(
            numerator, denominator, out=np.full(values.shape[1], np.nan),
            where=(denominator > 0) & (support >= minimum_support))
    return predictions, float(distances.min()), None


def selectors():
    output = {"all": None}
    for label, (low, high) in DEPTH_BINS.items():
        output[label] = lambda record, low=low, high=high: ((record["depth"] >= low) & (record["depth"] <= high))
    return output


def summarize_group(records):
    result = {"targets": len(records), "failures": dict(sorted(
        {reason: sum(record["failure"] == reason for record in records)
         for reason in {record["failure"] for record in records if record["failure"]}}.items()))}
    nearest = [record["nearest_context_km"] for record in records if record["nearest_context_km"] is not None]
    result["nearest_context_profile_km"] = {
        "n_profiles": len(nearest),
        "median": float(np.median(nearest)) if nearest else None,
        "max": float(np.max(nearest)) if nearest else None,
    }
    for variable in VARIABLES:
        result[variable] = {label: summarize(records, variable, selector)
                            for label, selector in selectors().items()}
        result[variable]["by_nearest_context_distance"] = {}
        for label, (low, high) in DISTANCE_BINS.items():
            selected = [record for record in records
                        if record["nearest_context_km"] is not None
                        and low <= record["nearest_context_km"] < high]
            result[variable]["by_nearest_context_distance"][label] = summarize(selected, variable)
    return result


def evaluate(profiles, bandwidths, target_roles, localization_factor, minimum_support):
    months = sorted({profile["month"] for profile in profiles})
    output = {
        "scope": "2018 development baseline; Q may select scale, fixed-platform E is report-only",
        "months": months,
        "roles": {month: {role: sum(profile["month"] == month and profile["role"] == role for profile in profiles)
                           for role in ("C", "Q", "E")} for month in months},
        "configuration": {
            "bandwidth_km": bandwidths,
            "localization_factor": localization_factor,
            "minimum_context_profiles_per_layer": minimum_support,
            "depth_bins": DEPTH_BINS,
            "distance_bins": {key: [low, None if np.isinf(high) else high]
                              for key, (low, high) in DISTANCE_BINS.items()},
        },
        "bandwidths": {},
    }
    by_month = {month: [profile for profile in profiles if profile["month"] == month] for month in months}
    for bandwidth in bandwidths:
        bandwidth_result = {}
        for role in target_roles:
            monthly_records = {}
            all_records = []
            for month in months:
                context = [profile for profile in by_month[month] if profile["role"] == "C"]
                targets = [profile for profile in by_month[month] if profile["role"] == role]
                records = []
                for target in targets:
                    prediction, nearest, failure = interpolate(
                        target, context, bandwidth, localization_factor, minimum_support)
                    records.append({**target, "prediction": prediction,
                                    "nearest_context_km": nearest, "failure": failure})
                monthly_records[month] = records
                all_records.extend(records)
            bandwidth_result[role] = {
                "monthly": {month: summarize_group(records) for month, records in monthly_records.items()},
                "cross_month": summarize_group(all_records),
            }
        output["bandwidths"][str(bandwidth)] = bandwidth_result
    if "Q" in target_roles:
        output["development_selection_Q_only"] = {}
        for variable in VARIABLES:
            candidates = []
            for bandwidth in bandwidths:
                value = output["bandwidths"][str(bandwidth)]["Q"]["cross_month"][variable]["all"]["rmse"]
                if value is not None:
                    candidates.append((value, bandwidth))
            output["development_selection_Q_only"][variable] = {
                "criterion": "minimum Q point-pooled RMSE on 2018 development months",
                "selected_bandwidth_km": min(candidates)[1] if candidates else None,
                "E_used_for_selection": False,
            }
    return output


def self_check():
    np.testing.assert_allclose(distance_km(0, 10, np.array([0]), np.array([10])), 0, atol=1e-9)
    np.testing.assert_allclose(distance_km(0, 179, np.array([0]), np.array([-179])), 222.39, rtol=.01)
    example = {"latitude": 0.0, "longitude": 0.0, "depth": np.array([5.0]),
               "temperature": np.array([1.0]), "salinity": np.array([35.0])}
    prediction, nearest, failure = interpolate(example, [example], 500, 3, 1)
    assert failure is None and nearest == 0
    assert prediction["temperature"][0] == 1 and prediction["salinity"][0] == 35


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles", type=Path, default=ROOT / "data/processed/argo_monthly/profile_roles.jsonl")
    parser.add_argument("--bandwidth-km", type=float, nargs="+", default=[500, 1000, 2000])
    parser.add_argument("--target-roles", nargs="+", choices=["Q", "E"], default=["Q", "E"])
    parser.add_argument("--localization-factor", type=float, default=3.0)
    parser.add_argument("--minimum-support", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "research/audits/sparse_baseline_2018.json")
    args = parser.parse_args()
    if any(value <= 0 for value in args.bandwidth_km) or args.localization_factor <= 0 or args.minimum_support < 1:
        raise ValueError("Bandwidth, localization, and support parameters must be positive")
    self_check()
    profiles = load_profiles(args.roles)
    months = sorted({profile["month"] for profile in profiles})
    expected = [f"2018{month:02d}" for month in range(1, 13)]
    if months != expected:
        raise ValueError(f"Expected exactly twelve ordered 2018 months, got {months}")
    result = evaluate(profiles, args.bandwidth_km, list(dict.fromkeys(args.target_roles)),
                      args.localization_factor, args.minimum_support)
    result.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "roles_path": str(args.roles.relative_to(ROOT)),
        "roles_sha256": sha256(args.roles),
        "cora_reference_warning": (
            "Same-month CORA may assimilate Argo and is not reported as independent truth or used for scale selection."),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"months": len(months), "bandwidths": args.bandwidth_km,
                      "target_roles": args.target_roles,
                      "selection": result.get("development_selection_Q_only")}, indent=2))


if __name__ == "__main__":
    main()
