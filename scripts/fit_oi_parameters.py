#!/usr/bin/env python3
"""Estimate local OI variance parameters from C profiles and LOMO backgrounds only."""
import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

from evaluate_sparse_baseline import load_profiles

ROOT = Path(__file__).resolve().parents[1]
VARIABLES = {
    'temperature': {
        'profile': 'temperature',
        'background': 'conservative_temperature_background',
        'background_variance': 'conservative_temperature_background_error_variance',
        'error_floor': 0.05,
        'units': 'degree_Celsius',
    },
    'salinity': {
        'profile': 'salinity',
        'background': 'absolute_salinity_background',
        'background_variance': 'absolute_salinity_background_error_variance',
        'error_floor': 0.01,
        'units': 'g kg-1',
    },
}
DEPTH_BANDS = {'5-100m': (5, 100), '110-300m': (110, 300), '310-500m': (310, 500)}


def sample_normalized(field, depth, latitude, longitude, profiles):
    values = np.asarray(field, dtype=float)
    valid = np.isfinite(values)
    kwargs = dict(bounds_error=False, fill_value=0.0)
    numerator = RegularGridInterpolator(
        (depth, latitude, longitude), np.where(valid, values, 0.0), **kwargs)
    weight = RegularGridInterpolator(
        (depth, latitude, longitude), valid.astype(float), **kwargs)
    sampled = []
    sampled_weight = []
    for profile in profiles:
        points = np.column_stack([
            profile['depth'],
            np.full(profile['depth'].shape, profile['latitude']),
            np.full(profile['depth'].shape, profile['longitude']),
        ])
        current_weight = weight(points)
        current = numerator(points)
        sampled.append(np.divide(current, current_weight,
                                 out=np.full(current.shape, np.nan),
                                 where=current_weight >= .5))
        sampled_weight.append(current_weight)
    return sampled, sampled_weight


def estimate_band(innovations, background_variances, floor):
    innovations = np.asarray(innovations, dtype=float)
    background_variances = np.asarray(background_variances, dtype=float)
    valid = np.isfinite(innovations)
    count = int(valid.sum())
    if count < 2:
        return {'n': count, 'innovation_variance': None, 'background_error_variance': None,
                'observation_error_variance': None, 'background_variance': None}
    innovation_variance = float(np.var(innovations[valid], ddof=1))
    finite_background = background_variances[np.isfinite(background_variances)]
    background_error_variance = (float(np.median(finite_background))
                                 if finite_background.size else innovation_variance * .75)
    observation_error_variance = max(floor ** 2, innovation_variance - background_error_variance)
    background_variance = max(floor ** 2, innovation_variance - observation_error_variance)
    return {
        'n': count,
        'innovation_variance': innovation_variance,
        'background_error_variance': background_error_variance,
        'observation_error_variance': observation_error_variance,
        'background_variance': background_variance,
        'observation_error_floor': floor,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roles', type=Path,
                        default=ROOT / 'data/processed/argo_monthly/profile_roles.jsonl')
    parser.add_argument('--background-dir', type=Path,
                        default=ROOT / 'data/processed/baselines')
    parser.add_argument('--horizontal-scales-km', nargs='+', type=float,
                        default=[500, 1000, 2000])
    parser.add_argument('--vertical-scale-m', type=float, default=50)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'research/audits/oi_parameters_2018.json')
    args = parser.parse_args()
    if any(scale <= 0 for scale in args.horizontal_scales_km) or args.vertical_scale_m <= 0:
        raise ValueError('OI scales must be positive')
    profiles = load_profiles(args.roles)
    context = [profile for profile in profiles if profile['role'] == 'C']
    if not context:
        raise ValueError('No C profiles available for OI parameter fitting')
    by_month = defaultdict(list)
    for profile in context:
        by_month[profile['month']].append(profile)
    expected_months = [f'2018{month:02d}' for month in range(1, 13)]
    if sorted(by_month) != expected_months:
        raise ValueError('OI parameter fit requires all twelve 2018 C months')

    collected = {variable: {band: {'innovation': [], 'background_variance': []}
                            for band in DEPTH_BANDS} for variable in VARIABLES}
    monthly_counts = {}
    for month in expected_months:
        background_path = args.background_dir / f'seasonal_background_lomo_2018_{month}.nc'
        month_profiles = by_month[month]
        monthly_counts[month] = len(month_profiles)
        with Dataset(background_path) as ds:
            depth = np.asarray(ds['depth'][:], dtype=float)
            latitude = np.asarray(ds['latitude'][:], dtype=float)
            longitude = np.asarray(ds['longitude'][:], dtype=float)
            for variable, spec in VARIABLES.items():
                background, _ = sample_normalized(
                    ds[spec['background']][:], depth, latitude, longitude, month_profiles)
                background_variance, _ = sample_normalized(
                    ds[spec['background_variance']][:], depth, latitude, longitude, month_profiles)
                for profile, profile_background, profile_variance in zip(
                        month_profiles, background, background_variance):
                    innovation = profile[spec['profile']] - profile_background
                    for band, (low, high) in DEPTH_BANDS.items():
                        keep = (profile['depth'] >= low) & (profile['depth'] <= high)
                        collected[variable][band]['innovation'].extend(innovation[keep].tolist())
                        collected[variable][band]['background_variance'].extend(profile_variance[keep].tolist())

    parameters = {}
    for variable, spec in VARIABLES.items():
        parameters[variable] = {
            'units': spec['units'],
            'depth_bands': {
                band: estimate_band(values['innovation'], values['background_variance'], spec['error_floor'])
                for band, values in collected[variable].items()
            },
        }
    output = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'completed',
        'fit_role': 'C only',
        'Q_used': False,
        'E_used': False,
        'fit_months': expected_months,
        'monthly_context_profiles': monthly_counts,
        'background': 'single-year leave-one-month-out engineering seasonal background',
        'horizontal_scale_candidates_km': args.horizontal_scales_km,
        'horizontal_scale_selection': '2018 Q only in evaluate_oi_baseline.py; E report-only',
        'vertical_scale_m': args.vertical_scale_m,
        'covariance': 'sigma_b^2 exp(-0.5*(horizontal/Lh)^2) exp(-abs(vertical)/Lz)',
        'observation_error': 'diag(max(physical floor squared, C innovation variance - median background error variance))',
        'parameters': parameters,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'fit_months': len(expected_months), 'context_profiles': len(context),
                      'parameters': parameters}, indent=2))


if __name__ == '__main__':
    main()
