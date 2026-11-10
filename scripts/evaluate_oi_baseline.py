#!/usr/bin/env python3
"""Evaluate local OI using the same C profiles on separate Q and fixed-platform E targets."""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from evaluate_sparse_baseline import DEPTH_BINS, DISTANCE_BINS, distance_km, load_profiles, pooled_metrics
from fit_oi_parameters import VARIABLES, sample_normalized
from oi_core import analyze

ROOT = Path(__file__).resolve().parents[1]


def depth_band(depth):
    for label, (low, high) in DEPTH_BINS.items():
        if low <= depth <= high:
            return label
    raise ValueError(f'Depth outside configured bands: {depth}')


def attach_background(profiles, background_dir):
    by_month = defaultdict(list)
    for profile in profiles:
        by_month[profile['month']].append(profile)
    for month, month_profiles in sorted(by_month.items()):
        path = background_dir / f'seasonal_background_lomo_2018_{month}.nc'
        with Dataset(path) as ds:
            depth = np.asarray(ds['depth'][:], dtype=float)
            latitude = np.asarray(ds['latitude'][:], dtype=float)
            longitude = np.asarray(ds['longitude'][:], dtype=float)
            for variable, spec in VARIABLES.items():
                background, _ = sample_normalized(
                    ds[spec['background']][:], depth, latitude, longitude, month_profiles)
                variance, _ = sample_normalized(
                    ds[spec['background_variance']][:], depth, latitude, longitude, month_profiles)
                for profile, values, variances in zip(month_profiles, background, variance):
                    profile[f'background_{variable}'] = values
                    profile[f'background_variance_{variable}'] = variances


def nearest_context_observations(target, target_depth, contexts, variable, vertical_radius_m):
    observations = []
    for context in contexts:
        values = context[VARIABLES[variable]['profile']]
        background = context[f'background_{variable}']
        candidates = np.flatnonzero(
            np.isfinite(values) & np.isfinite(background)
            & (np.abs(context['depth'] - target_depth) <= vertical_radius_m))
        if candidates.size == 0:
            continue
        index = candidates[np.abs(context['depth'][candidates] - target_depth).argmin()]
        observations.append((context, index))
    return observations


def analyze_profile(target, contexts, variable, parameters, horizontal_scale_km,
                    vertical_scale_m, vertical_radius_m, max_context_profiles,
                    localization_factor):
    prediction = np.full(target['depth'].shape, np.nan)
    analysis_variance = np.full(target['depth'].shape, np.nan)
    statuses = np.full(target['depth'].shape, 'target_missing', dtype=object)
    nearest = None
    if contexts:
        distances = distance_km(
            target['latitude'], target['longitude'],
            np.array([profile['latitude'] for profile in contexts]),
            np.array([profile['longitude'] for profile in contexts]))
        nearest = float(distances.min())
        order = np.argsort(distances)
        keep = [index for index in order
                if distances[index] <= horizontal_scale_km * localization_factor]
        keep = keep[:max_context_profiles]
        local_contexts = [contexts[index] for index in keep]
    else:
        local_contexts = []
    observed_target = target[VARIABLES[variable]['profile']]
    background_target = target[f'background_{variable}']
    for depth_index, depth in enumerate(target['depth']):
        if not np.isfinite(observed_target[depth_index]):
            continue
        band = depth_band(depth)
        band_parameters = parameters[variable]['depth_bands'][band]
        if band_parameters['background_variance'] is None:
            statuses[depth_index] = 'missing_parameters'
            prediction[depth_index] = background_target[depth_index]
            continue
        selected = nearest_context_observations(
            target, depth, local_contexts, variable, vertical_radius_m)
        observations = np.array([
            context[VARIABLES[variable]['profile']][index] for context, index in selected])
        observation_background = np.array([
            context[f'background_{variable}'][index] for context, index in selected])
        observation_latitude = np.array([context['latitude'] for context, _ in selected])
        observation_longitude = np.array([context['longitude'] for context, _ in selected])
        observation_depth = np.array([context['depth'][index] for context, index in selected])
        value, variance, diagnostics = analyze(
            background_target[depth_index], observation_background, observations,
            observation_latitude, observation_longitude, observation_depth,
            target['latitude'], target['longitude'], depth,
            band_parameters['background_variance'],
            band_parameters['observation_error_variance'],
            horizontal_scale_km, vertical_scale_m)
        prediction[depth_index] = value
        analysis_variance[depth_index] = variance
        statuses[depth_index] = diagnostics['status']
    return prediction, analysis_variance, statuses, nearest


def summarize(records, variable, selector=None):
    observed_parts, predicted_parts = [], []
    profile_mse, profile_mae, platforms = [], [], set()
    status_counts = Counter()
    for record in records:
        observed = record[VARIABLES[variable]['profile']]
        predicted = record[f'oi_{variable}']
        keep = np.ones(observed.shape, dtype=bool) if selector is None else selector(record)
        observed_parts.append(observed[keep])
        predicted_parts.append(predicted[keep])
        statuses = record[f'oi_status_{variable}'][keep]
        status_counts.update(str(status) for status in statuses)
        valid = np.isfinite(observed[keep]) & np.isfinite(predicted[keep])
        if valid.any():
            error = predicted[keep][valid] - observed[keep][valid]
            profile_mse.append(float(np.mean(error ** 2)))
            profile_mae.append(float(np.mean(np.abs(error))))
            platforms.add(record['platform'])
    observed = np.concatenate(observed_parts) if observed_parts else np.array([])
    predicted = np.concatenate(predicted_parts) if predicted_parts else np.array([])
    result = pooled_metrics(observed, predicted)
    result.update({
        'n_profiles': len(profile_mse), 'n_platforms': len(platforms),
        'profile_equal_rmse': float(np.sqrt(np.mean(profile_mse))) if profile_mse else None,
        'profile_equal_mae': float(np.mean(profile_mae)) if profile_mae else None,
        'layer_status_counts': dict(sorted(status_counts.items())),
    })
    return result


def summarize_group(records):
    result = {'target_profiles': len(records)}
    for variable in VARIABLES:
        result[variable] = {'all': summarize(records, variable)}
        for label, (low, high) in DEPTH_BINS.items():
            result[variable][label] = summarize(
                records, variable,
                lambda record, low=low, high=high: ((record['depth'] >= low) & (record['depth'] <= high)))
        result[variable]['by_nearest_context_distance'] = {}
        for label, (low, high) in DISTANCE_BINS.items():
            selected = [record for record in records
                        if record['nearest_context_km'] is not None
                        and low <= record['nearest_context_km'] < high]
            result[variable]['by_nearest_context_distance'][label] = summarize(selected, variable)
    return result


def evaluate(profiles, parameters, horizontal_scales, target_roles,
             vertical_radius_m, max_context_profiles, localization_factor):
    months = [f'2018{month:02d}' for month in range(1, 13)]
    by_month = {month: [profile for profile in profiles if profile['month'] == month]
                for month in months}
    output = {'months': months, 'horizontal_scales_km': horizontal_scales,
              'vertical_scale_m': parameters['vertical_scale_m'],
              'vertical_radius_m': vertical_radius_m,
              'max_context_profiles': max_context_profiles,
              'localization_factor': localization_factor,
              'roles': {}, 'scales': {}}
    for month, month_profiles in by_month.items():
        output['roles'][month] = {role: sum(profile['role'] == role for profile in month_profiles)
                                  for role in ('C', 'Q', 'E')}
    for horizontal_scale in horizontal_scales:
        scale_result = {}
        for role in target_roles:
            monthly = {}
            cross_month = []
            for month, month_profiles in by_month.items():
                contexts = [profile for profile in month_profiles if profile['role'] == 'C']
                targets = [profile for profile in month_profiles if profile['role'] == role]
                evaluated = []
                for target in targets:
                    nearest_values = []
                    for variable in VARIABLES:
                        prediction, variance, statuses, nearest = analyze_profile(
                            target, contexts, variable, parameters['parameters'], horizontal_scale,
                            parameters['vertical_scale_m'], vertical_radius_m,
                            max_context_profiles, localization_factor)
                        target[f'oi_{variable}'] = prediction
                        target[f'oi_variance_{variable}'] = variance
                        target[f'oi_status_{variable}'] = statuses
                        nearest_values.append(nearest)
                    target['nearest_context_km'] = next((value for value in nearest_values if value is not None), None)
                    evaluated.append(target)
                monthly[month] = summarize_group(evaluated)
                cross_month.extend(evaluated)
            scale_result[role] = {'monthly': monthly, 'cross_month': summarize_group(cross_month)}
        output['scales'][str(horizontal_scale)] = scale_result
    if 'Q' in target_roles:
        output['development_selection_Q_only'] = {}
        for variable in VARIABLES:
            candidates = []
            for scale in horizontal_scales:
                rmse = output['scales'][str(scale)]['Q']['cross_month'][variable]['all']['rmse']
                if rmse is not None:
                    candidates.append((rmse, scale))
            output['development_selection_Q_only'][variable] = {
                'criterion': 'minimum 2018 Q point-pooled RMSE',
                'selected_horizontal_scale_km': min(candidates)[1] if candidates else None,
                'E_used': False,
            }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roles', type=Path,
                        default=ROOT / 'data/processed/argo_monthly/profile_roles.jsonl')
    parser.add_argument('--background-dir', type=Path,
                        default=ROOT / 'data/processed/baselines')
    parser.add_argument('--parameters', type=Path,
                        default=ROOT / 'research/audits/oi_parameters_2018.json')
    parser.add_argument('--target-roles', nargs='+', choices=['Q', 'E'], default=['Q', 'E'])
    parser.add_argument('--vertical-radius-m', type=float, default=50)
    parser.add_argument('--max-context-profiles', type=int, default=8)
    parser.add_argument('--localization-factor', type=float, default=3)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'research/audits/oi_baseline_2018.json')
    args = parser.parse_args()
    parameters = json.loads(args.parameters.read_text(encoding='utf-8'))
    if parameters.get('Q_used') or parameters.get('E_used'):
        raise ValueError('OI parameters must not use Q or E')
    horizontal_scales = parameters['horizontal_scale_candidates_km']
    profiles = load_profiles(args.roles)
    months = sorted({profile['month'] for profile in profiles})
    if months != [f'2018{month:02d}' for month in range(1, 13)]:
        raise ValueError('OI evaluation requires exactly twelve 2018 months')
    attach_background(profiles, args.background_dir)
    result = evaluate(profiles, parameters, horizontal_scales,
                      list(dict.fromkeys(args.target_roles)), args.vertical_radius_m,
                      args.max_context_profiles, args.localization_factor)
    result.update({
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': '2018 development OI; Q scale selection and fixed-platform E report-only',
        'context_role': 'C only', 'Q_used_for_parameter_fit': False,
        'E_used_for_parameter_fit_or_selection': False,
        'background_warning': 'Single-year LOMO engineering seasonal background, not multi-year climatology',
        'cora_warning': 'CORA/Argo may be source-related; metrics are not independent truth claims',
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'months': len(months), 'scales': horizontal_scales,
                      'selection': result.get('development_selection_Q_only')}, indent=2))


if __name__ == '__main__':
    main()
