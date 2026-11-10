#!/usr/bin/env python3
"""Deduplicate Argo profiles, assign C/Q/E roles, and grid C inputs plus Q targets."""
import argparse
import calendar
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date

ROOT = Path(__file__).resolve().parents[1]


def hash_fraction(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') / 2**64


def platform_role(platform, evaluation_fraction, seed):
    """E is fixed by platform for the full archive; other platforms remain development candidates."""
    return 'E' if hash_fraction(f'{seed}:E:{platform}') < evaluation_fraction else 'development'


def profile_role(platform, cycle, direction, timestamp, context_fraction, evaluation_fraction, seed):
    if platform_role(platform, evaluation_fraction, seed) == 'E':
        return 'E'
    identity = f'{platform}:{cycle}:{direction}:{timestamp:.8f}'
    return 'C' if hash_fraction(f'{seed}:CQ:{identity}') < context_fraction else 'Q'


def nearest(values, point):
    return int(np.abs(values - point).argmin())


def load_candidates(input_dir, evaluation_fraction, context_fraction, seed):
    best = {}
    for path in sorted(input_dir.rglob('*.nc')):
        with Dataset(path) as ds:
            units = ds['time'].units
            for i in range(len(ds.dimensions['profile'])):
                platform = str(ds['platform'][i])
                cycle = int(ds['cycle'][i])
                direction = str(ds['direction'][i])
                timestamp = float(ds['time'][i])
                decoded = num2date(timestamp, units)
                month = decoded.strftime('%Y%m')
                temperature = np.ma.filled(ds['conservative_temperature_on_target_depth'][i], np.nan).astype(float)
                salinity = np.ma.filled(ds['absolute_salinity_on_target_depth'][i], np.nan).astype(float)
                score = (int(np.sum(np.isfinite(temperature) & np.isfinite(salinity))), int(np.isfinite(temperature).sum()))
                latitude, longitude = float(ds['latitude'][i]), float(ds['longitude'][i])
                if score == (0, 0) or not (-20 <= latitude <= 20 and 120 <= longitude <= 280):
                    continue
                key = (platform, cycle, direction, timestamp)
                days = calendar.monthrange(decoded.year, decoded.month)[1]
                day_value = decoded.day + (decoded.hour + decoded.minute / 60 + decoded.second / 3600) / 24
                candidate = dict(
                    month=month, platform=platform, cycle=cycle, direction=direction,
                    time=timestamp, time_units=units,
                    time_offset_days=float(day_value - (days + 1) / 2),
                    latitude=latitude, longitude=longitude,
                    temperature=temperature, salinity=salinity,
                    source=str(path.relative_to(ROOT)), source_profile_index=i,
                    role=profile_role(platform, cycle, direction, timestamp, context_fraction,
                                      evaluation_fraction, seed), score=score)
                if key not in best or score > best[key]['score']:
                    best[key] = candidate
    return list(best.values())


def accumulate(profiles, role_name, shape, lat, lon):
    sums_t, sums_s = np.zeros(shape), np.zeros(shape)
    counts_t, counts_s = np.zeros(shape, dtype=np.uint16), np.zeros(shape, dtype=np.uint16)
    offset_sums = {name: np.zeros(shape[1:]) for name in ('latitude', 'longitude', 'time')}
    offset_count = np.zeros(shape[1:], dtype=np.uint16)
    for profile in profiles:
        if profile['role'] != role_name:
            continue
        iy, ix = nearest(lat, profile['latitude']), nearest(lon, profile['longitude'])
        for values, sums, counts in ((profile['temperature'], sums_t, counts_t),
                                     (profile['salinity'], sums_s, counts_s)):
            valid = np.isfinite(values)
            sums[valid, iy, ix] += values[valid]
            counts[valid, iy, ix] += 1
        longitude_offset = (profile['longitude'] - lon[ix] + 180) % 360 - 180
        offset_sums['latitude'][iy, ix] += profile['latitude'] - lat[iy]
        offset_sums['longitude'][iy, ix] += longitude_offset
        offset_sums['time'][iy, ix] += profile['time_offset_days']
        offset_count[iy, ix] += 1
    return sums_t, sums_s, counts_t, counts_s, offset_sums, offset_count


def exclude_context_overlap(context, query):
    query = list(query)
    context_counts_t, context_counts_s = context[2], context[3]
    excluded_t = int(((query[2] > 0) & (context_counts_t > 0)).sum())
    excluded_s = int(((query[3] > 0) & (context_counts_s > 0)).sum())
    for sums_index, counts_index, context_counts in (
            (0, 2, context_counts_t), (1, 3, context_counts_s)):
        overlap = context_counts > 0
        query[sums_index][overlap] = 0
        query[counts_index][overlap] = 0
    return tuple(query), {'temperature': excluded_t, 'salinity': excluded_s}


def write_fields(out, prefix, accumulated, shape):
    sums_t, sums_s, counts_t, counts_s, offset_sums, offset_count = accumulated
    for suffix, sums, counts, units in [
            ('conservative_temperature', sums_t, counts_t, 'degree_Celsius'),
            ('absolute_salinity', sums_s, counts_s, 'g kg-1')]:
        values = np.divide(sums, counts, out=np.full(shape, np.nan), where=counts > 0)
        variable = out.createVariable(prefix + suffix, 'f4', ('depth', 'latitude', 'longitude'),
                                      zlib=True, fill_value=np.nan)
        variable[:] = values
        variable.units = units
        out.createVariable(prefix + suffix + '_count', 'u2',
                           ('depth', 'latitude', 'longitude'), zlib=True)[:] = counts
    for name, units in [('latitude', 'degrees_north'), ('longitude', 'degrees_east'), ('time', 'days')]:
        values = np.divide(offset_sums[name], offset_count,
                           out=np.full(shape[1:], np.nan), where=offset_count > 0)
        variable = out.createVariable(prefix + name + '_offset', 'f4', ('latitude', 'longitude'),
                                      zlib=True, fill_value=np.nan)
        variable[:] = values
        variable.units = units
    out.createVariable(prefix + 'profile_count', 'u2', ('latitude', 'longitude'), zlib=True)[:] = offset_count


def write_month(month, profiles, cora, output, evaluation_fraction, context_fraction, seed):
    with Dataset(cora) as grid:
        depth = np.asarray(grid['depth'][:])
        lat = np.asarray(grid['latitude'][:])
        lon = np.asarray(grid['longitude'][:])
        shape = (len(depth), len(lat), len(lon))
        context = accumulate(profiles, 'C', shape, lat, lon)
        query = accumulate(profiles, 'Q', shape, lat, lon)
        query, excluded_overlap = exclude_context_overlap(context, query)
        temporary = output.with_suffix('.tmp.nc')
        output.parent.mkdir(parents=True, exist_ok=True)
        with Dataset(temporary, 'w') as out:
            for name, values, units in [('depth', depth, 'm'), ('latitude', lat, 'degrees_north'), ('longitude', lon, 'degrees_east')]:
                out.createDimension(name, len(values))
                variable = out.createVariable(name, 'f4', (name,))
                variable[:] = values
                variable.units = units
            write_fields(out, 'argo_', context, shape)
            write_fields(out, 'argo_q_', query, shape)
            out.month = month
            out.source_cora_grid = str(cora.relative_to(ROOT))
            out.profile_role_rule = (
                f'E=SHA256({seed}:E:platform), fraction {evaluation_fraction:g}; '
                f'non-E profile C/Q=SHA256({seed}:CQ:profile), C fraction {context_fraction:g}')
            out.context_profiles = sum(profile['role'] == 'C' for profile in profiles)
            out.query_profiles = sum(profile['role'] == 'Q' for profile in profiles)
            out.evaluation_profiles = sum(profile['role'] == 'E' for profile in profiles)
            out.query_temperature_points_excluded_context_overlap = excluded_overlap['temperature']
            out.query_salinity_points_excluded_context_overlap = excluded_overlap['salinity']
        temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'data/processed/argo')
    parser.add_argument('--cora-dir', type=Path, default=ROOT / 'data/processed/cora')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed/argo_monthly')
    parser.add_argument('--evaluation-fraction', type=float, default=.2)
    parser.add_argument('--context-fraction', type=float, default=.5)
    parser.add_argument('--seed', default='argo-v2-cqe')
    args = parser.parse_args()
    if not 0 < args.evaluation_fraction < 1:
        raise ValueError('--evaluation-fraction must be in (0, 1)')
    if not 0 < args.context_fraction <= 1:
        raise ValueError('--context-fraction must be in (0, 1]')
    candidates = load_candidates(args.input_dir, args.evaluation_fraction, args.context_fraction, args.seed)
    if not candidates:
        raise FileNotFoundError('No processed Argo profiles')
    by_month = defaultdict(list)
    for profile in candidates:
        by_month[profile['month']].append(profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'profile_roles.jsonl').open('w', encoding='utf-8') as stream:
        for profile in sorted(candidates, key=lambda item: (item['month'], item['platform'], item['cycle'], item['direction'])):
            stream.write(json.dumps({key: value for key, value in profile.items()
                                     if key not in ('temperature', 'salinity', 'score')}) + '\n')
    for month, profiles in sorted(by_month.items()):
        matches = list(args.cora_dir.glob(f'cora_{month}_*.nc'))
        if len(matches) != 1:
            raise FileNotFoundError(f'Expected one CORA grid for {month}, found {len(matches)}')
        output = args.output_dir / f'argo_{month}_on_cora_grid.nc'
        write_month(month, profiles, matches[0], output, args.evaluation_fraction, args.context_fraction, args.seed)
        print(output.relative_to(ROOT))


if __name__ == '__main__':
    main()
