"""QC adjusted Argo measurements, preserving pressure and independent T/S masks.

QC and TEOS-10 conversion; no interpolation or train/test assignment.
Outputs one file per source to bound memory independently of archive size.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import gsw
from netCDF4 import Dataset, chartostring

ROOT = Path(__file__).resolve().parents[1]
TARGET_DEPTH = np.r_[np.arange(5, 101, 5), np.arange(110, 501, 10)].astype(float)


def thermodynamics(p, t, s, lon, lat):
    """Argo adjusted PRES (dbar), in-situ TEMP (ITS-90), PSAL (PSS-78).

    Depth is positive downward. Joint T/S diagnostics retain missing values;
    depth and the original temperature remain usable without salinity.
    """
    depth = -gsw.z_from_p(p, lat[:, None])
    sa = gsw.SA_from_SP(s, p, lon[:, None], lat[:, None])
    ct = gsw.CT_from_t(sa, t, p)
    return depth, sa, ct, gsw.sigma0(sa, ct)


def interpolate_profile(depth, values, target=TARGET_DEPTH, max_gap=50):
    """Linear interpolation inside observed brackets only; average exact duplicate depths."""
    valid = np.isfinite(depth) & np.isfinite(values)
    if valid.sum() < 2:
        return np.full(target.shape, np.nan)
    x, y = depth[valid], values[valid]
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique, inverse = np.unique(x, return_inverse=True)
    if unique.size != x.size:
        sums = np.bincount(inverse, weights=y)
        y = sums / np.bincount(inverse)
        x = unique
    right = np.searchsorted(x, target)
    result = np.full(target.shape, np.nan)
    exact = (right < len(x)) & np.isclose(x[np.minimum(right, len(x) - 1)], target, atol=1e-6)
    result[exact] = y[right[exact]]
    bracket = (right > 0) & (right < len(x)) & ~exact
    bracket &= (x[np.minimum(right, len(x) - 1)] - x[np.maximum(right - 1, 0)]) <= max_gap
    result[bracket] = np.interp(target[bracket], x, y)
    return result


def chars(value):
    a = np.ma.filled(value, b' ')
    return np.char.strip(chartostring(a).astype(str)) if a.ndim > 1 else np.char.strip(a.astype(str))


def qc_parameter(ds, name, accepted):
    shape = (len(ds.dimensions['N_PROF']), len(ds.dimensions['N_LEVELS']))
    if name + '_ADJUSTED' not in ds.variables or name + '_ADJUSTED_QC' not in ds.variables:
        return np.full(shape, np.nan)
    values = np.ma.filled(ds[name + '_ADJUSTED'][:], np.nan).astype(float)
    qc = np.ma.filled(ds[name + '_ADJUSTED_QC'][:], b' ').astype(str)
    valid = np.isin(qc, accepted) & np.isfinite(values)
    modes = np.asarray(chars(ds['DATA_MODE'][:]))
    if 'PARAMETER_DATA_MODE' in ds.variables:
        params = chars(ds['STATION_PARAMETERS'][:])
        parameter_modes = np.ma.filled(ds['PARAMETER_DATA_MODE'][:], b' ').astype(str)
        modes = np.array([parameter_modes[i, np.flatnonzero(params[i] == name)[0]]
                          if np.any(params[i] == name) else 'R' for i in range(shape[0])])
    valid &= np.isin(modes, ['A', 'D'])[:, None]
    return np.where(valid, values, np.nan)


def summarize_existing(source, destination, max_gap):
    with Dataset(destination) as ds:
        required = {
            'pressure', 'temperature', 'practical_salinity',
            'absolute_salinity', 'conservative_temperature', 'sigma0',
            'temperature_on_target_depth', 'absolute_salinity_on_target_depth',
            'conservative_temperature_on_target_depth', 'sigma0_on_target_depth',
            'platform', 'cycle', 'direction', 'time',
        }
        if not required.issubset(ds.variables):
            raise ValueError(f'{destination}: incomplete processed Argo variables')
        if str(Path(ds.source_file).resolve()) != str(source.resolve()):
            raise ValueError(f'{destination}: source file mismatch')
        expected_rule = f'maximum observed bracket {max_gap:g} m'
        if expected_rule not in ds.vertical_interpolation:
            raise ValueError(f'{destination}: vertical interpolation rule mismatch')
        profiles = len(ds.dimensions['profile'])
        pressure = np.ma.filled(ds['pressure'][:], np.nan)
        temperature = np.ma.filled(ds['temperature'][:], np.nan)
        salinity = np.ma.filled(ds['practical_salinity'][:], np.nan)
        target_temperature = np.ma.filled(ds['temperature_on_target_depth'][:], np.nan)
        target_salinity = np.ma.filled(ds['absolute_salinity_on_target_depth'][:], np.nan)
        return {
            'source': str(source), 'output': str(destination), 'profiles': profiles,
            'eligible_profiles': int(np.isfinite(pressure).any(axis=1).sum()),
            'temperature_points': int(np.isfinite(temperature).sum()),
            'salinity_points': int(np.isfinite(salinity).sum()),
            'joint_points': int((np.isfinite(temperature) & np.isfinite(salinity)).sum()),
            'target_temperature_points': int(np.isfinite(target_temperature).sum()),
            'target_joint_points': int((np.isfinite(target_temperature) & np.isfinite(target_salinity)).sum()),
            'duplicate_pressure_profiles': sum(
                len(values[np.isfinite(values)]) != len(np.unique(values[np.isfinite(values)]))
                for values in pressure),
            'resumed_existing': True,
        }


def process(source, destination, accepted, max_gap):
    with Dataset(source) as ds:
        p, t, s = [qc_parameter(ds, name, accepted) for name in ('PRES', 'TEMP', 'PSAL')]
        lat = np.ma.filled(ds['LATITUDE'][:], np.nan)
        lon = np.mod(np.ma.filled(ds['LONGITUDE'][:], np.nan), 360)
        time = np.ma.filled(ds['JULD'][:], np.nan)
        position_ok = np.isin(np.ma.filled(ds['POSITION_QC'][:], b' ').astype(str), accepted)
        time_ok = np.isin(np.ma.filled(ds['JULD_QC'][:], b' ').astype(str), accepted)
        eligible = position_ok & time_ok & np.isfinite(time) & (lat >= -20) & (lat <= 20) & (lon >= 120) & (lon <= 280)
        valid_p = np.isfinite(p) & (p >= 0) & eligible[:, None]
        t = np.where(valid_p, t, np.nan)
        s = np.where(valid_p, s, np.nan)
        p = np.where(valid_p, p, np.nan)
        depth, sa, ct, sigma0 = thermodynamics(p, t, s, lon, lat)
        target_t = np.stack([interpolate_profile(depth[i], t[i], max_gap=max_gap) for i in range(len(p))])
        target_s = np.stack([interpolate_profile(depth[i], s[i], max_gap=max_gap) for i in range(len(p))])
        target_p = gsw.p_from_z(-TARGET_DEPTH[None, :], lat[:, None])
        target_sa = gsw.SA_from_SP(target_s, target_p, lon[:, None], lat[:, None])
        target_ct = gsw.CT_from_t(target_sa, target_t, target_p)
        target_sigma0 = gsw.sigma0(target_sa, target_ct)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.tmp.nc')
        with Dataset(temporary, 'w') as out:
            out.createDimension('profile', p.shape[0])
            out.createDimension('level', p.shape[1])
            out.createDimension('target_depth', len(TARGET_DEPTH))
            for name, values, units in [('pressure', p, 'dbar'), ('temperature', t, 'degree_Celsius'), ('practical_salinity', s, '1'),
                                        ('depth', depth, 'm'), ('absolute_salinity', sa, 'g kg-1'),
                                        ('conservative_temperature', ct, 'degree_Celsius'), ('sigma0', sigma0, 'kg m-3')]:
                v = out.createVariable(name, 'f8', ('profile', 'level'), zlib=True, fill_value=np.nan)
                v[:] = values
                v.units = units
            out['depth'].positive = 'down'
            out['temperature'].standard_name = 'sea_water_temperature'
            out['sigma0'].long_name = 'Potential density anomaly referenced to 0 dbar'
            z = out.createVariable('target_depth', 'f8', ('target_depth',))
            z[:] = TARGET_DEPTH
            z.units, z.positive = 'm', 'down'
            for name, values, units in [
                    ('temperature_on_target_depth', target_t, 'degree_Celsius'),
                    ('practical_salinity_on_target_depth', target_s, '1'),
                    ('absolute_salinity_on_target_depth', target_sa, 'g kg-1'),
                    ('conservative_temperature_on_target_depth', target_ct, 'degree_Celsius'),
                    ('sigma0_on_target_depth', target_sigma0, 'kg m-3')]:
                v = out.createVariable(name, 'f8', ('profile', 'target_depth'), zlib=True, fill_value=np.nan)
                v[:] = values
                v.units = units
            for name, values, units in [('latitude', lat, 'degrees_north'), ('longitude', lon, 'degrees_east'), ('time', time, ds['JULD'].units)]:
                v = out.createVariable(name, 'f8', ('profile',))
                v[:] = values
                v.units = units
            for name, values in [('platform', chars(ds['PLATFORM_NUMBER'][:])), ('direction', chars(ds['DIRECTION'][:]))]:
                out.createVariable(name, str, ('profile',))[:] = values
            out.createVariable('cycle', 'i4', ('profile',))[:] = ds['CYCLE_NUMBER'][:]
            out.createVariable('source_profile_index', 'i4', ('profile',))[:] = np.arange(p.shape[0])
            out.source_file = str(source.resolve())
            out.qc_rule = 'Adjusted only; parameter mode A/D; QC ' + ','.join(accepted)
            out.processing_stage = 'Adjusted QC, TEOS-10 conversion, and bounded vertical interpolation; no split assignment'
            out.vertical_interpolation = f'Linear, no extrapolation, duplicate-depth mean, maximum observed bracket {max_gap:g} m'
            out.gsw_version = gsw.__version__
        temporary.replace(destination)
    return {'source': str(source), 'output': str(destination), 'profiles': int(p.shape[0]),
            'eligible_profiles': int(eligible.sum()), 'temperature_points': int(np.isfinite(t).sum()),
            'salinity_points': int(np.isfinite(s).sum()), 'joint_points': int((np.isfinite(t) & np.isfinite(s)).sum()),
            'target_temperature_points': int(np.isfinite(target_t).sum()),
            'target_joint_points': int((np.isfinite(target_t) & np.isfinite(target_s)).sum()),
            'duplicate_pressure_profiles': sum(len(x[np.isfinite(x)]) != len(np.unique(x[np.isfinite(x)])) for x in p)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'data/raw/argo')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed/argo')
    parser.add_argument('--qc', nargs='+', choices=['1', '2'], default=['1'])
    parser.add_argument('--max-gap', type=float, default=50, help='Largest observed depth bracket allowed for interpolation (m)')
    args = parser.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError('Raw and output directories must differ')
    sources = sorted(args.input_dir.rglob('*.nc'))
    if not sources:
        raise FileNotFoundError('No Argo inputs')
    if args.max_gap <= 0:
        raise ValueError('--max-gap must be positive')
    records = []
    resumed = 0
    for source in sources:
        destination = args.output_dir / source.relative_to(args.input_dir)
        if destination.exists():
            records.append(summarize_existing(source, destination, args.max_gap))
            resumed += 1
        else:
            records.append(process(source, destination, args.qc, args.max_gap))
    audit = args.output_dir / 'manifest.json'
    audit.write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps({'files': len(records), 'resumed_existing_files': resumed,
                      **{key: sum(r[key] for r in records) for key in
                     ['profiles', 'eligible_profiles', 'temperature_points', 'salinity_points', 'joint_points',
                      'target_temperature_points', 'target_joint_points', 'duplicate_pressure_profiles']}}, indent=2))


if __name__ == '__main__':
    main()
