#!/usr/bin/env python3
"""Regrid processed monthly surface fields to the matching CORA grid."""
import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {
    'oisst': ('sst', 'err'),
    'adt': ('adt',),
    'era5': ('u10', 'v10'),
    'smap': ('sss', 'mean_window_uncertainty'),
}


def normalized_linear(values, source_lat, source_lon, target_lat, target_lon, minimum_weight=0.5):
    """Mask-aware bilinear interpolation without extrapolation."""
    valid = np.isfinite(values)
    points = np.stack(np.meshgrid(target_lat, target_lon, indexing='ij'), axis=-1)
    kwargs = dict(bounds_error=False, fill_value=0.0)
    numerator = RegularGridInterpolator((source_lat, source_lon), np.where(valid, values, 0.0), **kwargs)(points)
    weight = RegularGridInterpolator((source_lat, source_lon), valid.astype(float), **kwargs)(points)
    return np.where(weight >= minimum_weight, numerator / np.where(weight > 0, weight, 1), np.nan), weight


def process(source, cora, destination, minimum_weight):
    with Dataset(source) as src, Dataset(cora) as target:
        source_lat = np.asarray(src['latitude'][:])
        source_lon = np.asarray(src['longitude'][:])
        target_lat = np.asarray(target['latitude'][:])
        target_lon = np.asarray(target['longitude'][:])
        kind = source.name.split('_')[0]
        temporary = destination.with_suffix('.tmp.nc')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Dataset(temporary, 'w') as out:
            out.createDimension('time', 1)
            out.createDimension('latitude', len(target_lat))
            out.createDimension('longitude', len(target_lon))
            for name, data, dims, units in [
                    ('time', target['time'][:], ('time',), target['time'].units),
                    ('latitude', target_lat, ('latitude',), 'degrees_north'),
                    ('longitude', target_lon, ('longitude',), 'degrees_east')]:
                v = out.createVariable(name, 'f8' if name == 'time' else 'f4', dims)
                v[:] = data
                v.units = units
            for name in FIELDS[kind]:
                if name not in src.variables:
                    raise KeyError(f'{source}: missing {name}')
                values = np.ma.filled(src[name][0], np.nan).astype(float)
                mapped, weight = normalized_linear(values, source_lat, source_lon, target_lat, target_lon, minimum_weight)
                v = out.createVariable(name, 'f4', ('time', 'latitude', 'longitude'), zlib=True, fill_value=np.nan)
                v[0] = mapped
                v.units = getattr(src[name], 'units', '')
                m = out.createVariable(name + '_regrid_weight', 'f4', ('time', 'latitude', 'longitude'), zlib=True)
                m[0] = weight
            out.source_file = str(source.relative_to(ROOT))
            out.target_grid_file = str(cora.relative_to(ROOT))
            out.regridding = f'mask-normalized bilinear; no extrapolation; minimum valid weight {minimum_weight:g}'
        temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=FIELDS)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--cora-dir', type=Path, default=ROOT / 'data/processed/cora')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed/on_cora_grid')
    parser.add_argument('--minimum-weight', type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.minimum_weight <= 1:
        raise ValueError('--minimum-weight must be in (0, 1]')
    input_dir = args.input_dir or ROOT / 'data/processed' / args.source
    files = sorted(input_dir.glob(f'{args.source}_??????_monthly.nc'))
    if not files:
        raise FileNotFoundError(f'No monthly {args.source} files in {input_dir}')
    for source in files:
        month = source.name.split('_')[1]
        matches = list(args.cora_dir.glob(f'cora_{month}_*.nc'))
        if len(matches) != 1:
            raise FileNotFoundError(f'Expected one CORA target for {month}, found {len(matches)}')
        destination = args.output_dir / f'{args.source}_{month}_on_cora_grid.nc'
        process(source, matches[0], destination, args.minimum_weight)
        print(destination.relative_to(ROOT))


if __name__ == '__main__':
    main()
