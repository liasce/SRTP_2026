"""Stream complete calendar months of OISST, DUACS ADT, or ERA5 on native grids.

No interpolation or training statistics. Daily samples must cover every day.
Per-pixel monthly means require >=80% finite daily values; counts are retained.
"""
import argparse
import calendar
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SPECS = {'oisst': ['sst', 'err', 'ice'], 'adt': ['adt'], 'era5': ['u10', 'v10']}


def normalize(ds):
    ds = ds.rename({k: v for k, v in [('lon', 'longitude'), ('lat', 'latitude'), ('valid_time', 'time')] if k in ds.dims})
    if 'zlev' in ds.dims:
        if ds.sizes['zlev'] != 1:
            raise ValueError('Unexpected multiple surface depths')
        ds = ds.isel(zlev=0, drop=True)
    ds = ds.assign_coords(longitude=ds.longitude % 360).sortby('longitude').sortby('latitude')
    ds = ds.sel(longitude=slice(120, 280), latitude=slice(-20, 20))
    if not ds.sizes['longitude'] or not ds.sizes['latitude']:
        raise ValueError('Empty region')
    for coord in ['longitude', 'latitude']:
        if not np.all(np.diff(ds[coord].values) > 0):
            raise ValueError('Duplicate or unordered coordinates')
    return ds


def calendar_check(days, month, monthly=False):
    expected = {month + '-01'} if monthly else {f'{month}-{d:02}' for d in range(1, calendar.monthrange(int(month[:4]), int(month[5:]))[1] + 1)}
    if set(days) != expected:
        raise ValueError(f'{month}: missing days {sorted(expected - set(days))}; unexpected {sorted(set(days) - expected)}')


def validate_monthly_output(path, source, month, expected_samples):
    with xr.open_dataset(path) as ds:
        if ds.sizes.get('time') != 1 or str(np.datetime64(ds.time.values[0], 'M')) != month:
            raise ValueError(f'{path}: monthly timestamp mismatch')
        if int(ds.attrs.get('temporal_samples', -1)) != expected_samples:
            raise ValueError(f'{path}: temporal sample count mismatch')
        for name in SPECS[source]:
            if name not in ds or name + '_count' not in ds:
                raise ValueError(f'{path}: missing {name} or count')
        required_finite = ['sst'] if source == 'oisst' else SPECS[source]
        if any(not bool(np.isfinite(ds[name]).any()) for name in required_finite):
            raise ValueError(f'{path}: required field entirely missing')
    return path


def process(source, input_dir, output_dir, min_fraction):
    # Index time metadata only; each day is loaded separately, bounded for long archives.
    groups = defaultdict(lambda: defaultdict(list))
    paths = sorted(input_dir.rglob('*.nc'))
    if not paths:
        raise FileNotFoundError(input_dir)
    for path in paths:
        with xr.open_dataset(path) as ds:
            time_name = 'valid_time' if 'valid_time' in ds else 'time'
            for i, stamp in enumerate(ds[time_name].values):
                day = str(np.datetime64(stamp, 'D'))
                groups[day[:7]][day].append((path, time_name, i))
    outputs = []
    for month, days in sorted(groups.items()):
        calendar_check(days, month, source == 'era5')
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f'{source}_{month.replace("-", "")}_monthly.nc'
        if target.exists():
            outputs.append(validate_monthly_output(target, source, month, len(days)))
            print('VALID', target)
            continue
        sums, counts, attrs = {}, {}, {}
        coordinates = None
        for day, entries in sorted(days.items()):
            parts = []
            for path, time_name, index in entries:
                with xr.open_dataset(path) as ds:
                    part = normalize(ds.isel({time_name: index}, drop=True))[SPECS[source]].load()
                    parts.append(part)
            if len(parts) > 1:
                if source not in ('adt', 'era5'):
                    raise ValueError(f'Duplicate {source} day: {day}')
                # Regional blocks overlap at 180 degrees; values there must agree.
                field = xr.concat(parts, dim='longitude').sortby('longitude')
                _, indices, counts_at_longitude = np.unique(
                    field.longitude.values, return_index=True, return_counts=True)
                for lon in field.longitude.values[indices[counts_at_longitude > 1]]:
                    for name in SPECS[source]:
                        repeated = field[name].sel(longitude=lon).values
                        if not np.allclose(repeated, repeated[..., :1], equal_nan=True):
                            raise ValueError(f'Conflicting {source} overlap for {name}')
                field = field.isel(longitude=np.sort(indices))
            else:
                field = parts[0]
            coords = (field.latitude.values, field.longitude.values)
            if coordinates is None:
                coordinates = coords
            elif not all(np.array_equal(a, b) for a, b in zip(coordinates, coords)):
                raise ValueError('Grid changed within month')
            for name in SPECS[source]:
                values = field[name].transpose('latitude', 'longitude').values.astype(float)
                valid = np.isfinite(values)
                if name not in sums:
                    sums[name] = np.zeros(values.shape)
                    counts[name] = np.zeros(values.shape, dtype=np.int32)
                    attrs[name] = dict(field[name].attrs)
                sums[name] += np.where(valid, values, 0)
                counts[name] += valid
        out = xr.Dataset(coords={'time': [np.datetime64(month + '-01', 'ns')], 'latitude': coordinates[0], 'longitude': coordinates[1]})
        for name in SPECS[source]:
            count = counts[name]
            mean = np.divide(sums[name], count, out=np.full(count.shape, np.nan), where=count > 0)
            mean[count < min_fraction * len(days)] = np.nan
            out[name] = (('time', 'latitude', 'longitude'), mean[None].astype('f4'), attrs[name])
            out[name + '_count'] = (('time', 'latitude', 'longitude'), count[None])
        out.attrs.update(source=source, processing='Native grid calendar-month mean', minimum_valid_fraction=min_fraction,
                         temporal_samples=len(days), input_files='\n'.join(str(p.resolve()) for p in sorted({e[0] for entries in days.values() for e in entries})),
                         uncertainty_note='err is mean input error estimate, not uncertainty of the monthly mean',
                         stage='Monthly native-grid intermediate; not regridded or standardized')
        temporary = target.with_suffix('.tmp.nc')
        out.to_netcdf(temporary, engine='netcdf4', encoding={n: {'zlib': True, 'complevel': 1} for n in out.data_vars})
        with xr.open_dataset(temporary) as check:
            assert check.sizes['time'] == 1
            for name in (['sst'] if source == 'oisst' else SPECS[source]):
                assert np.isfinite(check[name]).any(), name
        temporary.replace(target)
        outputs.append(target)
        print(target)
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', choices=SPECS)
    p.add_argument('--input-dir', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--min-fraction', type=float, default=.8)
    a = p.parse_args()
    if not 0 < a.min_fraction <= 1:
        p.error('min-fraction must be in (0,1]')
    process(a.source, a.input_dir or ROOT / 'data/raw' / a.source,
            a.output_dir or ROOT / 'data/processed' / a.source, a.min_fraction)


if __name__ == '__main__':
    main()
