"""Monthly mean of complete nominal-day SMAP 8-day running fields.

Preserves actual temporal support. Output is an average of overlapping windows,
not a daily-sampled monthly mean. No reference-salinity substitution.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import xarray as xr
from audit_smap_windows import audit, ROOT
from preprocess_surface import normalize


def process(input_dir, output_dir, months, min_fraction=.8):
    coverage = audit(input_dir, months)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'coverage.json').write_text(json.dumps(coverage, indent=2) + '\n')
    if not all(m['calendar_complete'] for m in coverage['months']):
        raise ValueError('Incomplete SMAP calendar; see coverage.json. No monthly field generated.')
    results = []
    for month in months:
        records = [r for r in coverage['files'] if r['nominal_date'].replace('-', '').startswith(month)]
        total = count = uncertainty = uncertainty_count = coords = None
        for record in records:
            with xr.open_dataset(input_dir / record['path']) as ds:
                ds = normalize(ds)
                sss = ds.sss_smap.transpose('lat' if 'lat' in ds.dims else 'latitude', 'longitude').values.astype(float)
                unc = ds.sss_smap_unc.values.astype(float)
                valid = np.isfinite(sss) & (sss >= 0) & (sss <= 45) & (ds.nobs.values > 0)
                current = (ds.latitude.values, ds.longitude.values)
                if coords is None:
                    coords = current
                    total = np.zeros(sss.shape)
                    count = np.zeros(sss.shape, dtype='i4')
                    uncertainty = np.zeros(sss.shape)
                    uncertainty_count = np.zeros(sss.shape, dtype='i4')
                elif not all(np.array_equal(a, b) for a, b in zip(coords, current)):
                    raise ValueError('SMAP grid changes within month')
                total += np.where(valid, sss, 0)
                count += valid
                good_unc = valid & np.isfinite(unc) & (unc >= 0)
                uncertainty += np.where(good_unc, unc, 0)
                uncertainty_count += good_unc
        mean = np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)
        mean[count < min_fraction * len(records)] = np.nan
        unc_mean = np.divide(uncertainty, uncertainty_count, out=np.full(total.shape, np.nan), where=uncertainty_count > 0)
        out = xr.Dataset(coords={'time': [np.datetime64(f'{month[:4]}-{month[4:]}-01', 'ns')],
                                 'latitude': coords[0], 'longitude': coords[1]})
        for name, values in [('sss', mean), ('sss_window_count', count), ('mean_window_uncertainty', unc_mean)]:
            out[name] = (('time', 'latitude', 'longitude'), values[None])
        out.sss.attrs.update(units='1e-3', long_name='Mean of SMAP 8-day salinity fields at nominal dates in month')
        out.attrs.update(support_start=min(r['support_start'] for r in records), support_end=max(r['support_end'] for r in records),
                         uncertainty_note='Mean per-window uncertainty; overlapping windows are correlated, not independent samples.',
                         minimum_valid_fraction=min_fraction, nominal_windows=len(records),
                         source_files='\n'.join(r['path'] for r in records))
        target = output_dir / f'smap_{month}_monthly.nc'
        part = target.with_suffix('.tmp.nc')
        out.to_netcdf(part, encoding={n: {'zlib': True, 'complevel': 1} for n in out.data_vars})
        with xr.open_dataset(part) as check:
            if not np.isfinite(check.sss).any():
                raise ValueError('No valid monthly SMAP salinity')
        part.replace(target)
        results.append(target)
        print(target)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=ROOT / 'data/raw/smap')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed/smap')
    p.add_argument('--months', nargs='+', required=True)
    p.add_argument('--min-fraction', type=float, default=.8)
    a = p.parse_args()
    if not 0 < a.min_fraction <= 1:
        p.error('min-fraction must be in (0,1]')
    process(a.input_dir, a.output_dir, a.months, a.min_fraction)


if __name__ == '__main__':
    main()
