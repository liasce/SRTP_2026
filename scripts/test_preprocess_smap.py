"""Known-answer SMAP window aggregation and missing-calendar rejection."""
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import xarray as xr
from preprocess_smap import process


def main():
    with tempfile.TemporaryDirectory() as folder:
        raw, out = Path(folder) / 'raw', Path(folder) / 'out'
        raw.mkdir()
        for day in range(1, 30):
            date = datetime(2020, 2, day, 12, tzinfo=timezone.utc)
            ds = xr.Dataset({'sss_smap': (('lat', 'lon'), [[float(day), 35.]]),
                             'sss_smap_unc': (('lat', 'lon'), [[.1, .2]]),
                             'nobs': (('lat', 'lon'), [[1, 0]])},
                            coords={'lat': [0.], 'lon': [150., 151.], 'time': [np.datetime64(date.replace(tzinfo=None), 'ns')]})
            ds.attrs.update(time_coverage_start=(date - timedelta(days=4)).isoformat(),
                            time_coverage_end=(date + timedelta(days=4)).isoformat())
            ds.to_netcdf(raw / f'{day:02}.nc')
        outputs = process(raw, out, ['202002'])
        with xr.open_dataset(outputs[0]) as ds:
            assert ds.sss.values[0, 0, 0] == 15
            assert np.isnan(ds.sss.values[0, 0, 1])
            assert ds.sss_window_count.values[0, 0, 0] == 29
            assert ds.attrs['support_start'].startswith('2020-01-28')
        (raw / '29.nc').rename(raw / '29.disabled')
        try:
            process(raw, out / 'incomplete', ['202002'])
        except ValueError:
            assert not list((out / 'incomplete').glob('*.nc'))
        else:
            raise AssertionError('Incomplete month accepted')
    print('SMAP known mean, observation mask, support window, missing-day checks passed')


if __name__ == '__main__':
    main()
