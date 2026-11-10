"""Check calendar failures and numerical aggregation independently of real files."""
import tempfile
from pathlib import Path
import numpy as np
import xarray as xr
from preprocess_surface import calendar_check, process


def main():
    try:
        calendar_check(['2020-02-01'], '2020-02')
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete month accepted')
    with tempfile.TemporaryDirectory() as folder:
        raw = Path(folder) / 'raw'
        raw.mkdir()
        days = np.arange(np.datetime64('2020-02-01'), np.datetime64('2020-03-01')).astype('datetime64[ns]')
        data = np.arange(29, dtype=float)[:, None, None] * np.ones((29, 2, 2))
        data[:10, 0, 0] = np.nan
        ds = xr.Dataset({'sst': (('time', 'lat', 'lon'), data), 'err': (('time', 'lat', 'lon'), np.ones_like(data)),
                         'ice': (('time', 'lat', 'lon'), np.zeros_like(data))},
                        coords={'time': days, 'lat': [1., 0.], 'lon': [-179., 179.]})
        ds.to_netcdf(raw / 'test.nc')
        outputs = process('oisst', raw, Path(folder) / 'out', .8)
        with xr.open_dataset(outputs[0]) as out:
            assert out.sst.sel(latitude=0, longitude=179).item() == 14
            assert np.isnan(out.sst.sel(latitude=1, longitude=181).item())
            assert out.sst_count.sel(latitude=1, longitude=181).item() == 19
            assert out.attrs['temporal_samples'] == 29

        era5_raw = Path(folder) / 'era5_raw'
        era5_raw.mkdir()
        latitude = [20.25, 0.0, -20.25]
        west_lon, east_lon = [120.0, 180.0], [-180.0, -80.0]
        for region, longitude in [('west', west_lon), ('east', east_lon)]:
            values = np.ones((1, len(latitude), len(longitude)), dtype=np.float32)
            if region == 'east':
                values[..., 0] = 1.0
            block = xr.Dataset(
                {'u10': (('valid_time', 'latitude', 'longitude'), values),
                 'v10': (('valid_time', 'latitude', 'longitude'), values * 2)},
                coords={'valid_time': [np.datetime64('2018-01-01')],
                        'latitude': latitude, 'longitude': longitude})
            block.to_netcdf(era5_raw / f'{region}.nc')
        era5_outputs = process('era5', era5_raw, Path(folder) / 'era5_out', .8)
        with xr.open_dataset(era5_outputs[0]) as out:
            assert out.sizes['longitude'] == 3
            assert np.array_equal(out.longitude.values, [120.0, 180.0, 280.0])
            assert out.attrs['temporal_samples'] == 1
    print('Surface calendar, coordinates, means, and coverage checks passed')


if __name__ == '__main__':
    main()
