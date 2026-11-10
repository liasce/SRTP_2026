#!/usr/bin/env python3
"""Pair QC-filtered TAO temperature/salinity rows and derive TEOS-10 quantities."""
import argparse
from pathlib import Path

import gsw
import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]


def rows(path, variable):
    with Dataset(path) as ds:
        station_ids = np.asarray(ds['station_id'][:]).astype(str)
        station = station_ids[np.asarray(ds['station_index'][:], dtype=int)]
        result = {}
        for i in range(len(ds.dimensions['observation'])):
            key = (station[i], int(ds['time'][i]), round(float(ds['depth'][i]), 3))
            result[key] = (float(ds[variable][i]), float(ds['latitude'][i]), float(ds['longitude'][i]))
        return result, ds['time'].units


def process(temperature_path, salinity_path, output):
    temperatures, time_units = rows(temperature_path, 'temperature')
    salinities, salinity_time_units = rows(salinity_path, 'practical_salinity')
    if time_units != salinity_time_units:
        raise ValueError('TAO time units differ between temperature and salinity')
    keys = sorted(temperatures.keys() & salinities.keys(), key=lambda x: (x[1], x[0], x[2]))
    if not keys:
        raise ValueError('No collocated TAO temperature/salinity rows')
    station = np.array([k[0] for k in keys])
    time = np.array([k[1] for k in keys], dtype=np.int64)
    depth = np.array([k[2] for k in keys])
    temperature = np.array([temperatures[k][0] for k in keys])
    salinity = np.array([salinities[k][0] for k in keys])
    lat = np.array([temperatures[k][1] for k in keys])
    lon = np.array([temperatures[k][2] for k in keys])
    pressure = gsw.p_from_z(-depth, lat)
    absolute_salinity = gsw.SA_from_SP(salinity, pressure, lon, lat)
    conservative_temperature = gsw.CT_from_t(absolute_salinity, temperature, pressure)
    sigma0 = gsw.sigma0(absolute_salinity, conservative_temperature)
    temporary = output.with_suffix('.tmp.nc')
    output.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(temporary, 'w') as out:
        out.createDimension('observation', len(keys))
        out.createVariable('station_id', str, ('observation',))[:] = station
        for name, values, dtype, units in [
                ('time', time, 'i8', time_units), ('latitude', lat, 'f4', 'degrees_north'),
                ('longitude', lon, 'f4', 'degrees_east'), ('depth', depth, 'f4', 'm'),
                ('pressure', pressure, 'f4', 'dbar'), ('temperature', temperature, 'f4', 'degree_Celsius'),
                ('practical_salinity', salinity, 'f4', '1'), ('absolute_salinity', absolute_salinity, 'f4', 'g kg-1'),
                ('conservative_temperature', conservative_temperature, 'f4', 'degree_Celsius'),
                ('sigma0', sigma0, 'f4', 'kg m-3')]:
            v = out.createVariable(name, dtype, ('observation',), zlib=True)
            v[:] = values
            v.units = units
        out.source_temperature_file = str(temperature_path.relative_to(ROOT))
        out.source_salinity_file = str(salinity_path.relative_to(ROOT))
        out.pairing = 'Exact station, daily timestamp, and depth rounded to 0.001 m'
        out.gsw_version = gsw.__version__
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'data/processed/tao')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed/tao')
    args = parser.parse_args()
    temperature_files = sorted(args.input_dir.glob('tao_??????_T_qc_5_500m.nc'))
    if not temperature_files:
        raise FileNotFoundError('No processed TAO temperature files')
    for temperature in temperature_files:
        month = temperature.name.split('_')[1]
        salinity = args.input_dir / f'tao_{month}_S_qc_5_500m.nc'
        if not salinity.exists():
            raise FileNotFoundError(salinity)
        output = args.output_dir / f'tao_{month}_TS_paired_5_500m.nc'
        process(temperature, salinity, output)
        with Dataset(output) as ds:
            assert np.isfinite(ds['sigma0'][:]).all()
            print(output.relative_to(ROOT), len(ds.dimensions['observation']))


if __name__ == '__main__':
    main()
