#!/usr/bin/env python3
"""Create repeatable tropical-Pacific CORA subsets from native monthly field files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gsw
import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw/cora/original_202603"
DEFAULT_OUTPUT = ROOT / "data/processed/cora"


def indices(lon: np.ndarray, lat: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon360 = np.mod(lon, 360)
    ix = np.flatnonzero((lon360 >= 120) & (lon360 <= 280))
    iy = np.flatnonzero((lat >= -20) & (lat <= 20))
    iz = np.flatnonzero((depth >= 5) & (depth <= 500))
    if not (ix.size and iy.size and iz.size):
        raise ValueError("Requested tropical-Pacific 5–500 m subset is empty")
    return ix, iy, iz


def copy_variable(out: Dataset, name: str, source, iz: np.ndarray, iy: np.ndarray, ix: np.ndarray) -> None:
    values = np.ma.filled(source[0, iz, iy, ix], np.nan).astype("f4")
    target = out.createVariable(name, "f4", ("time", "depth", "latitude", "longitude"),
                                zlib=True, complevel=1, fill_value=np.nan)
    target[0] = values
    for attr in ("units", "long_name", "standard_name"):
        if hasattr(source, attr):
            setattr(target, attr, getattr(source, attr))


def process(temp_path: Path, output_dir: Path, overwrite: bool) -> Path:
    sal_path = temp_path.with_name(temp_path.name.replace("_TEMP.nc", "_PSAL.nc"))
    if not sal_path.exists():
        raise FileNotFoundError(f"Missing paired salinity field: {sal_path.name}")
    month = temp_path.name.split("_")[2][:6]
    output = output_dir / f"cora_{month}_tropical_pacific_5_500m.nc"
    if output.exists() and not overwrite:
        return output

    with Dataset(temp_path) as temp_ds, Dataset(sal_path) as sal_ds:
        lon = np.asarray(temp_ds["longitude"][:])
        lat = np.asarray(temp_ds["latitude"][:])
        depth = np.asarray(temp_ds["depth"][:])
        ix, iy, iz = indices(lon, lat, depth)
        order = np.argsort(np.mod(lon[ix], 360))
        ix = ix[order]
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.tmp.nc')
        with Dataset(temporary, "w") as out:
            out.createDimension("time", 1)
            out.createDimension("depth", len(iz))
            out.createDimension("latitude", len(iy))
            out.createDimension("longitude", len(ix))
            for name, values, dims, units in [
                ("time", temp_ds["time"][:], ("time",), getattr(temp_ds["time"], "units", "")),
                ("depth", depth[iz], ("depth",), "m"),
                ("latitude", lat[iy], ("latitude",), "degrees_north"),
                ("longitude", np.mod(lon[ix], 360), ("longitude",), "degrees_east"),
            ]:
                variable = out.createVariable(name, "f8" if name == "time" else "f4", dims)
                variable[:] = values
                variable.units = units
            for name in ("TEMP", "TEMP_ERR", "TEMP_PCTVAR"):
                copy_variable(out, name, temp_ds[name], iz, iy, ix)
            for name in ("PSAL", "PSAL_ERR", "PSAL_PCTVAR"):
                copy_variable(out, name, sal_ds[name], iz, iy, ix)
            if getattr(temp_ds['TEMP'], 'standard_name', '').strip() != 'sea_water_temperature':
                raise ValueError('CORA TEMP is not explicitly identified as in-situ sea_water_temperature')
            temperature = np.ma.filled(temp_ds['TEMP'][0, iz, iy, ix], np.nan).astype(float)
            salinity = np.ma.filled(sal_ds['PSAL'][0, iz, iy, ix], np.nan).astype(float)
            latitude = lat[iy]
            longitude = np.mod(lon[ix], 360)
            pressure = gsw.p_from_z(-depth[iz, None, None], latitude[None, :, None])
            absolute_salinity = gsw.SA_from_SP(salinity, pressure, longitude[None, None, :], latitude[None, :, None])
            conservative_temperature = gsw.CT_from_t(absolute_salinity, temperature, pressure)
            sigma0 = gsw.sigma0(absolute_salinity, conservative_temperature)
            pressure_3d = np.broadcast_to(pressure, sigma0.shape)
            latitude_3d = np.broadcast_to(latitude[None, :, None], sigma0.shape)
            n2, pressure_mid = gsw.Nsquared(absolute_salinity, conservative_temperature,
                                           pressure_3d, latitude_3d, axis=0)
            out.createDimension('depth_mid', len(iz) - 1)
            mid = out.createVariable('depth_mid', 'f4', ('depth_mid',))
            mid[:] = (depth[iz][:-1] + depth[iz][1:]) / 2
            mid.units, mid.positive = 'm', 'down'
            for name, values, dims, units in [
                    ('pressure', pressure[:, :, 0], ('depth', 'latitude'), 'dbar'),
                    ('absolute_salinity', absolute_salinity[None], ('time', 'depth', 'latitude', 'longitude'), 'g kg-1'),
                    ('conservative_temperature', conservative_temperature[None], ('time', 'depth', 'latitude', 'longitude'), 'degree_Celsius'),
                    ('sigma0', sigma0[None], ('time', 'depth', 'latitude', 'longitude'), 'kg m-3'),
                    ('N2', n2[None], ('time', 'depth_mid', 'latitude', 'longitude'), 's-2'),
                    ('pressure_mid', pressure_mid[None], ('time', 'depth_mid', 'latitude', 'longitude'), 'dbar')]:
                variable = out.createVariable(name, 'f4', dims, zlib=True, complevel=1, fill_value=np.nan)
                variable[:] = values
                variable.units = units
            out.source_temp_file = str(temp_path.relative_to(ROOT))
            out.source_psal_file = str(sal_path.relative_to(ROOT))
            out.spatial_subset = "120E–280E, 20S–20N"
            out.depth_subset = "5–500 m"
            out.longitude_convention = "0–360 degrees east, increasing"
            out.thermodynamic_status = "TEMP CF sea_water_temperature treated as in-situ ITS-90; PSAL PSS-78; TEOS-10 via GSW"
            out.gsw_version = gsw.__version__
        temporary.replace(output)
    return output


def verify(path: Path) -> None:
    with Dataset(path) as ds:
        lon, lat, depth = ds["longitude"][:], ds["latitude"][:], ds["depth"][:]
        assert np.all(np.diff(lon) > 0) and lon.min() >= 120 and lon.max() <= 280
        assert np.all(np.diff(lat) > 0) and lat.min() >= -20 and lat.max() <= 20
        assert np.all(np.diff(depth) > 0) and depth.min() >= 5 and depth.max() <= 500
        assert ds["TEMP"].shape == ds["PSAL"].shape
        assert np.isfinite(ds["TEMP"][:]).any() and np.isfinite(ds["PSAL"][:]).any()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    files = sorted(args.input_dir.glob("OA_CORA5.2_*_fld_TEMP.nc"))
    if not files:
        raise FileNotFoundError(f"No CORA field-temperature files in {args.input_dir}")
    records = []
    for file in files:
        output = process(file, args.output_dir, args.overwrite)
        verify(output)
        print(output.relative_to(ROOT))
        with Dataset(output) as ds:
            n2 = np.ma.filled(ds['N2'][:], np.nan)
            finite = np.isfinite(n2)
            records.append({'output': str(output.relative_to(ROOT)), 'finite_N2_layer_pairs': int(finite.sum()),
                            'negative_N2_layer_pairs': int((finite & (n2 < 0)).sum()),
                            'negative_N2_fraction': float((n2[finite] < 0).mean()),
                            'fraction_below_threshold': {str(limit): float((n2[finite] < -limit).mean())
                                                         for limit in (1e-6, 1e-5, 1e-4)}})
    (args.output_dir / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == "__main__":
    main()
