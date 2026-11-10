#!/usr/bin/env python3
"""Apply one fixed QC rule to daily TAO/TRITON temperature and salinity rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, chartostring


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw/tao_daily"
DEFAULT_OUTPUT = ROOT / "data/processed/tao"
SPECS = {
    "T": ("T_20", "QT_5020", "temperature", "degree_C"),
    "S": ("S_41", "QS_5041", "practical_salinity", "PSU"),
}


def strings(variable) -> np.ndarray:
    values = np.asarray(variable[:])
    return values.astype(str) if values.ndim == 1 and values.dtype.kind == "U" else chartostring(values).astype(str)


def process(source: Path, output_dir: Path, overwrite: bool) -> Path:
    kind = source.name.removeprefix("pmelTaoDy")[0]
    value_name, qc_name, output_name, units = SPECS[kind]
    month = source.stem.split("_")[-1]
    output = output_dir / f"tao_{month}_{kind}_qc_5_500m.nc"
    if output.exists() and not overwrite:
        return output

    with Dataset(source) as ds:
        station = strings(ds["station"])
        time = np.asarray(ds["time"][:], dtype=np.int64)
        lat = np.ma.filled(ds["latitude"][:], np.nan).astype(float)
        lon = np.mod(np.ma.filled(ds["longitude"][:], np.nan).astype(float), 360)
        depth = np.ma.filled(ds["depth"][:], np.nan).astype(float)
        value = np.ma.filled(ds[value_name][:], np.nan).astype(float)
        qc = np.ma.filled(ds[qc_name][:], np.nan).astype(float)
        wmo = np.ma.filled(ds["wmo_platform_code"][:], -1).astype(np.int64)
        time_units = ds["time"].units

    keep = (np.isfinite(lat) & np.isfinite(lon) & np.isfinite(depth) & np.isfinite(value)
            & np.isin(qc, [1, 2, 3]) & (lat >= -20) & (lat <= 20)
            & (lon >= 120) & (lon <= 280) & (depth >= 5) & (depth <= 500))
    order = np.lexsort((depth[keep], station[keep], time[keep]))
    station, time, lat, lon, depth, value, qc, wmo = [x[keep][order]
                                                       for x in (station, time, lat, lon, depth, value, qc, wmo)]
    station_ids, station_index = np.unique(station, return_inverse=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Dataset(output, "w") as out:
        out.createDimension("observation", len(value))
        out.createDimension("station", len(station_ids))
        out.createVariable("station_id", str, ("station",))[:] = station_ids
        for name, values, dtype in [
            ("station_index", station_index, "i4"), ("time", time, "i8"),
            ("latitude", lat, "f4"), ("longitude", lon, "f4"),
            ("depth", depth, "f4"), ("wmo_platform_code", wmo, "i8"),
            ("source_qc", qc, "i1"), (output_name, value, "f4"),
        ]:
            variable = out.createVariable(name, dtype, ("observation",), zlib=True, complevel=1)
            variable[:] = values
        out["time"].units = time_units
        out["latitude"].units, out["longitude"].units, out["depth"].units = "degrees_north", "degrees_east", "m"
        out[output_name].units = units
        out.source_file = str(source.relative_to(ROOT))
        out.qc_rule = f"{qc_name} in {{1,2,3}}; 20S–20N, 120E–280E, 5–500 m"
        out.longitude_convention = "0–360 degrees east"
        out.thermodynamic_status = "Native units retained; SA/CT conversion pending GSW"
    return output


def verify(path: Path) -> None:
    with Dataset(path) as ds:
        assert ds.dimensions["observation"].size > 0
        assert np.isfinite(ds["depth"][:]).all() and (ds["depth"][:] >= 5).all() and (ds["depth"][:] <= 500).all()
        assert np.isin(ds["source_qc"][:], [1, 2, 3]).all()
        assert (ds["longitude"][:] >= 120).all() and (ds["longitude"][:] <= 280).all()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    files = sorted(args.input_dir.glob("pmelTaoDy[TS]_*.nc"))
    if not files:
        raise FileNotFoundError(f"No TAO daily T/S files in {args.input_dir}")
    for source in files:
        output = process(source, args.output_dir, args.overwrite)
        verify(output)
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
