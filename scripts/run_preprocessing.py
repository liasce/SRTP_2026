#!/usr/bin/env python3
"""Run the reusable monthly preprocessing pipeline and write a training-file manifest."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
FULL_START = '201504'


def argo_preprocessing_complete():
    manifest_path = ROOT / 'data/processed/argo/manifest.json'
    if not manifest_path.exists():
        return False
    try:
        records = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    raw_files = sorted((ROOT / 'data/raw/argo').rglob('*.nc'))
    if len(records) != len(raw_files):
        return False
    expected_sources = {str(path) for path in raw_files}
    recorded_sources = {record.get('source') for record in records}
    return expected_sources == recorded_sources and all(
        record.get('output') and Path(record['output']).exists() for record in records)


def run(*args):
    subprocess.run([sys.executable, str(ROOT / 'scripts' / args[0]), *args[1:]], check=True)


def split(month):
    value = int(month)
    if value <= 201503: return 'pretrain'
    if value <= 202011: return 'finetune'
    if value == 202012: return 'isolation'
    if value <= 202212: return 'validation'
    if value <= 202412: return 'test'
    if value <= 202506: return 'forward_test'
    return 'unassigned'


def previous_month(month, offset):
    year, number = int(month[:4]), int(month[4:])
    absolute = year * 12 + number - 1 + offset
    return f'{absolute // 12:04d}{absolute % 12 + 1:02d}'


def manifest(mode, selected_months=None):
    records = []
    selected_months = set(selected_months or [])
    for cora in sorted((ROOT / 'data/processed/cora').glob('cora_??????_*.nc')):
        month = cora.name.split('_')[1]
        if selected_months and month not in selected_months:
            continue
        if mode == 'full' and month < FULL_START:
            continue
        paths = {
            'cora': cora,
            'oisst': ROOT / f'data/processed/on_cora_grid/oisst_{month}_on_cora_grid.nc',
            'adt': ROOT / f'data/processed/on_cora_grid/adt_{month}_on_cora_grid.nc',
            'era5': ROOT / f'data/processed/on_cora_grid/era5_{month}_on_cora_grid.nc',
            'argo': ROOT / f'data/processed/argo_monthly/argo_{month}_on_cora_grid.nc',
        }
        if mode == 'full':
            paths['smap'] = ROOT / f'data/processed/on_cora_grid/smap_{month}_on_cora_grid.nc'
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f'{month}: missing processed sources {missing}')
        with Dataset(cora) as reference:
            lat, lon = reference['latitude'][:], reference['longitude'][:]
        for path in paths.values():
            with Dataset(path) as ds:
                np.testing.assert_array_equal(ds['latitude'][:], lat)
                np.testing.assert_array_equal(ds['longitude'][:], lon)
        surface_history = {}
        for source in ('oisst', 'adt', 'era5', *(('smap',) if mode == 'full' else ())):
            history = []
            for offset in (-2, -1, 0):
                history_month = previous_month(month, offset)
                candidate = ROOT / f'data/processed/on_cora_grid/{source}_{history_month}_on_cora_grid.nc'
                if candidate.exists():
                    with Dataset(candidate) as history_ds:
                        np.testing.assert_array_equal(history_ds['latitude'][:], lat)
                        np.testing.assert_array_equal(history_ds['longitude'][:], lon)
                    history.append(str(candidate.relative_to(ROOT)))
                else:
                    history.append(None)
            surface_history[source] = history
        records.append({'month': month, 'split': split(month),
                        'surface_history_offsets': [-2, -1, 0],
                        'surface_history': surface_history,
                        'files': {name: str(path.relative_to(ROOT)) for name, path in paths.items()}})
    if not records:
        raise FileNotFoundError('No processed CORA months')
    actual_months = [record['month'] for record in records]
    if selected_months and set(actual_months) != selected_months:
        raise FileNotFoundError(
            f'Manifest month mismatch; missing={sorted(selected_months - set(actual_months))}; '
            f'unexpected={sorted(set(actual_months) - selected_months)}')
    if actual_months != sorted(set(actual_months)):
        raise ValueError('Manifest months must be unique and chronological')
    output = ROOT / f'data/processed/training_manifest_{mode}.json'
    output.write_text(json.dumps({'mode': mode, 'months': records}, indent=2) + '\n')
    print(output.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['common', 'full'], default='common',
                        help='common excludes SMAP; full requires SMAP')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--months', nargs='+', metavar='YYYYMM',
                        help='Restrict the final manifest to these months')
    args = parser.parse_args()
    if args.months and (len(args.months) != len(set(args.months))
                        or any(len(month) != 6 or not month.isdigit() for month in args.months)):
        parser.error('--months must be unique YYYYMM values')
    overwrite = ['--overwrite'] if args.overwrite else []
    run('preprocess_cora.py', *overwrite)
    if not argo_preprocessing_complete():
        run('preprocess_argo.py')
    if (ROOT / 'data/raw/tao_daily').exists():
        run('preprocess_tao.py', *overwrite)
        run('pair_tao.py')
    for source in ('oisst', 'adt', 'era5'):
        run('preprocess_surface.py', source)
    months = [p.name.split('_')[1] for p in sorted((ROOT / 'data/processed/cora').glob('cora_??????_*.nc'))]
    if args.mode == 'full':
        run('preprocess_smap.py', '--months', *(month for month in months if month >= FULL_START))
    for source in (*('oisst', 'adt', 'era5'), *(('smap',) if args.mode == 'full' else ())):
        run('regrid_surface.py', source)
    run('grid_argo_monthly.py')
    manifest(args.mode, args.months)
    run('compute_normalization.py', '--manifest', str(ROOT / f'data/processed/training_manifest_{args.mode}.json'))
    run('validate_pipeline.py',
        '--manifest', str(ROOT / f'data/processed/training_manifest_{args.mode}.json'),
        '--normalization', str(ROOT / f'data/processed/normalization_{args.mode}.json'),
        '--output', str(ROOT / f'research/audits/pipeline_validation_{args.mode}.json'))


if __name__ == '__main__':
    main()
