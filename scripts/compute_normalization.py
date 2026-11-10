#!/usr/bin/env python3
"""Compute finite-value normalization from pretrain/finetune months only."""
import argparse
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
SURFACE = {'oisst': ('sst',), 'adt': ('adt',), 'era5': ('u10', 'v10'), 'smap': ('sss',)}
VOLUME = {'cora': ('conservative_temperature', 'absolute_salinity'),
          'argo': ('argo_conservative_temperature', 'argo_absolute_salinity')}


def select_fit_months(records):
    months = [record for record in records if record['split'] in ('pretrain', 'finetune')]
    forbidden = [record['month'] for record in months if int(record['month']) >= 202012]
    if forbidden:
        raise ValueError(f'Isolation/evaluation dates cannot enter normalization: {forbidden}')
    return months


def finish(total, square, count):
    mean = np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)
    variance = np.divide(square, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0) - mean**2
    return {'mean': mean.tolist(), 'std': np.sqrt(np.maximum(variance, 0)).tolist(), 'count': count.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/processed/training_manifest_common.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    months = select_fit_months(manifest['months'])
    if not months:
        raise ValueError('Manifest contains no pretrain/finetune months')
    accumulators = {}
    for item in months:
        for source, variables in {**SURFACE, **VOLUME}.items():
            if source not in item['files']:
                continue
            with Dataset(ROOT / item['files'][source]) as ds:
                for variable in variables:
                    values = np.ma.filled(ds[variable][:], np.nan).astype(float)
                    if source in VOLUME:
                        values = values.reshape(values.shape[-3:])
                        axes = (1, 2)
                        size = values.shape[0]
                    else:
                        axes = None
                        size = 1
                    key = f'{source}.{variable}'
                    total, square, count = accumulators.setdefault(key, [np.zeros(size), np.zeros(size), np.zeros(size, dtype=np.int64)])
                    finite = np.isfinite(values)
                    total += np.where(finite, values, 0).sum(axis=axes).reshape(-1)
                    square += np.where(finite, values * values, 0).sum(axis=axes).reshape(-1)
                    count += finite.sum(axis=axes).reshape(-1)
    output = args.output or args.manifest.with_name(args.manifest.stem.replace('training_manifest', 'normalization') + '.json')
    output.write_text(json.dumps({'source_manifest': str(args.manifest.relative_to(ROOT)),
                                  'fitted_splits': ['pretrain', 'finetune'],
                                  'fitted_months': [m['month'] for m in months],
                                  'statistics': {k: finish(*v) for k, v in accumulators.items()}}, indent=2) + '\n')
    print(output.relative_to(ROOT))


if __name__ == '__main__':
    main()
