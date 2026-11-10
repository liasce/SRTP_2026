#!/usr/bin/env python3
"""Framework-neutral loader returning normalized arrays and explicit validity masks."""
import argparse
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]


def read(path, variables):
    with Dataset(ROOT / path) as ds:
        return [np.ma.filled(ds[name][:], np.nan).astype(np.float32).squeeze(axis=0)
                if ds[name].dimensions[0] == 'time' else np.ma.filled(ds[name][:], np.nan).astype(np.float32)
                for name in variables]


def scale(values, statistics, keys, per_depth=False):
    output, masks = [], []
    for value, key in zip(values, keys):
        mask = np.isfinite(value)
        mean = np.asarray(statistics[key]['mean'], dtype=np.float32)
        std = np.asarray(statistics[key]['std'], dtype=np.float32)
        if per_depth:
            mean, std = mean[:, None, None], std[:, None, None]
        standardized = (value - mean) / np.where(std > 0, std, 1)
        output.append(np.where(mask, standardized, 0))
        masks.append(mask.astype(np.float32))
    return np.stack(output), np.stack(masks)


def load_sample(record, statistics):
    surface_sources = [('oisst', 'sst'), ('adt', 'adt'), ('era5', 'u10'), ('era5', 'v10')]
    if 'smap' in record['files']:
        surface_sources.append(('smap', 'sss'))
    surface, surface_keys = [], []
    for source, variable in surface_sources:
        current = read(record['files'][source], [variable])[0]
        history = record.get('surface_history', {}).get(source, [record['files'][source]])
        if len(history) != 3:
            raise ValueError(f"{record['month']}: {source} history must contain t-2:t")
        for path in history:
            value = np.full_like(current, np.nan) if path is None else read(path, [variable])[0]
            surface.append(value)
            surface_keys.append(f'{source}.{variable}')
    surface, surface_mask = scale(surface, statistics, surface_keys)
    sparse_names = ['argo_conservative_temperature', 'argo_absolute_salinity']
    sparse = read(record['files']['argo'], sparse_names)
    # Sparse observations and targets share physical units, so use target-scale statistics.
    sparse, sparse_mask = scale(sparse, statistics,
                                ['cora.conservative_temperature', 'cora.absolute_salinity'], True)
    position_names = ['argo_latitude_offset', 'argo_longitude_offset', 'argo_time_offset']
    position = read(record['files']['argo'], position_names)
    position_mask = np.stack([np.isfinite(value).astype(np.float32) for value in position])
    position = np.stack([np.where(np.isfinite(value), value, 0).astype(np.float32) for value in position])
    query_names = ['argo_q_conservative_temperature', 'argo_q_absolute_salinity']
    query = read(record['files']['argo'], query_names)
    query, query_mask = scale(query, statistics,
                              ['cora.conservative_temperature', 'cora.absolute_salinity'], True)
    target_names = ['conservative_temperature', 'absolute_salinity']
    target = read(record['files']['cora'], target_names)
    target, target_mask = scale(target, statistics, [f'cora.{n}' for n in target_names], True)
    return {'surface': surface, 'surface_mask': surface_mask, 'sparse': sparse,
            'sparse_mask': sparse_mask, 'sparse_position': position,
            'sparse_position_mask': position_mask, 'query_target': query,
            'query_mask': query_mask, 'target': target, 'target_mask': target_mask}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/processed/training_manifest_common.json')
    parser.add_argument('--normalization', type=Path, default=ROOT / 'data/processed/normalization_common.json')
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    statistics = json.loads(args.normalization.read_text())['statistics']
    record = manifest['months'][args.index]
    sample = load_sample(record, statistics)
    print(record['month'], record['split'], {k: list(v.shape) for k, v in sample.items()})
    for value in sample.values():
        assert np.isfinite(value).all()


if __name__ == '__main__':
    main()
