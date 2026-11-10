#!/usr/bin/env python3
"""Run shared profile metrics on processed CORA fields as a coverage audit."""
import argparse
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from scientific_metrics import d20, density_mld, heat_content

ROOT = Path(__file__).resolve().parents[1]


def summary(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return {'count': int(values.size), 'median': float(np.median(values)),
            'p05': float(np.quantile(values, .05)), 'p95': float(np.quantile(values, .95))} if values.size else {'count': 0}


def evaluate(path):
    with Dataset(path) as ds:
        depth = np.asarray(ds['depth'][:])
        temperature = np.ma.filled(ds['TEMP'][0], np.nan)
        ct = np.ma.filled(ds['conservative_temperature'][0], np.nan)
        sigma0 = np.ma.filled(ds['sigma0'][0], np.nan)
        n2 = np.ma.filled(ds['N2'][0], np.nan)
    metrics = {'d20_m': [], 'density_mld_m': [], 'heat_content_5_300_J_m2': []}
    for iy, ix in np.ndindex(temperature.shape[1:]):
        metrics['d20_m'].append(d20(depth, temperature[:, iy, ix]))
        metrics['density_mld_m'].append(density_mld(depth, sigma0[:, iy, ix]))
        metrics['heat_content_5_300_J_m2'].append(heat_content(depth, ct[:, iy, ix]))
    finite = np.isfinite(n2)
    return {'file': str(path.relative_to(ROOT)), **{name: summary(values) for name, values in metrics.items()},
            'finite_N2_layer_pairs': int(finite.sum()), 'negative_N2_fraction': float((n2[finite] < 0).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'data/processed/cora')
    parser.add_argument('--output', type=Path, default=ROOT / 'research/audits/field_metric_coverage.json')
    args = parser.parse_args()
    files = sorted(args.input_dir.glob('cora_??????_*.nc'))
    if not files:
        raise FileNotFoundError('No processed CORA files')
    result = [evaluate(path) for path in files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
