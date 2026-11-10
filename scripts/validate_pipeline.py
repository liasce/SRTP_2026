#!/usr/bin/env python3
"""Audit every manifest month as a training-readable sample without fitting on evaluation periods."""
import argparse
import json
from pathlib import Path

import numpy as np

from training_data import ROOT, load_sample


def previous_month(month, offset):
    year, number = int(month[:4]), int(month[4:])
    absolute = year * 12 + number - 1 + offset
    return f'{absolute // 12:04d}{absolute % 12 + 1:02d}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/processed/training_manifest_common.json')
    parser.add_argument('--normalization', type=Path, default=ROOT / 'data/processed/normalization_common.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'research/audits/pipeline_validation.json')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    normalization = json.loads(args.normalization.read_text())
    records = manifest['months']
    months = [r['month'] for r in records]
    if months != sorted(set(months)):
        raise ValueError('Manifest months must be unique and chronological')
    expected_fit = [r['month'] for r in records if r['split'] in ('pretrain', 'finetune')]
    if normalization['fitted_months'] != expected_fit:
        raise ValueError('Normalization months do not exactly match training-side manifest months')
    if any(r['split'] in ('validation', 'test', 'forward_test') for r in records
           if r['month'] in normalization['fitted_months']):
        raise ValueError('Evaluation month leaked into normalization')
    statistics = normalization['statistics']
    checks, shapes = [], None
    for record in records:
        history_availability = {}
        if record.get('surface_history_offsets') != [-2, -1, 0]:
            raise ValueError(f"{record['month']}: surface history offsets are not t-2:t")
        for source, history in record.get('surface_history', {}).items():
            if len(history) != 3:
                raise ValueError(f"{record['month']}: {source} history length is not three")
            if history[-1] is None:
                raise ValueError(f"{record['month']}: current {source} field is missing")
            history_availability[source] = [path is not None for path in history]
            for offset, path in zip((-2, -1, 0), history):
                if path is not None and previous_month(record['month'], offset) not in Path(path).name:
                    raise ValueError(f"{record['month']}: incorrect or future {source} history path")
        sample = load_sample(record, statistics)
        current_shapes = {key: list(value.shape) for key, value in sample.items()}
        if shapes is None:
            shapes = current_shapes
        elif current_shapes != shapes:
            raise ValueError(f"{record['month']}: tensor shapes changed")
        for key, value in sample.items():
            if not np.isfinite(value).all():
                raise ValueError(f"{record['month']}: nonfinite training array {key}")
        for key in ('surface_mask', 'sparse_mask', 'sparse_position_mask',
                    'query_mask', 'target_mask'):
            if not np.isin(sample[key], [0, 1]).all():
                raise ValueError(f"{record['month']}: non-binary {key}")
        if np.any((sample['sparse_mask'] > 0) & (sample['query_mask'] > 0)):
            raise ValueError(f"{record['month']}: Q target overlaps C input grid-depth points")
        if (sample['target_mask'].sum() == 0 or sample['surface_mask'].sum() == 0
                or sample['query_mask'].sum() == 0):
            raise ValueError(f"{record['month']}: no valid target, query, or surface values")
        checks.append({'month': record['month'], 'split': record['split'],
                       'surface_history_available_t_minus_2_to_t': history_availability,
                       'surface_valid_fraction': float(sample['surface_mask'].mean()),
                       'sparse_valid_points': int(sample['sparse_mask'].sum()),
                       'query_valid_points': int(sample['query_mask'].sum()),
                       'position_valid_points': int(sample['sparse_position_mask'].sum()),
                       'target_valid_fraction': float(sample['target_mask'].mean())})
    roles_path = ROOT / 'data/processed/argo_monthly/profile_roles.jsonl'
    roles = [json.loads(line) for line in roles_path.read_text().splitlines()]
    valid_roles = {'C', 'Q', 'E'}
    if any(item['role'] not in valid_roles for item in roles):
        raise ValueError('Profile role outside C/Q/E')
    identities = [(item['platform'], item['cycle'], item['direction'], item['time']) for item in roles]
    if len(identities) != len(set(identities)):
        raise ValueError('A deduplicated Argo profile appears more than once')
    by_platform = {}
    for item in roles:
        is_evaluation = item['role'] == 'E'
        previous = by_platform.setdefault(item['platform'], is_evaluation)
        if previous != is_evaluation:
            raise ValueError('An E platform appears in C or Q in another month')
    role_counts = {role: sum(item['role'] == role for item in roles) for role in sorted(valid_roles)}
    if any(count == 0 for count in role_counts.values()):
        raise ValueError('At least one C/Q/E role is empty')
    report = {'status': 'passed', 'mode': manifest['mode'], 'months': len(records), 'shapes': shapes,
              'normalization_fitted_months': expected_fit, 'platforms': len(by_platform),
              'profile_role_counts': role_counts, 'checks': checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'mode', 'months', 'shapes', 'platforms')}, indent=2))


if __name__ == '__main__':
    main()
