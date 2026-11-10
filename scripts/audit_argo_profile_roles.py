#!/usr/bin/env python3
"""Audit monthly raw, deduplicated, QC-valid, and C/Q/E Argo profile counts."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from netCDF4 import Dataset, num2date

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-dir', type=Path, default=ROOT / 'data/processed/argo')
    parser.add_argument('--roles', type=Path,
                        default=ROOT / 'data/processed/argo_monthly/profile_roles.jsonl')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'research/audits/argo_2018_profile_counts.json')
    args = parser.parse_args()
    role_records = [json.loads(line) for line in args.roles.read_text(encoding='utf-8').splitlines()]
    role_identities = {
        (record['platform'], int(record['cycle']), record['direction'], float(record['time'])): record
        for record in role_records
    }
    if len(role_identities) != len(role_records):
        raise ValueError('Duplicate profile identities remain in role manifest')

    raw_counts = Counter()
    deduplicated = defaultdict(set)
    unreadable = []
    for path in sorted(args.processed_dir.rglob('*.nc')):
        try:
            with Dataset(path) as ds:
                units = ds['time'].units
                for index in range(len(ds.dimensions['profile'])):
                    timestamp = float(ds['time'][index])
                    try:
                        month = num2date(timestamp, units).strftime('%Y%m')
                    except Exception:
                        month = 'unassigned'
                    raw_counts[month] += 1
                    identity = (str(ds['platform'][index]), int(ds['cycle'][index]),
                                str(ds['direction'][index]), timestamp)
                    deduplicated[month].add(identity)
        except Exception as error:
            unreadable.append({'path': str(path.relative_to(ROOT)), 'error_type': type(error).__name__})

    months = [f'2018{month:02d}' for month in range(1, 13)]
    records = []
    for month in months:
        month_roles = [record for record in role_records if record['month'] == month]
        record = {
            'month': month,
            'raw_profile_records': raw_counts[month],
            'deduplicated_profile_identities': len(deduplicated[month]),
            'qc_valid_profiles': len(month_roles),
            'roles': {role: sum(item['role'] == role for item in month_roles) for role in ('C', 'Q', 'E')},
            'platforms': {role: len({item['platform'] for item in month_roles if item['role'] == role})
                          for role in ('C', 'Q', 'E')},
        }
        if sum(record['roles'].values()) != record['qc_valid_profiles']:
            raise ValueError(f'{month}: C/Q/E counts do not sum to QC-valid profiles')
        records.append(record)
    e_platforms = {record['platform'] for record in role_records if record['role'] == 'E'}
    leakage = [record for record in role_records
               if record['platform'] in e_platforms and record['role'] != 'E']
    if leakage:
        raise ValueError('E platform leaked into C or Q')
    if any(record['roles'][role] == 0 for record in records for role in ('C', 'Q', 'E')):
        raise ValueError('At least one month lacks C, Q, or E profiles')
    output = {
        'status': 'passed' if not unreadable else 'failed',
        'scope': '2018 processed Argo profile records and C/Q/E role manifest',
        'processed_files': sum(1 for _ in args.processed_dir.rglob('*.nc')),
        'role_records': len(role_records),
        'fixed_e_platforms': len(e_platforms),
        'unreadable_files': unreadable,
        'months': records,
        'totals': {
            'raw_profile_records': sum(record['raw_profile_records'] for record in records),
            'deduplicated_profile_identities': sum(record['deduplicated_profile_identities'] for record in records),
            'qc_valid_profiles': sum(record['qc_valid_profiles'] for record in records),
            'roles': {role: sum(record['roles'][role] for record in records) for role in ('C', 'Q', 'E')},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(output['totals'], indent=2))
    if output['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
