"""Audit SMAP nominal dates and actual support before monthly aggregation."""
import argparse
import calendar
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from netCDF4 import Dataset, num2date

ROOT = Path(__file__).resolve().parents[1]


def inspect(path):
    with Dataset(path) as ds:
        if ds['time'].size != 1:
            raise ValueError(f'Expected one nominal date: {path}')
        nominal = num2date(ds['time'][:], ds['time'].units)[0].strftime('%Y-%m-%d')
        start = datetime.fromisoformat(ds.time_coverage_start.replace('Z', '+00:00'))
        end = datetime.fromisoformat(ds.time_coverage_end.replace('Z', '+00:00'))
        if end - start != timedelta(days=8):
            raise ValueError(f'Unexpected SMAP duration: {path}')
        for name in ['sss_smap', 'sss_smap_unc', 'nobs']:
            if name not in ds.variables:
                raise ValueError(f'Missing {name}: {path}')
        return {'path': path.name, 'nominal_date': nominal,
                'support_start': start.isoformat(), 'support_end': end.isoformat()}


def audit(directory, months):
    records = [inspect(p) for p in sorted(directory.glob('*.nc'))]
    by_date = defaultdict(list)
    for record in records:
        by_date[record['nominal_date']].append(record)
    result = []
    for month in months:
        year, number = int(month[:4]), int(month[4:])
        expected = [f'{year:04}-{number:02}-{day:02}' for day in range(1, calendar.monthrange(year, number)[1] + 1)]
        missing = [day for day in expected if day not in by_date]
        duplicates = [day for day in expected if len(by_date[day]) > 1]
        result.append({'month': month, 'expected_nominal_dates': len(expected),
                       'present_nominal_dates': len(expected) - len(missing), 'missing_dates': missing,
                       'duplicate_dates': duplicates, 'calendar_complete': not missing and not duplicates})
    return {'months': result, 'files': records,
            'interpretation': 'Calendar completeness only; monthly aggregate retains overlapping 8-day support and is not an independent daily-observation mean.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=ROOT / 'data/raw/smap')
    p.add_argument('--months', nargs='+', required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'research/audits/smap_window_coverage.json')
    a = p.parse_args()
    result = audit(a.input_dir, a.months)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['months'], indent=2))
    if not all(m['calendar_complete'] for m in result['months']):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
