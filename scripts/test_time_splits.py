"""Check frozen temporal split boundaries and normalization eligibility."""
from compute_normalization import select_fit_months
from run_preprocessing import previous_month, split


def main():
    expected = {
        '201503': 'pretrain',
        '201504': 'finetune',
        '202011': 'finetune',
        '202012': 'isolation',
        '202101': 'validation',
        '202212': 'validation',
        '202301': 'test',
        '202412': 'test',
        '202501': 'forward_test',
        '202506': 'forward_test',
        '202507': 'unassigned',
    }
    actual = {month: split(month) for month in expected}
    assert actual == expected, actual
    training_splits = {'pretrain', 'finetune'}
    assert split('202012') not in training_splits
    assert all(split(month) not in training_splits for month in ('202101', '202212', '202301', '202412', '202501'))
    assert [previous_month('201801', offset) for offset in (-2, -1, 0)] == [
        '201711', '201712', '201801']
    assert [previous_month('201812', offset) for offset in (-2, -1, 0)] == [
        '201810', '201811', '201812']
    assert [record['month'] for record in select_fit_months([
        {'month': '202011', 'split': 'finetune'},
        {'month': '202012', 'split': 'isolation'},
        {'month': '202101', 'split': 'validation'},
    ])] == ['202011']
    for month in ('202012', '202101', '202301'):
        try:
            select_fit_months([{'month': month, 'split': 'finetune'}])
        except ValueError:
            pass
        else:
            raise AssertionError(f'Forbidden normalization month accepted: {month}')
    print('Temporal split boundaries, t-2:t history, and normalization isolation checks passed')


if __name__ == '__main__':
    main()
