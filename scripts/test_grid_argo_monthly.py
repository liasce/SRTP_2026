"""Determinism checks for Argo C/Q/E roles and nearest-grid selection."""
import numpy as np

from grid_argo_monthly import exclude_context_overlap, nearest, platform_role, profile_role
from run_preprocessing import FULL_START, split


def main():
    assert platform_role('5900000', .2, 'x') == platform_role('5900000', .2, 'x')
    assert profile_role('5900000', 1, 'A', 1.0, .5, .2, 'x') == profile_role('5900000', 1, 'A', 1.0, .5, .2, 'x')
    assert nearest(np.array([0., 2., 4.]), 3.6) == 2
    assert FULL_START == '201504' and split('201503') == 'pretrain' and split('201504') == 'finetune'
    assert split('202011') == 'finetune' and split('202012') == 'isolation' and split('202101') == 'validation'
    platforms = [str(index) for index in range(1000)]
    fraction = sum(platform_role(platform, .2, 'x') == 'E' for platform in platforms) / len(platforms)
    assert .15 < fraction < .25
    for platform in platforms:
        roles = {profile_role(platform, cycle, 'A', float(cycle), .5, .2, 'x') for cycle in range(5)}
        if platform_role(platform, .2, 'x') == 'E':
            assert roles == {'E'}
        else:
            assert roles <= {'C', 'Q'}
    shape = (2, 1, 2)
    context = (np.ones(shape), np.ones(shape),
               np.array([[[1, 0]], [[0, 1]]]), np.array([[[0, 1]], [[1, 0]]]),
               {'latitude': np.zeros((1, 2)), 'longitude': np.zeros((1, 2)), 'time': np.zeros((1, 2))},
               np.ones((1, 2), dtype=np.uint16))
    query = (np.ones(shape), np.ones(shape), np.ones(shape, dtype=np.uint16),
             np.ones(shape, dtype=np.uint16),
             {'latitude': np.zeros((1, 2)), 'longitude': np.zeros((1, 2)), 'time': np.zeros((1, 2))},
             np.ones((1, 2), dtype=np.uint16))
    filtered, excluded = exclude_context_overlap(context, query)
    assert not np.any((filtered[2] > 0) & (context[2] > 0))
    assert not np.any((filtered[3] > 0) & (context[3] > 0))
    assert excluded == {'temperature': 2, 'salinity': 2}
    print('Argo C/Q/E gridding, non-input Q, and temporal boundary checks passed')


if __name__ == '__main__':
    main()
