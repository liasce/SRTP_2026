"""Numerical checks for shared ocean profile metrics."""
import numpy as np
from scientific_metrics import CP0, RHO0, buoyancy_frequency, d20, density_mld, heat_content


def main():
    assert d20([0, 50, 100], [25, 22, 18]) == 75
    assert np.isclose(density_mld([5, 10, 20, 30], [24, 24, 24.02, 24.06]), 22.5)
    expected = RHO0 * CP0 * 20 * 295
    assert np.isclose(heat_content([5, 100, 300], [20, 20, 20]), expected)
    assert np.isnan(heat_content([10, 100, 300], [20, 20, 20]))
    depth = np.array([5., 50., 100.])
    mid, n2 = buoyancy_frequency(depth, [35, 35, 35], [28, 20, 10], 0)
    assert len(mid) == 2 and np.all(n2 > 0)
    _, n2 = buoyancy_frequency(depth, [35, 35, 35], [10, 20, 28], 0)
    assert np.all(n2 < 0)
    print('Scientific metric checks passed')


if __name__ == '__main__':
    main()
