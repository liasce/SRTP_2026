"""Small numerical check for mask-aware surface regridding."""
import numpy as np
from regrid_surface import normalized_linear


def main():
    lat = lon = np.array([0., 1., 2.])
    field = lat[:, None] + 2 * lon[None, :]
    target = np.array([.5, 1.5])
    got, weight = normalized_linear(field, lat, lon, target, target)
    np.testing.assert_allclose(got, target[:, None] + 2 * target[None, :])
    np.testing.assert_allclose(weight, 1)
    field[0, 0] = np.nan
    got, weight = normalized_linear(field, lat, lon, np.array([.5]), np.array([.5]), .8)
    assert np.isnan(got[0, 0]) and weight[0, 0] == .75
    print('Surface regridding checks passed')


if __name__ == '__main__':
    main()
