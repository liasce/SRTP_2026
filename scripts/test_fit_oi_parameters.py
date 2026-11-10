"""Synthetic checks for OI parameter estimation."""
import numpy as np

from fit_oi_parameters import estimate_band, sample_normalized


def main():
    innovations = np.array([1.0, -1.0, 2.0, -2.0])
    background = np.full(4, .5)
    result = estimate_band(innovations, background, .1)
    assert result['n'] == 4
    assert result['observation_error_variance'] >= .01
    assert result['background_variance'] > 0

    field = np.arange(8, dtype=float).reshape(2, 2, 2)
    profile = {'depth': np.array([0.0, 1.0]), 'latitude': .5, 'longitude': .5}
    sampled, weights = sample_normalized(field, np.array([0.0, 1.0]),
                                          np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                                          [profile])
    np.testing.assert_allclose(sampled[0], [1.5, 5.5])
    np.testing.assert_allclose(weights[0], 1)

    field[:, 0, 0] = np.nan
    sampled, weights = sample_normalized(field, np.array([0.0, 1.0]),
                                          np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                                          [profile])
    assert np.isfinite(sampled[0]).all()
    assert np.all(weights[0] >= .5)
    print('OI parameter variance and normalized background sampling checks passed')


if __name__ == '__main__':
    main()
