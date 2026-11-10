"""Analytic and numerical checks for local optimal interpolation."""
import numpy as np

from oi_core import analyze, covariance, distance_km


def one_observation(error_variance):
    return analyze(
        background_target=0.0,
        background_observations=np.array([0.0]), observations=np.array([10.0]),
        observation_latitude=np.array([0.0]), observation_longitude=np.array([180.0]),
        observation_depth=np.array([100.0]), target_latitude=0.0,
        target_longitude=-180.0, target_depth=100.0,
        background_variance=4.0, observation_error_variance=error_variance,
        horizontal_scale_km=500.0, vertical_scale_m=50.0)


def main():
    np.testing.assert_allclose(distance_km(0, 179, 0, -179), 222.39, rtol=.01)
    assert covariance(0, 0, 100, 0, 0, 150, 4, 500, 50) < 4

    analysis, variance, diagnostics = one_observation(1e12)
    np.testing.assert_allclose(analysis, 0.0, atol=1e-8)
    np.testing.assert_allclose(variance, 4.0, atol=1e-8)
    assert diagnostics['status'] == 'ok'

    analysis, variance, diagnostics = one_observation(1e-10)
    np.testing.assert_allclose(analysis, 10.0, atol=1e-6)
    assert variance < 1e-6 and diagnostics['status'] == 'ok'

    analysis, variance, diagnostics = analyze(
        2.0, np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),
        0.0, 0.0, 100.0, 4.0, np.array([]), 500.0, 50.0)
    assert analysis == 2 and variance == 4 and diagnostics['status'] == 'no_observations'

    analysis, variance, diagnostics = analyze(
        0.0, np.array([0.0, 0.0]), np.array([1.0, 1.0]),
        np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([100.0, 100.0]),
        0.0, 0.0, 100.0, 4.0, np.array([1e-12, 1e-12]), 500.0, 50.0)
    assert np.isfinite(analysis) and variance >= 0
    assert diagnostics['status'] == 'ok'

    analysis, variance, diagnostics = analyze(
        np.nan, np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([0.0]),
        np.array([100.0]), 0.0, 0.0, 100.0, 4.0, 1.0, 500.0, 50.0)
    assert np.isnan(analysis) and np.isnan(variance) and diagnostics['status'] == 'missing_background'
    print('OI distance, covariance, R limits, jitter, and fallback checks passed')


if __name__ == '__main__':
    main()
