"""Synthetic integration checks for local C-to-Q/E OI evaluation."""
import numpy as np

from evaluate_oi_baseline import analyze_profile, summarize_group


def profile(role, value, longitude=180.0):
    depth = np.array([5.0, 50.0, 100.0])
    return {
        'role': role, 'platform': role + '1', 'latitude': 0.0, 'longitude': longitude,
        'depth': depth, 'temperature': np.full(3, value), 'salinity': np.full(3, 35 + value / 100),
        'background_temperature': np.zeros(3), 'background_salinity': np.full(3, 35.0),
        'background_variance_temperature': np.ones(3),
        'background_variance_salinity': np.ones(3) * .01,
    }


def main():
    parameters = {
        variable: {'depth_bands': {
            band: {'background_variance': variance, 'observation_error_variance': error}
            for band in ('5-100m', '110-300m', '310-500m')}}
        for variable, variance, error in [('temperature', 4.0, .01), ('salinity', .04, .0001)]
    }
    context = profile('C', 10.0)
    target = profile('Q', 10.0, longitude=-180.0)
    for variable in ('temperature', 'salinity'):
        prediction, variance, statuses, nearest = analyze_profile(
            target, [context], variable, parameters, 500, 50, 50, 8, 3)
        assert nearest == 0
        assert np.isfinite(prediction).all() and np.all(variance >= 0)
        assert np.all(statuses == 'ok')
        target[f'oi_{variable}'] = prediction
        target[f'oi_variance_{variable}'] = variance
        target[f'oi_status_{variable}'] = statuses
    target['nearest_context_km'] = 0.0
    summary = summarize_group([target])
    assert summary['temperature']['all']['n_points'] == 3
    assert summary['salinity']['all']['prediction_coverage'] == 1
    print('Local OI C input, vertical selection, dateline, and metric checks passed')


if __name__ == '__main__':
    main()
