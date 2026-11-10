"""Numerically stable local optimal-interpolation primitives."""
import numpy as np

EARTH_RADIUS_KM = 6371.0


def distance_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlat = lat2 - lat1
    dlon = np.radians((np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float) + 180) % 360 - 180)
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def covariance(lat1, lon1, depth1, lat2, lon2, depth2,
               background_variance, horizontal_scale_km, vertical_scale_m):
    horizontal = distance_km(lat1, lon1, lat2, lon2)
    vertical = np.abs(np.asarray(depth2, dtype=float) - np.asarray(depth1, dtype=float))
    return background_variance * np.exp(-0.5 * (horizontal / horizontal_scale_km) ** 2) * np.exp(-vertical / vertical_scale_m)


def analyze(background_target, background_observations, observations,
            observation_latitude, observation_longitude, observation_depth,
            target_latitude, target_longitude, target_depth,
            background_variance, observation_error_variance,
            horizontal_scale_km, vertical_scale_m,
            max_jitter_attempts=5, condition_limit=1e12):
    observations = np.asarray(observations, dtype=float)
    background_observations = np.asarray(background_observations, dtype=float)
    observation_latitude = np.asarray(observation_latitude, dtype=float)
    observation_longitude = np.asarray(observation_longitude, dtype=float)
    observation_depth = np.asarray(observation_depth, dtype=float)
    error_variance = np.broadcast_to(np.asarray(observation_error_variance, dtype=float), observations.shape)
    valid = (np.isfinite(observations) & np.isfinite(background_observations)
             & np.isfinite(observation_latitude) & np.isfinite(observation_longitude)
             & np.isfinite(observation_depth) & np.isfinite(error_variance) & (error_variance > 0))
    if not np.isfinite(background_target):
        return np.nan, np.nan, {'status': 'missing_background', 'observations': 0, 'jitter': None}
    if not valid.any():
        return float(background_target), float(background_variance), {
            'status': 'no_observations', 'observations': 0, 'jitter': None}
    observations = observations[valid]
    background_observations = background_observations[valid]
    observation_latitude = observation_latitude[valid]
    observation_longitude = observation_longitude[valid]
    observation_depth = observation_depth[valid]
    error_variance = error_variance[valid]
    if background_variance <= 0 or horizontal_scale_km <= 0 or vertical_scale_m <= 0:
        raise ValueError('OI variance and scales must be positive')

    lat_i = observation_latitude[:, None]
    lon_i = observation_longitude[:, None]
    depth_i = observation_depth[:, None]
    matrix = covariance(lat_i, lon_i, depth_i,
                        observation_latitude[None, :], observation_longitude[None, :],
                        observation_depth[None, :], background_variance,
                        horizontal_scale_km, vertical_scale_m)
    matrix = (matrix + matrix.T) / 2
    matrix[np.diag_indices_from(matrix)] += error_variance
    cross = covariance(target_latitude, target_longitude, target_depth,
                       observation_latitude, observation_longitude, observation_depth,
                       background_variance, horizontal_scale_km, vertical_scale_m)
    innovation = observations - background_observations
    base_jitter = max(1e-12, 1e-10 * float(np.median(np.diag(matrix))))
    last_condition = None
    for attempt in range(max_jitter_attempts + 1):
        jitter = 0.0 if attempt == 0 else base_jitter * 10 ** (attempt - 1)
        candidate = matrix.copy()
        candidate[np.diag_indices_from(candidate)] += jitter
        try:
            condition = float(np.linalg.cond(candidate))
            last_condition = condition
            if not np.isfinite(condition) or condition > condition_limit:
                raise np.linalg.LinAlgError('condition limit exceeded')
            factor = np.linalg.cholesky(candidate)
            weights = np.linalg.solve(factor.T, np.linalg.solve(factor, innovation))
            covariance_weights = np.linalg.solve(factor.T, np.linalg.solve(factor, cross))
            analysis = float(background_target + cross @ weights)
            analysis_variance = float(background_variance - cross @ covariance_weights)
            if analysis_variance < -1e-8 * background_variance:
                raise np.linalg.LinAlgError('negative analysis variance')
            analysis_variance = max(0.0, analysis_variance)
            return analysis, analysis_variance, {
                'status': 'ok', 'observations': int(len(observations)),
                'jitter': jitter, 'condition': condition, 'attempts': attempt + 1}
        except np.linalg.LinAlgError:
            continue
    return float(background_target), float(background_variance), {
        'status': 'linear_algebra_failure', 'observations': int(len(observations)),
        'jitter': base_jitter * 10 ** (max_jitter_attempts - 1),
        'condition': last_condition, 'attempts': max_jitter_attempts + 1}
