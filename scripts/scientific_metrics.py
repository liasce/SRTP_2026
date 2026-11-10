"""Shared profile metrics for observations, CORA, and future model outputs."""
import numpy as np
import gsw

CP0 = 3991.86795711963
RHO0 = 1025.0


def d20(depth, temperature):
    depth, temperature = _clean(depth, temperature)
    crossing = np.flatnonzero((temperature[:-1] >= 20) & (temperature[1:] < 20))
    return _cross(depth, temperature, int(crossing[0]), 20) if crossing.size else np.nan


def density_mld(depth, sigma0, reference_depth=10, threshold=.03):
    depth, sigma0 = _clean(depth, sigma0)
    if depth.size < 2 or reference_depth < depth[0] or reference_depth > depth[-1]:
        return np.nan
    reference = np.interp(reference_depth, depth, sigma0)
    anomaly = sigma0 - reference
    crossing = np.flatnonzero((depth[:-1] >= reference_depth) & (anomaly[:-1] < threshold) & (anomaly[1:] >= threshold))
    return _cross(depth, anomaly, int(crossing[0]), threshold) if crossing.size else np.nan


def heat_content(depth, conservative_temperature, shallow=5, deep=300):
    depth, temperature = _clean(depth, conservative_temperature)
    if depth.size < 2 or depth[0] > shallow or depth[-1] < deep:
        return np.nan
    inside = (depth > shallow) & (depth < deep)
    z = np.r_[shallow, depth[inside], deep]
    t = np.interp(z, depth, temperature)
    return RHO0 * CP0 * np.trapezoid(t, z)


def buoyancy_frequency(depth, absolute_salinity, conservative_temperature, latitude):
    depth, absolute_salinity, conservative_temperature = _clean(depth, absolute_salinity, conservative_temperature)
    pressure = gsw.p_from_z(-depth, latitude)
    n2, pressure_mid = gsw.Nsquared(absolute_salinity, conservative_temperature, pressure, lat=latitude)
    return -gsw.z_from_p(pressure_mid, latitude), n2


def _clean(depth, *values):
    arrays = [np.asarray(x, dtype=float) for x in (depth, *values)]
    valid = np.logical_and.reduce([np.isfinite(x) for x in arrays])
    arrays = [x[valid] for x in arrays]
    if not arrays[0].size:
        return tuple(arrays)
    order = np.argsort(arrays[0])
    return tuple(x[order] for x in arrays)


def _cross(depth, value, i, threshold):
    return float(depth[i] + (threshold - value[i]) * (depth[i + 1] - depth[i]) / (value[i + 1] - value[i]))
