"""Synthetic adjusted-value QC and TEOS-10 roundtrip/sign checks."""
import tempfile
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
import gsw
from preprocess_argo import interpolate_profile, qc_parameter, thermodynamics


def main():
    z = np.array([0., 10., 10., 20., 100.])
    v = np.array([0., 10., 12., 20., 100.])
    target = np.array([0., 10., 15., 50., 100., 101.])
    got = interpolate_profile(z, v, target, max_gap=30)
    np.testing.assert_allclose(got[:3], [0., 11., 15.5])
    assert np.isnan(got[3]) and got[4] == 100 and np.isnan(got[5])
    p = np.array([[0., 100., 500.]])
    t = np.array([[28., 20., 8.]])
    s = np.full_like(p, 35.)
    lat, lon = np.array([0.]), np.array([160.])
    depth, sa, ct, sigma = thermodynamics(p, t, s, lon, lat)
    np.testing.assert_allclose(gsw.p_from_z(-depth, lat[:, None]), p, atol=1e-7)
    np.testing.assert_allclose(gsw.t_from_CT(sa, ct, p), t, atol=1e-10)
    np.testing.assert_allclose(gsw.SP_from_SA(sa, p, lon[:, None], lat[:, None]), s, atol=1e-10)
    assert np.all(np.diff(depth[0]) > 0) and np.all(np.diff(sigma[0]) > 0)
    assert np.all(gsw.Nsquared(sa[0], ct[0], p[0], lat=0)[0] > 0)
    assert np.all(gsw.Nsquared(sa[0], ct[0, ::-1], p[0], lat=0)[0] < 0)
    s[0, 1] = np.nan
    depth, sa, ct, sigma = thermodynamics(p, t, s, lon, lat)
    assert np.isfinite(depth).all() and np.isfinite(t).all()
    assert np.isnan(ct[0, 1]) and np.isnan(sigma[0, 1])
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'fixture.nc'
        with Dataset(path, 'w') as ds:
            ds.createDimension('N_PROF', 2)
            ds.createDimension('N_LEVELS', 3)
            ds.createVariable('DATA_MODE', 'S1', ('N_PROF',))[:] = np.array(['D', 'R'], dtype='S1')
            for name in ('PRES', 'TEMP', 'PSAL'):
                ds.createVariable(name + '_ADJUSTED', 'f8', ('N_PROF', 'N_LEVELS'))[:] = [[1, 2, 3], [4, 5, 6]]
                ds.createVariable(name + '_ADJUSTED_QC', 'S1', ('N_PROF', 'N_LEVELS'))[:] = np.array([['1', '2', '4'], ['1', '1', '1']], dtype='S1')
        with Dataset(path) as ds:
            x = qc_parameter(ds, 'TEMP', ['1'])
            assert x[0, 0] == 1 and np.isnan(x[0, 1:]).all()
            assert np.isnan(x[1]).all(), 'Raw mode must not use adjusted values'
            x = qc_parameter(ds, 'TEMP', ['1', '2'])
            assert x[0, 1] == 2 and np.isnan(x[0, 2])
    print('Argo synthetic QC and TEOS-10 roundtrip/sign checks passed')


if __name__ == '__main__':
    main()
