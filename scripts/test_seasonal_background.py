"""Synthetic checks for leave-one-month-out seasonal background fitting."""
import numpy as np

from build_seasonal_background import fit_lomo, harmonic_design


def main():
    fit_months = list(range(1, 12))
    design = harmonic_design(fit_months)
    coefficients = np.array([[10.0, 35.0], [2.0, 0.5], [-1.0, 0.2]])
    values = (design @ coefficients).reshape(11, 1, 2)
    fitted, variance, support, fallback = fit_lomo(values, fit_months, 12, 6)
    expected = harmonic_design([12])[0] @ coefficients
    np.testing.assert_allclose(fitted[0], expected, atol=1e-5)
    np.testing.assert_allclose(variance, 0, atol=1e-10)
    assert np.all(support == 11) and np.all(fallback == 0)

    partial = values.copy()
    partial[:4, 0, 0] = np.nan
    fitted, _, support, fallback = fit_lomo(partial, fit_months, 12, 6)
    np.testing.assert_allclose(fitted[0, 0], np.nanmean(partial[:, 0, 0]))
    assert support[0, 0] == 7 and fallback[0, 0] == 1

    insufficient = values.copy()
    insufficient[:6, 0, 1] = np.nan
    fitted, _, support, fallback = fit_lomo(insufficient, fit_months, 12, 6)
    assert np.isnan(fitted[0, 1]) and support[0, 1] == 5 and fallback[0, 1] == 2

    extreme_target_month = 1e6
    unchanged, _, _, _ = fit_lomo(values, fit_months, 12, 6)
    assert extreme_target_month not in values
    np.testing.assert_allclose(unchanged[0], expected, atol=1e-5)
    print('LOMO seasonal fit, fallback, support, and target-exclusion checks passed')


if __name__ == '__main__':
    main()
