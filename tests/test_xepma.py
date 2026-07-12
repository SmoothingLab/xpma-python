"""Tests for the XEPMA endpoint and its off-axis corrected siblings.

XEPMA is the zero-lag two-rate endpoint (quadratic-exact at s = 1 only; zero-lag
m1 = 0 at every s, with m2 = ((1-s)/s) L^2 != 0 at s >= 2). QuadraticXEPMA absorbs
the curvature correction c(s) * Delta^2 IFEMA(p, s+2) and is quadratic-exact at
every real s >= 1 (identical to the endpoint at s = 1). DampedXEPMA is the
min-overshoot sibling.

Covers:
 (a) QuadraticXEPMA == XEPMA endpoint at s = 1 (correction coefficient is zero there).
 (b) Quadratic exactness: QuadraticXEPMA tracks a + b t + c t^2 with zero
     steady-state error at integer and fractional s (the endpoint trails by c * m2).
 (c) Zero lag preserved: kernel m0 = 1, m1 = 0 (and m2 = 0) for QuadraticXEPMA.
 (d) DampedXEPMA overshoot strictly below the endpoint's, its coefficient matches
     the verify-script references, and it keeps zero lag.

Kernel extraction is preceded by a long zero warm-up so the impulse response is the
true LTI kernel, not the "first output = first input" seeding artefact.

Reference min-overshoot coefficients are independently verified values.
"""

import sys
import math

sys.path.insert(0, '..')

from xpma import XEPMA, QuadraticXEPMA, DampedXEPMA


TEST_DATA = []
for i in range(300):
    TEST_DATA.append(100.0 + i * 0.1 + 5.0 * math.sin(i * 0.15) + 2.0 * math.sin(i * 0.47))


def _quadratic_trail(cls, period, s, a, b, c, n=5000):
    """Feed a + b t + c t^2 and return (output - input) at the last bar, plus input."""
    filt = cls(period, s)
    out = inp = None
    for t in range(n):
        inp = a + b * t + c * t * t
        out = filt.get_next(inp)
    return out - inp, inp


def _kernel_moments(cls, period, s, n=20000, warm=200):
    """m0, m1, m2 of the from-rest impulse response (long zero warm-up first)."""
    filt = cls(period, s)
    for _ in range(warm):
        filt.get_next(0.0)
    h = [filt.get_next(1.0)]
    for _ in range(n - 1):
        h.append(filt.get_next(0.0))
    m0 = math.fsum(h)
    m1 = math.fsum(k * h[k] for k in range(len(h)))
    m2 = math.fsum(k * k * h[k] for k in range(len(h)))
    return m0, m1, m2


def _step_overshoot(cls, period, s, settle=200, n_step=2000):
    """Max value above 1 of the from-rest unit-step response."""
    filt = cls(period, s)
    for _ in range(settle):
        filt.get_next(0.0)
    peak = -math.inf
    for _ in range(n_step):
        peak = max(peak, filt.get_next(1.0))
    return peak - 1.0


def test_quadratic_xepma_s1_identical_to_endpoint():
    """At s = 1 the correction coefficient is zero, so QuadraticXEPMA == XEPMA endpoint."""
    period = 21.0
    quad = QuadraticXEPMA(period, 1.0)
    endpoint = XEPMA(period, 1.0)

    max_err = 0.0
    for val in TEST_DATA:
        max_err = max(max_err, abs(quad.get_next(val) - endpoint.get_next(val)))

    assert max_err == 0.0, f"QuadraticXEPMA(s=1) not bit-identical to endpoint: {max_err}"
    print(f"PASS: QuadraticXEPMA(s=1) == XEPMA endpoint (max diff {max_err:.1e})")


def test_quadratic_xepma_quadratic_exact():
    """QuadraticXEPMA tracks a parabola with zero steady-state error at integer and
    fractional s. The XEPMA endpoint trails by c * m2 (m2 = ((1-s)/s) L^2 != 0 at
    s >= 2); QuadraticXEPMA's trail is machine zero."""
    a, b, c = 10.0, 0.5, 0.01
    for s in (2.0, 3.0, 4.0, 2.5):
        quad_trail, inp = _quadratic_trail(QuadraticXEPMA, 20.0, s, a, b, c)
        endpoint_trail, _ = _quadratic_trail(XEPMA, 20.0, s, a, b, c)

        rel = abs(quad_trail) / abs(inp)
        assert rel < 1e-9, f"QuadraticXEPMA(s={s}) not quadratic-exact: rel trail {rel:.2e}"
        assert abs(endpoint_trail) > 1e-3, (
            f"test input too flat: endpoint trail {endpoint_trail:.2e} should be clearly nonzero"
        )
        assert abs(quad_trail) < abs(endpoint_trail) / 1000.0, (
            f"QuadraticXEPMA(s={s}) trail {quad_trail:.2e} not far below endpoint {endpoint_trail:.2e}"
        )
    print("PASS: QuadraticXEPMA quadratic-exact at s = 2, 3, 4, 2.5 (rel trail < 1e-9)")


def test_quadratic_xepma_zero_lag_kernel():
    """QuadraticXEPMA kernel has m0 = 1, m1 = 0 (zero lag) and m2 = 0 (quadratic-exact)."""
    for s in (2.0, 3.0, 4.0, 2.5):
        for p in (20.0, 40.0):
            m0, m1, m2 = _kernel_moments(QuadraticXEPMA, p, s)
            assert abs(m0 - 1.0) < 1e-9, f"QuadraticXEPMA(s={s},p={p}) m0 = {m0}"
            assert abs(m1) < 1e-6, f"QuadraticXEPMA(s={s},p={p}) m1 = {m1} (lag not zero)"
            assert abs(m2) < 1e-6, f"QuadraticXEPMA(s={s},p={p}) m2 = {m2} (not quadratic-exact)"
    print("PASS: QuadraticXEPMA kernel m0 = 1, m1 = 0, m2 = 0 at sampled (s, p)")


def test_quadratic_xepma_calc_next_stateless():
    """calc_next is a stateless probe matching a subsequent get_next (SecantSolver contract)."""
    for s in (1.0, 2.0, 2.5):
        filt = QuadraticXEPMA(20.0, s)
        for val in TEST_DATA[:60]:
            filt.get_next(val)
        probe = TEST_DATA[60]
        c1 = filt.calc_next(probe)
        filt.calc_next(999.0)
        filt.calc_next(-500.0)
        c2 = filt.calc_next(probe)
        assert abs(c1 - c2) < 1e-14, f"QuadraticXEPMA(s={s}) calc_next not stateless: {c1} vs {c2}"
        g = filt.get_next(probe)
        assert abs(c1 - g) < 1e-14, f"QuadraticXEPMA(s={s}) calc_next {c1} != get_next {g}"
    print("PASS: QuadraticXEPMA calc_next stateless and consistent with get_next")


# Verified min-overshoot coefficients (absolute, in the correction term's units).
# The s = 1 references are absolute; the s >= 2 references are gamma* which
# multiplies c(s) = ((s-1)/(2s)) L^2.
_DAMPED_REF_S1 = {(1, 10): 2.276977, (1, 20): 4.293697, (1, 40): 22.684273}
_DAMPED_REF_GAMMA_STAR = {(2, 20): 0.363466, (3, 20): 0.389671, (4, 20): 0.410107}


def test_damped_xepma_coefficient_matches_reference():
    """DampedXEPMA's construction-time coefficient matches the verify-script values."""
    for (s, p), ref in _DAMPED_REF_S1.items():
        factor = DampedXEPMA(float(p), float(s))._factor
        assert abs(factor - ref) < 1e-4, (
            f"DampedXEPMA(s={s},p={p}) factor {factor:.6f} != reference {ref}"
        )
    for (s, p), gamma_star in _DAMPED_REF_GAMMA_STAR.items():
        ma_lag = (p - 1.0) / 2.0
        c = (s - 1.0) / (2.0 * s) * ma_lag * ma_lag
        ref = gamma_star * c
        factor = DampedXEPMA(float(p), float(s))._factor
        assert abs(factor - ref) < 2e-3, (
            f"DampedXEPMA(s={s},p={p}) factor {factor:.6f} != gamma* c {ref:.6f}"
        )
    print("PASS: DampedXEPMA coefficient matches verify_qxepma_fractional references")


def test_damped_xepma_overshoot_below_endpoint():
    """DampedXEPMA step overshoot is strictly below the XEPMA endpoint's, incl. s = 1."""
    for s in (1.0, 2.0, 3.0, 4.0):
        for p in (20.0, 40.0):
            endpoint_ov = _step_overshoot(XEPMA, p, s)
            damped_ov = _step_overshoot(DampedXEPMA, p, s)
            assert damped_ov < endpoint_ov, (
                f"DampedXEPMA(s={s},p={p}) overshoot {damped_ov:.6f} not below endpoint {endpoint_ov:.6f}"
            )
    print("PASS: DampedXEPMA overshoot strictly below XEPMA endpoint (incl. s = 1)")


def test_damped_xepma_zero_lag_kernel():
    """DampedXEPMA preserves zero lag: kernel m0 = 1, m1 = 0 (m2 stays nonzero)."""
    for s in (1.0, 2.0, 3.0):
        for p in (20.0, 40.0):
            m0, m1, _ = _kernel_moments(DampedXEPMA, p, s)
            assert abs(m0 - 1.0) < 1e-9, f"DampedXEPMA(s={s},p={p}) m0 = {m0}"
            assert abs(m1) < 1e-6, f"DampedXEPMA(s={s},p={p}) m1 = {m1} (lag not zero)"
    print("PASS: DampedXEPMA kernel m0 = 1, m1 = 0 at sampled (s, p)")


def test_damped_xepma_calc_next_stateless():
    """DampedXEPMA calc_next is a stateless probe consistent with get_next."""
    for s in (1.0, 2.0):
        filt = DampedXEPMA(20.0, s)
        for val in TEST_DATA[:60]:
            filt.get_next(val)
        probe = TEST_DATA[60]
        c1 = filt.calc_next(probe)
        filt.calc_next(999.0)
        c2 = filt.calc_next(probe)
        assert abs(c1 - c2) < 1e-14, f"DampedXEPMA(s={s}) calc_next not stateless"
        g = filt.get_next(probe)
        assert abs(c1 - g) < 1e-14, f"DampedXEPMA(s={s}) calc_next {c1} != get_next {g}"
    print("PASS: DampedXEPMA calc_next stateless and consistent with get_next")


if __name__ == '__main__':
    tests = [
        test_quadratic_xepma_s1_identical_to_endpoint,
        test_quadratic_xepma_quadratic_exact,
        test_quadratic_xepma_zero_lag_kernel,
        test_quadratic_xepma_calc_next_stateless,
        test_damped_xepma_coefficient_matches_reference,
        test_damped_xepma_overshoot_below_endpoint,
        test_damped_xepma_zero_lag_kernel,
        test_damped_xepma_calc_next_stateless,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
    if failed > 0:
        sys.exit(1)
