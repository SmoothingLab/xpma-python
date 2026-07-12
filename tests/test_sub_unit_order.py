"""Sub-unit order semantics: order-0 raises, IFEMA C1 shelf on (0, 1).

  - the lag-matched EMA cascade has no finite order-0 member, so MultiEMA,
    ReverseMultiEMA, IFEMA and XPMA all raise at order 0 (XPMA(p, 0, r=1) is the
    sole order-0 object, the identity on the zero-lag line);
  - IFEMA on (0, 1) is the moment-exact C1 shelf (mean lag L, kappa2 = L^2/s + L,
    non-negative kernel);
  - XEPMA keeps zero lag at fractional 0 < s < 1 with the C1 base.
"""

import random
import sys

sys.path.insert(0, '..')

import pytest

from xpma import EMA, ReverseMultiEMA, MultiEMA, IFEMA, XEPMA, XPMA


# ----------------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------------

def _measure_moments(make_filter, warm=3000, tol=1e-13, cap=200000):
    """Impulse response after a zero warm-up; return (mass, mean, kappa2).

    Accumulates incrementally and stops once the residual mass is below tol."""
    f = make_filter()
    for _ in range(warm):
        f.get_next(0.0)
    S = s1 = s2 = 0.0
    k = 0
    val = f.get_next(1.0)
    while True:
        S += val
        s1 += k * val
        s2 += k * k * val
        k += 1
        if k > 50 and S >= 1.0 - tol:
            break
        if k >= cap:
            break
        val = f.get_next(0.0)
    mean = s1 / S
    kappa2 = s2 / S - mean * mean
    return S, mean, kappa2


def _step_response(make_filter, warm=3000, span=8000):
    """Feed a zero warm-up then a held unit step; return the output stream."""
    f = make_filter()
    for _ in range(warm):
        f.get_next(0.0)
    return [f.get_next(1.0) for _ in range(span)]


def _random_walk(seed, n=400, start=100.0):
    rng = random.Random(seed)
    series = [start]
    for _ in range(n):
        series.append(series[-1] + rng.gauss(0.0, 1.0))
    return series


# ----------------------------------------------------------------------------
# (a) Order 0 raises across the cascade family.
# ----------------------------------------------------------------------------

def test_order_zero_raises():
    with pytest.raises(ValueError):
        MultiEMA(21.0, 0)
    with pytest.raises(ValueError):
        ReverseMultiEMA(21.0, 0)
    with pytest.raises(ValueError):
        IFEMA(21.0, 0.0)
    with pytest.raises(ValueError):
        IFEMA(21.0, -1.0)
    for r in (None, 0.0, 0.5):
        with pytest.raises(ValueError):
            XPMA(21.0, 0.0, r)
    print("PASS: order 0 raises across MultiEMA / ReverseMultiEMA / IFEMA / XPMA")


# ----------------------------------------------------------------------------
# (b) C1 moment exactness: mean lag L, kappa2 = L^2/s + L.
# ----------------------------------------------------------------------------

def test_c1_moment_exactness():
    for p in (5.0, 21.0, 100.0):
        L = (p - 1.0) / 2.0
        for s in (0.25, 0.5, 0.75):
            mass, mean, kappa2 = _measure_moments(lambda p=p, s=s: IFEMA(p, s))
            assert abs(mass - 1.0) < 1e-9, f"p={p} s={s}: mass {mass}"
            assert abs(mean - L) <= 1e-6 * L, f"p={p} s={s}: mean {mean} != L {L}"
            expected = L * L / s + L
            assert abs(kappa2 - expected) <= 1e-6 * expected, (
                f"p={p} s={s}: kappa2 {kappa2} != {expected}")
    print("PASS: IFEMA C1 shelf is moment-exact (mean=L, kappa2=L^2/s+L)")


# ----------------------------------------------------------------------------
# (c) No overshoot: unit step response stays within [0, 1].
# ----------------------------------------------------------------------------

def test_c1_no_overshoot():
    for p in (5.0, 21.0, 100.0):
        for s in (0.25, 0.5, 0.75):
            step = _step_response(lambda p=p, s=s: IFEMA(p, s))
            lo, hi = min(step), max(step)
            assert lo >= -1e-12, f"p={p} s={s}: step undershoot {lo}"
            assert hi <= 1.0 + 1e-12, f"p={p} s={s}: step overshoot {hi - 1.0}"
    print("PASS: IFEMA C1 step response stays within [0, 1] (no overshoot)")


# ----------------------------------------------------------------------------
# (d) Near-continuity at s = 1, and bit-identity at s = 1.
# ----------------------------------------------------------------------------

def test_c1_continuity_at_one():
    series = _random_walk(seed=20260711)
    approach = IFEMA(21.0, 0.999)
    ema = EMA(21.0)
    max_dev = max(abs(approach.get_next(x) - ema.get_next(x)) for x in series)
    assert max_dev < 0.05, f"IFEMA(0.999) not close to EMA(21): {max_dev}"

    exact = IFEMA(21.0, 1.0)
    multi = MultiEMA(21.0, 1)
    max_bit = max(abs(exact.get_next(x) - multi.get_next(x)) for x in series)
    assert max_bit == 0.0, f"IFEMA(1.0) != MultiEMA(21,1) bit-for-bit: {max_bit}"
    print("PASS: IFEMA -> EMA(p) as s -> 1; IFEMA(1.0) == MultiEMA(21,1) exactly")


# ----------------------------------------------------------------------------
# (e) XPMA(p, 0, 1) is the identity.
# ----------------------------------------------------------------------------

def test_xpma_order_zero_identity():
    series = _random_walk(seed=99)
    f = XPMA(21.0, 0.0, 1.0)
    max_dev = max(abs(f.get_next(x) - x) for x in series)
    assert max_dev == 0.0, f"XPMA(21,0,1) not identity: {max_dev}"
    # calc_next probe is also the identity.
    g = XPMA(21.0, 0.0, 1.0)
    assert g.calc_next(3.14) == 3.14
    print("PASS: XPMA(p, 0, 1) is the identity stream")


# ----------------------------------------------------------------------------
# (f) XEPMA zero lag at fractional 0 < s < 1 (C1 base preserves lag cancel).
# ----------------------------------------------------------------------------

def test_xepma_fractional_zero_lag():
    for p in (21.0, 50.0):
        f = XEPMA(p, 0.5)
        err = 0.0
        for t in range(4000):
            out = f.get_next(float(t))
            err = out - t
        assert abs(err) < 1e-6, f"XEPMA(p={p}, s=0.5) ramp lag: {err}"
    print("PASS: XEPMA at s=0.5 has zero steady-state lag on a linear ramp")


if __name__ == "__main__":
    tests = [
        test_order_zero_raises,
        test_c1_moment_exactness,
        test_c1_no_overshoot,
        test_c1_continuity_at_one,
        test_xpma_order_zero_identity,
        test_xepma_fractional_zero_lag,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
    if failed:
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")
