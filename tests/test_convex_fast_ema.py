"""Tests for ConvexFastEMA (the r_crit^C member of the XPMA family).

r_crit^C is the convexity boundary: the discrete kernel is non-increasing at s = 1
and unimodal at s >= 2. The continuous constant sits safely inside the discrete
boundary (O(1/p) margin), so at r_crit^C the discrete kernel is unimodal with room
to spare. At fractional s the output-level realisation preserves unimodality where
the internal cascade blend would not.

Kernels are extracted with a long zero warm-up before the impulse (so the cascade
starts at rest), and the unimodality scan is restricted to the support where the
kernel exceeds 1e-11 x peak so tail floating-point noise cannot create spurious
flips.
"""

import math
import sys

sys.path.insert(0, '..')

from xpma import ConvexFastEMA, r_crit_c, EMA


def _impulse_kernel(filt, n=8000, warm=400):
    for _ in range(warm):
        filt.get_next(0.0)
    h = [filt.get_next(1.0)]
    for _ in range(n - 1):
        h.append(filt.get_next(0.0))
    return h


def _kernel_upcrossings(h, rel_tol=1e-11):
    """Count '- -> +' sign changes of Delta h over the significant support.

    A non-increasing kernel (s = 1) and a unimodal kernel (s >= 2) both have zero:
    the first difference is <= 0 throughout (s = 1) or >= 0 then <= 0 (unimodal),
    so it never turns back up."""
    peak = max(h)
    thr = rel_tol * peak
    # Significant support: from the first index above threshold to the last.
    idx = [i for i, v in enumerate(h) if v > thr]
    lo, hi = idx[0], idx[-1]
    upcross = 0
    eps = 1e-13 * peak
    for n in range(lo + 2, hi + 1):
        d_prev = h[n - 1] - h[n - 2]
        d_curr = h[n] - h[n - 1]
        if d_prev < -eps and d_curr > eps:
            upcross += 1
    return upcross


def test_convex_kernel_non_increasing_s1():
    """At s = 1 the ConvexFastEMA kernel is non-increasing (mode at n = 0)."""
    for p in (15.0, 21.0, 40.0, 100.0):
        h = _impulse_kernel(ConvexFastEMA(p, 1.0))
        assert h.index(max(h)) == 0, f"ConvexFastEMA(p={p},s=1) mode not at n=0"
        assert _kernel_upcrossings(h) == 0, f"ConvexFastEMA(p={p},s=1) kernel re-accelerates"
    print("PASS: ConvexFastEMA kernel non-increasing at s = 1")


def test_convex_kernel_unimodal_integer_s():
    """At s = 2..4 the ConvexFastEMA kernel is unimodal (one interior mode, no rebound)."""
    for s in (2.0, 3.0, 4.0):
        for p in (21.0, 40.0):
            h = _impulse_kernel(ConvexFastEMA(p, s))
            assert _kernel_upcrossings(h) == 0, (
                f"ConvexFastEMA(p={p},s={s}) kernel not unimodal (rate re-accelerates)"
            )
    print("PASS: ConvexFastEMA kernel unimodal at s = 2, 3, 4")


def test_convex_kernel_unimodal_fractional_s():
    """At fractional s the output-level realisation keeps the kernel unimodal."""
    for s in (1.5, 2.5):
        for p in (40.0, 100.0):
            h = _impulse_kernel(ConvexFastEMA(p, s))
            assert _kernel_upcrossings(h) == 0, (
                f"ConvexFastEMA(p={p},s={s}) fractional kernel not unimodal"
            )
    print("PASS: ConvexFastEMA kernel unimodal at fractional s = 1.5, 2.5")


def test_convex_lag_reduction_value_and_ramp():
    """ConvexFastEMA carries lag (1 - r_crit^C(s)) * (p-1)/2, less than EMA's."""
    for s in (1.0, 2.0, 3.0):
        p = 21.0
        r = r_crit_c(s)
        assert abs(ConvexFastEMA(p, s).lag_reduction - r) < 1e-12
        conv = ConvexFastEMA(p, s)
        ema = EMA(p)
        for i in range(2000):
            rc = conv.get_next(float(i))
            re = ema.get_next(float(i))
        conv_lag = 1999.0 - rc
        ema_lag = 1999.0 - re
        expected = (1.0 - r) * (p - 1.0) / 2.0
        assert abs(conv_lag - expected) < 1e-2, f"ConvexFastEMA(s={s}) lag {conv_lag:.4f} != {expected:.4f}"
        assert conv_lag < ema_lag, "ConvexFastEMA not faster than EMA"
    print("PASS: ConvexFastEMA lag = (1 - r_crit^C) * (p-1)/2, faster than EMA")


if __name__ == "__main__":
    tests = [
        test_convex_kernel_non_increasing_s1,
        test_convex_kernel_unimodal_integer_s,
        test_convex_kernel_unimodal_fractional_s,
        test_convex_lag_reduction_value_and_ramp,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
    if failed:
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")
