"""Tests for ConvexLeadEMA (the lag-matched r_crit^C member of the XPMA family).

ConvexLeadEMA is to ConvexFastEMA what LeadEMA is to FastEMA: XPMA locked to the
convexity boundary r_crit^C(s), with the averaging period inflated by
1/(1 - r_crit^C(s)) so the output lag matches EMA(period)'s lag (period - 1)/2. It
spends the convexity-boundary lag reduction on tracking closer at matched lag rather
than on speed.

The convexity property (kernel non-increasing at s = 1, unimodal at s >= 2) is
period-independent in the continuous limit and the discrete boundary sits above the
continuous constant with an O(1/p) margin; inflation only increases the working
period, so the guarantee carries over. It is asserted here, not assumed. At
fractional s the output-level realisation preserves it where the internal cascade
blend would not.

Kernels are extracted with a long zero warm-up before the impulse (so the cascade
starts at rest), and the unimodality scan is restricted to the support where the
kernel exceeds 1e-11 x peak so tail floating-point noise cannot create spurious
flips.
"""

import math
import sys

sys.path.insert(0, '..')

from xpma import ConvexLeadEMA, LeadEMA, ConvexFastEMA, r_crit_c, r_crit_m, EMA


def _impulse_kernel(filt, n=12000, warm=600):
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


def _kernel_lag(h):
    """First moment (group delay) of the impulse response: sum(n h[n]) / sum(h[n])."""
    total = math.fsum(h)
    moment = math.fsum(n * v for n, v in enumerate(h))
    return moment / total


def test_convex_lead_kernel_non_increasing_s1():
    """At s = 1 the ConvexLeadEMA kernel is non-increasing (mode at n = 0)."""
    for p in (15.0, 21.0, 40.0, 100.0):
        h = _impulse_kernel(ConvexLeadEMA(p, 1.0))
        assert h.index(max(h)) == 0, f"ConvexLeadEMA(p={p},s=1) mode not at n=0"
        assert _kernel_upcrossings(h) == 0, f"ConvexLeadEMA(p={p},s=1) kernel re-accelerates"
    print("PASS: ConvexLeadEMA kernel non-increasing at s = 1")


def test_convex_lead_kernel_unimodal_integer_s():
    """At s = 2..4 the ConvexLeadEMA kernel is unimodal (one interior mode, no rebound)."""
    for s in (2.0, 3.0, 4.0):
        for p in (21.0, 40.0):
            h = _impulse_kernel(ConvexLeadEMA(p, s))
            assert _kernel_upcrossings(h) == 0, (
                f"ConvexLeadEMA(p={p},s={s}) kernel not unimodal (rate re-accelerates)"
            )
    print("PASS: ConvexLeadEMA kernel unimodal at s = 2, 3, 4")


def test_convex_lead_kernel_unimodal_fractional_s():
    """At fractional s the output-level realisation keeps the kernel unimodal."""
    for s in (1.5, 2.5):
        for p in (40.0, 100.0):
            h = _impulse_kernel(ConvexLeadEMA(p, s))
            assert _kernel_upcrossings(h) == 0, (
                f"ConvexLeadEMA(p={p},s={s}) fractional kernel not unimodal"
            )
    print("PASS: ConvexLeadEMA kernel unimodal at fractional s = 1.5, 2.5")


def test_convex_lead_lag_matches_ema():
    """ConvexLeadEMA(p, s)'s kernel first moment equals EMA(p)'s lag (p-1)/2."""
    for s in (1.0, 2.0, 3.0, 4.0, 1.5, 2.5):
        for p in (20.0, 50.0):
            filt = ConvexLeadEMA(p, s)
            assert abs(filt.time_lag - (p - 1.0) / 2.0) < 1e-12, (
                f"ConvexLeadEMA(p={p},s={s}).time_lag={filt.time_lag} != {(p-1)/2}"
            )
            h = _impulse_kernel(filt)
            lag = _kernel_lag(h)
            ema_lag = (p - 1.0) / 2.0
            assert abs(lag - ema_lag) < 1e-4, (
                f"ConvexLeadEMA(p={p},s={s}) kernel lag {lag:.6f} != EMA lag {ema_lag:.6f}"
            )
    print("PASS: ConvexLeadEMA kernel first moment = EMA(p) lag (p-1)/2")


def test_convex_lead_equals_convex_fast_at_adjP():
    """ConvexLeadEMA(p, s) == ConvexFastEMA(adjP, s), adjP = 1 + (p-1)/(1 - r_crit^C(s))."""
    import random
    rng = random.Random(37)
    for s in (1.0, 2.0, 3.0):
        r = r_crit_c(s)
        for p in (15.0, 30.0):
            adj_p = 1.0 + (p - 1.0) / (1.0 - r)
            lead = ConvexLeadEMA(p, s)
            fast = ConvexFastEMA(adj_p, s)
            diffs = []
            for _ in range(500):
                v = rng.uniform(50.0, 150.0)
                a = lead.get_next(v)
                b = fast.get_next(v)
                if a is not None and b is not None:
                    diffs.append(abs(a - b))
            assert max(diffs) < 1e-12, (
                f"ConvexLeadEMA(p={p},s={s}) != ConvexFastEMA(adjP={adj_p:.4f},s={s}): "
                f"max diff {max(diffs):.2e}"
            )
    print("PASS: ConvexLeadEMA(p) == ConvexFastEMA(inflated p)")


def test_convex_lead_vs_lead_relationship():
    """Same lag as LeadEMA, but smaller r and smaller inflation (r_crit^C < r_crit^M)."""
    for s in (1.0, 2.0, 3.0, 4.0):
        for p in (20.0, 50.0):
            conv = ConvexLeadEMA(p, s)
            lead = LeadEMA(p, s)
            # Both match EMA(p)'s lag.
            assert abs(conv.time_lag - lead.time_lag) < 1e-12, (
                f"ConvexLeadEMA/LeadEMA lag mismatch at (p={p},s={s})"
            )
            rc, rm = r_crit_c(s), r_crit_m(s)
            assert rc < rm, f"r_crit^C({s})={rc} not < r_crit^M({s})={rm}"
            infl_c = 1.0 / (1.0 - rc)
            infl_m = 1.0 / (1.0 - rm)
            assert infl_c < infl_m, (
                f"ConvexLeadEMA inflation {infl_c:.4f} not < LeadEMA inflation {infl_m:.4f} at s={s}"
            )
    print("PASS: ConvexLeadEMA matches LeadEMA lag with smaller r and smaller inflation")


def test_calc_next_stateless():
    """calc_next is a stateless probe consistent with get_next."""
    import random
    rng = random.Random(11)
    filt = ConvexLeadEMA(20.0, 2.0)
    for _ in range(100):
        filt.get_next(rng.uniform(80.0, 120.0))
    probe = rng.uniform(80.0, 120.0)
    c1 = filt.calc_next(probe)
    c2 = filt.calc_next(probe)
    assert c1 is not None, "calc_next returned None after warmup"
    assert abs(c1 - c2) < 1e-14, f"calc_next not stateless ({c1} vs {c2})"
    g = filt.get_next(probe)
    assert abs(c1 - g) < 1e-14, f"calc_next {c1} != get_next {g}"
    print("PASS: ConvexLeadEMA calc_next stateless and consistent")


if __name__ == "__main__":
    tests = [
        test_convex_lead_kernel_non_increasing_s1,
        test_convex_lead_kernel_unimodal_integer_s,
        test_convex_lead_kernel_unimodal_fractional_s,
        test_convex_lead_lag_matches_ema,
        test_convex_lead_equals_convex_fast_at_adjP,
        test_convex_lead_vs_lead_relationship,
        test_calc_next_stateless,
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
