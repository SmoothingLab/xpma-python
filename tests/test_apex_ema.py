"""Tests for ApexFastEMA and ApexLeadEMA (the r_crit^O members of the XPMA family).

r_crit^O is the no-overshoot boundary: the largest lag reduction whose unit-step
response never crosses the target. At exactly r_crit^O the apex of the step
response touches the target in the continuous limit (the step error is tangent to
zero), so the response rises fast, grazes the target, dips below it, then settles.
Unlike FastEMA (monotone) and ConvexFastEMA (convex step error) the Apex members
are NON-MONOTONE - the sub-target dip is intrinsic - and the retained guarantee is
no overshoot only.

Discrete safety of r_crit^O is numerical: locking at the continuous constant never
overshoots at any tested (s, p) with s = 1..4 and p in {5, 10, 20, 50, 200} (worst
step-response overshoot 0.0), consistent with the O(1/p) margin from above.

Step responses and kernels are extracted with a long zero warm-up before the step
or impulse (so the cascade starts at rest). Fractional s is realised at the OUTPUT
level: no overshoot is preserved because a convex blend of two no-overshoot (step
error >= 0) components has step error >= 0.
"""

import math
import sys

sys.path.insert(0, '..')

from xpma import (
    ApexFastEMA, ApexLeadEMA, FastEMA, LeadEMA, ConvexFastEMA, ConvexLeadEMA,
    r_crit_o, r_crit_m, r_crit_c, r_crit_o_effective, EMA,
)


def _step_response(filt, n=40000, warm=2000):
    for _ in range(warm):
        filt.get_next(0.0)
    return [filt.get_next(1.0) for _ in range(n)]


def _impulse_kernel(filt, n=20000, warm=2000):
    for _ in range(warm):
        filt.get_next(0.0)
    h = [filt.get_next(1.0)]
    for _ in range(n - 1):
        h.append(filt.get_next(0.0))
    return h


def _kernel_lag(h):
    """First moment (group delay) of the impulse response: sum(n h[n]) / sum(h[n])."""
    total = math.fsum(h)
    moment = math.fsum(n * v for n, v in enumerate(h))
    return moment / total


def _first_apex(y):
    """Index and value of the first local maximum (the apex, before the sub-target dip)."""
    for i in range(1, len(y)):
        if y[i] < y[i - 1]:
            return i - 1, y[i - 1]
    return len(y) - 1, max(y)  # monotone: no interior apex


def test_apex_no_overshoot_integer_and_fractional():
    """The discrete step response never exceeds the target (tol 1e-9), integer and fractional s."""
    target = 1.0
    for cls in (ApexFastEMA, ApexLeadEMA):
        for s in (1.0, 2.0, 3.0, 4.0, 1.5, 2.5):
            for p in (10.0, 20.0, 50.0):
                peak = max(_step_response(cls(p, s)))
                assert peak <= target + 1e-9, (
                    f"{cls.__name__}(p={p},s={s}) overshoots: max {peak:.12f} > 1"
                )
    print("PASS: ApexFastEMA/ApexLeadEMA no overshoot at integer and fractional s")


def test_apex_non_monotone_dip():
    """The distinguishing shape: a genuine sub-target dip after the apex (s = 1, p = 20).

    This separates Apex from FastEMA, whose step response is monotone at the same
    (s, p) with no dip and no negative first differences."""
    y = _step_response(ApexFastEMA(20.0, 1.0), n=3000)
    apex_i, apex_v = _first_apex(y)
    neg_diffs = sum(1 for i in range(apex_i + 1, len(y)) if y[i] - y[i - 1] < -1e-15)
    assert neg_diffs > 0, "ApexFastEMA(20, 1) step response has no sub-target dip"
    dip = min(y[apex_i:])
    assert dip < apex_v - 1e-6, (
        f"ApexFastEMA(20, 1) dip {dip:.9f} not below apex {apex_v:.9f}"
    )
    # Contrast: FastEMA at the same (p, s) is monotone (no dip).
    yf = _step_response(FastEMA(20.0, 1.0), n=3000)
    fast_neg = sum(1 for i in range(1, len(yf)) if yf[i] - yf[i - 1] < -1e-15)
    assert fast_neg == 0, "FastEMA(20, 1) should be monotone (no dip)"
    print(f"PASS: ApexFastEMA non-monotone (dip, {neg_diffs} descending steps); FastEMA monotone")


def test_apex_near_target_and_margin_ordering():
    """The apex approaches the target as p grows (the tangency is the continuous limit).

    The first local maximum sits below the target and its margin shrinks with p, so at
    p = 200 the apex is within a small margin below 1 and closer than at p = 10."""
    margins = {}
    for p in (10.0, 50.0, 200.0):
        y = _step_response(ApexFastEMA(p, 1.0), n=8000)
        apex_i, apex_v = _first_apex(y)
        assert apex_v <= 1.0 + 1e-12, f"apex {apex_v} above target at p={p}"
        margins[p] = 1.0 - apex_v
    assert margins[200.0] < margins[50.0] < margins[10.0], (
        f"apex margin not shrinking with p: {margins}"
    )
    assert margins[200.0] < 1e-2, f"apex still far below target at p=200: {margins[200.0]:.4e}"
    print(f"PASS: ApexFastEMA apex approaches target as p grows (margins {margins})")


def test_apex_lead_lag_matches_ema():
    """ApexLeadEMA(p, s)'s kernel first moment equals EMA(p)'s lag (p-1)/2."""
    for s in (1.0, 2.0, 3.0, 4.0, 1.5, 2.5):
        for p in (20.0, 50.0):
            filt = ApexLeadEMA(p, s)
            assert abs(filt.time_lag - (p - 1.0) / 2.0) < 1e-12, (
                f"ApexLeadEMA(p={p},s={s}).time_lag={filt.time_lag} != {(p-1)/2}"
            )
            lag = _kernel_lag(_impulse_kernel(filt))
            assert abs(lag - (p - 1.0) / 2.0) < 1e-4, (
                f"ApexLeadEMA(p={p},s={s}) kernel lag {lag:.6f} != EMA lag {(p-1)/2:.6f}"
            )
    print("PASS: ApexLeadEMA kernel first moment = EMA(p) lag (p-1)/2")


def test_apex_ladder_ordering():
    """Nominal lags ApexFast < Fast < ConvexFast; inflations ApexLead > Lead > ConvexLead.

    Follows from r_crit^O > r_crit^M > r_crit^C: larger r means less lag at the nominal
    period and more inflation to restore EMA's lag."""
    for s in (1.0, 2.0, 3.0, 4.0):
        for p in (21.0, 50.0):
            la = ApexFastEMA(p, s).time_lag
            lf = FastEMA(p, s).time_lag
            lc = ConvexFastEMA(p, s).time_lag
            assert la < lf < lc, (
                f"nominal lag order broken at (p={p},s={s}): Apex {la:.5f}, Fast {lf:.5f}, Convex {lc:.5f}"
            )
        ro, rm, rc = r_crit_o(s), r_crit_m(s), r_crit_c(s)
        assert rc < rm < ro, f"r_crit ladder broken at s={s}: {rc}, {rm}, {ro}"
        infl_o = 1.0 / (1.0 - ro)
        infl_m = 1.0 / (1.0 - rm)
        infl_c = 1.0 / (1.0 - rc)
        assert infl_o > infl_m > infl_c, (
            f"inflation order broken at s={s}: Apex {infl_o:.4f}, Lead {infl_m:.4f}, Convex {infl_c:.4f}"
        )
        # Lag-matched siblings all carry EMA(p)'s lag.
        assert abs(ApexLeadEMA(50.0, s).time_lag - LeadEMA(50.0, s).time_lag) < 1e-12
        assert abs(ApexLeadEMA(50.0, s).time_lag - ConvexLeadEMA(50.0, s).time_lag) < 1e-12
    print("PASS: nominal lags ApexFast < Fast < ConvexFast; inflations ApexLead > Lead > ConvexLead")


def test_apex_effective_lag_reduction():
    """lag_reduction equals r_crit^O(s) at integer s and the convex blend at fractional s."""
    for s in (1.0, 2.0, 3.0):
        assert abs(ApexFastEMA(21.0, s).lag_reduction - r_crit_o(s)) < 1e-12
        assert abs(ApexLeadEMA(21.0, s).lag_reduction - r_crit_o(s)) < 1e-12
    for s in (1.5, 2.5):
        eff = r_crit_o_effective(s)
        assert abs(ApexFastEMA(21.0, s).lag_reduction - eff) < 1e-12
        # The blend lies strictly between the bracketing integer values.
        lo, hi = r_crit_o(math.floor(s)), r_crit_o(math.ceil(s))
        assert hi < eff < lo, f"effective r {eff} not between {hi} and {lo} at s={s}"
    print("PASS: ApexFastEMA/ApexLeadEMA effective lag reduction correct (integer and fractional)")


def test_apex_calc_next_stateless():
    """calc_next is a stateless probe consistent with get_next."""
    import random
    rng = random.Random(23)
    for filt in (ApexFastEMA(20.0, 2.0), ApexLeadEMA(20.0, 2.5)):
        for _ in range(100):
            filt.get_next(rng.uniform(80.0, 120.0))
        probe = rng.uniform(80.0, 120.0)
        c1 = filt.calc_next(probe)
        c2 = filt.calc_next(probe)
        assert c1 is not None, "calc_next returned None after warmup"
        assert abs(c1 - c2) < 1e-14, f"calc_next not stateless ({c1} vs {c2})"
        g = filt.get_next(probe)
        assert abs(c1 - g) < 1e-14, f"calc_next {c1} != get_next {g}"
    print("PASS: ApexFastEMA/ApexLeadEMA calc_next stateless and consistent")


if __name__ == "__main__":
    tests = [
        test_apex_no_overshoot_integer_and_fractional,
        test_apex_non_monotone_dip,
        test_apex_near_target_and_margin_ordering,
        test_apex_lead_lag_matches_ema,
        test_apex_ladder_ordering,
        test_apex_effective_lag_reduction,
        test_apex_calc_next_stateless,
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
