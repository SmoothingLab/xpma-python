"""Tests for new FastEMA and LeadEMA semantics.

FastEMA (new): nominal period, r_crit^M lag reduction. Genuinely faster than EMA(p).
LeadEMA (new): inflated period, r_crit^M lag reduction. Same lag as EMA(p) but less undershoot.
"""

import sys
import math

from xpma import LeadEMA, FastEMA, EMA, max_monotone_lag_reduction, lead_ema_max_lag_reduction


def _ramp_lag(filt, n=2000, warmup=500):
    """Estimate lag by comparing filter output to a ramp input."""
    slope = 1.0
    for i in range(warmup):
        filt.get_next(float(i) * slope)
    # After warmup, filter output lags by 'lag' bars on a ramp: output = input - lag*slope
    last_in = float(warmup - 1) * slope
    last_out = None
    for i in range(warmup, n):
        v = float(i) * slope
        out = filt.get_next(v)
        if out is not None:
            last_in = v
            last_out = out
    return last_in - last_out  # lag in bars


def _settled_step(filt, n=2000):
    """Return unit-step response values starting from first non-None output."""
    results = []
    for i in range(n):
        v = 100.0 if i >= 1 else 0.0
        out = filt.get_next(v)
        if out is not None:
            results.append(out / 100.0)
    return results


def test_fast_ema_lag_formula():
    """FastEMA lag = (1 - r_crit^M(s)) * (p-1)/2, strictly less than EMA's (p-1)/2."""
    for s in [1, 2, 3, 4]:
        r = max_monotone_lag_reduction(float(s))
        for p in [20, 50]:
            ema_lag = (p - 1) / 2.0
            expected_lag = (1.0 - r) * ema_lag
            filt = FastEMA(float(p), float(s))
            measured = _ramp_lag(filt)
            assert abs(measured - expected_lag) < 1e-3, (
                f"FastEMA(p={p}, s={s}): expected lag {expected_lag:.4f}, got {measured:.4f}"
            )
            assert measured < ema_lag - 0.5, (
                f"FastEMA(p={p}, s={s}): lag {measured:.4f} not strictly less than EMA lag {ema_lag:.4f}"
            )
    print("PASS test_fast_ema_lag_formula")


def test_lead_ema_lag_formula():
    """LeadEMA lag = (p-1)/2 to ~1e-3 on ramp, matching EMA(p)'s lag."""
    for s in [1, 2, 3, 4]:
        for p in [20, 50]:
            ema_lag = (p - 1) / 2.0
            filt = LeadEMA(float(p), float(s))
            measured = _ramp_lag(filt)
            assert abs(measured - ema_lag) < 1e-3, (
                f"LeadEMA(p={p}, s={s}): expected lag {ema_lag:.4f}, got {measured:.4f}"
            )
            assert abs(filt.time_lag - ema_lag) < 1e-12, (
                f"LeadEMA(p={p}, s={s}).time_lag={filt.time_lag} != {ema_lag}"
            )
    print("PASS test_lead_ema_lag_formula")


def test_integer_smoothness_no_overshoot():
    """Both FastEMA and LeadEMA step responses are monotone and overshoot-free at integer s."""
    for s in [1, 2, 3]:
        for p in [15, 40]:
            for cls, name in [(FastEMA, "FastEMA"), (LeadEMA, "LeadEMA")]:
                filt = cls(float(p), float(s))
                y = _settled_step(filt)
                max_val = max(y)
                assert max_val - 1.0 < 1e-9, (
                    f"{name}(p={p}, s={s}): overshoot {max_val - 1.0:.2e}"
                )
                diffs = [y[i] - y[i - 1] for i in range(1, len(y))]
                min_diff = min(diffs)
                assert min_diff > -1e-9, (
                    f"{name}(p={p}, s={s}): non-monotone step (min diff {min_diff:.2e})"
                )
    print("PASS test_integer_smoothness_no_overshoot")


def test_fractional_smoothness_no_overshoot():
    """FastEMA and LeadEMA no overshoot at fractional s (blend does not guarantee monotone)."""
    for s in [1.5, 2.5, 3.5]:
        for p in [40, 100]:
            for cls, name in [(FastEMA, "FastEMA"), (LeadEMA, "LeadEMA")]:
                filt = cls(float(p), float(s))
                y = _settled_step(filt)
                max_val = max(y)
                assert max_val - 1.0 < 1e-9, (
                    f"{name}(p={p}, s={s:.1f}): overshoot {max_val - 1.0:.2e}"
                )
    print("PASS test_fractional_smoothness_no_overshoot")


def test_lead_ema_equals_fast_ema_at_adjP():
    """LeadEMA(p, s) == FastEMA(adjP, s) where adjP = 1 + (p-1)/(1 - r_crit^M(s))."""
    import random
    rng = random.Random(42)
    for s in [1, 2, 3]:
        r = max_monotone_lag_reduction(float(s))
        for p in [15, 30]:
            adj_p = 1.0 + (p - 1.0) / (1.0 - r)
            lead = LeadEMA(float(p), float(s))
            fast = FastEMA(adj_p, float(s))
            diffs = []
            for _ in range(500):
                v = rng.uniform(50.0, 150.0)
                a = lead.get_next(v)
                b = fast.get_next(v)
                if a is not None and b is not None:
                    diffs.append(abs(a - b))
            assert max(diffs) < 1e-12, (
                f"LeadEMA(p={p},s={s}) != FastEMA(adjP={adj_p:.4f},s={s}): max diff {max(diffs):.2e}"
            )
    print("PASS test_lead_ema_equals_fast_ema_at_adjP")


def test_fractional_fast_ema_equals_manual_blend():
    """FastEMA(40, 2.5) == wlo*FastEMA(40,2) + whi*FastEMA(40,3) with moment-matched weights."""
    import random
    rng = random.Random(99)
    s = 2.5
    p = 40.0
    frac = s % 1.0
    floor_s = math.floor(s)
    ceil_s = math.ceil(s)
    wlo = (1.0 - frac) * floor_s / s
    whi = frac * ceil_s / s
    filt = FastEMA(p, s)
    lo = FastEMA(p, float(floor_s))
    hi = FastEMA(p, float(ceil_s))
    diffs = []
    for _ in range(500):
        v = rng.uniform(50.0, 150.0)
        out = filt.get_next(v)
        a = lo.get_next(v)
        b = hi.get_next(v)
        if out is not None and a is not None and b is not None:
            manual = wlo * a + whi * b
            diffs.append(abs(out - manual))
    assert max(diffs) < 1e-12, f"FastEMA fractional blend mismatch: max diff {max(diffs):.2e}"
    print("PASS test_fractional_fast_ema_equals_manual_blend")


def test_calc_next_stateless():
    """calc_next is stateless for both FastEMA and LeadEMA."""
    import random
    rng = random.Random(7)
    for cls, name in [(FastEMA, "FastEMA"), (LeadEMA, "LeadEMA")]:
        filt = cls(20.0, 2.0)
        # Warm up
        for _ in range(100):
            filt.get_next(rng.uniform(80.0, 120.0))
        # calc_next repeated probe should return same value
        probe = rng.uniform(80.0, 120.0)
        c1 = filt.calc_next(probe)
        c2 = filt.calc_next(probe)
        assert c1 is not None, f"{name}: calc_next returned None after warmup"
        assert abs(c1 - c2) < 1e-14, f"{name}: calc_next not stateless ({c1} vs {c2})"
        # And it should match a subsequent get_next
        g = filt.get_next(probe)
        assert abs(c1 - g) < 1e-14, f"{name}: calc_next {c1} != get_next {g}"
    print("PASS test_calc_next_stateless")


if __name__ == "__main__":
    tests = [
        test_fast_ema_lag_formula,
        test_lead_ema_lag_formula,
        test_integer_smoothness_no_overshoot,
        test_fractional_smoothness_no_overshoot,
        test_lead_ema_equals_fast_ema_at_adjP,
        test_fractional_fast_ema_equals_manual_blend,
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
