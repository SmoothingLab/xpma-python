"""Validate the xpma library against known mathematical properties."""

import sys
import math
sys.path.insert(0, '..')

from xpma import EMA, ReverseEMA, MultiEMA, IFEMA, XEPMA, XPMA, FastEMA, LeadEMA, SecantSolver, lead_ema_max_lag_reduction


# Test data: 100 points of a sine wave + linear trend + noise-like pattern
TEST_DATA = []
for i in range(200):
    TEST_DATA.append(100.0 + i * 0.1 + 5.0 * math.sin(i * 0.15) + 2.0 * math.sin(i * 0.47))


def test_ema_basic():
    """EMA should converge towards constant input."""
    ema = EMA(10)
    for _ in range(100):
        result = ema.get_next(50.0)
    assert abs(result - 50.0) < 0.001, f"EMA didn't converge: {result}"
    print("PASS: EMA converges to constant input")


def test_ema_first_value():
    """First EMA output should equal first input."""
    ema = EMA(20)
    result = ema.get_next(42.5)
    assert result == 42.5, f"First value mismatch: {result}"
    print("PASS: EMA first value equals input")


def test_reverse_ema_roundtrip():
    """EMA followed by ReverseEMA should recover original input."""
    period = 15.0
    ema = EMA(period)
    rev = ReverseEMA(period)

    max_error = 0.0
    for i, val in enumerate(TEST_DATA):
        smoothed = ema.get_next(val)
        recovered = rev.get_next(smoothed)
        if i > 0:  # Skip first value (initialisation)
            error = abs(recovered - val)
            max_error = max(max_error, error)

    assert max_error < 1e-10, f"Roundtrip error too large: {max_error}"
    print(f"PASS: EMA -> ReverseEMA roundtrip (max error: {max_error:.2e})")


def test_multi_ema_period_adjustment():
    """MultiEMA(p, n) should have the same lag as EMA(p)."""
    # Test by feeding a linear ramp and measuring the offset
    period = 21.0

    ema = EMA(period)
    multi = MultiEMA(period, 3)

    # Feed enough linear ramp data for both to settle
    for i in range(500):
        val = float(i)
        e = ema.get_next(val)
        m = multi.get_next(val)

    # Both should lag by (period-1)/2 = 10 behind value 499
    ema_lag = 499.0 - e
    multi_lag = 499.0 - m

    assert abs(ema_lag - multi_lag) < 0.01, f"Lag mismatch: EMA={ema_lag:.4f}, Multi={multi_lag:.4f}"
    print(f"PASS: MultiEMA has same lag as EMA (EMA={ema_lag:.4f}, Multi={multi_lag:.4f})")


def test_multi_ema_smoother():
    """MultiEMA(p, 2) should be smoother than EMA(p) on noisy data."""
    period = 21.0
    ema = EMA(period)
    multi = MultiEMA(period, 2)

    ema_turns = 0
    multi_turns = 0
    prev_ema = None
    prev_multi = None
    prev_ema_dir = None
    prev_multi_dir = None

    for val in TEST_DATA:
        e = ema.get_next(val)
        m = multi.get_next(val)

        if prev_ema is not None:
            ema_dir = e > prev_ema
            multi_dir = m > prev_multi
            if prev_ema_dir is not None:
                if ema_dir != prev_ema_dir:
                    ema_turns += 1
                if multi_dir != prev_multi_dir:
                    multi_turns += 1
            prev_ema_dir = ema_dir
            prev_multi_dir = multi_dir

        prev_ema = e
        prev_multi = m

    assert multi_turns < ema_turns, f"MultiEMA not smoother: {multi_turns} vs {ema_turns} turns"
    print(f"PASS: MultiEMA smoother than EMA ({multi_turns} vs {ema_turns} direction changes)")


def test_xepma_zero_time_lag():
    """XEPMA should have zero average time lag on a linear ramp."""
    period = 21.0
    xepma = XEPMA(period, smoothness=1.0)

    # Feed linear ramp - XEPMA should track it perfectly after settling
    for i in range(500):
        val = float(i)
        result = xepma.get_next(val)

    # With zero time lag on a linear ramp, output should equal input
    error = abs(result - 499.0)
    assert error < 0.01, f"XEPMA time lag on linear ramp: {error:.6f}"
    print(f"PASS: XEPMA zero time lag on linear ramp (error: {error:.6f})")


def test_xepma_higher_order():
    """XEPMA^[2] should also have zero time lag on linear ramp."""
    period = 21.0
    xepma = XEPMA(period, smoothness=2.0)

    for i in range(500):
        val = float(i)
        result = xepma.get_next(val)

    error = abs(result - 499.0)
    assert error < 0.01, f"XEPMA^[2] time lag: {error:.6f}"
    print(f"PASS: XEPMA^[2] zero time lag on linear ramp (error: {error:.6f})")


def test_xpma_r0_equals_ifema():
    """XPMA with r=0 should equal IFEMA."""
    period = 21.0
    smoothness = 2.0

    xpma_filter = XPMA(period, smoothness, lag_reduction=0.0)
    ifema = IFEMA(period, smoothness)

    max_error = 0.0
    for val in TEST_DATA:
        n = xpma_filter.get_next(val)
        f = ifema.get_next(val)
        error = abs(n - f)
        max_error = max(max_error, error)

    assert max_error < 1e-12, f"XPMA(r=0) != IFEMA: {max_error}"
    print(f"PASS: XPMA(r=0) equals IFEMA (max error: {max_error:.2e})")


def test_xpma_r1_equals_xepma_endpoint():
    """XPMA with r=1 should equal the XEPMA endpoint.

    The r = 1 endpoint of the lag-reduction family is XEPMA itself (the zero-lag
    endpoint). QuadraticXEPMA and DampedXEPMA sit off the family axis and are not
    the r = 1 member."""
    period = 21.0
    smoothness = 2.0

    xpma_filter = XPMA(period, smoothness, lag_reduction=1.0)
    endpoint = XEPMA(period, smoothness)

    max_error = 0.0
    for val in TEST_DATA:
        n = xpma_filter.get_next(val)
        b = endpoint.get_next(val)
        error = abs(n - b)
        max_error = max(max_error, error)

    assert max_error < 1e-10, f"XPMA(r=1) != XEPMA endpoint: {max_error}"
    print(f"PASS: XPMA(r=1) equals XEPMA endpoint (max error: {max_error:.2e})")


def test_ifema_fractional():
    """IFEMA^[1.5] should be between IFEMA^[1] and IFEMA^[2]."""
    period = 21.0
    ma1 = IFEMA(period, 1.0)
    ma15 = IFEMA(period, 1.5)
    ma2 = IFEMA(period, 2.0)

    between_count = 0
    total = 0

    for val in TEST_DATA:
        r1 = ma1.get_next(val)
        r15 = ma15.get_next(val)
        r2 = ma2.get_next(val)

        lo = min(r1, r2)
        hi = max(r1, r2)
        if lo <= r15 <= hi:
            between_count += 1
        total += 1

    pct = 100.0 * between_count / total
    assert pct > 95.0, f"IFEMA^[1.5] not between ^[1] and ^[2] often enough: {pct:.1f}%"
    print(f"PASS: IFEMA^[1.5] between ^[1] and ^[2] {pct:.1f}% of the time")


def test_calc_next_stateless():
    """calc_next should not change internal state."""
    period = 21.0
    ema = EMA(period)

    # Advance a few steps
    for val in TEST_DATA[:10]:
        ema.get_next(val)

    # Call calc_next multiple times with different values
    state_before = ema.result
    ema.calc_next(999.0)
    ema.calc_next(0.0)
    ema.calc_next(-500.0)
    state_after = ema.result

    assert state_before == state_after, "calc_next changed state!"
    print("PASS: calc_next is stateless")


def test_secant_solver():
    """SecantSolver should reverse XEPMA."""
    period = 15.0
    smoothness = 1.0

    xepma = XEPMA(period, smoothness)
    rev_xepma = XEPMA(period, smoothness)
    solver = SecantSolver(rev_xepma)

    # Forward pass
    smoothed = []
    for val in TEST_DATA:
        smoothed.append(xepma.get_next(val))

    # Reverse pass
    max_error = 0.0
    prev_result = None
    for i, s in enumerate(smoothed):
        estimate = s if prev_result is None else prev_result
        recovered = solver.solve(s, estimate)
        prev_result = recovered

        if i > 5:  # Allow settling
            error = abs(recovered - TEST_DATA[i])
            max_error = max(max_error, error)

    assert max_error < 0.01, f"SecantSolver reversal error too large: {max_error}"
    print(f"PASS: SecantSolver reverses XEPMA (max error: {max_error:.6f})")


def test_fast_ema_no_overshoot():
    """FastEMA should not overshoot on a step input."""
    period = 21.0
    fast = FastEMA(period, smoothness=1.0)

    # Step from 0 to 100
    for _ in range(50):
        fast.get_next(0.0)

    max_val = 0.0
    for _ in range(200):
        result = fast.get_next(100.0)
        max_val = max(max_val, result)

    assert max_val <= 100.001, f"FastEMA overshoots: {max_val}"
    print(f"PASS: FastEMA no overshoot on step input (max: {max_val:.6f})")


def test_fast_ema_faster_than_ema():
    """FastEMA (nominal, r_crit^M) is faster than EMA: strictly less time lag."""
    period = 21.0
    ema = EMA(period)
    fast = FastEMA(period, smoothness=1.0)

    for i in range(500):
        val = float(i)
        e = ema.get_next(val)
        f = fast.get_next(val)

    ema_lag = 499.0 - e
    fast_lag = 499.0 - f
    pred = (1.0 - lead_ema_max_lag_reduction(1.0)) * (period - 1.0) / 2.0

    assert fast_lag < ema_lag - 0.5, f"FastEMA not faster: EMA={ema_lag:.4f}, Fast={fast_lag:.4f}"
    assert abs(fast_lag - pred) < 0.1, f"FastEMA lag {fast_lag:.4f} != predicted {pred:.4f}"
    print(f"PASS: FastEMA faster than EMA (EMA={ema_lag:.4f}, Fast={fast_lag:.4f})")


def test_lead_ema_same_lag_as_ema():
    """LeadEMA (inflated, r_crit^M) matches EMA's time lag for the given period."""
    period = 21.0
    ema = EMA(period)
    lead = LeadEMA(period, smoothness=1.0)

    for i in range(500):
        val = float(i)
        e = ema.get_next(val)
        le = lead.get_next(val)

    ema_lag = 499.0 - e
    lead_lag = 499.0 - le

    assert abs(ema_lag - lead_lag) < 0.1, f"Lag mismatch: EMA={ema_lag:.4f}, Lead={lead_lag:.4f}"
    print(f"PASS: LeadEMA same lag as EMA (EMA={ema_lag:.4f}, Lead={lead_lag:.4f})")


if __name__ == '__main__':
    tests = [
        test_ema_basic,
        test_ema_first_value,
        test_reverse_ema_roundtrip,
        test_multi_ema_period_adjustment,
        test_multi_ema_smoother,
        test_xepma_zero_time_lag,
        test_xepma_higher_order,
        test_xpma_r0_equals_ifema,
        test_xpma_r1_equals_xepma_endpoint,
        test_ifema_fractional,
        test_calc_next_stateless,
        test_secant_solver,
        test_fast_ema_no_overshoot,
        test_fast_ema_faster_than_ema,
        test_lead_ema_same_lag_as_ema,
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
