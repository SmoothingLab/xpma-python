"""Tests for LeadXEPMA."""

import sys
sys.path.insert(0, '..')

from xpma import LeadXEPMA


def test_lead_xepma_zero_time_lag():
    """LeadXEPMA should have zero time lag on a linear ramp."""
    period = 21.0
    fxep = LeadXEPMA(period, smoothness=2.0)

    for i in range(500):
        result = fxep.get_next(float(i))

    error = abs(result - 499.0)
    assert error < 0.1, f"LeadXEPMA time lag: {error:.6f}"
    print(f"PASS: LeadXEPMA zero time lag (error: {error:.6f})")


def test_lead_xepma_zero_lag_multiple_orders():
    """LeadXEPMA is built on LeadEMA (lag-matched), so it is zero-lag but does not
    track closer to raw price than XEPMA. This verifies the zero-lag property at
    several smoothness orders with a tight threshold."""
    for smoothness in (1.0, 2.0, 3.0):
        fxep = LeadXEPMA(21.0, smoothness=smoothness)
        for i in range(500):
            result = fxep.get_next(float(i))
        error = abs(result - 499.0)
        assert error < 0.01, f"LeadXEPMA(s={smoothness}) not zero-lag: {error:.6f}"
    print("PASS: LeadXEPMA zero time lag at s=1,2,3 (LeadEMA-based building block)")


if __name__ == '__main__':
    tests = [
        test_lead_xepma_zero_time_lag,
        test_lead_xepma_zero_lag_multiple_orders,
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
