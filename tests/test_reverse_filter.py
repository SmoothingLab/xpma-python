"""ReverseFilter: generic reversal of any filter via the secant solver.

Round-trips forward filters that have no algebraic inverse (XEPMA,
DampedXEPMA, LeadEMA), cross-checks against the exact algebraic ReverseEMA,
and pins the shared interface conventions (bad-input guard, stateless
calc_next).
"""

import math
import sys

sys.path.insert(0, '..')

from xpma import (
    EMA, ReverseEMA, ReverseFilter, XEPMA, DampedXEPMA, LeadEMA, QuadraticXEPMA,
)


DATA = [100.0 + i * 0.1 + 5.0 * math.sin(i * 0.15) + 2.0 * math.sin(i * 0.47) for i in range(300)]


def _roundtrip_max_error(make_forward, warm=6):
    forward = make_forward()
    reverse = ReverseFilter(make_forward())
    max_err = 0.0
    for i, x in enumerate(DATA):
        recovered = reverse.get_next(forward.get_next(x))
        if i > warm:
            max_err = max(max_err, abs(recovered - x))
    return max_err


def test_roundtrip_no_algebraic_inverse():
    """ReverseFilter recovers the input of filters with no algebraic inverse."""
    for name, make in [
        ("XEPMA", lambda: XEPMA(21.0, 2.0)),
        ("DampedXEPMA", lambda: DampedXEPMA(21.0, 2.0)),
        ("QuadraticXEPMA", lambda: QuadraticXEPMA(21.0, 2.0)),
        ("LeadEMA", lambda: LeadEMA(21.0, 1.0)),
    ]:
        err = _roundtrip_max_error(make)
        assert err < 1e-3, f"ReverseFilter({name}) round-trip error {err:.2e}"
    print("PASS: ReverseFilter round-trips XEPMA, DampedXEPMA, QuadraticXEPMA, LeadEMA")


def test_agrees_with_algebraic_reverse_ema():
    """On an EMA stream, ReverseFilter matches the exact algebraic ReverseEMA."""
    period = 15.0
    forward = EMA(period)
    exact = ReverseEMA(period)
    generic = ReverseFilter(EMA(period))

    max_gap = 0.0
    for i, x in enumerate(DATA):
        smoothed = forward.get_next(x)
        a = exact.get_next(smoothed)
        b = generic.get_next(smoothed)
        if i > 6:
            max_gap = max(max_gap, abs(a - b))

    assert max_gap < 1e-3, f"ReverseFilter vs ReverseEMA gap {max_gap:.2e}"
    print(f"PASS: ReverseFilter agrees with algebraic ReverseEMA (max gap {max_gap:.2e})")


def test_bad_input_guard():
    """None/nan return None and leave the reversal state untouched."""
    reference = ReverseFilter(XEPMA(21.0))
    guarded = ReverseFilter(XEPMA(21.0))
    forward = XEPMA(21.0)

    stream = [forward.get_next(x) for x in DATA[:40]]
    for s in stream[:20]:
        reference.get_next(s)
        guarded.get_next(s)

    assert guarded.get_next(float("nan")) is None
    assert guarded.get_next(None) is None

    for s in stream[20:]:
        a = reference.get_next(s)
        b = guarded.get_next(s)
        assert a == b, f"state disturbed by bad input: {a} != {b}"
    print("PASS: ReverseFilter bad-input guard leaves state untouched")


def test_calc_next_stateless():
    """calc_next probes the reversal without advancing it."""
    forward = XEPMA(21.0)
    reverse = ReverseFilter(XEPMA(21.0))
    stream = [forward.get_next(x) for x in DATA[:30]]
    for s in stream[:-1]:
        reverse.get_next(s)

    probe1 = reverse.calc_next(stream[-1])
    probe2 = reverse.calc_next(stream[-1])
    committed = reverse.get_next(stream[-1])

    assert probe1 == probe2, "calc_next not repeatable"
    assert abs(probe1 - committed) < 1e-9, f"probe {probe1} != committed {committed}"
    print("PASS: ReverseFilter calc_next is a stateless probe")


if __name__ == "__main__":
    tests = [
        test_roundtrip_no_algebraic_inverse,
        test_agrees_with_algebraic_reverse_ema,
        test_bad_input_guard,
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
