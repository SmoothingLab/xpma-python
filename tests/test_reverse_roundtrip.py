"""Round-trip tests at fractional smoothness.

The forward XPMA cascade uses the second-moment-matched blend (IFEMA) at
fractional s. The reverse is realised by the forward-filter-agnostic SecantSolver,
so it is consistent with whatever forward filter it wraps. These tests pin that
consistency: forward then reverse recovers the input to solver tolerance at
fractional s, for both the bare IFEMA cascade and the XPMA family.
"""

import math
import sys

sys.path.insert(0, '..')

from xpma import IFEMA, XPMA, SecantSolver


DATA = [100.0 + i * 0.1 + 5.0 * math.sin(i * 0.15) + 2.0 * math.sin(i * 0.47) for i in range(300)]


def _roundtrip_max_error(make_forward, warm=6):
    """Forward through make_forward(), reverse via SecantSolver(make_forward()), and
    return the max recovery error after a short settling window."""
    forward = make_forward()
    reverse = SecantSolver(make_forward())
    smoothed = [forward.get_next(v) for v in DATA]
    max_err = 0.0
    prev = None
    for i, s in enumerate(smoothed):
        est = s if prev is None else prev
        recovered = reverse.solve(s, est)
        prev = recovered
        if i > warm:
            max_err = max(max_err, abs(recovered - DATA[i]))
    return max_err


def test_ifema_fractional_roundtrip():
    """Forward IFEMA (moment-matched) then reverse recovers the input at fractional s."""
    for s in (1.5, 2.5, 3.54):
        err = _roundtrip_max_error(lambda s=s: IFEMA(21.0, s))
        assert err < 1e-3, f"IFEMA(s={s}) round-trip error {err:.2e}"
    print("PASS: IFEMA fractional-s round-trip recovers input")


def test_xpma_fractional_roundtrip():
    """Forward XPMA (fractional s, explicit r) then reverse recovers the input."""
    for s in (2.5, 3.54):
        for r in (0.0, 0.3, 1.0):
            err = _roundtrip_max_error(lambda s=s, r=r: XPMA(21.0, s, r))
            assert err < 1e-3, f"XPMA(s={s},r={r}) round-trip error {err:.2e}"
    print("PASS: XPMA fractional-s round-trip recovers input")


if __name__ == "__main__":
    tests = [
        test_ifema_fractional_roundtrip,
        test_xpma_fractional_roundtrip,
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
