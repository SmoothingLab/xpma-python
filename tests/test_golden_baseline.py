"""Golden-baseline regression tests.

Every filter must reproduce the pinned numeric baseline in
tests/golden_baseline.json over a shared input series. The baseline is keyed by
class name.

  - Direct implementations match the baseline BIT-IDENTICALLY:
        XEPMA, QuadraticXEPMA, DampedXEPMA, XPMA(p, s, r) with explicit r, IFEMA.
  - The thin wrappers built on XPMA plus a criticality constant match to
    MACHINE PRECISION (the outputs are algebraically identical; only the final
    blend arithmetic is reassociated):
        FastEMA, LeadEMA, LeadXEPMA.

Also covers the exact-kernel EIFEMA: integer-order equivalence to MultiEMA, and
closeness of the moment-matched IFEMA blend at fractional order.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xpma
from xpma import MultiEMA

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN = json.load(open(os.path.join(_HERE, "golden_baseline.json")))
_SERIES = _GOLDEN["input"]

# Filters that must match the baseline bit-identically vs to machine precision only.
_BIT_IDENTICAL = {"XEPMA", "QuadraticXEPMA", "DampedXEPMA", "XPMA", "IFEMA"}
_MACHINE_PRECISION = {"FastEMA", "LeadEMA", "LeadXEPMA"}
_WRAPPER_TOL = 1e-9


def _construct(case):
    lab, p, s, ex = case["label"], case["period"], case["smoothness"], case["extra"]
    if lab == "XEPMA":
        return xpma.XEPMA(p, s)
    if lab == "QuadraticXEPMA":
        return xpma.QuadraticXEPMA(p, s)
    if lab == "DampedXEPMA":
        return xpma.DampedXEPMA(p, s)
    if lab == "XPMA":
        return xpma.XPMA(p, s, ex["r"])
    if lab == "IFEMA":
        return xpma.IFEMA(p, s)
    if lab == "FastEMA":
        return xpma.FastEMA(p, s)
    if lab == "LeadEMA":
        return xpma.LeadEMA(p, s)
    if lab == "LeadXEPMA":
        return xpma.LeadXEPMA(p, s)
    raise KeyError(lab)


def _run(filt):
    return [filt.get_next(v) for v in _SERIES]


def _check_group(labels, bit_identical):
    worst = 0.0
    worst_case = None
    for case in _GOLDEN["cases"]:
        if case["label"] not in labels:
            continue
        got = _run(_construct(case))
        assert len(got) == len(case["values"])
        for a, b in zip(case["values"], got):
            if a is None or b is None:
                assert a is b, f"{case['label']}: None mismatch (golden {a}, got {b})"
                continue
            diff = abs(a - b)
            if diff > worst:
                worst, worst_case = diff, (case["label"], case["period"], case["smoothness"])
            if bit_identical:
                assert a == b, (
                    f"{case['label']}(p={case['period']},s={case['smoothness']}) not "
                    f"bit-identical: golden {a!r} vs {b!r}"
                )
            else:
                assert diff < _WRAPPER_TOL, (
                    f"{case['label']}(p={case['period']},s={case['smoothness']}) diff "
                    f"{diff:.2e} exceeds {_WRAPPER_TOL:.0e}"
                )
    return worst, worst_case


def test_direct_filters_bit_identical():
    """The direct filters reproduce the golden baseline bit-for-bit."""
    worst, worst_case = _check_group(_BIT_IDENTICAL, bit_identical=True)
    assert worst == 0.0, f"expected bit-identical, worst {worst:.2e} at {worst_case}"
    print(f"PASS: direct filters bit-identical to golden (worst diff {worst:.1e})")


def test_wrapper_filters_machine_precision():
    """FastEMA/LeadEMA family reproduce the golden baseline to machine precision."""
    worst, worst_case = _check_group(_MACHINE_PRECISION, bit_identical=False)
    assert worst < _WRAPPER_TOL, f"worst {worst:.2e} at {worst_case}"
    print(f"PASS: wrapper filters within {_WRAPPER_TOL:.0e} of golden (worst diff {worst:.1e} at {worst_case})")


# EIFEMA is a truncated FIR (renormalised through warm-up), so it agrees with the
# IIR cascades only in steady state; settle past its longest weight window first.
_SETTLE = [100.0] * 4000
_VARY = [100.0 + 8.0 * math.sin(i * 0.11) + 3.0 * math.sin(i * 0.37) for i in range(300)]


def test_eifema_integer_equals_multiema():
    """The exact-kernel EIFEMA reduces to MultiEMA at integer order (steady state)."""
    for s in (1, 2, 3, 4):
        for p in (21.0, 50.0):
            ei = xpma.EIFEMA(p, float(s))
            me = MultiEMA(p, s)
            for v in _SETTLE:
                ei.get_next(v)
                me.get_next(v)
            worst = 0.0
            for v in _VARY:
                worst = max(worst, abs(ei.get_next(v) - me.get_next(v)))
            assert worst < 1e-9, f"EIFEMA(p={p},s={s}) != MultiEMA: {worst:.2e}"
    print("PASS: EIFEMA integer order matches MultiEMA in steady state")


def test_eifema_moment_matched_close_to_ifema():
    """IFEMA (moment-matched) tracks the exact EIFEMA closely at fractional order."""
    for s in (2.5, 3.5, 3.54):
        ei = xpma.EIFEMA(21.0, s)
        im = xpma.IFEMA(21.0, s)
        for v in _SETTLE:
            ei.get_next(v)
            im.get_next(v)
        worst = 0.0
        for v in _VARY:
            a = ei.get_next(v)
            b = im.get_next(v)
            worst = max(worst, abs(a - b) / max(1.0, abs(a)))
        # The moment-matched blend reproduces EIFEMA to about 0.02-0.03%.
        assert worst < 1e-3, f"IFEMA(s={s}) not close to EIFEMA: rel {worst:.2e}"
    print("PASS: IFEMA moment-matched close to exact EIFEMA at fractional order")


if __name__ == "__main__":
    tests = [
        test_direct_filters_bit_identical,
        test_wrapper_filters_machine_precision,
        test_eifema_integer_equals_multiema,
        test_eifema_moment_matched_close_to_ifema,
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
