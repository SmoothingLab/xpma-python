"""Tests for the critical lag-reduction constants (xpma.r_crit).

Verifies r_crit^M, r_crit^O and r_crit^C against their closed forms and reference
constant tables, including the s = 1 degenerate Cardano case (the cubic
discriminant is exactly zero there and rounding can push the inner square root
negative).
"""

import math
import sys

sys.path.insert(0, '..')

from xpma import r_crit as _rc_module  # noqa: F401  (ensure submodule import path works)
from xpma import r_crit_m, r_crit_o, r_crit_c
from xpma.r_crit import tau0_m, tau_p_o, tau1_c


# Reference tables (continuous-limit constants).
_R_CRIT_M = {1: 0.56021113, 2: 0.52291260, 3: 0.45519806, 4: 0.39646970}
_R_CRIT_O = {1: 0.6795705, 2: 0.5788861, 3: 0.4853691, 4: 0.4145623}
_R_CRIT_C = {1: 0.46181601, 2: 0.46869231, 3: 0.42427891, 4: 0.37746845}
_TAU1_C = {1: 2.0, 2: 2.04133361, 3: 2.05667739, 4: 2.06162959}


def test_r_crit_m_closed_form():
    """r_crit^M(s) matches the closed form and the table; stall tau_0 = (2s+1)/(s+1)."""
    for s, ref in _R_CRIT_M.items():
        got = r_crit_m(float(s))
        assert abs(got - ref) < 1e-7, f"r_crit^M({s}) = {got:.8f} != {ref}"
        assert abs(tau0_m(float(s)) - (2.0 * s + 1.0) / (s + 1.0)) < 1e-15
    print("PASS: r_crit^M matches closed form and table")


def test_r_crit_o_table_and_e_over_4():
    """r_crit^O(s) matches the table, and equals e/4 exactly at s = 1."""
    for s, ref in _R_CRIT_O.items():
        got = r_crit_o(float(s))
        assert abs(got - ref) < 1e-6, f"r_crit^O({s}) = {got:.8f} != {ref}"
    assert r_crit_o(1.0) == math.e / 4.0, f"r_crit^O(1) != e/4: {r_crit_o(1.0)!r}"
    # tau_p anchors.
    assert abs(tau_p_o(1.0) - 1.0) < 1e-9
    assert abs(tau_p_o(2.0) - (1.0 + math.sqrt(17.0)) / 4.0) < 1e-7
    print("PASS: r_crit^O matches table and equals e/4 at s = 1")


def test_r_crit_o_rejects_fractional():
    """r_crit^O is elementary only at integer s; a fractional call raises clearly."""
    for s in (1.5, 2.5, 3.54):
        try:
            r_crit_o(s)
            assert False, f"r_crit^O({s}) should have raised"
        except ValueError:
            pass
    print("PASS: r_crit^O raises for fractional s")


def test_r_crit_c_cardano_and_anchors():
    """r_crit^C(s) matches the table; s = 1 anchors e^2/16 and tau_1 = 2 exactly (T5)."""
    for s, ref in _R_CRIT_C.items():
        got = r_crit_c(float(s))
        assert abs(got - ref) < 1e-7, f"r_crit^C({s}) = {got:.8f} != {ref}"
    for s, ref in _TAU1_C.items():
        assert abs(tau1_c(float(s)) - ref) < 1e-7, f"tau_1({s}) = {tau1_c(float(s)):.8f} != {ref}"
    # s = 1 degenerate Cardano case: exact anchors (tau_1 = 2 exactly, r = e^2/16).
    assert tau1_c(1.0) == 2.0, f"tau_1(1) != 2 exactly: {tau1_c(1.0)!r}"
    assert abs(r_crit_c(1.0) - math.e ** 2 / 16.0) < 1e-15, f"r_crit^C(1) != e^2/16: {r_crit_c(1.0)!r}"
    print("PASS: r_crit^C matches table; s = 1 anchors e^2/16 and tau_1 = 2 exact")


def test_r_crit_c_near_s1_stable():
    """r_crit^C is finite and near e^2/16 just above s = 1 (no Cardano rounding blow-up)."""
    base = math.e ** 2 / 16.0
    for s in (1.0, 1.0 + 1e-9, 1.0 + 1e-6, 1.001, 1.01):
        val = r_crit_c(s)
        assert math.isfinite(val), f"r_crit^C({s}) not finite"
        assert abs(val - base) < 1e-2, f"r_crit^C({s}) = {val} jumped from e^2/16 = {base}"
    print("PASS: r_crit^C stable across the s = 1 degeneracy")


def test_ladder_ordering():
    """The ladder r_crit^C < r_crit^M < r_crit^O holds at every integer s = 1..4."""
    for s in (1, 2, 3, 4):
        c, m, o = r_crit_c(float(s)), r_crit_m(float(s)), r_crit_o(float(s))
        assert c < m < o, f"ladder broken at s={s}: {c:.6f}, {m:.6f}, {o:.6f}"
    print("PASS: r_crit^C < r_crit^M < r_crit^O at s = 1..4")


def test_r_crit_c_fractional_real_analytic():
    """r_crit^C is real-analytic for real s > 1 (peak near s ~ 1.484, value ~ 0.479)."""
    peak_s, peak_v = max(((s / 1000.0, r_crit_c(s / 1000.0))
                          for s in range(1100, 2000)), key=lambda t: t[1])
    assert abs(peak_s - 1.4841) < 5e-3, f"r_crit^C peak at s = {peak_s:.4f}, expected ~1.4841"
    assert abs(peak_v - 0.47902) < 5e-4, f"r_crit^C peak value {peak_v:.5f}, expected ~0.47902"
    print(f"PASS: r_crit^C peak at s = {peak_s:.4f}, value {peak_v:.5f}")


if __name__ == "__main__":
    tests = [
        test_r_crit_m_closed_form,
        test_r_crit_o_table_and_e_over_4,
        test_r_crit_o_rejects_fractional,
        test_r_crit_c_cardano_and_anchors,
        test_r_crit_c_near_s1_stable,
        test_ladder_ordering,
        test_r_crit_c_fractional_real_analytic,
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
