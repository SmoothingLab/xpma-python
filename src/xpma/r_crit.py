"""Critical lag-reduction constants for the two-rate XPMA family.

Three step-response boundaries form a ladder on the base two-rate line (the r
axis from the EMA cascade at r = 0 to the XEPMA endpoint at r = 1), each a
tangency one derivative deeper:

    r_crit^C(s) < r_crit^M(s) < r_crit^O(s)

  - r_crit^C (the convexity boundary): the largest r at which the step error
    stays convex (decays at a monotonically slowing rate, like the EMA's pure
    exponential); equivalently the kernel is non-increasing at s = 1 and
    unimodal at s >= 2. ConvexFastEma sits here (nominal period).
  - r_crit^M (the maximal monotone boundary): the largest r whose unit-step
    response stays monotone (no sub-target dip, no overshoot). FastEma and
    LeadEma sit here.
  - r_crit^O (the no-overshoot boundary): the largest r with no overshoot,
    allowing a sub-target dip. ApexFastEma and ApexLeadEma sit here.

These are CONTINUOUS-LIMIT values. Discrete safety (the discrete boundary
staying at or above the continuous constant at finite period p) is proven for
r_crit^M and numerical for r_crit^C and r_crit^O (the discrete boundary sits at
or above the constant at every tested (s, p), with an O(1/p) margin). r_crit^O is
elementary at integer s (a single positive root of the degree-s polynomial P_s);
a general fractional-s form would need incomplete-gamma functions and is not
provided, so the Apex members realise a fractional order by output-level
interpolation of the bracketing integer-order instances.
"""

import math


# ----------------------------------------------------------------------------------
# r_crit^M: maximal monotone lag reduction (closed form, all real s > 0).
# ----------------------------------------------------------------------------------
def r_crit_m(smoothness: float) -> float:
    """r_crit^M(s) = (s^(s+1) / (s+1)^(s+2)) * e^((2s+1)/(s+1))."""
    s = float(smoothness)
    return (s ** (s + 1.0) / (s + 1.0) ** (s + 2.0)) * math.exp((2.0 * s + 1.0) / (s + 1.0))


def tau0_m(smoothness: float) -> float:
    """Stall time tau_0(s) = (2s + 1) / (s + 1) (the r_crit^M step-error stall)."""
    s = float(smoothness)
    return (2.0 * s + 1.0) / (s + 1.0)


# ----------------------------------------------------------------------------------
# r_crit^O: no-overshoot boundary (integer s, via the degree-s polynomial P_s).
# ----------------------------------------------------------------------------------
def _fact(n: int) -> float:
    return float(math.factorial(int(n)))


def _q_o(tau: float, s: int) -> float:
    """Q(tau) = sum_{j=0}^{s-1} (s tau)^j / j! (s-stage cascade step-error factor)."""
    return math.fsum((s * tau) ** j / _fact(j) for j in range(int(s)))


def _p_s(tau: float, s: int) -> float:
    """P_s(tau) = Q(tau)(s - (s+1)tau) + s^s tau^s / (s-1)!; its positive root is tau_p."""
    return _q_o(tau, s) * (s - (s + 1) * tau) + s ** s * tau ** s / _fact(s - 1)


def _require_integer(smoothness: float, name: str) -> int:
    s = float(smoothness)
    if s % 1.0 != 0.0:
        raise ValueError(
            "%s is elementary only at integer s (a positive root of P_s); at fractional "
            "s realise the r(s) filter by output-level interpolation of the two bracketing "
            "integer-order instances." % name
        )
    return int(round(s))


def tau_p_o(smoothness: float, hi: float = 40.0, iters: int = 200) -> float:
    """Relevant positive root tau_p of P_s (integer s). P_s(0^+) > 0, leading coeff < 0,
    so a single sign change on (0, inf) is isolated by bisection."""
    s = _require_integer(smoothness, "r_crit^O")
    lo = 1e-9
    if _p_s(lo, s) <= 0.0:
        raise ValueError("unexpected sign of P_s at tau -> 0")
    hi_b = hi
    while _p_s(hi_b, s) > 0.0:
        hi_b *= 2.0
    a, b = lo, hi_b
    for _ in range(iters):
        m = 0.5 * (a + b)
        if _p_s(m, s) > 0.0:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def r_crit_o(smoothness: float) -> float:
    """r_crit^O(s) = s! Q(tau_p) e^{tau_p} / ((s+1)^{s+1} tau_p^s) (integer s).

    At s = 1 this is e/4 exactly."""
    s = _require_integer(smoothness, "r_crit^O")
    tau_p = tau_p_o(s)
    return _fact(s) * _q_o(tau_p, s) * math.exp(tau_p) / ((s + 1) ** (s + 1) * tau_p ** s)


def r_crit_o_effective(smoothness: float) -> float:
    """Effective (advertised) lag reduction of the r_crit^O members at any real s.

    r_crit^O itself is elementary only at integer s. The Apex filters realise a
    fractional order by an OUTPUT-LEVEL convex blend of the two bracketing
    integer-order instances, so their effective lag reduction is the same convex
    blend of the bracketing
    integer r_crit^O values (the first moment is linear under an output-level
    blend). Weights match FractionalSmoothness. At integer s this is r_crit^O(s)."""
    s = float(smoothness)
    if s % 1.0 == 0.0:
        return r_crit_o(int(round(s)))
    frac = s % 1.0
    floor_order = math.floor(s)
    ceil_order = math.ceil(s)
    lo_weight = 0.0 if floor_order == 0 else (1.0 - frac) * floor_order / s
    hi_weight = 1.0 - lo_weight
    result = 0.0
    if lo_weight > 0.0:
        result += lo_weight * r_crit_o(floor_order)
    if hi_weight > 0.0:
        result += hi_weight * r_crit_o(ceil_order)
    return result


# ----------------------------------------------------------------------------------
# r_crit^C: convexity (monotone-rate) boundary (Cardano, all real s > 1; s = 1 exact).
# ----------------------------------------------------------------------------------
def _cbrt(x: float) -> float:
    """Real cube root (Python's ** on a negative base returns a complex number)."""
    if x >= 0.0:
        return x ** (1.0 / 3.0)
    return -((-x) ** (1.0 / 3.0))


def _q_c(tau: float, s: float) -> float:
    """q(tau) = s(s-1) - 2s(s+1)tau + (s+1)^2 tau^2 (post-mode binding denominator)."""
    return s * (s - 1.0) - 2.0 * s * (s + 1.0) * tau + (s + 1.0) ** 2 * tau * tau


def tau1_c(smoothness: float) -> float:
    """tau_1(s): the unique real root beyond u_+ of the cubic

        P(u) = s(s+1)^2 u^3 - (s+1)(4s^2+s-1) u^2 + (5s^3+s^2-4s-2) u - s(s-1)(2s+1),

    via Cardano. disc(P) < 0 for real s > 1 (one real root); at s = 1 the cubic
    degenerates (disc = 0) to 4 u^2 (u - 2), so tau_1 = 2 exactly."""
    s = float(smoothness)
    # s = 1 degenerate case: tau_1 = 2 exactly (avoids a disc = 0 rounding artefact).
    if abs(s - 1.0) < 1e-12:
        return 2.0
    a3 = s * (s + 1.0) ** 2
    a2 = -(s + 1.0) * (4.0 * s * s + s - 1.0)
    a1 = 5.0 * s ** 3 + s * s - 4.0 * s - 2.0
    a0 = -s * (s - 1.0) * (2.0 * s + 1.0)
    shift = a2 / (3.0 * a3)
    p3 = (3.0 * a3 * a1 - a2 * a2) / (3.0 * a3 * a3)
    q3 = (2.0 * a2 ** 3 - 9.0 * a3 * a2 * a1 + 27.0 * a3 * a3 * a0) / (27.0 * a3 ** 3)
    delta = (q3 / 2.0) ** 2 + (p3 / 3.0) ** 3
    # disc(P) < 0 <=> delta > 0 (one real root). Clamp a tiny negative delta from
    # rounding near s = 1 up to zero.
    if delta < 0.0:
        if delta > -1e-12:
            delta = 0.0
        else:
            raise ValueError("r_crit^C: unexpected three-real-root regime at s = %r" % s)
    sq = math.sqrt(delta)
    v = _cbrt(-q3 / 2.0 + sq) + _cbrt(-q3 / 2.0 - sq)
    return v - shift


def r_crit_c(smoothness: float) -> float:
    """r_crit^C(s) = (s/(s+1))^{s+1} e^{tau_1} (s tau_1 - s + 1) / q(tau_1).

    At s = 1 this is e^2 / 16 exactly."""
    s = float(smoothness)
    tau1 = tau1_c(s)
    return (s / (s + 1.0)) ** (s + 1.0) * math.exp(tau1) * (s * tau1 - s + 1.0) / _q_c(tau1, s)
