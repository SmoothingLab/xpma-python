"""ApexFastEMA: the r_crit^O member of the XPMA family (nominal period).

Thin wrapper on XPMA at the no-overshoot boundary r_crit^O(s): the LARGEST
lag reduction whose unit-step response never crosses the target. At exactly
r_crit^O the response's apex touches the target in the continuous limit (the step
error is tangent to zero): it rises fast, grazes the target at its apex, recedes
into a sub-target dip, then settles back up. One-sided convergence, never beyond
the target - the apex of its step response sits on the target, which names it.

Unlike FastEMA (monotone step response) and ConvexFastEMA (convex step error), the
Apex members are NON-MONOTONE: the sub-target dip is intrinsic for r in
(r_crit^M, r_crit^O]. The retained guarantee is no overshoot only. ApexFastEMA
trades FastEMA's monotonicity for the maximal lag reduction at the nominal period,
so it sits outside FastEMA on the ladder r_crit^C < r_crit^M < r_crit^O and is the
fastest of the three at the nominal period (smallest time lag).

Discrete safety of r_crit^O is numerical: the discrete no-overshoot boundary sits
at or above the continuous constant at every tested (s, p) with s = 1..4 and p in
{5, 10, 20, 50, 200} (worst step-response overshoot 0.0 at the continuous
constant), consistent with the O(1/p) margin from above. So the fixed choice of the
continuous constant is conservative at finite period.

Fractional smoothness is interpolated at the OUTPUT level between the two
bracketing integer-order cores (r_crit^O is elementary only at integer s). No
overshoot is preserved by the convex output blend: each integer component's step
error is >= 0 (it never crosses the target), so the blend's is too, exactly as the
Convex wrappers preserve their boundary; the internal cascade blend would not. At
integer s the interpolator delegates to a single core.

Its lag-matched sibling ApexLeadEMA inflates the period to restore EMA's lag; see
apex_lead_ema.py.
"""

from .xpma import XPMA
from .r_crit import r_crit_o, r_crit_o_effective
from .fractional_smoothness import FractionalSmoothness


def _apex_core(period: float, order: int) -> XPMA:
    """Integer-order ApexFastEMA core: XPMA at the nominal period with r = r_crit^O."""
    return XPMA(period, order, r_crit_o(order))


class ApexFastEMA:
    def __init__(self, period: float, smoothness: float = 1.0):
        self.smoothness = smoothness
        self.lag_reduction = r_crit_o_effective(smoothness)
        self.ma_lag = (period - 1.0) / 2.0
        self.time_lag = (1.0 - self.lag_reduction) * self.ma_lag
        self._interp = FractionalSmoothness(smoothness, lambda order: _apex_core(period, order))

    def get_next(self, value):
        return self._interp.get_next(value)

    def calc_next(self, value):
        return self._interp.calc_next(value)
