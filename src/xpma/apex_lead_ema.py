"""ApexLeadEMA: matches EMA's lag but undershoots least via the no-overshoot boundary (inflated, r_crit^O).

Thin wrapper on XPMA at the no-overshoot boundary r_crit^O(s), with the
averaging period inflated by 1/(1 - r_crit^O(s)) so the output lag is pushed back
up to EMA(period)'s lag (period - 1)/2. Not faster than EMA: it spends the maximal
no-overshoot lag reduction on tracking closer at matched lag, still never crossing
the target. It is to ApexFastEMA what LeadEMA is to FastEMA: the lag-matched
sibling that inflates the period rather than spending the reduction on speed.

Because r_crit^O > r_crit^M, the inflation 1/(1 - r_crit^O) is larger than LeadEMA's
1/(1 - r_crit^M), so ApexLeadEMA works at a longer period than LeadEMA at the same
nominal (period, smoothness) while carrying the identical lag, and undershoots
least of the three lag-matched members.

Like ApexFastEMA the step response is NON-MONOTONE (a sub-target dip is intrinsic
for r in (r_crit^M, r_crit^O]); the retained guarantee is no overshoot only.
ApexLeadEMA trades LeadEMA's monotonicity for minimal undershoot at matched lag.
Discrete safety of r_crit^O is numerical (see apex_fast_ema.py).

Fractional smoothness is interpolated at the OUTPUT level between the two
bracketing integer-order cores (each inflated by its own order's r_crit^O). No
overshoot is preserved by the convex output blend: each integer component's step
error is >= 0 (it never crosses the target), so the blend's is too, exactly as the
Convex wrappers preserve their boundary. At integer s the interpolator delegates to
a single core.
"""

from .xpma import XPMA
from .r_crit import r_crit_o, r_crit_o_effective
from .fractional_smoothness import FractionalSmoothness


def _apex_lead_core(period: float, order: int) -> XPMA:
    """Integer-order ApexLeadEMA core: XPMA at the inflated period with r = r_crit^O."""
    r = r_crit_o(order)
    ma_period = 1.0 + (period - 1.0) / (1.0 - r)
    return XPMA(ma_period, order, r_crit_o(order))


class ApexLeadEMA:
    def __init__(self, period: float, smoothness: float = 1.0):
        self.smoothness = smoothness
        self.lag_reduction = r_crit_o_effective(smoothness)
        self.time_lag = (period - 1.0) / 2.0  # matches EMA(period)
        self._interp = FractionalSmoothness(smoothness, lambda order: _apex_lead_core(period, order))

    def get_next(self, value):
        return self._interp.get_next(value)

    def calc_next(self, value):
        return self._interp.calc_next(value)
