"""ConvexLeadEMA: matches EMA's lag but tracks closer via the convexity boundary (inflated, r_crit^C).

Thin wrapper on XPMA at the convexity boundary r_crit^C(s), but with the
averaging period inflated by 1/(1 - r_crit^C(s)) so the output lag is pushed back up
to EMA(period)'s lag (period - 1)/2. Not faster than EMA, but it spends the
convexity-boundary lag reduction on tracking closer at matched lag, with the step
error convex throughout (no stall, no rate re-acceleration). It is to ConvexFastEMA
what LeadEMA is to FastEMA: the lag-matched sibling that inflates the period rather
than spending the reduction on speed.

Because r_crit^C < r_crit^M, the inflation 1/(1 - r_crit^C) is smaller than LeadEMA's
1/(1 - r_crit^M), so ConvexLeadEMA works at a shorter period than LeadEMA at the same
nominal (period, smoothness) while carrying the identical lag.

Fractional smoothness is interpolated at the OUTPUT level between the two bracketing
integer-order cores (each inflated by its own order's r_crit^C), preserving the
convexity guarantee. This is load-bearing for r_crit^C specifically: XPMA's internal
moment-matched cascade blend is proven-by-measurement to breach the convexity boundary
in the s in (2, 3) band, whereas the output-level realisation preserved it in every
tested case. At integer s the interpolator delegates to a single core.
"""

from .xpma import XPMA
from .r_crit import r_crit_c
from .fractional_smoothness import FractionalSmoothness


def _convex_lead_core(period: float, order: int) -> XPMA:
    """Integer-order ConvexLeadEMA core: XPMA at the inflated period with r = r_crit^C."""
    r = r_crit_c(order)
    ma_period = 1.0 + (period - 1.0) / (1.0 - r)
    return XPMA(ma_period, order, r_crit_c(order))


class ConvexLeadEMA:
    def __init__(self, period: float, smoothness: float = 1.0):
        self.smoothness = smoothness
        self.lag_reduction = r_crit_c(smoothness)
        self.time_lag = (period - 1.0) / 2.0  # matches EMA(period)
        self._interp = FractionalSmoothness(smoothness, lambda order: _convex_lead_core(period, order))

    def get_next(self, value):
        return self._interp.get_next(value)

    def calc_next(self, value):
        return self._interp.calc_next(value)
