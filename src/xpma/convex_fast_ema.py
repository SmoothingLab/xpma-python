"""ConvexFastEMA: the r_crit^C member of the XPMA family (nominal period).

Thin wrapper on XPMA at the convexity boundary r_crit^C(s): the largest
lag reduction at which the step error stays convex (decays at a monotonically
slowing rate, like the EMA's pure exponential). Equivalently the discrete kernel
is non-increasing at s = 1 and unimodal at s >= 2. It sits just inside FastEMA on
the ladder r_crit^C < r_crit^M < r_crit^O, so it is slightly slower but has no
step-error stall (no rate re-acceleration).

Fractional smoothness is interpolated at the OUTPUT level between the two
bracketing integer-order cores. This is load-bearing for r_crit^C specifically:
XPMA's internal moment-matched cascade blend is proven-by-measurement to breach
the convexity boundary in the s in (2, 3) band, whereas the output-level
realisation preserved it in every tested case. At integer s the interpolator
delegates to a single core.

Its lag-matched sibling ConvexLeadEMA inflates the period to restore EMA's lag
(exactly as LeadEMA does for FastEMA); see convex_lead_ema.py.
"""

from .xpma import XPMA
from .r_crit import r_crit_c
from .fractional_smoothness import FractionalSmoothness


def _convex_core(period: float, order: int) -> XPMA:
    """Integer-order ConvexFastEMA core: XPMA at the nominal period with r = r_crit^C."""
    return XPMA(period, order, r_crit_c(order))


class ConvexFastEMA:
    def __init__(self, period: float, smoothness: float = 1.0):
        self.smoothness = smoothness
        self.lag_reduction = r_crit_c(smoothness)
        self.ma_lag = (period - 1.0) / 2.0
        self.time_lag = (1.0 - self.lag_reduction) * self.ma_lag
        self._interp = FractionalSmoothness(smoothness, lambda order: _convex_core(period, order))

    def get_next(self, value):
        return self._interp.get_next(value)

    def calc_next(self, value):
        return self._interp.calc_next(value)
