"""FastEMA: faster than EMA - same bandwidth, less lag (nominal period, r_crit^M).

Thin wrapper on XPMA at the maximal monotone lag reduction r_crit^M(s), at
the NOMINAL period: it shares EMA(p)'s bandwidth but carries less lag,
(1 - r_crit^M(s)) * (p-1)/2 < EMA(p)'s (p-1)/2, so it is genuinely faster than
EMA while staying overshoot-free. Its lag-matched sibling LeadEMA inflates the
period to restore EMA's lag.

Fractional smoothness is interpolated at the OUTPUT level between the two
bracketing integer-order cores (FractionalSmoothness), never through XPMA's
internal cascade blend: r_crit^M depends on s, so a cascade-level blend would
break the no-overshoot guarantee. At integer s the interpolator delegates to a
single core.
"""

from .xpma import XPMA
from .r_crit import r_crit_m
from .fractional_smoothness import FractionalSmoothness


def _fast_core(period: float, order: int) -> XPMA:
    """Integer-order FastEMA core: XPMA at the nominal period with r = r_crit^M."""
    return XPMA(period, order, r_crit_m(order))


class FastEMA:
    def __init__(self, period: float, smoothness: float = 1.0):
        self.smoothness = smoothness
        self.lag_reduction = r_crit_m(smoothness)
        self.ma_lag = (period - 1.0) / 2.0
        self.time_lag = (1.0 - self.lag_reduction) * self.ma_lag
        self._interp = FractionalSmoothness(smoothness, lambda order: _fast_core(period, order))

    def get_next(self, value):
        return self._interp.get_next(value)

    def calc_next(self, value):
        return self._interp.calc_next(value)
