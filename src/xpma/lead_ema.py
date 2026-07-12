"""LeadEMA: matches EMA's lag but leads price via reduced undershoot (inflated, r_crit^M).

Thin wrapper on XPMA at r_crit^M(s), but with the averaging period inflated
by 1/(1 - r_crit^M(s)) so the output lag is pushed back up to EMA(period)'s lag
(period - 1)/2. Not faster than EMA, but leads price toward its level with much
less undershoot (and no overshoot). Its un-inflated sibling FastEMA spends the lag
reduction on speed instead.

Fractional smoothness is interpolated at the OUTPUT level between the two
bracketing integer-order cores (each inflated by its own order's r_crit^M),
preserving the no-overshoot guarantee. At integer s the interpolator delegates to
a single core.
"""

from .xpma import XPMA
from .r_crit import r_crit_m
from .fractional_smoothness import FractionalSmoothness


def _lead_core(period: float, order: int) -> XPMA:
    """Integer-order LeadEMA core: XPMA at the inflated period with r = r_crit^M."""
    r = r_crit_m(order)
    ma_period = 1.0 + (period - 1.0) / (1.0 - r)
    return XPMA(ma_period, order, r_crit_m(order))


class LeadEMA:
    def __init__(self, period: float, smoothness: float = 1.0):
        self.smoothness = smoothness
        self.lag_reduction = r_crit_m(smoothness)
        self.time_lag = (period - 1.0) / 2.0  # matches EMA(period)
        self._interp = FractionalSmoothness(smoothness, lambda order: _lead_core(period, order))

    def get_next(self, value):
        return self._interp.get_next(value)

    def calc_next(self, value):
        return self._interp.calc_next(value)
