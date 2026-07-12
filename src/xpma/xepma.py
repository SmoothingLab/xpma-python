"""eXponential End Point Moving Average: the zero time-lag endpoint filter.

XEPMA is the endpoint of the two-rate regression line: the r = 1 member of the
XPMA lag-reduction family, and the smoother the FastEMA / LeadEMA / ConvexFastEMA
family build on. It is a level-plus-trend nowcast,

    XEPMA^[s](p) = IFEMA(p, s) + L * Delta IFEMA(p, s+1),   L = (p - 1) / 2,

quadratic-exact at s = 1 (m1 = m2 = 0) and zero-lag at every real s (m1 = 0),
with m2 = ((1 - s)/s) L^2 at general s. It is the WMA/EPMA analogue for the
exponential family: WMA is to EPMA what FastEMA is to XEPMA.

Two named points sit off the family axis (they are not members of the
lag-reduction line): QuadraticXEPMA (quadratic_xepma.py), the curvature-corrected
quadratic-exact sibling, and DampedXEPMA (damped_xepma.py), the min-overshoot
sibling.
"""

import math

from .ifema import IFEMA


class XEPMA:
    """Zero time-lag endpoint: IFEMA(p, s) + L * Delta IFEMA(p, s+1), L = (p-1)/2.

    The r = 1 endpoint of the XPMA family. Quadratic-exact at s = 1 only; zero-lag
    (m1 = 0) at every s. XPMA (r = 1), FastEMA / LeadEMA / ConvexFastEMA,
    QuadraticXEPMA and DampedXEPMA all build on this endpoint.
    """

    def __init__(self, period: float, smoothness: float = 1.0):
        self.smoothness = smoothness
        self.ma_lag = (period - 1.0) / 2.0

        if self.smoothness != 0.0:
            self._ma1 = IFEMA(period, self.smoothness)
            self._ma2 = IFEMA(period, self.smoothness + 1.0)
        else:
            self._ma1 = None
            self._ma2 = None
        self._prev_ma2 = None

    def get_next(self, value: float) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        if self.smoothness == 0.0:
            return value

        ma1 = self._ma1.get_next(value)
        ma2 = self._ma2.get_next(value)

        if self._prev_ma2 is None:
            result = value
        else:
            result = ma1 + self.ma_lag * (ma2 - self._prev_ma2)
        self._prev_ma2 = ma2
        return result

    def calc_next(self, value: float) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        if self.smoothness == 0.0 or self._prev_ma2 is None:
            return value

        ma1 = self._ma1.calc_next(value)
        ma2 = self._ma2.calc_next(value)
        return ma1 + self.ma_lag * (ma2 - self._prev_ma2)
