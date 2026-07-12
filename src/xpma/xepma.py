"""eXponential End Point Moving Average: the zero time-lag endpoint filter.

XEPMA is the endpoint of the two-rate regression line: the r = 1 member of the
XPMA lag-reduction family, and the smoother the FastEMA / LeadEMA / ConvexFastEMA
family build on. It is a level-plus-trend nowcast,

    XEPMA^[s](p) = IFEMA(p, s) + L * Delta IFEMA(p, s+1),   L = (p - 1) / 2,

quadratic-exact at s = 1 (m1 = m2 = 0) and zero-lag at every real s (m1 = 0),
with m2 = ((1 - s)/s) L^2 at general s. It is the WMA/EPMA analogue for the
exponential family: WMA is to EPMA what FastEMA is to XEPMA.

Two named points sit off the family axis (they are not members of the
lag-reduction line):

  - QuadraticXEPMA adds the lag- and gain-preserving curvature correction
    c(s) Delta^2 IFEMA(p, s+2), c = ((s-1)/(2s)) L^2, cancelling m2 at every
    order (quadratic-exact for all real s >= 1; identical to XEPMA at s = 1,
    where c = 0). It is the mathematically complete solution; its noise gain
    rises with s, so it is primarily a mathematical exercise.
  - DampedXEPMA (in damped_xepma.py) adds the exact discrete min-overshoot
    coefficient instead: the useful sibling, below the endpoint's overshoot at
    every (s, p) including s = 1.

Neither takes a tuning parameter.
"""

import math

from .ifema import IFEMA


class XEPMA:
    """Zero time-lag endpoint: IFEMA(p, s) + L * Delta IFEMA(p, s+1), L = (p-1)/2.

    The r = 1 endpoint of the XPMA family. Quadratic-exact at s = 1 only; zero-lag
    (m1 = 0) at every s. XPMA (r = 1), ReverseXepma, FastEMA / LeadEMA /
    ConvexFastEMA and DampedXEPMA all build on this endpoint.
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


def _quadratic_correction_factor(period: float, smoothness: float) -> float:
    """Coefficient c(s) = ((s-1)/(2s)) L^2 that cancels m2 of the endpoint.

    Zero at s = 1 (the endpoint is already quadratic-exact there)."""
    ma_lag = (period - 1.0) / 2.0
    return (smoothness - 1.0) / (2.0 * smoothness) * ma_lag * ma_lag


class QuadraticXEPMA:
    """Quadratic-exact eXponential End Point Moving Average (off the family axis).

    XEPMA endpoint plus factor * Delta^2 IFEMA(p, s+2); factor = c(s) cancels the
    kernel's second moment (quadratic-exact) while preserving m0 = 1 and m1 = 0.
    At s = 1, factor = 0 and the output is the endpoint bit-for-bit.
    """

    def __init__(self, period: float, smoothness: float = 1.0):
        self.smoothness = smoothness
        self._base = XEPMA(period, smoothness)

        if smoothness != 0.0:
            self._factor = _quadratic_correction_factor(period, smoothness)
        else:
            self._factor = 0.0

        if self._factor != 0.0:
            self._ma3 = IFEMA(period, smoothness + 2.0)
        else:
            self._ma3 = None
        self._prev_ma3 = None
        self._prev_ma3_2 = None

    def get_next(self, value: float) -> float | None:
        if value is None or not math.isfinite(value):
            return None

        result = self._base.get_next(value)
        if self._ma3 is None:
            return result

        ma3 = self._ma3.get_next(value)
        # Second difference needs two prior cascade values; before that the
        # output is the endpoint (the correction is not yet defined).
        if self._prev_ma3_2 is not None:
            result += self._factor * (ma3 - 2.0 * self._prev_ma3 + self._prev_ma3_2)
        self._prev_ma3_2 = self._prev_ma3
        self._prev_ma3 = ma3
        return result

    def calc_next(self, value: float) -> float | None:
        if value is None or not math.isfinite(value):
            return None

        result = self._base.calc_next(value)
        if self._ma3 is None or self._prev_ma3_2 is None:
            return result

        ma3 = self._ma3.calc_next(value)
        return result + self._factor * (ma3 - 2.0 * self._prev_ma3 + self._prev_ma3_2)
