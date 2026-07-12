"""QuadraticXEPMA: the quadratic-exact endpoint correction (off the family axis).

The XEPMA endpoint plus the lag- and gain-preserving curvature correction
c(s) Delta^2 IFEMA(p, s+2), with c(s) = ((s-1)/(2s)) L^2, cancelling the
kernel's second moment at every order: quadratic-exact for all real s >= 1,
and identical to XEPMA at s = 1, where c = 0. It is the mathematically
complete solution; its noise gain rises with s, so for practical use
DampedXEPMA (the min-overshoot correction) is usually the better sibling.

Takes no tuning parameter.
"""

import math

from .ifema import IFEMA
from .xepma import XEPMA


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
