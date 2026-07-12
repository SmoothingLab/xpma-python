"""Lead XEPMA: zero time-lag built on the lag-matched LeadEMA smoother.

Its endpoint correction uses ma_lag = (period-1)/2, which requires the smoother to
carry exactly EMA(period)'s lag, the property LeadEMA provides.
"""

import math

from .lead_ema import LeadEMA


class LeadXEPMA:
    def __init__(self, period: float, smoothness: float = 1.0):
        self.smoothness = smoothness
        self.ma_lag = (period - 1.0) / 2.0

        if self.smoothness != 0.0:
            self._ma1 = LeadEMA(period, self.smoothness)
            self._ma2 = LeadEMA(period, self.smoothness + 1.0)
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
