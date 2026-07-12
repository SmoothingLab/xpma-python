"""XPMA (eXponential Polynomial Moving Average): the r-parameterised family.

The base two-rate line, running from the EMA cascade (lag reduction r = 0) to the
XEPMA endpoint (r = 1):

    XPMA^[s](p, r) = IFEMA(p, s) + r * (XEPMA(p, s) - IFEMA(p, s)).

Controllable smoothness (s) and lag reduction (r). FastEMA, LeadEMA and
ConvexFastEMA are the r_crit^M / r_crit^M(inflated) / r_crit^C members of this
family; QuadraticXEPMA and DampedXEPMA sit off the axis and are not members.

The lag reduction is fixed at construction (an IIR filter cannot change r
mid-stream). The named members pass their critical constant explicitly; the
constants themselves live in xpma.r_crit.
"""

import math

from .ifema import IFEMA
from .xepma import XEPMA


class XPMA:
    def __init__(self, period: float, smoothness: float, lag_reduction: float):
        self.period = period
        self.smoothness = smoothness
        self.ma_lag = (period - 1.0) / 2.0

        if smoothness < 0.0:
            raise ValueError("XPMA smoothness must be >= 0")

        if smoothness == 0.0:
            # Order 0 exists only on the zero-lag r = 1 line, where the two-rate
            # family degenerates to the identity (XEPMA(period, 0), mean lag 0).
            # It is constructed directly, never via IFEMA(period, 0) (which raises).
            if lag_reduction != 1.0:
                raise ValueError(
                    "XPMA order 0 exists only on the r = 1 line, where it is the "
                    "identity; construct XPMA(period, 0, lag_reduction=1). Any other "
                    "lag reduction has no finite order-0 member."
                )
            self._identity = True
            self.lag_reduction = 1.0
            self._ma = None
            self._xepma = None
            return

        self._identity = False
        self._ma = IFEMA(period, smoothness)
        self.lag_reduction = lag_reduction
        self._xepma = XEPMA(period, smoothness) if lag_reduction != 0.0 else None

    # -- streaming -----------------------------------------------------------------
    def get_next(self, value: float) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        if self._identity:
            return value

        ma = self._ma.get_next(value)
        if self.lag_reduction == 0.0:
            return ma

        xepma = self._xepma.get_next(value)
        return ma + self.lag_reduction * (xepma - ma)

    def calc_next(self, value: float) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        if self._identity:
            return value

        ma = self._ma.calc_next(value)
        if self.lag_reduction == 0.0:
            return ma

        xepma = self._xepma.calc_next(value)
        return ma + self.lag_reduction * (xepma - ma)
