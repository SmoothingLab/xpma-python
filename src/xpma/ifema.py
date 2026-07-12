"""IFEMA: the standard fractional-order EMA cascade, with three smoothness regimes.

The lag-matched gamma family has second cumulant kappa2(s) = L^2 / s + L,
L = (period - 1) / 2, which diverges as s -> 0: there is no finite order-0 member
(the s -> 0 limit is the identity, which loses the lag). IFEMA therefore covers:

  - smoothness <= 0: raises. Sub-unit orders start strictly above 0; the only
    order-0 object is XPMA(period, 0, lag_reduction = 1), the identity on the
    zero-lag line.

  - 0 < smoothness < 1: the C1 shelf, a single EMA(p1) followed by a ReverseEMA(q1)
    with p1 = L (1 + s) / s + 1 and q1 = L (1 - s) / s + 1. It is moment-exact
    (mean lag L, kappa2 = L^2 / s + L exactly), its kernel is non-negative (so no
    overshoot), and it collapses continuously to EMA(p) as s -> 1 (there q1 -> 1,
    the identity).

  - smoothness >= 1: the second-moment-matched blend of the two integer EMA
    cascades either side of s,

        w_floor = (1 - frac) * floor(n) / n,   w_ceil = frac * ceil(n) / n,

    chosen so the blended kernel's second moment matches the true fractional-order
    cascade. The neighbours and the true fractional filter share the same first
    moment (mean lag L), so the blend variance is linear in the weight and the
    match has a closed form; the (period - 1) factor cancels, leaving a
    period-independent weighting. The weights sum to 1 and reduce to a single
    cascade at integer n, where the output is bit-identical to MultiEMA(period, n).

This is the standard fractional cascade used by XPMA, XEPMA, QuadraticXEPMA and
DampedXEPMA. The exact reference realisation (single fractional pole, a gamma /
negative-binomial kernel) is EIFEMA.
"""

import math

from .ema import EMA
from .reverse_ema import ReverseEMA
from .multi_ema import MultiEMA


class IFEMA:
    def __init__(self, period: float, smoothness: float):
        if smoothness <= 0.0:
            raise ValueError(
                "IFEMA order %s does not exist: the lag-matched EMA cascade has no "
                "finite order-0 member; sub-unit orders start strictly above 0. Use "
                "0 < smoothness < 1 for the C1 shelf, or XPMA(period, 0, "
                "lag_reduction=1) for the identity fast leg." % smoothness
            )

        self.smoothness = smoothness

        if smoothness < 1.0:
            # C1 shelf: the genuine sub-unit order (mean lag L, kappa2 = L^2/s + L).
            self._mode = "c1"
            ma_lag = (period - 1.0) / 2.0
            p1 = ma_lag * (1.0 + smoothness) / smoothness + 1.0
            q1 = ma_lag * (1.0 - smoothness) / smoothness + 1.0
            self._stage1 = EMA(p1)
            self._stage2 = ReverseEMA(q1)
            return

        self._mode = "blend"
        self.fraction = smoothness % 1.0
        if self.fraction == 0.0:
            self._ma = MultiEMA(period, int(smoothness))
            self._ma1 = self._ma2 = None
        else:
            f = math.floor(smoothness)
            c = math.ceil(smoothness)
            self._ma1 = MultiEMA(period, f)
            self._ma2 = MultiEMA(period, c)
            self._w1 = (1.0 - self.fraction) * f / smoothness
            self._w2 = self.fraction * c / smoothness

    def get_next(self, value):
        if value is None or not math.isfinite(value):
            return None
        if self._mode == "c1":
            # EMA first, then ReverseEMA (forward-then-reverse shelf).
            return self._stage2.get_next(self._stage1.get_next(value))
        if self.fraction == 0.0:
            return self._ma.get_next(value)
        return self._w1 * self._ma1.get_next(value) + self._w2 * self._ma2.get_next(value)

    def calc_next(self, value):
        if value is None or not math.isfinite(value):
            return None
        if self._mode == "c1":
            return self._stage2.calc_next(self._stage1.calc_next(value))
        if self.fraction == 0.0:
            return self._ma.calc_next(value)
        return self._w1 * self._ma1.calc_next(value) + self._w2 * self._ma2.calc_next(value)
