"""DampedXEPMA: minimum-overshoot sibling of the XEPMA endpoint.

Same construction as QuadraticXEPMA (the XEPMA endpoint plus factor * Delta^2
IFEMA(p, s+2), which leaves m0 = 1 and m1 = 0 untouched, so zero time lag
survives), but the coefficient is the exact discrete minimiser of step-response
overshoot for the given (period, smoothness) rather than the quadratic-exact value.

The step response is affine in the coefficient, so its overshoot is convex
piecewise-linear and the minimum is the crossing of the two branches at the
adjacent times straddling the mode of the (s+2)-cascade's kernel. The coefficient
is computed once at construction from the from-rest step responses (a leading zero
seeds all internal EMA states at rest). Overshoot sits below the XEPMA endpoint's at
every (s, p), including s = 1, where the quadratic-exact coefficient is zero but the
min-overshoot correction still exists (it trades the endpoint's quadratic
exactness for less overshoot).
"""

import math

from .ifema import IFEMA
from .xepma import XEPMA


def _min_overshoot_factor(period: float, smoothness: float) -> float:
    """Exact discrete min-overshoot correction coefficient for (period, smoothness).

    Simulates the from-rest step responses of the XEPMA endpoint and the (s+2)
    correction cascade, then returns the branch-crossing value at the adjacent times
    straddling the cascade kernel's mode."""
    base = XEPMA(period, smoothness)
    cascade = IFEMA(period, smoothness + 2.0)
    bars = int(math.ceil(period * 4.0)) + 10

    # Seed all internal states at rest with a leading zero.
    base.get_next(0.0)
    cascade.get_next(0.0)

    e_x = []          # base-endpoint step error
    w = []            # Delta^2 of the cascade step response
    g1 = None
    g2 = None
    mode_index = 0
    max_impulse = -math.inf
    for t in range(bars):
        e_x.append(base.get_next(1.0) - 1.0)
        g0 = cascade.get_next(1.0)
        if g2 is not None:
            w.append(g0 - 2.0 * g1 + g2)
        elif g1 is not None:
            w.append(g0 - 2.0 * g1)
        else:
            w.append(g0)
        impulse = g0 if g1 is None else g0 - g1
        if impulse > max_impulse:
            max_impulse = impulse
            mode_index = t
        g2 = g1
        g1 = g0

    t2 = mode_index
    t1 = mode_index + 1
    return (e_x[t1] - e_x[t2]) / (w[t2] - w[t1])


class DampedXEPMA:
    """Minimum-overshoot eXponential End Point Moving Average.

    XEPMA endpoint plus factor * Delta^2 IFEMA(p, s+2); factor is the exact discrete
    min-overshoot coefficient for (period, smoothness). Preserves m0 = 1 and m1 = 0
    (zero time lag) at every s. Meaningful at s = 1 (unlike QuadraticXEPMA's
    correction, which vanishes there).
    """

    def __init__(self, period: float, smoothness: float = 1.0):
        self.smoothness = smoothness
        self._base = XEPMA(period, smoothness)

        if smoothness != 0.0:
            self._factor = _min_overshoot_factor(period, smoothness)
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
        # output is the base endpoint (the correction is not yet defined).
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
