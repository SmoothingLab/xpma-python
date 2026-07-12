"""Top-level fractional-order interpolation.

Realises a fractional smoothness order by interpolating, at the OUTPUT level, between the
two bracketing integer-order instances of a filter (not inside its cascades). A convex
blend of two non-negative-kernel filters is non-negative, so a no-overshoot filter
(FastEMA, LeadEMA) keeps its no-overshoot guarantee exactly at fractional orders, while a
cascade-level blend would lose it. Lag (shared by the integer instances) and unit DC gain
are preserved. Use this for filters whose lag reduction depends on s (FastEMA, LeadEMA,
ConvexFastEMA); for fixed-lag-reduction filters (XEPMA, XPMA) use the IFEMA cascade
instead.

At integer smoothness the class delegates to the single integer-order instance (no
blend), bit-identical to constructing that instance directly, so consumers can route
every order through this class without branching on s % 1.

Sub-unit orders (0 < s < 1) are handled differently by the two fractional-order
mechanisms in this package. For the EMA-cascade family (IFEMA and everything built
on it) a genuine sub-unit order is realised by IFEMA's moment-exact C1 shelf. This
top-level output-level class deliberately does NOT reach into that regime: for
0 < s < 1 the floor order is 0, its weight is 0, and the class collapses to the
single ceil (order-1) instance. That is intentional, because its consumers are the
r(s) lag-reduction filters (FastEMA, LeadEMA, ConvexFastEMA), whose lag reduction
r_crit(s) has no sub-unit definition, so there is no order-0 member for them to
blend toward.
"""

import math


class FractionalSmoothness:
    def __init__(self, smoothness: float, make_filter):
        # Integer order: a single instance, no blend (bit-identical to direct
        # construction). Keeps consumers from having to branch on s % 1.
        if smoothness % 1.0 == 0.0:
            self._single = make_filter(int(smoothness))
            self._lo = self._hi = None
            self.lo_weight = 1.0
            self.hi_weight = 0.0
            return

        self._single = None
        frac = smoothness % 1.0
        floor_order = math.floor(smoothness)
        ceil_order = math.ceil(smoothness)
        # IFEMA second-moment-matched, convex weights (sum to 1).
        self.lo_weight = 0.0 if floor_order == 0 else (1.0 - frac) * floor_order / smoothness
        self.hi_weight = 1.0 - self.lo_weight
        self._lo = make_filter(floor_order) if self.lo_weight > 0.0 else None
        self._hi = make_filter(ceil_order) if self.hi_weight > 0.0 else None

    def get_next(self, value):
        if self._single is not None:
            return self._single.get_next(value)
        a = self._lo.get_next(value) if self._lo is not None else None
        b = self._hi.get_next(value) if self._hi is not None else None
        if (self._lo is not None and a is None) or (self._hi is not None and b is None):
            return None
        result = 0.0
        if self._lo is not None:
            result += self.lo_weight * a
        if self._hi is not None:
            result += self.hi_weight * b
        return result

    def calc_next(self, value):
        if self._single is not None:
            return self._single.calc_next(value)
        a = self._lo.calc_next(value) if self._lo is not None else None
        b = self._hi.calc_next(value) if self._hi is not None else None
        if (self._lo is not None and a is None) or (self._hi is not None and b is None):
            return None
        result = 0.0
        if self._lo is not None:
            result += self.lo_weight * a
        if self._hi is not None:
            result += self.hi_weight * b
        return result
