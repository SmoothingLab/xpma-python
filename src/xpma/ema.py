"""Exponential Moving Average with fractional period support."""

import math


class EMA:
    def __init__(self, period: float):
        self.alpha = 2.0 / (period + 1.0)
        self.result = None

    def get_next(self, value: float) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        self.result = self.calc_next(value)
        return self.result

    def calc_next(self, value: float) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        if self.result is None:
            return value
        return self.result + (value - self.result) * self.alpha
