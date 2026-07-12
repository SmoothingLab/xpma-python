"""Reverse EMA: algebraic inversion of Brown's formula."""

import math


class ReverseEMA:
    def __init__(self, period: float):
        self.alpha = 2.0 / (period + 1.0)
        self.previous_input = None
        self.result = None

    def get_next(self, value: float) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        self.result = self.calc_next(value)
        self.previous_input = value
        return self.result

    def calc_next(self, value: float) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        if self.result is None:
            return value
        return self.previous_input + (value - self.previous_input) / self.alpha
