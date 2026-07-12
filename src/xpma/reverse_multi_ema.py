"""Reverse multi-ordered EMA: chain of reverse EMAs."""

import math

from .reverse_ema import ReverseEMA


class ReverseMultiEMA:
    def __init__(self, period: float, num_smooths: int):
        num_smooths = int(num_smooths)
        if num_smooths <= 0:
            raise ValueError(
                "ReverseMultiEMA order 0 does not exist: the lag-matched family "
                "has no finite order-0 member (its s -> 0 limit is the identity). "
                "Invert IFEMA(period, s) with 0 < s < 1 for a genuine sub-unit order."
            )

        self._filters = []
        ema_period = (period - 1.0) / num_smooths + 1.0
        for _ in range(num_smooths):
            self._filters.append(ReverseEMA(ema_period))

    def get_next(self, value: float) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        result = value
        for f in self._filters:
            result = f.get_next(result)
        return result

    def calc_next(self, value: float) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        result = value
        for f in self._filters:
            result = f.calc_next(result)
        return result
