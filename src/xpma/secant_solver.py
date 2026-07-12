"""Secant solver: iteratively reverses any filter with get_next/calc_next."""

import math


class SecantSolver:
    def __init__(self, indicator, max_iterations: int = 5, max_error: float = 1e-6):
        if not hasattr(indicator, 'calc_next') or not hasattr(indicator, 'get_next'):
            raise ValueError("Indicator must have calc_next and get_next methods")
        self.indicator = indicator
        self.max_iterations = max_iterations
        self.max_error = max_error

    def solve(self, target: float, estimate: float) -> float:
        if not math.isfinite(target) or not math.isfinite(estimate):
            raise ValueError("Target and estimate must be finite numbers")

        if abs(self.indicator.calc_next(estimate) - target) < self.max_error:
            self.indicator.get_next(estimate)
            return estimate

        prev_estimate = None
        prev_diff = None

        for i in range(1, self.max_iterations + 1):
            estimate_result = self.indicator.calc_next(estimate)
            diff = estimate_result - target

            if abs(diff) < self.max_error:
                break

            if prev_estimate is None:
                prev_estimate = estimate
                estimate += diff
            elif estimate == prev_estimate:
                estimate -= diff / 2.0
            else:
                slope = (diff - prev_diff) / (estimate - prev_estimate)
                prev_estimate = estimate
                estimate -= diff / slope

            prev_diff = diff
        else:
            raise RuntimeError("Max iterations exceeded in SecantSolver")

        self.indicator.get_next(estimate)
        return estimate
