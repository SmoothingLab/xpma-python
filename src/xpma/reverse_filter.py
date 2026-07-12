"""ReverseFilter: run any filter in the package backwards.

Feed it a filter's output stream and it recovers, sample by sample, the input
series that would have produced it. ReverseEMA and ReverseMultiEMA invert their
filters algebraically and exactly; ReverseFilter covers every other filter by
solving each step numerically with SecantSolver, seeding each solve with the
previously recovered input.

The wrapped indicator is driven by the reversal (each recovered input is
committed to it), so pass a dedicated instance constructed with the same
parameters as the forward filter, not the instance smoothing your live stream.
"""

import math

from .secant_solver import SecantSolver


class ReverseFilter:
    def __init__(self, indicator, max_iterations: int = 5, max_error: float = 1e-6):
        self._solver = SecantSolver(indicator, max_iterations, max_error)
        self._prev = None

    def get_next(self, value: float) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        estimate = value if self._prev is None else self._prev
        recovered = self._solver.solve(value, estimate)
        self._prev = recovered
        return recovered

    def calc_next(self, value: float) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        estimate = value if self._prev is None else self._prev
        return self._solver.solve(value, estimate, commit=False)
