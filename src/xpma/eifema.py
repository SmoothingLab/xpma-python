"""EIFEMA: exact fractional-order EMA via the fractional power of a single EMA pole.

Where IFEMA approximates a fractional smoothness order by blending two integer EMA
cascades (two different pole rates), EIFEMA computes the true fractional cascade
(1 - lambda z^-1)^-n directly, with one pole rate. Its impulse response is the
normalised negative-binomial / gamma weighting

    w_0 = alpha^n,   w_k = w_{k-1} * lambda * (n + k - 1) / k,

which sums to 1 and reduces to the exact n-fold EMA cascade (= MultiEMA) at
integer n. The pole sub-period e = (period - 1) / n + 1 is the same formula
MultiEMA uses, so the total time lag is (period - 1) / 2 for any real n.

Realised as a truncated FIR filter (the tail decays geometrically). During
warm-up the weighted sum is renormalised by the partial weight sum, so the first
output equals the raw input (matching the EMA convention) and the output
converges to the full convolution once the window fills.

This is the reference realisation that validates the moment-matched IFEMA weights
and the r_crit^C safe fractional fallback.
"""

import math


class EIFEMA:
    def __init__(self, period: float, smoothness: float):
        self.period = period
        self.smoothness = smoothness

        # Single pole rate; sub-period matches MultiEMA so the integer case agrees.
        ema_period = (period - 1.0) / smoothness + 1.0
        alpha = 2.0 / (ema_period + 1.0)
        self.lam = 1.0 - alpha

        # Precompute the exact fractional-cascade impulse-response weights. The
        # weights sum to 1, so generate terms until the remaining tail is below
        # tolerance.
        self.weights = []
        w = alpha ** smoothness
        cumulative = w
        self.weights.append(w)
        k = 1
        max_terms = 100000
        while 1.0 - cumulative > 1e-13 and k < max_terms:
            w = w * self.lam * (smoothness + k - 1.0) / k
            self.weights.append(w)
            cumulative += w
            k += 1

        # Most-recent-last buffer of the last len(weights) inputs.
        self.history = []

    def get_next(self, value: float):
        if value is None or not math.isfinite(value):
            return None

        # Commit the new input, capping the buffer at the weight count.
        self.history.append(value)
        if len(self.history) > len(self.weights):
            self.history.pop(0)

        # Convolve weights with the buffer (w_0 pairs with the newest input),
        # renormalised by the weight used so far.
        count = len(self.history)
        last = count - 1
        total = 0.0
        weight_sum = 0.0
        for i in range(count):
            wi = self.weights[i]
            total += wi * self.history[last - i]
            weight_sum += wi
        return total / weight_sum

    def calc_next(self, value: float):
        if value is None or not math.isfinite(value):
            return None

        # Treat value as the newest sample without committing it to the buffer.
        stored = len(self.history)
        available = stored + 1
        count = available if available < len(self.weights) else len(self.weights)
        last = stored - 1

        total = self.weights[0] * value
        weight_sum = self.weights[0]
        for i in range(1, count):
            wi = self.weights[i]
            total += wi * self.history[last - (i - 1)]
            weight_sum += wi
        return total / weight_sum
