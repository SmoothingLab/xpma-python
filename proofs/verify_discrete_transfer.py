"""Parity check for the paper's discrete-definition block (Section 2.5).

Builds a direct, from-scratch implementation of the printed z-domain transfer and
its state recurrences

    H_{s,p}(z) = A_{s,p}(z) + r L (1 - z^{-1}) B_{s+1,p}(z),

with the lag-matched cascades' explicit pole parameters (per-stage sub-periods
1 + (p-1)/s and 1 + (p-1)/(s+1); per-stage alpha = 2/(subperiod + 1),
beta = 1 - alpha; L = (p-1)/2), seed-on-first-input states, and the
backward-difference correction L (B_n - B_{n-1}) with the term taken as zero at
the first sample. Compares it against the package filter xpma.XPMA(p, s, r) over a
random series with a long zero warm-up, at (p, s, r) = (20, 1, 0.5), (20, 2, 1.0)
and (50, 3, 0.4). Agreement to ~1e-12 is asserted.

Run: python proofs/verify_discrete_transfer.py
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

from xpma import XPMA


def cascade_subperiod(p, order):
    """Lag-matched per-stage period q = 1 + (p-1)/order for an order-stage cascade."""
    return 1.0 + (p - 1.0) / order


def ema_pole(subperiod):
    """Per-stage alpha, beta for an EMA at the given (sub-)period."""
    alpha = 2.0 / (subperiod + 1.0)
    return alpha, 1.0 - alpha


class DirectCascade:
    """Order-stage EMA cascade at a fixed sub-period; every stage seeds on first input."""

    def __init__(self, subperiod, order):
        self.alpha, self.beta = ema_pole(subperiod)
        self.order = order
        self.state = None

    def step(self, x):
        if self.state is None:
            self.state = [x] * self.order
            return x
        v = x
        for j in range(self.order):
            self.state[j] = self.beta * self.state[j] + self.alpha * v
            v = self.state[j]
        return v


class DirectXPMA:
    """H_{s,p}(z) = A_{s,p}(z) + r L (1 - z^{-1}) B_{s+1,p}(z) by direct recurrence."""

    def __init__(self, p, s, r):
        self.L = (p - 1.0) / 2.0
        self.r = r
        self.A = DirectCascade(cascade_subperiod(p, s), s)          # base s-cascade
        self.B = DirectCascade(cascade_subperiod(p, s + 1), s + 1)  # lead (s+1)-cascade
        self.prev_B = None

    def step(self, x):
        a = self.A.step(x)
        b = self.B.step(x)
        if self.prev_B is None:
            out = a  # first sample: backward-difference term taken as zero
        else:
            out = a + self.r * self.L * (b - self.prev_B)
        self.prev_B = b
        return out


def _run_direct(seq, p, s, r):
    f = DirectXPMA(p, s, r)
    return np.array([f.step(float(x)) for x in seq])


def _run_package(seq, p, s, r):
    f = XPMA(p, s, r)
    return np.array([f.get_next(float(x)) for x in seq])


def main():
    rng = np.random.default_rng(20260710)
    warm = np.zeros(200)                       # long zero warm-up
    body = rng.standard_normal(800)
    seq = np.concatenate([warm, body])

    cases = [(20, 1, 0.5), (20, 2, 1.0), (50, 3, 0.4)]
    worst = 0.0
    for (p, s, r) in cases:
        direct = _run_direct(seq, p, s, r)
        pkg = _run_package(seq, p, s, r)
        d = np.abs(direct - pkg)[len(warm):]   # compare on the post-warm-up body
        m = float(np.max(d))
        worst = max(worst, m)
        print(f"(p={p}, s={s}, r={r}): max|direct - XPMA| = {m:.3e}")
        assert m < 1e-12, f"parity failure at (p={p}, s={s}, r={r}): {m:.3e}"
    print(f"OK: worst-case discrepancy {worst:.3e} < 1e-12")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
