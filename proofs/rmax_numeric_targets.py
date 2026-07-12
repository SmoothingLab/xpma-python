"""Numerical ground truth for the r_crit^C (concavity / monotone-rate) boundary
of the base two-rate family XPMA^[s](p, r).

Produces the reference target values for the r_crit^C boundary:
  1. Continuous-limit r_crit^C(s) by bisection on the number of minus-to-plus sign
     changes of h'(tau) (dip-bump birth), s = 1..4.
  2. Tangency location tau_1(s): argmax of h' at the boundary (h' touches 0).
  3. Discrete boundaries at p in {20, 50, 200} for s = 1 (largest r with a
     non-increasing kernel), via XPMA with a long zero warm-up (the constituent
     EMAs seed on the first input, so a zero warm-up puts every cascade at rest).

Continuous-limit kernels, time in lag units (L = 1):
  h1 = lag-matched EMA^[s] cascade (per-stage rate s):
       h1(t) = s^s t^(s-1) e^(-s t) / (s-1)!
  h2 = lag-matched EMA^[s+1] cascade (per-stage rate s+1)
  family kernel h = h1 + r * h2'   (XPMA^[s] at lag reduction r)

s = 1 closed form: r_crit^C(1) = e^2/16, tangency at tau = 2.

Run: python proofs/rmax_numeric_targets.py
"""

from math import factorial, e

import numpy as np


def h_prime(s, r, t):
    """h'(t) = h1'(t) + r * h2''(t) for the base family, continuous limit."""
    c1 = s ** s / factorial(s - 1)
    if s >= 2:
        h1p = c1 * np.exp(-s * t) * ((s - 1) * t ** (s - 2) - s * t ** (s - 1))
    else:
        h1p = -np.exp(-t)
    c2 = (s + 1) ** (s + 1) / factorial(s)
    h2pp = c2 * np.exp(-(s + 1) * t) * (
        s * (s - 1) * t ** (s - 2)
        - 2 * s * (s + 1) * t ** (s - 1)
        + (s + 1) ** 2 * t ** s
    )
    return h1p + r * h2pp


def n_upcross(s, r, t):
    """Count minus-to-plus sign changes of h' (the dip-bump signature)."""
    sgn = np.sign(h_prime(s, r, t))
    return int((np.diff(sgn) > 0).sum())


def continuous_rmax(s, t):
    lo, hi = 0.01, 0.99
    for _ in range(52):
        mid = 0.5 * (lo + hi)
        if n_upcross(s, mid, t) == 0:
            lo = mid
        else:
            hi = mid
    return lo


def tangency(s, r):
    t = np.linspace(1.2, 6, 4_000_001)
    v = h_prime(s, r, t)
    i = int(v.argmax())
    return float(t[i]), float(v[i])


def discrete_kernel(p, s, r, n=4000, warm=6000):
    from xpma import XPMA

    f = XPMA(p, float(s), r)
    for _ in range(warm):
        f.get_next(0.0)  # zero warm-up BEFORE the impulse (seeding trap)
    out = [f.get_next(1.0)]
    for _ in range(n - 1):
        out.append(f.get_next(0.0))
    return np.array(out)


def discrete_rmax(p, s=1):
    def concave(r):
        return bool(np.all(np.diff(discrete_kernel(p, s, r)) <= 1e-13))

    lo, hi = 0.30, 0.60
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if concave(mid):
            lo = mid
        else:
            hi = mid
    return lo


def main():
    t = np.linspace(1e-9, 40, 2_000_001)
    print("s = 1 closed form: e^2/16 = %.10f" % (e ** 2 / 16))
    print()
    print("Continuous-limit targets:")
    for s in (1, 2, 3, 4):
        rmax = continuous_rmax(s, t)
        tau1, resid = tangency(s, rmax)
        r_m = (s ** (s + 1) / (s + 1) ** (s + 2)) * np.exp((2 * s + 1) / (s + 1))
        print(
            "  s=%d  r_crit^C = %.8f   tau_1 = %.6f (h' residual %+.1e)"
            "   r_crit^M = %.8f   ratio = %.4f"
            % (s, rmax, tau1, resid, r_m, rmax / r_m)
        )
    print()
    print("Discrete boundaries (s = 1; expect approach to e^2/16 from above):")
    for p in (20.0, 50.0, 200.0):
        print("  p=%5.0f  r_crit^C = %.6f" % (p, discrete_rmax(p)))


if __name__ == "__main__":
    main()
