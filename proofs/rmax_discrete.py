"""Discrete safety of the continuous r_crit^C(s) constant.

The discrete boundary r_crit^C_disc(s, p) is the largest lag reduction r for which
the discrete kernel h_n of XPMA(p, s, r) is unimodal, i.e. its first difference
Delta h_n = h_n - h_{n-1} has no minus-to-plus sign change past the kernel mode
(no rate re-acceleration).  At s = 1 the kernel mode is n = 0, so this is exactly
"largest r with a non-increasing discrete kernel"; at s >= 2 it is the discrete
image of the continuous "exactly one sign change of h'".

Discrete kernel via XPMA(period, smoothness, lag_reduction), with a long zero
warm-up before the impulse (the constituent EMAs seed on the first input, so a
zero warm-up puts every cascade at rest and the subsequent impulse response is
the true LTI kernel).

We verify r_crit^C_disc(s, p) > r_crit^C(s) (continuous) for every tested (s, p), so
the closed-form constant is safe to deploy at any period, and report whether the
margin shrinks with p (expected O(1/p) from above, mirroring r_crit^M).

Run: python proofs/rmax_discrete.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmax_closed_form import tau1_cardano, R as Rpost  # noqa: E402
from xpma import XPMA  # noqa: E402


def rmax_continuous(s):
    return float(Rpost(s, tau1_cardano(s)))


def discrete_kernel(p, s, r):
    """Impulse response with a long zero warm-up BEFORE the impulse."""
    n = int(max(6000, 30 * p))
    warm = int(max(8000, 40 * p))
    f = XPMA(float(p), float(s), float(r))
    for _ in range(warm):
        f.get_next(0.0)
    out = [f.get_next(1.0)]
    for _ in range(n - 1):
        out.append(f.get_next(0.0))
    return np.array(out)


def rate_violation(p, s, r):
    """max first-difference past the kernel mode (>0 => rate re-accelerated).

    Restricted to the support where the kernel is non-negligible so the decaying
    tail's floating-point noise does not create spurious sign flips.
    """
    h = discrete_kernel(p, s, r)
    peak = h.max()
    mode = int(h.argmax())
    d = np.diff(h)  # Delta h_n
    # window: from just past the mode to where the kernel decays below eps*peak
    tail = np.where(h[mode:] > 1e-11 * peak)[0]
    end = mode + (tail[-1] if len(tail) else len(h) - mode - 1)
    seg = d[mode:end]
    return float(seg.max()) if len(seg) else -1.0


def discrete_rmax(p, s, tol=1e-13):
    lo, hi = 0.02, 0.98
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if rate_violation(p, s, mid) <= tol * abs(discrete_kernel(p, s, mid).max()):
            lo = mid
        else:
            hi = mid
    return lo


def banner(x):
    print("\n" + "=" * 72)
    print(x)
    print("=" * 72)


def main():
    banner("Calibration vs reference targets (s=1)")
    print("  p      discrete r_crit^C     reference")
    ref = {20: 0.506497, 50: 0.480045, 200: 0.466419}
    for p in (20, 50, 200):
        dr = discrete_rmax(p, 1)
        print("  %-5d  %.6f          %.6f" % (p, dr, ref[p]))

    banner("Discrete safety table:  s = 1..4,  p in {5,10,20,50,200,1000}")
    print("  continuous r_crit^C(s):", {s: round(rmax_continuous(s), 6) for s in (1, 2, 3, 4)})
    print()
    print("   s |    p |  disc r_crit^C  | cont r_crit^C  |  margin (disc-cont)")
    margins = {}
    for s in (1, 2, 3, 4):
        rc = rmax_continuous(s)
        margins[s] = []
        for p in (5, 10, 20, 50, 200, 1000):
            dr = discrete_rmax(p, s)
            m = dr - rc
            margins[s].append((p, m))
            flag = "OK" if m > 0 else "*** BELOW ***"
            print("  %2d | %4d | %11.7f | %10.7f | %+11.7f  %s" % (s, p, dr, rc, m, flag))
        print()

    banner("Margin shrinkage with p (expect ~O(1/p), halving as p doubles)")
    for s in (1, 2, 3, 4):
        print("  s=%d:" % s)
        for (p, m), (p2, m2) in zip(margins[s][:-1], margins[s][1:]):
            ratio = m / m2 if m2 else float("nan")
            print("     p %4d->%4d  margin %+.6f -> %+.6f   ratio %.2f  p*margin=%.4f"
                  % (p, p2, m, m2, ratio, p2 * m2))


if __name__ == "__main__":
    main()
