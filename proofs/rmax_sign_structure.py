"""Sign-change structure of h'(tau) for the base two-rate family, and the
demonstration that only the post-mode region (u_+, infinity) can give birth to
an extra dip-bump pair (the pre-mode region never binds).

h'(tau) = tau^{s-2} e^{-s tau} F(tau),   F(tau) = c1 A(tau) + r c2 e^{-tau} q(tau)
  c1 = s^s/(s-1)!,  c2 = (s+1)^{s+1}/s!,  A = (s-1) - s tau,
  q = s(s-1) - 2s(s+1)tau + (s+1)^2 tau^2.
Region edges (s >= 2):  u_- = (s-sqrt s)/(s+1) < (s-1)/s < u_+ = (s+sqrt s)/(s+1).
  (0, u_-):        A>0, q>0  => F>0            (rising, intrinsic)
  (u_-, (s-1)/s):  A>0, q<0  => pre-mode; the single mode crossing lives here
  ((s-1)/s, u_+):  A<0, q<0  => F<0            (falling after mode)
  (u_+, infty):    A<0, q>0  => post-mode; the extra dip-bump is born here

Claims verified here (NUMERICAL; analytic reasons stated in the doc):
  (A) F(0)>0 and F<0 on ((s-1)/s, u_+): the two same-sign regions.
  (B) On (0,(s-1)/s) the zero-set curve r = R2(tau) is strictly monotone
      decreasing, so F has exactly one zero there (the mode) for EVERY r>=0:
      the pre-mode region never births an extra pair.  d(lnR2)/dtau < 0 checked.
  (C) Post-mode R(tau) -> +inf at both ends of (u_+,infty), unique interior
      min = r_crit^C (single real root of the stationarity cubic).
  (D) Bisection r_crit^C from the total upcrossing count == closed form (all
      regions, so the post-mode branch is the binding one).

numpy float64 is used for the dense grid scans (sign counts, monotonicity);
mpmath (dps=40) for the closed-form values.

Run: python proofs/rmax_sign_structure.py
"""

import os
import sys

import numpy as np
import mpmath as mp

mp.mp.dps = 40
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmax_closed_form import tau1_cardano, R as Rpost  # noqa: E402


def c2_over_c1(s):
    # c2/c1 = ((s+1)^{s+1}/s!) / (s^s/(s-1)!) = ((s+1)/s)^{s+1}
    return ((s + 1) / s) ** (s + 1)


def Af(s, t):
    return (s - 1) - s * t


def qf(s, t):
    return s * (s - 1) - 2 * s * (s + 1) * t + (s + 1) ** 2 * t**2


def Ff(s, r, t):
    # F/c1 (same sign as F, no overflow): A + r (c2/c1) e^{-t} q
    return Af(s, t) + r * c2_over_c1(s) * np.exp(-t) * qf(s, t)


def edges(s):
    return (s - np.sqrt(s)) / (s + 1), (s - 1) / s, (s + np.sqrt(s)) / (s + 1)


def R2(s, t):
    # zero-set r = -A / ((c2/c1) e^{-t} q); ratio form avoids overflow
    return -Af(s, t) / (c2_over_c1(s) * np.exp(-t) * qf(s, t))


def rmax_cf(s):
    return float(Rpost(s, tau1_cardano(s)))


def banner(x):
    print("\n" + "=" * 72)
    print(x)
    print("=" * 72)


banner("A.  Same-sign regions:  F(0+)>0  and  F<0 on ((s-1)/s, u_+)")
for s in (2, 3, 4, 6, 10, 30, 100):
    um, a0, up = edges(s)
    rm = rmax_cf(s)
    f0 = Ff(s, rm, 1e-9)
    grid = np.linspace(a0, up, 2000)[1:-1]
    fmax = Ff(s, rm, grid).max()
    print("  s=%3d  F(0+)=%+.4f (>0)   max F on ((s-1)/s,u_+)=%+.3e (<0)"
          % (s, f0, fmax))

banner("B.  Pre-mode zero-set R2(tau) strictly monotone => one mode zero, all r")
print("  central-difference d(ln R2)/dtau over (u_-, (s-1)/s)")
for s in (2, 3, 4, 6, 10, 30, 100, 1000):
    um, a0, up = edges(s)
    t = np.linspace(um, a0, 200002)[1:-1]
    lr = np.log(R2(s, t))
    d = np.diff(lr)  # sign of increments = sign of derivative
    allneg = bool(np.all(d < 0))
    allpos = bool(np.all(d > 0))
    print("  s=%4d  increments  min=%+.3e max=%+.3e  monotone=%s"
          % (s, d.min(), d.max(),
             "yes(decr)" if allneg else ("yes(incr)" if allpos else "NO")))

banner("C.  Post-mode R(tau): +inf at both ends, unique interior min = r_crit^C")
for s in (2, 4, 10, 100):
    um, a0, up = edges(s)
    t1 = float(tau1_cardano(s))
    near = up + (t1 - up) * 1e-4
    far = t1 * 30
    print("  s=%3d  R(u_+^+)=%.3e  R(min@%.5f)=%.7f  R(30*tau1)=%.3e"
          % (s, float(Rpost(s, near)), t1, float(Rpost(s, t1)), float(Rpost(s, far))))

banner("D.  Bisection r_crit^C (total upcrossings, dense grid) == closed form")


def n_upcross(s, r, ts):
    v = Ff(s, r, ts)
    sgn = np.sign(v)
    return int((np.diff(sgn) > 0).sum())


ts = np.linspace(1e-7, 45, 3_000_001)
for s in (1, 2, 3, 4, 5, 8):
    lo, hi = 0.05, 0.95
    for _ in range(52):
        mid = 0.5 * (lo + hi)
        if n_upcross(s, mid, ts) == 0:
            lo = mid
        else:
            hi = mid
    cf = rmax_cf(s)
    print("  s=%d  bisection r_crit^C=%.7f  closed form=%.7f  diff=%.1e"
          % (s, lo, cf, abs(lo - cf)))

if __name__ == "__main__":
    pass
