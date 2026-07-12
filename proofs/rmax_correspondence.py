"""Correspondence of the concavity, monotonicity and no-overshoot boundaries
across the three lead-correction families.

Adds the concavity / monotone-rate boundary (r_crit^C) to the existing
monotone (r_crit^M) and no-overshoot (r_crit^O) correspondence across the three
lead-correction families.  Verified claims:

  (a) FIR / uniform-advance family (WMA line): the concavity, monotonicity and
      overshoot boundaries all COINCIDE at the WMA edge a_crit = (p+1)/6
      (continuous r_crit^FIR = 1/3).  Reason: the kernel's negative region is a
      trailing segment abutting the window edge, so the last weight going
      negative breaks all three properties at once (the window-edge jump from a
      negative last weight up to zero is itself the rate re-acceleration).

  (b) Same-rate exponential family ((1+r)EMA - r EMA^2, Brown/GDEMA line): no
      r > 0 satisfies ANY of the three.  Kernel h(tau) = e^{-tau}(1 + r - r tau),
      h'(tau) = e^{-tau}(r tau - (1 + 2r)) changes sign (- to +) at
      tau = 2 + 1/r for every r > 0 (concavity fails); the step error
      beta^{n+1}(1 - r alpha (n+1)) changes sign at n+1 = 1/(r alpha) for every
      r > 0 (monotonicity and overshoot fail).  All three boundaries are 0.

  (c) Two-rate family (this work): the three split fully,
      0 < r_crit^C < r_crit^M < r_crit^O, for every s.

Run: python proofs/rmax_correspondence.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmax_closed_form import tau1_cardano, R as Rpost  # noqa: E402


def banner(x):
    print("\n" + "=" * 72)
    print(x)
    print("=" * 72)


TOL = 1e-9


def step_props(step):
    """(monotone, concave, no_overshoot) booleans for a step response array."""
    d1 = np.diff(step)
    d2 = np.diff(d1)
    monotone = bool(np.all(d1 >= -TOL))
    concave = bool(np.all(d2 <= TOL))
    no_over = bool(step.max() <= 1 + TOL)
    return monotone, concave, no_over


# ---------------------------------------------------------------------------
# (a) FIR uniform-advance family
# ---------------------------------------------------------------------------
def fir_kernel_by_lag(p, a):
    """Uniform-window OLS line advanced by a; kernel as a function of lag j."""
    kbar = (p - 1) / 2
    sig2 = (p**2 - 1) / 12
    j = np.arange(p)  # lag: 0 = newest
    return (1 / p) * (1 + a * (kbar - j) / sig2)


def fir_step(p, a, n=None):
    w = fir_kernel_by_lag(p, a)
    if n is None:
        n = 3 * p
    # step response of an FIR filter: s_n = sum_{j=0}^{min(n,p-1)} w[j]
    cs = np.cumsum(w)
    s = np.concatenate([cs, np.full(n - p, cs[-1])])
    return s


banner("(a) FIR uniform-advance family: three boundaries coincide at a_crit")
for p in (11, 21, 51):
    a_crit = (p + 1) / 6
    # find each boundary by bisection on a
    def largest_a(prop_idx):
        lo, hi = 0.0, 3 * a_crit
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if step_props(fir_step(p, mid))[prop_idx]:
                lo = mid
            else:
                hi = mid
        return lo
    a_mono = largest_a(0)
    a_conc = largest_a(1)
    a_over = largest_a(2)
    # r_crit^FIR = advance / (T/2) with T = p-1 (continuous lag ratio); here in
    # discrete lag units the maximal reduction is a_crit/kbar = (p+1)/(3(p-1)).
    kbar = (p - 1) / 2
    print("  p=%2d  a_crit=(p+1)/6=%.5f | mono=%.5f conc=%.5f over=%.5f | r_crit^FIR=a/kbar=%.5f (->1/3)"
          % (p, a_crit, a_mono, a_conc, a_over, a_crit / kbar))

# ---------------------------------------------------------------------------
# (b) Same-rate exponential family: (1+r) EMA - r EMA^2
# ---------------------------------------------------------------------------
from xpma.ema import EMA  # noqa: E402


def gdema_step(p, r, n=4000, warm=4000):
    e1, e2 = EMA(p), EMA(p)
    for _ in range(warm):
        v = e1.get_next(0.0)
        e2.get_next(v)
    out = []
    for _ in range(n):
        a1 = e1.get_next(1.0)
        a2 = e2.get_next(a1)
        out.append((1 + r) * a1 - r * a2)
    return np.array(out)


banner("(b) Same-rate exponential ((1+r)EMA - r EMA^2): all three fail for r>0")
print("  analytic kernel h(tau)=e^-tau(1+r-r tau); h'(tau)=e^-tau(r tau-(1+2r))")
print("  => concavity breaks at tau=2+1/r; step error sign change at n+1=1/(r alpha)")
print("  discrete p=20 (violations exponentially small but present):")
for r in (0.05, 0.1, 0.3, 0.56):
    step = gdema_step(20, r)
    mono, conc, over = step_props(step)
    print("    r=%.2f  monotone=%-5s concave=%-5s no_overshoot=%-5s  max_step=%.7f"
          % (r, mono, conc, over, step.max()))
# concavity sign-change location check (continuous)
print("  concavity re-acceleration point tau*=2+1/r (analytic):")
for r in (0.1, 0.3, 0.56):
    tau = np.linspace(0.01, 30, 600000)
    hp = np.exp(-tau) * (r * tau - (1 + 2 * r))
    # first minus->plus of h' after the initial region
    sc = np.where((hp[:-1] < 0) & (hp[1:] > 0))[0]
    loc = tau[sc[0]] if len(sc) else float("nan")
    print("    r=%.2f  measured h' up-cross at tau=%.4f   2+1/r=%.4f" % (r, loc, 2 + 1 / r))

# ---------------------------------------------------------------------------
# (c) Two-rate family: fully split
# ---------------------------------------------------------------------------
def r_crit_M(s):
    return (s ** (s + 1) / (s + 1) ** (s + 2)) * np.exp((2 * s + 1) / (s + 1))


def r_crit_O_s(s):
    """No-overshoot boundary: root tau_p of P_s(tau)=Q(tau)(s-(s+1)tau)+s^s tau^s/(s-1)!,
    then r = s! Q(tau_p) e^{tau_p}/((s+1)^{s+1} tau_p^s).  s=1..4."""
    from math import factorial, exp

    def Q(tau):
        return sum((s * tau) ** j / factorial(j) for j in range(s))

    def Ps(tau):
        return Q(tau) * (s - (s + 1) * tau) + s**s * tau**s / factorial(s - 1)

    # bracket the positive root in (0.5, 3)
    lo, hi = 0.5, 3.0
    flo, fhi = Ps(lo), Ps(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Ps(mid) * flo <= 0:
            hi = mid
        else:
            lo, flo = mid, Ps(mid)
    taup = 0.5 * (lo + hi)
    return factorial(s) * Q(taup) * exp(taup) / ((s + 1) ** (s + 1) * taup**s)


banner("(c) Two-rate family: 0 < r_crit^C < r_crit^M < r_crit^O  (fully split)")
print("   s |   r_crit^C    | r_crit^M   | r_crit^O   | ordering ok")
for s in (1, 2, 3, 4):
    rmax = float(Rpost(s, tau1_cardano(s)))
    rM = r_crit_M(s)
    rO = r_crit_O_s(s)
    ok = 0 < rmax < rM < rO
    print("  %2d | %.7f | %.7f | %.7f | %s" % (s, rmax, rM, rO, ok))

if __name__ == "__main__":
    pass
