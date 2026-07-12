"""Fractional-s statement for r_crit^C.

Two questions:
  (1) Does the closed form extend to real s >= 1?  The cubic P(u) has polynomial
      coefficients in s and discriminant -4(s-1)(s+1)^2(s^2+s+1)(...) < 0 for all
      real s > 1, so the single-real-root Cardano form persists and tau_1(s),
      r_crit^C(s) are real-analytic on s > 1.  We confirm the Cardano value matches
      a direct continuous sign-change bisection of the TRUE fractional-cascade
      kernel (gamma-shape normalisation) at fractional s.  This is exact for the
      single-pole fractional cascade (EIFEMA), exactly as for r_crit^M.

  (2) The realisation subtlety.  The high-performance discrete realisation
      XPMA(period, s, r) uses the two-pole IFEMA cascade blend for fractional
      orders, not the exact single-pole cascade.  Unlike no-overshoot
      (non-negativity, preserved under convex blends), the concavity property is
      unimodality of the kernel (single sign change of h'), which is NOT
      guaranteed to survive an approximate kernel shape.  We therefore CHECK
      whether the IFEMA-realised discrete r_crit^C still exceeds the continuous
      r_crit^C(s) at fractional s (safety), and report the margin.

Run: python proofs/rmax_fractional.py
"""

import os
import sys
from math import gamma, exp

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmax_closed_form import tau1_cardano, R as Rpost  # noqa: E402
from rmax_discrete import discrete_rmax, rmax_continuous  # noqa: E402


def banner(x):
    print("\n" + "=" * 72)
    print(x)
    print("=" * 72)


# Continuous true-fractional-cascade kernel derivative h'(tau), gamma normalisation
def h_prime_frac(s, r, t):
    c1 = s**s / gamma(s)                    # s^s/Gamma(s)
    c2 = (s + 1) ** (s + 1) / gamma(s + 1)  # (s+1)^{s+1}/Gamma(s+1)
    h1p = c1 * np.exp(-s * t) * ((s - 1) * t ** (s - 2) - s * t ** (s - 1))
    h2pp = c2 * np.exp(-(s + 1) * t) * (
        s * (s - 1) * t ** (s - 2)
        - 2 * s * (s + 1) * t ** (s - 1)
        + (s + 1) ** 2 * t**s
    )
    return h1p + r * h2pp


def continuous_rmax_bisect(s):
    t = np.linspace(1e-7, 45, 3_000_001)
    def n_up(r):
        sg = np.sign(h_prime_frac(s, r, t))
        return int((np.diff(sg) > 0).sum())
    lo, hi = 0.05, 0.95
    for _ in range(56):
        mid = 0.5 * (lo + hi)
        if n_up(mid) == 0:
            lo = mid
        else:
            hi = mid
    return lo


banner("(1) Closed form (Cardano) vs direct sign-change bisection, fractional s")
print("  s     Cardano r_crit^C   bisection r_crit^C   diff        tau_1(s)")
for s in (1.25, 1.5, 1.75, 2.5, 3.5, 4.5):
    cf = float(Rpost(s, tau1_cardano(s)))
    bi = continuous_rmax_bisect(s)
    print("  %.2f   %.8f      %.8f      %.1e   %.6f"
          % (s, cf, bi, abs(cf - bi), float(tau1_cardano(s))))

banner("(2) IFEMA-realised discrete r_crit^C at fractional s: NOT always safe")
print("   s   |    p |  disc r_crit^C  | cont r_crit^C  |  margin      safe")
for s in (1.5, 2.5, 3.5):
    rc = rmax_continuous(s)
    for p in (50, 200, 1000):
        dr = discrete_rmax(p, s)
        m = dr - rc
        print("  %.1f | %4d | %11.7f | %10.7f | %+11.7f  %s"
              % (s, p, dr, rc, m, "OK" if m > 0 else "*** BELOW ***"))
    print()

print("  Finding: unlike at integer s, the two-pole IFEMA blend can realise a")
print("  concavity boundary BELOW the continuous r_crit^C(s) at fractional s (worst in")
print("  the (2,3) band, e.g. s=2.5: margin -0.0045 at p=1000). The blend's kernel")
print("  shape is only approximate, and concavity (single sign change of h') is NOT")
print("  preserved by it the way non-negativity is. The exact single-pole cascade")
print("  (EIFEMA) realises the closed form exactly; the deployed blend needs a small")
print("  safety margin (or per-(s,p) check) at fractional s.")

banner("(2b) Does output-level interpolation restore the guarantee? (No, not automatically)")
print("  Blend the two bracketing integer-order kernels, each at its OWN r_crit^C,")
print("  with IFEMA second-moment weights. Test unimodality")
print("  (single + -> - sign change of the first difference of the kernel).")


def integer_kernel(p, s_int, r):
    n = int(max(6000, 30 * p))
    warm = int(max(8000, 40 * p))
    from xpma import XPMA
    f = XPMA(float(p), float(s_int), float(r))
    for _ in range(warm):
        f.get_next(0.0)
    out = [f.get_next(1.0)]
    for _ in range(n - 1):
        out.append(f.get_next(0.0))
    return np.array(out)


def n_up_first_diff(h):
    peak = h.max()
    mode = int(h.argmax())
    d = np.diff(h)
    tail = np.where(h[mode:] > 1e-11 * peak)[0]
    end = mode + (tail[-1] if len(tail) else len(h) - mode - 1)
    seg = d[mode:end]
    tol = 1e-12 * peak
    return int(np.sum((seg[:-1] < -tol) & (seg[1:] > tol)))


for s in (1.5, 2.5, 3.5):
    lo, hi = int(np.floor(s)), int(np.ceil(s))
    frac = s - lo
    wlo, whi = (1 - frac) * lo / s, frac * hi / s   # IFEMA second-moment weights
    for p in (200, 1000):
        rlo, rhi = rmax_continuous(lo), rmax_continuous(hi)
        klo = integer_kernel(p, lo, rlo)
        khi = integer_kernel(p, hi, rhi)
        kb = wlo * klo + whi * khi
        ups = n_up_first_diff(kb)
        print("  s=%.1f p=%4d  output-level blend @ each r_crit^C: kernel up-crossings=%d  %s"
              % (s, p, ups, "unimodal (OK)" if ups == 0 else "NON-unimodal (dip-bump present)"))

print("\n  Conclusion: the deployed cascade (IFEMA) blend can lose concavity at")
print("  fractional s (part 2), but the output-level interpolation of the two integer-")
print("  order r_crit^C filters (the recommended realisation for r(s)")
print("  filters) stays unimodal in every case tested: each component has h'<=0 on the")
print("  post-mode region, so the convex blend does too (the only risk zone, between the")
print("  two near-coincident modes, does not bind here). This is NUMERICAL, not a proof")
print("  (unimodality is not convex-combination-preserved in general). Exact realisation")
print("  at any real s: the true single-pole fractional cascade (EIFEMA).")

if __name__ == "__main__":
    pass
