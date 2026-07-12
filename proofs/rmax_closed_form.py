"""Closed form for tau_1(s) and r_crit^C(s), high-precision verification, the
non-monotone peak, and the s -> infinity behaviour of r_crit^C / r_crit^M.

The monotone-rate boundary of the base two-rate family XPMA^[s](p, r):

  tau_1(s) = the unique real root, beyond u_q := (s + sqrt(s))/(s+1), of
      P(u) = s(s+1)^2 u^3 - (s+1)(4s^2+s-1) u^2
             + (5s^3+s^2-4s-2) u - s(s-1)(2s+1) = 0.
  Cardano (disc(P) < 0 for real s > 1, so exactly one real root):
      depress by u = v - a2/(3 a3);  v^3 + P3 v + Q3 = 0;
      v = cbrt(-Q3/2 + sqrt(Delta)) + cbrt(-Q3/2 - sqrt(Delta)),  Delta = (Q3/2)^2 + (P3/3)^3 > 0.
  r_crit^C(s) = R(tau_1) = (s/(s+1))^{s+1} e^{tau_1} (s tau_1 - s + 1) / q(tau_1),
      q(u) = s(s-1) - 2s(s+1)u + (s+1)^2 u^2.

At s = 1: tau_1 = 2 exactly, r_crit^C = e^2/16.

Run: python proofs/rmax_closed_form.py
"""

import mpmath as mp

mp.mp.dps = 50


def coeffs(s):
    a3 = s * (s + 1) ** 2
    a2 = -(s + 1) * (4 * s**2 + s - 1)
    a1 = 5 * s**3 + s**2 - 4 * s - 2
    a0 = -s * (s - 1) * (2 * s + 1)
    return a3, a2, a1, a0


def q_poly(s, u):
    return s * (s - 1) - 2 * s * (s + 1) * u + (s + 1) ** 2 * u**2


def R(s, u):
    """Binding ratio R(u) = r for which h' has a root at u on the post-mode branch."""
    return (s / (s + 1)) ** (s + 1) * mp.e**u * (s * u - s + 1) / q_poly(s, u)


def u_q(s):
    """Lower edge of the post-mode binding region: larger root of q."""
    return (s + mp.sqrt(s)) / (s + 1)


def tau1_cardano(s):
    """Real root of P beyond u_q via Cardano (disc<0 branch, real cube roots)."""
    s = mp.mpf(s)
    if s == 1:
        return mp.mpf(2)
    a3, a2, a1, a0 = coeffs(s)
    # depressed cubic v^3 + P3 v + Q3, u = v - a2/(3 a3)
    shift = a2 / (3 * a3)
    P3 = (3 * a3 * a1 - a2**2) / (3 * a3**2)
    Q3 = (2 * a2**3 - 9 * a3 * a2 * a1 + 27 * a3**2 * a0) / (27 * a3**3)
    Delta = (Q3 / 2) ** 2 + (P3 / 3) ** 3
    assert Delta > 0, "expected one real root (Delta>0) for s>1"
    sq = mp.sqrt(Delta)
    v = mp.cbrt(-Q3 / 2 + sq) + mp.cbrt(-Q3 / 2 - sq)
    return v - shift


def tau1_direct(s):
    """Independent root find of P beyond u_q, for cross-check."""
    s = mp.mpf(s)
    a3, a2, a1, a0 = coeffs(s)
    P = lambda u: ((a3 * u + a2) * u + a1) * u + a0
    return mp.findroot(P, (mp.mpf("2.0"), mp.mpf("2.2")), solver="anderson",
                       tol=mp.mpf("1e-45")) if s > 1 else mp.mpf(2)


def rmax(s):
    return R(s, tau1_cardano(s))


def r_crit_M(s):
    s = mp.mpf(s)
    return (s ** (s + 1) / (s + 1) ** (s + 2)) * mp.e ** ((2 * s + 1) / (s + 1))


def banner(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def banner(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def main():
    import numpy as np

    banner("1.  Closed form vs reference targets (dps=50)")
    print("  s |   r_crit^C (Cardano)      tau_1 (Cardano)   | Cardano vs direct | P(tau_1)")
    for s in (1, 2, 3, 4):
        t1 = tau1_cardano(s)
        t1d = tau1_direct(s)
        rm = R(s, t1)
        a3, a2, a1, a0 = coeffs(mp.mpf(s))
        Pval = ((a3 * t1 + a2) * t1 + a1) * t1 + a0
        print("  %d | %s   %s | tau diff %.2e | %+.1e"
              % (s, mp.nstr(rm, 12), mp.nstr(t1, 12), float(abs(t1 - t1d)), float(Pval)))
    print("\n  s=1 exact checks:")
    print("   r_crit^C(1) - e^2/16 =", mp.nstr(rmax(1) - mp.e**2 / 16, 5))
    print("   tau_1(1) - 2      =", mp.nstr(tau1_cardano(1) - 2, 5))

    banner("2.  Post-mode ordering  u_q < tau_1  and R'(tau_1)=0 (interior min)")
    for s in (1, 2, 3, 4, 8, 20):
        t1 = tau1_cardano(s)
        uq = u_q(mp.mpf(s))
        dR = mp.diff(lambda u: R(mp.mpf(s), u), t1)
        d2R = mp.diff(lambda u: R(mp.mpf(s), u), t1, 2)
        print("  s=%2d  u_q=%s < tau_1=%s   R'(tau_1)=%+.1e  R''(tau_1)=%+.3f (min if >0)"
              % (s, mp.nstr(uq, 8), mp.nstr(t1, 8), float(dR), float(d2R)))

    banner("3.  Ratio r_crit^C / r_crit^M and the non-monotone peak of r_crit^C(s)")
    print("  s      r_crit^C         r_crit^M      ratio         tau_1")
    for s in [1, 2, 3, 4, 5, 6, 8, 10, 20, 50, 100, 200, 500, 1000]:
        rm = rmax(s)
        rc = r_crit_M(s)
        print("  %-5d  %s  %s  %s  %s"
              % (s, mp.nstr(rm, 8), mp.nstr(rc, 8), mp.nstr(rm / rc, 8),
                 mp.nstr(tau1_cardano(s), 8)))

    banner("3b.  Peak of r_crit^C(s) over real s (maximiser)")
    grid = np.linspace(1.2, 3.5, 24)
    best = max((float(rmax(mp.mpf(g))), g) for g in grid)
    peak = mp.findroot(lambda sv: mp.diff(lambda x: rmax(x), sv), mp.mpf(best[1]))
    print("  coarse-grid best near s =", round(best[1], 3), " r_crit^C =", round(best[0], 8))
    print("  refined peak: s* =", mp.nstr(peak, 10), " r_crit^C(s*) =", mp.nstr(rmax(peak), 10))
    print("  integer maximum is s = 2 (0.4687); continuous peak s* ~ 1.484")

    banner("4.  s -> infinity:  tau_1 -> 2,  r_crit^C/r_crit^M -> 1 (from below)")
    print("  s        tau_1 - 2        s*(tau_1-2)     1 - ratio        s^2*(1-ratio)")
    for s in [10, 20, 50, 100, 200, 500, 1000, 5000, 20000]:
        t1 = tau1_cardano(s)
        ratio = rmax(s) / r_crit_M(s)
        print("  %-6d  %s   %s   %s   %s"
              % (s, mp.nstr(t1 - 2, 8), mp.nstr(s * (t1 - 2), 8),
                 mp.nstr(1 - ratio, 8), mp.nstr(s**2 * (1 - ratio), 8)))
    print("  (analytic: tau_1 = 2 + 1/s - 8/s^2 + ...; ratio = 1 - 2/s^2 + ...;")
    print("   see rmax_asymptotics.py for the exact series)")


if __name__ == "__main__":
    main()
