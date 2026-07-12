"""Symbolic derivation of the r_crit^C (monotone-rate) boundary of the base
two-rate family XPMA^[s](p, r), continuous limit.

Establishes, purely symbolically (sympy), the objects the closed form needs:

  1. The sign function of h'(tau): F(tau) = c1 A(tau) + r c2 e^{-tau} q(tau),
     with A(tau) = (s-1) - s tau,  q(tau) = s(s-1) - 2s(s+1)tau + (s+1)^2 tau^2.
  2. The binding-ratio R(u) on the post-mode region (A < 0, q > 0):
         R(u) = (s/(s+1))^{s+1} e^u (s u - s + 1) / q(u).
  3. The stationarity cubic P(u) = 0 whose relevant root is tau_1(s), via
     d(ln R)/du = 0.  Verify it equals the head-start cubic
         P(u) = s(s+1)^2 u^3 - (s+1)(4s^2+s-1)u^2
                + (5s^3+s^2-4s-2)u - s(s-1)(2s+1).
  4. Exact identities: P(2) = -s(s-1), P'(2) = s^3+5s^2+8s+2,
     s=1 degeneracy P = 4u^2(u-2), large-s limiting cubic (u-1)^2(u-2).
  5. Cardano closed form for the relevant real root, and the cubic discriminant
     sign as a function of s.

Run: python proofs/rmax_symbolic.py
"""

import sympy as sp

s, u, r = sp.symbols("s u r", positive=True)


def banner(txt):
    print("\n" + "=" * 72)
    print(txt)
    print("=" * 72)


# ---------------------------------------------------------------------------
# 1. Kernel derivative sign function F(tau)
# ---------------------------------------------------------------------------
banner("1.  h1', h2'' and the sign function F(tau) of h'(tau)")

fac = sp.factorial
c1_full = s**s / fac(s - 1)          # h1(t) = c1_full t^{s-1} e^{-s t}
c2_full = (s + 1) ** (s + 1) / fac(s)  # h2(t) = c2_full t^{s} e^{-(s+1)t}

t = sp.symbols("t", positive=True)
h1 = c1_full * t ** (s - 1) * sp.exp(-s * t)
h2 = c2_full * t ** s * sp.exp(-(s + 1) * t)

h1p = sp.diff(h1, t)
h2pp = sp.diff(h2, t, 2)

# Proposed factorisations
A = (s - 1) - s * t
q = s * (s - 1) - 2 * s * (s + 1) * t + (s + 1) ** 2 * t**2
c1 = s**s / fac(s - 1)
c2 = (s + 1) ** (s + 1) / fac(s)

h1p_claim = c1 * t ** (s - 2) * sp.exp(-s * t) * A
h2pp_claim = c2 * t ** (s - 2) * sp.exp(-(s + 1) * t) * q

print("h1' - claim  simplifies to:", sp.simplify(h1p - h1p_claim))
print("h2'' - claim simplifies to:", sp.simplify(h2pp - h2pp_claim))

# c1/c2
print("c1/c2 =", sp.simplify(c1 / c2), " (expect s^(s+1)/(s+1)^(s+1))")
print("check :", sp.simplify(c1 / c2 - s ** (s + 1) / (s + 1) ** (s + 1)))

# q roots
qr = sp.solve(q, t)
print("q(tau) roots:", [sp.simplify(x) for x in qr], " (expect (s +- sqrt s)/(s+1))")

# ---------------------------------------------------------------------------
# 2. Binding ratio R(u) and stationarity -> cubic
# ---------------------------------------------------------------------------
banner("2.  Stationarity d(ln R)/du = 0  ->  cubic P(u)")

# R(u) = (c1/c2) e^u (-A)/q  with -A = s u - (s-1)
qu = s * (s - 1) - 2 * s * (s + 1) * u + (s + 1) ** 2 * u**2
negA = s * u - (s - 1)
lnR = u + sp.log(negA) - sp.log(qu)           # drop const (c1/c2)
dlnR = sp.together(sp.diff(lnR, u))
num_dlnR = sp.numer(dlnR)                       # numerator = 0 at stationarity
num_dlnR = sp.expand(num_dlnR)
print("numerator of d(ln R)/du (expanded):")
print("  ", num_dlnR)

# Head-start cubic
P_head = (
    s * (s + 1) ** 2 * u**3
    - (s + 1) * (4 * s**2 + s - 1) * u**2
    + (5 * s**3 + s**2 - 4 * s - 2) * u
    - s * (s - 1) * (2 * s + 1)
)
# The stationarity numerator should be -P_head (or a positive multiple).
ratio = sp.simplify(num_dlnR / P_head)
print("numerator / P_head =", ratio, " (expect a nonzero constant, sign fixes orientation)")

# Confirm the identity  (s u + 1) q(u) = 2(s+1)((s+1)u - s)(s u - s + 1)  matches P
lhs = (s * u + 1) * qu
rhs = 2 * (s + 1) * ((s + 1) * u - s) * (s * u - s + 1)
print("head-start identity  (su+1)q - 2(s+1)((s+1)u-s)(su-s+1) - P_head:",
      sp.simplify(sp.expand(lhs - rhs) - (-P_head)))

# ---------------------------------------------------------------------------
# 3. Exact identities at u = 2 and s = 1, and large-s
# ---------------------------------------------------------------------------
banner("3.  Anchors:  P(2), P'(2), s=1 degeneracy, large-s limit")

print("P(2)      =", sp.simplify(P_head.subs(u, 2)), " (expect -s(s-1))")
Pp = sp.diff(P_head, u)
print("P'(2)     =", sp.expand(Pp.subs(u, 2)), " (expect s^3+5s^2+8s+2)")

P_s1 = sp.expand(P_head.subs(s, 1))
print("P(u; s=1) =", P_s1, " factor:", sp.factor(P_s1), " (expect 4u^2(u-2))")

# large s: divide by s^3, let s->oo
P_over_s3 = sp.expand(P_head / s**3)
lim_cubic = sp.limit(P_over_s3, s, sp.oo)
print("lim_{s->oo} P/s^3 =", sp.expand(lim_cubic), " factor:", sp.factor(lim_cubic),
      " (expect (u-1)^2(u-2))")

# ---------------------------------------------------------------------------
# 4. Cardano closed form and discriminant sign
# ---------------------------------------------------------------------------
banner("4.  Cardano root and discriminant of P(u)")

a3 = s * (s + 1) ** 2
a2 = -(s + 1) * (4 * s**2 + s - 1)
a1 = 5 * s**3 + s**2 - 4 * s - 2
a0 = -s * (s - 1) * (2 * s + 1)
print("coeffs: a3=%s  a2=%s  a1=%s  a0=%s" % (a3, sp.expand(a2), a1, sp.expand(a0)))

disc = sp.discriminant(P_head, u)
disc = sp.factor(disc)
print("discriminant (factored):")
print("  ", disc)
# Evaluate discriminant sign for a range of s
for sv in [sp.Rational(3, 2), 2, 3, 4, 5, 10, 50]:
    dv = sp.discriminant(P_head.subs(s, sv), u)
    print("  s=%s  disc=%s  (neg => one real root)" % (sv, sp.nsimplify(sp.N(dv))))

# Cardano for the single real root (depressed cubic)
banner("4b.  Symbolic real root via sympy (for reference)")
roots_s2 = sp.solve(sp.Poly(P_head.subs(s, 2), u), u)
print("s=2 roots (one real):")
for rt in roots_s2:
    print("   ", sp.N(rt, 12))

if __name__ == "__main__":
    pass
