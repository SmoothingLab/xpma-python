"""Analytic s -> infinity expansion of tau_1(s) and r_crit^C(s)/r_crit^M(s).

Sets sigma = 1/s and expands the cubic root tau_1 = 2 + c1 sigma + c2 sigma^2
+ ... then the ratio r_crit^C/r_crit^M = 1 + d1 sigma + d2 sigma^2 + ...  The
numeric harness rmax_closed_form.py suffers catastrophic cancellation in
1 - ratio at very large s; this script gets the exact rational coefficients
symbolically so no cancellation is involved.

Run: python proofs/rmax_asymptotics.py
"""

import sympy as sp

sig = sp.symbols("sigma", positive=True)   # sigma = 1/s
w = sp.symbols("w")                          # tau_1 = 2 + w
ORD = 6


def banner(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


# Cubic in terms of s = 1/sigma. Multiply through by sigma^3 to keep it a
# polynomial with a finite sigma->0 limit ( (u-2)(u-1)^2 ).
s = 1 / sig
u = 2 + w
a3 = s * (s + 1) ** 2
a2 = -(s + 1) * (4 * s**2 + s - 1)
a1 = 5 * s**3 + s**2 - 4 * s - 2
a0 = -s * (s - 1) * (2 * s + 1)
P = a3 * u**3 + a2 * u**2 + a1 * u + a0
P = sp.expand(P * sig**3)   # polynomial in sigma and w, finite at sigma=0

banner("1.  tau_1 = 2 + w(sigma) expansion (sigma = 1/s)")
# Solve order by order: w = c1 sigma + c2 sigma^2 + ...
cs = sp.symbols("c1:%d" % (ORD + 1))
wser = sum(cs[i] * sig ** (i + 1) for i in range(ORD))
Psub = sp.series(P.subs(w, wser), sig, 0, ORD + 1).removeO()
Psub = sp.expand(Psub)
sol = {}
for k in range(1, ORD + 1):
    coeff = Psub.coeff(sig, k).subs(sol)
    c = sp.solve(coeff, cs[k - 1])
    if c:
        sol[cs[k - 1]] = sp.simplify(c[0])
wexpr = sum(sol.get(cs[i], 0) * sig ** (i + 1) for i in range(ORD))
print("  tau_1 - 2 =", wexpr, " + O(sigma^%d)" % (ORD + 1))
print("  i.e. tau_1 = 2 + 1/s + c2/s^2 + ... ,  c1 =", sol[cs[0]], " c2 =", sol[cs[1]])

# ---------------------------------------------------------------------------
banner("2.  Ratio r_crit^C / r_crit^M = 1 + ... expansion")
# r_crit^C = R(tau_1) = (s/(s+1))^{s+1} e^{tau_1} (s tau_1 - s + 1)/q(tau_1)
# r_crit^M = (s^{s+1}/(s+1)^{s+2}) e^{(2s+1)/(s+1)}
# ratio = (s+1) e^{tau_1 - (2s+1)/(s+1)} (s tau_1 - s + 1)/q(tau_1)
tau1 = 2 + wexpr
q = s * (s - 1) - 2 * s * (s + 1) * tau1 + (s + 1) ** 2 * tau1**2
expo = tau1 - (2 * s + 1) / (s + 1)
ratio = (s + 1) * sp.exp(expo) * (s * tau1 - s + 1) / q
rseries = sp.series(ratio, sig, 0, 4).removeO()
rseries = sp.simplify(rseries)
print("  ratio ~", sp.nsimplify(sp.expand(rseries)))
# leading correction
diff1 = sp.simplify(sp.limit((ratio - 1) / sig, sig, 0))
diff2 = sp.simplify(sp.limit((ratio - 1) / sig**2, sig, 0))
print("  lim (ratio-1)/sigma   =", diff1, "  (coeff of 1/s)")
print("  lim (ratio-1)/sigma^2 =", diff2, "  (coeff of 1/s^2)")
print("  => 1 - r_crit^C/r_crit^M ~ %s / s^2  (approaches 1 from below)" % (-diff2))

if __name__ == "__main__":
    pass
