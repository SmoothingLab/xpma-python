"""Symbolic backing for the pre-mode non-binding proof.

Claim: on the pre-mode region (u_-, (s-1)/s), where u_- = (s-sqrt s)/(s+1),
the zero-set curve of F,
    R2(tau) = (c1/c2) e^tau A(tau) / (-q(tau)),   A = (s-1)-s tau > 0,  q < 0,
is strictly decreasing.  Hence F has exactly one zero there (the kernel mode)
for every r >= 0, so no extra sign-change pair is ever born before the mode.

Proof reduces to  d(ln R2)/dtau = 1 - s/A - q'/q < 0  on the region, via:
  (i)  0 < A < s-1  =>  s/A > s/(s-1) > 1,
  (ii) q < 0 and q' = 2(s+1)((s+1)tau - s) < 0 (since tau < (s-1)/s < s/(s+1))
       =>  q'/q > 0,
so  1 - s/A - q'/q < 1 - 1 - 0 = 0.  This script confirms each sign fact
symbolically and confirms d(ln R2)/dtau = 1 - s/A - q'/q.

Run: python proofs/rmax_premode_proof.py
"""

import sympy as sp

s, t = sp.symbols("s t", positive=True)

A = (s - 1) - s * t
q = s * (s - 1) - 2 * s * (s + 1) * t + (s + 1) ** 2 * t**2
# R2 up to the positive constant c1/c2 (drops out of d ln R2/dt)
lnR2 = t + sp.log(A) - sp.log(-q)
dlnR2 = sp.simplify(sp.diff(lnR2, t))
claim = 1 - s / A - sp.diff(q, t) / q
print("d(ln R2)/dt - (1 - s/A - q'/q) simplifies to:",
      sp.simplify(dlnR2 - claim), "  (expect 0)")

# (i) A < s-1 on the region (tau>0): A = (s-1) - s t < s-1 for t>0.  s/A > s/(s-1) > 1.
print("A(0) = s-1 (max of A at left):", A.subs(t, 0))
print("s/(s-1) > 1 for s>1: s/(s-1) - 1 =", sp.simplify(s / (s - 1) - 1), " (>0)")

# (ii) q' sign on region: q' = 2(s+1)((s+1)t - s), negative iff t < s/(s+1).
qp = sp.factor(sp.diff(q, t))
print("q' factored:", qp, " => q'<0 iff t < s/(s+1)")
print("region right edge (s-1)/s < s/(s+1)?  s/(s+1) - (s-1)/s =",
      sp.simplify(s / (s + 1) - (s - 1) / s), " (>0 for s>1)")

# q < 0 strictly between its roots u_- and u_+; region (u_-,(s-1)/s) lies inside.
u_minus = (s - sp.sqrt(s)) / (s + 1)
u_plus = (s + sp.sqrt(s)) / (s + 1)
print("q(u_-)=", sp.simplify(q.subs(t, u_minus)), " q(u_+)=", sp.simplify(q.subs(t, u_plus)))
print("(s-1)/s in (u_-,u_+):  (s-1)/s - u_- =", sp.simplify((s - 1) / s - u_minus),
      "  u_+ - (s-1)/s =", sp.simplify(u_plus - (s - 1) / s))

# numeric sanity: d(ln R2)/dt < 0 at sample interior points for several s
print("\nnumeric d(ln R2)/dt on interior sample points (all must be < 0):")
for sv in (2, 3, 5, 10, 50):
    um = float((sv - sv**0.5) / (sv + 1))
    a0 = (sv - 1) / sv
    for frac in (0.25, 0.5, 0.75):
        tv = um + (a0 - um) * frac
        val = float(claim.subs({s: sv, t: tv}))
        assert val < 0, (sv, tv, val)
    print("  s=%2d  ok (interior d(lnR2)/dt < 0)" % sv)

if __name__ == "__main__":
    pass
