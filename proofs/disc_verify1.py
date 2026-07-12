import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
import mpmath as mp
mp.mp.dps = 60

def rcrit_M(s):
    s = mp.mpf(s)
    return (s**(s+1) / (s+1)**(s+2)) * mp.e**((2*s+1)/(s+1))

def kernels(s, p, N):
    """Return a_n, b_n arrays for n=0..N-1 at high precision.
    a = s-stage cascade, per-stage alpha_s = s/(L+s); L=(p-1)/2.
    b = (s+1)-stage cascade, per-stage alpha_{s+1}=(s+1)/(L+s+1).
    negative-binomial: m-stage w_n = C(n+m-1,m-1) alpha^m beta^n."""
    s = int(s)
    L = (mp.mpf(p) - 1) / 2
    al_s   = s / (L + s)
    be_s   = 1 - al_s
    al_s1  = (s+1) / (L + s + 1)
    be_s1  = 1 - al_s1
    a = []
    b = []
    for n in range(N):
        # a_n = C(n+s-1, s-1) al_s^s be_s^n
        an = mp.binomial(n+s-1, s-1) * al_s**s * be_s**n
        bn = mp.binomial(n+s,   s)   * al_s1**(s+1) * be_s1**n
        a.append(an); b.append(bn)
    return a, b, L, al_s, be_s, al_s1, be_s1

def disc_boundary(s, p, N=None):
    """True discrete boundary r_crit^disc = min_{n: D_n>0} a_n/(L D_n)."""
    s = int(s)
    L = (mp.mpf(p) - 1) / 2
    if N is None:
        # need enough n to cover the minimum; min near n0 + 1/ln(rho)
        N = int(mp.nint(2 * (s*L/(s+1)) + 50/ max(1e-9, float(mp.log(1+1/(L+s)))) )) + 200
        N = max(N, 400)
    a, b, L, al_s, be_s, al_s1, be_s1 = kernels(s, p, N)
    n0 = s*L/(s+1)
    best = None
    argbest = None
    for n in range(1, N):
        Dn = b[n-1] - b[n]
        if Dn > 0:
            Rn = a[n] / (L * Dn)
            if best is None or Rn < best:
                best = Rn; argbest = n
    return best, argbest, n0, N

def R_formula(s, p, n):
    """R_n via the closed reduction:
       R_n = [s al_s^s be_s1 / (L al_s1^{s+2})] * rho^n / (n - n0),  rho=be_s/be_s1"""
    s = int(s)
    L = (mp.mpf(p) - 1) / 2
    al_s   = s / (L + s); be_s = 1 - al_s
    al_s1  = (s+1)/(L+s+1); be_s1 = 1 - al_s1
    rho = be_s / be_s1
    n0 = s*L/(s+1)
    P = s * al_s**s * be_s1 / (L * al_s1**(s+2))
    return P * rho**n / (n - n0)

def R_cont_min(s, p):
    """Continuous relaxation min of R(x)=P rho^x/(x-n0):
       = P * e * ln(rho) * rho^{n0}, at x*=n0+1/ln(rho)."""
    s = int(s)
    L = (mp.mpf(p) - 1) / 2
    al_s   = s / (L + s); be_s = 1 - al_s
    al_s1  = (s+1)/(L+s+1); be_s1 = 1 - al_s1
    rho = be_s / be_s1
    n0 = s*L/(s+1)
    P = s * al_s**s * be_s1 / (L * al_s1**(s+2))
    return P * mp.e * mp.log(rho) * rho**n0, n0 + 1/mp.log(rho)

print("=== Base fact reproduction: discrete boundary vs r_crit^M, and R-formula check ===")
print(f"{'s':>2} {'p':>5} {'r*=rcritM':>14} {'disc_bnd':>14} {'disc-r*':>12} {'argmin':>7} "
      f"{'R_cont_min':>14} {'Rc-r*':>12} {'formula_ok':>10}")
for s in (1,2,3,4):
    for p in (5,10,20,50,200):
        rstar = rcrit_M(s)
        db, arg, n0, N = disc_boundary(s, p)
        # cross-check R formula vs direct at the argmin
        a,b,L,_,_,_,_ = kernels(s,p, arg+3)
        Rn_direct = a[arg]/(L*(b[arg-1]-b[arg]))
        Rn_form   = R_formula(s,p,arg)
        formula_ok = abs(Rn_direct - Rn_form) < mp.mpf(10)**(-40)
        rc, xstar = R_cont_min(s,p)
        print(f"{s:>2} {p:>5} {mp.nstr(rstar,10):>14} {mp.nstr(db,10):>14} "
              f"{mp.nstr(db-rstar,6):>12} {arg:>7} {mp.nstr(rc,10):>14} "
              f"{mp.nstr(rc-rstar,6):>12} {str(bool(formula_ok)):>10}")
