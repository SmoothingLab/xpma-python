import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v]="1"
import sympy as sp

n, L = sp.symbols('n L', positive=True)

def check_reduction(s):
    s = sp.Integer(s)
    al_s  = s/(L+s);       be_s  = 1-al_s
    al_s1 = (s+1)/(L+s+1); be_s1 = 1-al_s1
    # kernels via negative binomial (Gamma for symbolic n)
    a_n   = sp.binomial(n+s-1, s-1)*al_s**s   * be_s**n
    b_n   = sp.binomial(n+s,   s)  *al_s1**(s+1)*be_s1**n
    b_nm1 = b_n.subs(n, n-1)
    Rn = sp.simplify(a_n/(L*(b_nm1 - b_n)))
    # claimed closed form
    rho = be_s/be_s1
    n0  = s*L/(s+1)
    P   = s*al_s**s*be_s1/(L*al_s1**(s+2))
    Rn_claim = sp.simplify(P*rho**n/(n-n0))
    diff = sp.simplify(Rn - Rn_claim)
    # P closed form check: P == s^{s+1}/(s+1)^{s+2} * (M+1)^{s+1}/M^s ,  M=L+s
    M = L+s
    P_claim = s**(s+1)/(s+1)**(s+2) * (M+1)**(s+1)/M**s
    Pdiff = sp.simplify(P - P_claim)
    return diff, Pdiff

for s in [1,2,3,4]:
    d, pd = check_reduction(s)
    print(f"s={s}:  R_n - P rho^n/(n-n0) = {d}    ;   P - closedform = {pd}")

# Symbolic: continuous min of R(x)=P rho^x/(x-n0) is P e ln(rho) rho^{n0}
x = sp.symbols('x', positive=True)
P, rho, n0 = sp.symbols('P rho n0', positive=True)
Rx = P*rho**x/(x-n0)
dR = sp.diff(sp.log(Rx), x)
xstar = sp.solve(sp.Eq(dR,0), x)
print("\nContinuous minimiser x* solves d/dx ln R =0 :", xstar)
xs = n0 + 1/sp.log(rho)
Rmin = sp.simplify(Rx.subs(x, xs))
print("R(x*) simplified:", Rmin, "  == P*e*ln(rho)*rho^n0 ?",
      sp.simplify(Rmin - P*sp.E*sp.log(rho)*rho**n0)==0)

# Symbolic: g clean form == A - mu C
xx, mu = sp.symbols('x mu', positive=True)
phi = xx/(sp.exp(xx)-1)
g   = sp.log(phi) + (2-mu)*xx + (1-mu)*(phi-1)
A   = sp.log(phi) + 2*xx + phi - 1
C   = xx + phi - 1
print("\ng - (A - mu*C) simplifies to:", sp.simplify(g - (A - mu*C)))

# Symbolic small-x leading coeff of g at mu: (from clean) = (2- mu - (1-mu)/2 -1/2)?? just series
ser = sp.series(g, xx, 0, 2).removeO()
print("g series to O(x):", sp.simplify(ser), "  (coeff of x should be 1 - mu/2)")
print("coeff x:", sp.simplify(ser.coeff(xx,1)), " ; 1-mu/2 =", sp.simplify(1-mu/2))
