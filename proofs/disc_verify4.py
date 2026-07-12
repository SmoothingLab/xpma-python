import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import mpmath as mp
mp.mp.dps = 60

def rcrit_M(s):
    s = mp.mpf(s)
    return (s**(s+1) / (s+1)**(s+2)) * mp.e**((2*s+1)/(s+1))

def phi(x):
    x = mp.mpf(x)
    return mp.mpf(1) if x==0 else x/(mp.e**x - 1)

# ---------- GROUND TRUTH: direct kernel h_n(r*) >= 0 for all n ----------
def ground_truth_min_h(s, p, r):
    """Directly build negative-binomial kernels a_n, b_n and check min_n h_n(r)."""
    s = int(s)
    L = (mp.mpf(p)-1)/2
    al_s  = s/(L+s);      be_s  = 1-al_s
    al_s1 = (s+1)/(L+s+1); be_s1 = 1-al_s1
    # choose N large enough to pass the minimum (near n0 + 1/ln rho) with margin
    rho = be_s/be_s1; n0 = s*L/(s+1)
    N = int(mp.nint(n0 + 40/mp.log(rho))) + 400
    N = max(N, 600)
    minh = None; argmin=None
    b_prev = mp.mpf(0)  # b_{-1}=0
    for n in range(N):
        an = mp.binomial(n+s-1, s-1) * al_s**s * be_s**n
        bn = mp.binomial(n+s,   s)   * al_s1**(s+1) * be_s1**n
        hn = an + r*L*(bn - b_prev)   # = a_n + r L (b_n - b_{n-1})
        if minh is None or hn < minh: minh = hn; argmin=n
        b_prev = bn
    return minh, argmin, N

print("=== GROUND TRUTH: min_n h_n(r*) directly from kernels; must be >= 0 ===")
print(f"{'s':>3} {'p':>9} {'min h_n(r*)':>16} {'argmin':>7} {'>=0?':>5}")
allok=True
for s in [1,2,3,4,6,10,20]:
    for p in [mp.mpf('1.01'), mp.mpf(2), mp.mpf(5), mp.mpf(20), mp.mpf(200), mp.mpf(10000)]:
        r = rcrit_M(s)
        mh, arg, N = ground_truth_min_h(s, p, r)
        ok = mh >= -mp.mpf(10)**(-45)   # tolerance for rounding
        allok = allok and ok
        print(f"{s:>3} {mp.nstr(p,7):>9} {mp.nstr(mh,8):>16} {arg:>7} {str(bool(ok)):>5}")
print("ALL ground-truth h_n(r*)>=0:", allok)

# ---------- The reduction chain identities ----------
print("\n=== Chain identity: g(u,s) == ln(R_cont_min / r*) ===")
def R_cont_min_over_rstar(s,p):
    s=int(s); L=(mp.mpf(p)-1)/2
    al_s=s/(L+s); be_s=1-al_s; al_s1=(s+1)/(L+s+1); be_s1=1-al_s1
    rho=be_s/be_s1; n0=s*L/(s+1)
    P = s*al_s**s*be_s1/(L*al_s1**(s+2))
    Rc = P*mp.e*mp.log(rho)*rho**n0
    return Rc/rcrit_M(s)
def g_clean_xmu(s,p):
    s=int(s); L=(mp.mpf(p)-1)/2; M=L+s
    x=mp.log(1+1/M); mu=mp.mpf(1)/(s+1)
    return mp.log(phi(x)) + (2-mu)*x + (1-mu)*(phi(x)-1)
maxd=mp.mpf(0)
for s in [1,2,3,5,10,20]:
    for p in [mp.mpf('1.01'),mp.mpf(2),mp.mpf(7),mp.mpf(50),mp.mpf(500)]:
        d = abs(mp.log(R_cont_min_over_rstar(s,p)) - g_clean_xmu(s,p))
        maxd=max(maxd,d)
print("max |ln(Rc/r*) - g_clean| :", mp.nstr(maxd,6))

# ---------- C(x) > 0 ; linearity in mu ----------
print("\n=== C(x)=x+phi-1 > 0 on (0,ln2); and g = A - mu C (linear in mu) ===")
ln2=mp.log(2)
minC=None
for k in range(1,20000):
    x=ln2*mp.mpf(k)/20000
    C=x+phi(x)-1
    if minC is None or C<minC: minC=C
print("min C(x) on (0,ln2):", mp.nstr(minC,8), " (should be >0; note C>x/2)")
# linearity check: g(x,mu) vs A(x)-mu C(x)
def A(x): x=mp.mpf(x); return mp.log(phi(x))+2*x+phi(x)-1
def Cx(x): x=mp.mpf(x); return x+phi(x)-1
def g_xmu(x,mu): x=mp.mpf(x);mu=mp.mpf(mu); return mp.log(phi(x))+(2-mu)*x+(1-mu)*(phi(x)-1)
maxlin=mp.mpf(0)
for x in [mp.mpf('0.05'),mp.mpf('0.3'),mp.mpf('0.6')]:
    for mu in [mp.mpf('0.1'),mp.mpf('0.3'),mp.mpf('0.5')]:
        maxlin=max(maxlin, abs(g_xmu(x,mu)-(A(x)-mu*Cx(x))))
print("max |g - (A - mu C)| :", mp.nstr(maxlin,6))

# ---------- s=1 lemma bounds ----------
print("\n=== s=1 lemma: I(x)=g(x,1/2). Bounds: ln(phi)+x/2 >= -x^2/24 ; phi >= 1-x/2 ===")
print("   => I(x) >= 3x/4 - x^2/24 = x(18-x)/24 > 0 on (0,18) ===")
b1_ok=True; b2_ok=True; I_ok=True; tight=None
for k in range(1,40000):
    x=ln2*mp.mpf(k)/40000
    # bound 1
    lhs1 = mp.log(phi(x))+x/2
    if lhs1 < -x*x/24 - mp.mpf(10)**(-40): b1_ok=False
    # bound 2
    if phi(x) < 1 - x/2 - mp.mpf(10)**(-40): b2_ok=False
    # I(x) and the lower bound
    I = mp.log(phi(x)) + mp.mpf(3)/2*x + phi(x)/2 - mp.mpf(1)/2
    lb = 3*x/4 - x*x/24
    if I < lb - mp.mpf(10)**(-40): I_ok=False
    slack = I - lb
    if tight is None or slack<tight: tight=slack
print("bound1 (ln phi + x/2 >= -x^2/24) holds:", b1_ok)
print("bound2 (phi >= 1-x/2) holds:", b2_ok)
print("I(x) >= x(18-x)/24 holds:", I_ok, " min slack I-lb:", mp.nstr(tight,6))
# also confirm I>0 directly and I(ln2)
print("I(ln2)=", mp.nstr(mp.log(phi(ln2))+mp.mpf(3)/2*ln2+phi(ln2)/2-mp.mpf(1)/2, 8),
      "  lower bound x(18-x)/24 at ln2=", mp.nstr(3*ln2/4-ln2*ln2/24,8))
