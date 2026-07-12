import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import mpmath as mp
mp.mp.dps = 50

def phi(x):
    x = mp.mpf(x)
    if x == 0: return mp.mpf(1)
    return x/(mp.e**x - 1)

# clean form g(x,mu) = ln(phi) + (2-mu)x + (1-mu)(phi-1),  mu=1/(s+1), x=ln(1+1/(L+s))
def g_clean(x, mu):
    x = mp.mpf(x); mu = mp.mpf(mu)
    return mp.log(phi(x)) + (2-mu)*x + (1-mu)*(phi(x)-1)

# original g(u,s) (the messy but exact one) for cross-check
def g_u(u, s):
    u = mp.mpf(u); s = mp.mpf(s)
    l = mp.log(1+u)
    return mp.log(l/u) + (s+1)*l + s*(1-s*u)*l/((s+1)*u) - s/(s+1)

# ---- cross-check clean form vs original, and vs the direct R_cont_min>=r* ----
print("=== Cross-check: clean g(x,mu) == original g(u,s), for random (s,p) ===")
import random
random.seed(0)
maxdiff = mp.mpf(0)
for _ in range(200):
    s = random.choice([1,2,3,4,5,7,10])
    p = mp.mpf(1) + mp.mpf(random.uniform(0.01, 500))
    L = (p-1)/2
    M = L + s
    u = 1/M
    x = mp.log(1+1/M)
    mu = mp.mpf(1)/(s+1)
    d = abs(g_clean(x,mu) - g_u(u,s))
    maxdiff = max(maxdiff, d)
print("max |g_clean - g_u| over 200 random (s,p):", mp.nstr(maxdiff, 6))

# ---- Endpoint (I): mu=1/2 (s=1), x in (0, ln2) ----
print("\n=== Endpoint (I): g(x,1/2) >= 0 on (0, ln2) ? ===")
ln2 = mp.log(2)
mn = None; at=None
for k in range(1, 20000):
    x = ln2*mp.mpf(k)/20000
    v = g_clean(x, mp.mpf(1)/2)
    if mn is None or v<mn: mn=v; at=x
print("min g(x,1/2) =", mp.nstr(mn,10), "at x=", mp.nstr(at,8), " (x/ln2=",mp.nstr(at/ln2,4),")")

# ---- Endpoint (II): mu=1-e^{-x}, x in (0, ln2) ----
print("\n=== Endpoint (II): g(x, 1-e^{-x}) >= 0 on (0, ln2) ? ===")
mn2=None; at2=None
for k in range(1, 20000):
    x = ln2*mp.mpf(k)/20000
    mu = 1 - mp.e**(-x)
    v = g_clean(x, mu)
    if mn2 is None or v<mn2: mn2=v; at2=x
print("min g(x,1-e^-x) =", mp.nstr(mn2,10), "at x=", mp.nstr(at2,8), " (x/ln2=",mp.nstr(at2/ln2,4),")")

# ---- Confirm two-endpoint reduction: for integer s, g(x,mu) between the two endpoint lines ----
print("\n=== Direct check g(x,mu)>=0 on actual integer-s slices (sanity) ===")
worst=None
for s in [1,2,3,4,6,10,20,50]:
    mu = mp.mpf(1)/(s+1)
    xmax = -mp.log(1-mu)
    mn=None
    for k in range(1,5000):
        x = xmax*mp.mpf(k)/5000
        v=g_clean(x,mu)
        if mn is None or v<mn: mn=v
    if worst is None or mn<worst[0]: worst=(mn,s)
    print(f"  s={s:>3}  mu={mp.nstr(mu,6):>10}  xmax={mp.nstr(xmax,6):>9}  min_g={mp.nstr(mn,8):>14}")
print("worst integer-slice min_g:", mp.nstr(worst[0],8), "at s=",worst[1])
