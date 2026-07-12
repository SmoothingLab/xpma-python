import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import mpmath as mp
mp.mp.dps = 50

# g(u,s) = ln(ln(1+u)/u) + (s+1)ln(1+u) + s(1-su)ln(1+u)/((s+1)u) - s/(s+1)
# domain u in (0, 1/s). Claim: g >= 0, with g -> 0 as u->0+, leading coeff (3s+2)/(2(s+1)) u.
def g(u, s):
    u = mp.mpf(u); s = mp.mpf(s)
    l = mp.log(1+u)
    return mp.log(l/u) + (s+1)*l + s*(1-s*u)*l/((s+1)*u) - s/(s+1)

print("=== Sweep g(u,s) over full domain u in (0,1/s), many s ===")
print("Looking for any u where g<0 (would break MAIN). Report min g and its location.")
print(f"{'s':>3} {'min_g':>16} {'at u':>14} {'u/(1/s) frac':>12} {'g(near 0)':>14} {'g(near 1/s)':>14}")
worst_overall = None
for s in [1,2,3,4,5,6,8,10,12,16,24,32,48,64,100]:
    umax = mp.mpf(1)/s
    # scan u across (0, umax) on a fine grid, including near both ends (log + linear)
    mins = None; at = None
    # geometric grid near 0, plus linear grid to near umax
    grid = []
    for k in range(1, 4000):
        grid.append(umax * mp.mpf(k)/4000)
    # add very small u
    for e in range(1, 40):
        grid.append(umax * mp.mpf(10)**(-e/2))
    for u in grid:
        if u <= 0 or u >= umax: continue
        val = g(u, s)
        if mins is None or val < mins:
            mins = val; at = u
    g_lo = g(umax*mp.mpf(10)**(-12), s)
    g_hi = g(umax*(1-mp.mpf(10)**(-8)), s)
    print(f"{s:>3} {mp.nstr(mins,8):>16} {mp.nstr(at,8):>14} {mp.nstr(at/umax,4):>12} "
          f"{mp.nstr(g_lo,6):>14} {mp.nstr(g_hi,6):>14}")
    if worst_overall is None or mins < worst_overall[0]:
        worst_overall = (mins, s, at)
print()
print("Worst (most negative) g found across all s:", mp.nstr(worst_overall[0],10),
      "at s=",worst_overall[1], "u=", mp.nstr(worst_overall[2],8))
