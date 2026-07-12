"""Verification for QuadraticXEPMA, the quadratic-exact extension of XEPMA
(abbreviated QXEPMA in the tables below).

Reproduces / verifies:
 1. Moment-knob lemma: appending T = c * Delta^k * G (G any unit-DC-gain causal
    filter) shifts m_k by exactly (-1)^k * k! * c and leaves m_0 .. m_{k-1}
    untouched, independent of G's period.
 2. XEPMA^[s] second moment: m2 = ((1-s)/s) * L^2 with L = (p-1)/2 (the NOMINAL
    lag), s = 1..4, p = 10, 20, 40; cross-checked against the xpma package.
 3. QXEPMA^[s] = XEPMA^[s] + c * Delta^2 * MultiEMA^[s+2](p), c = ((s-1)/(2s)) L^2:
    m0 = 1, m1 ~ 0, m2 ~ 0 at machine precision; ramp / parabola end-to-end.
 4. Sub-period independence of m2; what the sub-period does control (m3 affine in
    G's lag with slope 6c, noise gain, overshoot); the m3-nulling cubic-exact member.
 5. Characterisation at p = 20: noise-gain ladder, step overshoot / ringing, m3.
 6. Damped variant gamma * c sweep.

All kernels are extracted from at-rest (zero initial state) recursions so they are
the true LTI impulse responses; the xpma classes' "first output = first input"
warm-up is consumed with leading zeros where the package is used directly.

Run: python proofs/verify_qxepma.py
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
from xpma import XEPMA

N_KER = 20000   # impulse-response length for moment extraction
N_STEP = 2500   # step-response length
N_TRK = 4000    # ramp / parabola tracking length


# ---------- filter primitives (at-rest, true LTI kernels) ----------
def ema_atrest(x, per):
    """EMA started from rest (state 0): impulse gives alpha*beta^k, not a seed spike."""
    a = 2.0 / (per + 1.0)
    y = np.empty_like(x)
    s = 0.0
    for i, v in enumerate(x):
        s += a * (v - s)
        y[i] = s
    return y


def cascade_q(x, q, n):
    """n EMA passes, each at per-pass period q."""
    y = x
    for _ in range(int(n)):
        y = ema_atrest(y, q)
    return y


def multi_ema(x, p, n):
    """MultiEMA(p, n): n passes at q = 1 + (p-1)/n (total lag (p-1)/2, matching EMA(p))."""
    return cascade_q(x, 1.0 + (p - 1.0) / n, n)


def diff1(y):
    """First difference with y_{-1} = 0 (at-rest)."""
    return np.diff(y, prepend=0.0)


def xepma_atrest(x, p, s):
    """XEPMA^[s](p) = MultiEMA(p,s) + L * Delta MultiEMA(p,s+1), L = (p-1)/2."""
    L = (p - 1.0) / 2.0
    return multi_ema(x, p, s) + L * diff1(multi_ema(x, p, s + 1))


def corr_stream(x, p, s, q_G=None):
    """Delta^2 of the (s+2)-stage correction cascade.
    Default per-pass period adjP = 1 + (p-1)/(s+2) (cascade lag = L).
    Returns (stream, lag of G)."""
    n_G = s + 2
    if q_G is None:
        q_G = 1.0 + (p - 1.0) / n_G
    g = cascade_q(x, q_G, n_G)
    return diff1(diff1(g)), n_G * (q_G - 1.0) / 2.0


def qxepma_atrest(x, p, s, c, q_G=None):
    """QXEPMA^[s](p) = XEPMA^[s](p) + c * Delta^2 MultiEMA-style cascade (order s+2)."""
    t, _ = corr_stream(x, p, s, q_G)
    return xepma_atrest(x, p, s) + c * t


def impulse(N):
    x = np.zeros(N)
    x[0] = 1.0
    return x


def moments(h, upto=3):
    n = np.arange(len(h), dtype=float)
    return [float((n ** j * h).sum()) for j in range(upto + 1)]


def step_metrics(y, tol=1e-9):
    """(max overshoot above 1, number of sign changes of the step error)."""
    e = y - 1.0
    overshoot = float(y.max() - 1.0)
    signs = np.sign(e[np.abs(e) > tol])
    changes = int(np.count_nonzero(signs[1:] != signs[:-1]))
    return overshoot, changes


# ---------- closed forms under test ----------
def m2_xepma_closed(p, s):
    L = (p - 1.0) / 2.0
    return (1.0 - s) / s * L * L


def m3_xepma_closed(p, s):
    L = (p - 1.0) / 2.0
    M3 = L ** 3 * ((s + 1.0) * (s + 2.0) / s ** 2 - 3.0 * (s + 2.0) / (s + 1.0))
    M2 = (1.0 - s) / s * L * L
    return M3 + 3.0 * M2  # m3 = M3 + 3 M2 + M1, M1 = 0


def c_closed(p, s):
    L = (p - 1.0) / 2.0
    return (s - 1.0) / (2.0 * s) * L * L


def mu_g_star(p, s):
    """Correction-cascade lag that nulls m3 (cubic-exact member), s >= 2."""
    L = (p - 1.0) / 2.0
    return L * (s + 2.0) * (2.0 * s ** 2 - 2.0 * s - 1.0) / (3.0 * s * (s + 1.0) * (s - 1.0))


# ====================================================================
print("=" * 76)
print("1. MOMENT-KNOB LEMMA: T = c*Delta^k*G shifts m_k by (-1)^k k! c only")
print("=" * 76)
print("  G = unit-DC EMA cascades of assorted order/period; c = 3.7")
print(f"  {'k':>2} {'G(order,per)':>14} {'m0(T)':>10} {'m1(T)':>12} {'m2(T)':>12} "
      f"{'m3(T)':>14} {'predicted shift':>22}")
c_test = 3.7
for k in (1, 2, 3):
    for (nG, qG) in [(2, 5.0), (3, 12.7), (4, 30.0)]:
        g = cascade_q(impulse(N_KER), qG, nG)
        t = g
        for _ in range(k):
            t = diff1(t)
        t = c_test * t
        m0, m1, m2, m3 = moments(t)
        pred = (-1.0) ** k * float(np.prod(np.arange(1, k + 1))) * c_test
        target = {1: m1, 2: m2, 3: m3}[k]
        print(f"  {k:>2} {f'({nG}, {qG})':>14} {m0:10.2e} {m1:12.4e} {m2:12.4e} "
              f"{m3:14.4e} m{k} -> {pred:+9.4f} (err {abs(target - pred):.1e})")
print("  -> m_j = 0 for j < k; m_k = (-1)^k k! c for ANY G period (G enters via G(1)=1).")
print("  k=2 corollary: m3(T) = 6c(mu_G + 1), mu_G = lag of G "
      "(checked in section 4).")

# ====================================================================
print("\n" + "=" * 76)
print("2. XEPMA^[s] SECOND MOMENT: m2 = ((1-s)/s) L^2, L = (p-1)/2 (nominal lag)")
print("=" * 76)
print(f"  {'s':>2} {'p':>4} {'m1_num':>12} {'m2_num':>14} {'m2_closed':>14} {'abs_err':>10}")
max_err2 = 0.0
for s in (1, 2, 3, 4):
    for p in (10, 20, 40):
        h = xepma_atrest(impulse(N_KER), p, s)
        m0, m1, m2, _ = moments(h)
        m2c = m2_xepma_closed(p, s)
        err = abs(m2 - m2c)
        max_err2 = max(max_err2, err)
        print(f"  {s:>2} {p:>4} {m1:12.2e} {m2:14.6f} {m2c:14.6f} {err:10.2e}")
print(f"\n  max |m2_num - ((1-s)/s)L^2| = {max_err2:.3e}")
print("  Lag in the formula = NOMINAL (p-1)/2: the period adjustment 1+(p-1)/n pins")
print("  every constituent cascade's total lag to L, so no sub-period lag appears.")

print("\n  xpma package cross-check (warm zeros consume the class warm-up):")
for (p, s) in [(20, 1), (20, 2), (10, 3)]:
    warm = 5
    f = XEPMA(p, float(s))
    seq = [0.0] * warm + [1.0] + [0.0] * (N_KER - 1)
    out = np.array([f.get_next(v) for v in seq])[warm:]
    mine = xepma_atrest(impulse(N_KER), p, s)
    print(f"    XEPMA(p={p}, s={s}): max |xpma - at-rest recursion| = "
          f"{np.abs(out - mine).max():.3e}")

# ====================================================================
print("\n" + "=" * 76)
print("3. QXEPMA^[s] = XEPMA^[s] + c*Delta^2*MultiEMA^[s+2](p), c = -m2(XEPMA)/2")
print("=" * 76)
print(f"  {'s':>2} {'p':>4} {'c_meas':>12} {'c_closed':>12} {'m0(Q)':>12} "
      f"{'m1(Q)':>12} {'m2(Q)':>12} {'sum h^2':>10}")
for s in (2, 3, 4):
    for p in (10, 20, 40):
        imp = impulse(N_KER)
        hx = xepma_atrest(imp, p, s)
        _, _, m2x, _ = moments(hx)
        c_meas = -m2x / 2.0
        cc = c_closed(p, s)
        hq = qxepma_atrest(imp, p, s, c_meas)
        m0q, m1q, m2q, _ = moments(hq)
        sumsq = float((hq * hq).sum())
        print(f"  {s:>2} {p:>4} {c_meas:12.6f} {cc:12.6f} {m0q:12.9f} "
              f"{m1q:12.2e} {m2q:12.2e} {sumsq:10.5f}")
print("  -> c_meas matches ((s-1)/(2s))L^2; m0 = 1, m1 ~ 0, m2 ~ 0 at machine")
print("     precision; sum h^2 < 1 everywhere: genuine smoothing, unlike Holt's")
print("     identity-only m2 = 0 member.")

print("\n  End-to-end tracking at p = 20 (trail = y - x at n = %d):" % (N_TRK - 1))
ramp = np.arange(N_TRK, dtype=float)
para = ramp ** 2
print(f"  {'s':>2} {'XEPMA ramp':>12} {'QXEPMA ramp':>13} {'XEPMA parab':>13} "
      f"{'(=m2)':>10} {'QXEPMA parab':>14}")
for s in (2, 3, 4):
    p = 20
    _, _, m2x, _ = moments(xepma_atrest(impulse(N_KER), p, s))
    c = -m2x / 2.0
    trails = []
    for x in (ramp, para):
        yx = xepma_atrest(x, p, s)
        yq = qxepma_atrest(x, p, s, c)
        trails.append((yx[-1] - x[-1], yq[-1] - x[-1]))
    print(f"  {s:>2} {trails[0][0]:+12.4f} {trails[0][1]:+13.4f} "
          f"{trails[1][0]:+13.4f} {m2x:+10.3f} {trails[1][1]:+14.4f}")
print("  -> both ramp-exact; on the parabola XEPMA trails by exactly m2, QXEPMA by 0.")

# ====================================================================
print("\n" + "=" * 76)
print("4. SUB-PERIOD FREEDOM: m2 invariant to the correction cascade's period")
print("=" * 76)
p = 20
step_in = np.ones(N_STEP)
for s in (2, 3, 4):
    imp = impulse(N_KER)
    _, _, m2x, m3x = moments(xepma_atrest(imp, p, s))
    c = -m2x / 2.0
    adjP = 1.0 + (p - 1.0) / (s + 2)
    print(f"\n  s = {s}  (c = {c:.4f}, adjP = {adjP:.4f}, m3(XEPMA) = {m3x:+.3f})")
    print(f"  {'q_G':>9} {'lag(G)':>8} {'m1(Q)':>10} {'m2(Q)':>10} {'m3_num':>12} "
          f"{'m3_pred':>12} {'sum h^2':>9} {'ovsh':>8} {'sgnchg':>6}")
    for fac in (0.5, 1.0, 2.0):
        qg = fac * adjP
        hq = qxepma_atrest(imp, p, s, c, q_G=qg)
        _, mug = corr_stream(imp, p, s, q_G=qg)
        m0q, m1q, m2q, m3q = moments(hq)
        m3pred = m3_xepma_closed(p, s) + 6.0 * c * (mug + 1.0)
        sumsq = float((hq * hq).sum())
        ov, sc = step_metrics(qxepma_atrest(step_in, p, s, c, q_G=qg))
        print(f"  {qg:9.4f} {mug:8.4f} {m1q:10.1e} {m2q:10.1e} {m3q:12.4f} "
              f"{m3pred:12.4f} {sumsq:9.5f} {ov:8.4f} {sc:6d}")
    # m3-nulling (cubic-exact) member
    mug_star = mu_g_star(p, s)
    qg_star = 1.0 + 2.0 * mug_star / (s + 2)
    hq = qxepma_atrest(imp, p, s, c, q_G=qg_star)
    m0q, m1q, m2q, m3q = moments(hq)
    sumsq = float((hq * hq).sum())
    ov, sc = step_metrics(qxepma_atrest(step_in, p, s, c, q_G=qg_star))
    print(f"  {qg_star:9.4f} {mug_star:8.4f} {m1q:10.1e} {m2q:10.1e} {m3q:12.4f} "
          f"{'(m3* = 0)':>12} {sumsq:9.5f} {ov:8.4f} {sc:6d}   <- cubic-exact member")
print("\n  -> m2 ~ 0 for every sub-period, as the lemma predicts (G enters only via")
print("     G(1) = 1). The sub-period DOES control: m3 (affine in lag(G), slope 6c),")
print("     noise gain (shorter G period -> sharper Delta^2 -> higher sum h^2) and")
print("     step overshoot. Setting lag(G) = L(s+2)(2s^2-2s-1)/(3s(s+1)(s-1)) nulls")
print("     m3 too: a CUBIC-exact member with no extra term.")

# ====================================================================
print("\n" + "=" * 76)
print("5. CHARACTERISATION AT p = 20")
print("=" * 76)
p = 20
imp = impulse(N_KER)
alpha_p = 2.0 / (p + 1.0)
ema_sumsq = alpha_p / (2.0 - alpha_p)
print("\n  5a. Noise-gain ladder (sum h^2, ratio vs EMA(20) = %.5f):" % ema_sumsq)
print(f"  {'s':>2} {'MultiEMA':>10} {'ratio':>7} {'XEPMA':>10} {'ratio':>7} "
      f"{'QXEPMA':>10} {'ratio':>7}")
for s in (1, 2, 3, 4):
    hm = multi_ema(imp, p, s)
    hx = xepma_atrest(imp, p, s)
    _, _, m2x, _ = moments(hx)
    c = -m2x / 2.0
    hq = qxepma_atrest(imp, p, s, c)
    sm, sx, sq = (float((h * h).sum()) for h in (hm, hx, hq))
    print(f"  {s:>2} {sm:10.5f} {sm/ema_sumsq:6.2f}x {sx:10.5f} {sx/ema_sumsq:6.2f}x "
          f"{sq:10.5f} {sq/ema_sumsq:6.2f}x")
print("  (XEPMA s=1 must reproduce 0.28762 / 5.75x from the Holt evaluation;")
print("   QXEPMA s=1 has c = 0 and equals XEPMA.)")

print("\n  5b. Step response (unit step from rest): overshoot and ringing:")
print(f"  {'s':>2} {'XEPMA ovsh':>11} {'XEPMA sgnchg':>13} {'QXEPMA ovsh':>12} "
      f"{'QXEPMA sgnchg':>14}")
for s in (1, 2, 3, 4):
    _, _, m2x, _ = moments(xepma_atrest(imp, p, s))
    c = -m2x / 2.0
    ovx, scx = step_metrics(xepma_atrest(step_in, p, s))
    ovq, scq = step_metrics(qxepma_atrest(step_in, p, s, c))
    print(f"  {s:>2} {ovx:11.4f} {scx:13d} {ovq:12.4f} {scq:14d}")

print("\n  5c. Third moments (default sub-period; feeds the d_crit investigation):")
print(f"  {'s':>2} {'MultiEMA m3':>12} {'XEPMA m3_num':>13} {'XEPMA m3_closed':>16} "
      f"{'QXEPMA m3_num':>14} {'QXEPMA m3_closed':>17}")
L = (p - 1.0) / 2.0
for s in (1, 2, 3, 4):
    _, _, _, m3m = moments(multi_ema(imp, p, s))
    _, _, m2x, m3x = moments(xepma_atrest(imp, p, s))
    c = -m2x / 2.0
    _, _, _, m3q = moments(qxepma_atrest(imp, p, s, c))
    m3xc = m3_xepma_closed(p, s)
    m3qc = m3xc + 6.0 * c * (L + 1.0)
    print(f"  {s:>2} {m3m:12.3f} {m3x:13.3f} {m3xc:16.3f} {m3q:14.3f} {m3qc:17.3f}")

# ====================================================================
print("\n" + "=" * 76)
print("6. DAMPED VARIANT: gamma * c, gamma in {0, 0.25, 0.5, 0.75, 1} (p = 20)")
print("=" * 76)
for s in (2, 3, 4):
    _, _, m2x, _ = moments(xepma_atrest(imp, p, s))
    c = -m2x / 2.0
    print(f"\n  s = {s} (m2(XEPMA) = {m2x:+.4f}, c = {c:.4f}); "
          f"prediction m2(gamma) = (1-gamma)*m2(XEPMA):")
    print(f"  {'gamma':>6} {'m2_num':>12} {'m2_pred':>12} {'sum h^2':>10} "
          f"{'ovsh':>8} {'sgnchg':>7}")
    vals = []
    for gam in (0.0, 0.25, 0.5, 0.75, 1.0):
        hq = qxepma_atrest(imp, p, s, gam * c)
        _, _, m2q, _ = moments(hq)
        sumsq = float((hq * hq).sum())
        ov, sc = step_metrics(qxepma_atrest(step_in, p, s, gam * c))
        vals.append((gam, m2q, sumsq, ov, sc))
        print(f"  {gam:6.2f} {m2q:12.4f} {(1-gam)*m2x:12.4f} {sumsq:10.5f} "
              f"{ov:8.4f} {sc:7d}")
    mono_ng = all(b[2] >= a[2] for a, b in zip(vals, vals[1:]))
    mono_ov = all(b[3] >= a[3] for a, b in zip(vals, vals[1:]))
    print(f"  monotone in gamma: noise gain {mono_ng}, overshoot {mono_ov}")

print("\n" + "=" * 76)
print("DONE")
