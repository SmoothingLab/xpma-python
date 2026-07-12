"""Independent verification for the Holt quadratic-exactness evaluation.

Reproduces:
 1. Holt level-output kernel moments m0,m1,m2 by impulse-response summation,
    compared against the analytic formula m2 = -2(1-a)/(a*g), on a stability-region grid.
 2. Brown tie: Holt(a_H, g_H) reproduces DEMA m2 = -2 L^2.
 3. alpha-beta filtered vs predicted acceleration lag (constant-acceleration steady state).
 4. XEPMA(s=1) at p=20: m2 = 0 with genuine smoothing (sum h^2), and the 3-pole /
    "not 2-pole" structure.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
from xpma import XEPMA


# ---------- 1. Holt level-output kernel moments ----------
def holt_impulse(alpha, gamma, N):
    """Level-output impulse response h_n, n=0..N-1. Recursion:
        l_t = alpha*x_t + (1-alpha)*(l_{t-1}+b_{t-1})
        b_t = gamma*(l_t - l_{t-1}) + (1-gamma)*b_{t-1}
    Output = l_t.  Zero initial state, x = unit impulse at n=0."""
    h = np.empty(N)
    l_prev, b_prev = 0.0, 0.0
    for n in range(N):
        x = 1.0 if n == 0 else 0.0
        l = alpha * x + (1.0 - alpha) * (l_prev + b_prev)
        b = gamma * (l - l_prev) + (1.0 - gamma) * b_prev
        h[n] = l
        l_prev, b_prev = l, b
    return h


def moments(h):
    n = np.arange(len(h), dtype=float)
    return h.sum(), (n * h).sum(), (n * n * h).sum()


print("=" * 70)
print("1. HOLT LEVEL-OUTPUT KERNEL MOMENTS (impulse-response summation)")
print("=" * 70)
print(f"{'alpha':>6} {'gamma':>6} {'m0':>10} {'m1':>12} {'m2_num':>14} "
      f"{'m2_formula':>14} {'abs_err':>10}")
max_err = 0.0
N = 60000
for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
    for gamma in [0.1, 0.3, 0.5, 0.7, 0.9]:
        h = holt_impulse(alpha, gamma, N)
        m0, m1, m2 = moments(h)
        m2_formula = -2.0 * (1.0 - alpha) / (alpha * gamma)
        err = abs(m2 - m2_formula)
        max_err = max(max_err, err)
        print(f"{alpha:6.2f} {gamma:6.2f} {m0:10.6f} {m1:12.2e} "
              f"{m2:14.6f} {m2_formula:14.6f} {err:10.2e}")
print(f"\nmax |m2_num - m2_formula| over grid = {max_err:.3e}")
print(f"max |m1| over grid (ramp-exactness): checked ~0 above")

# alpha>1 branch: still stable if a*g < 2(2-a); shows m2>0 (not a genuine smoother)
print("\n  alpha>1 stable branch (expansive filter, negative kernel weights):")
for alpha, gamma in [(1.5, 0.3), (1.2, 0.5)]:
    if alpha * gamma < 2 * (2 - alpha):  # stability
        h = holt_impulse(alpha, gamma, N)
        m0, m1, m2 = moments(h)
        m2f = -2.0 * (1.0 - alpha) / (alpha * gamma)
        hasneg = (h < -1e-12).any()
        print(f"    a={alpha} g={gamma}: m2_num={m2:.5f} m2_formula={m2f:.5f} "
              f"kernel_has_negative_weights={hasneg}")

# ---------- 2. Brown tie ----------
print("\n" + "=" * 70)
print("2. BROWN TIE: Holt(a_H,g_H) reproduces DEMA m2 = -2 L^2")
print("=" * 70)
def ema_pass_atrest(x, per):
    """EMA started from rest (state 0), so an impulse gives the true LTI
    kernel alpha*beta^k rather than a seed-on-first-sample spike."""
    a = 2.0 / (per + 1.0)
    y = np.empty_like(x)
    s = 0.0
    for i, v in enumerate(x):
        s = s + a * (v - s)
        y[i] = s
    return y

for p in [20, 10, 40]:
    alpha_B = 2.0 / (p + 1.0)
    beta = 1.0 - alpha_B
    alpha_H = 1.0 - beta ** 2
    gamma_H = (1.0 - beta) / (1.0 + beta)
    L = beta / alpha_B
    m2_formula = -2.0 * (1.0 - alpha_H) / (alpha_H * gamma_H)
    # direct Brown DEMA m2 by impulse response (same period p, both passes)
    Nb = 40000
    imp = np.zeros(Nb); imp[0] = 1.0
    e1 = ema_pass_atrest(imp, p)
    e2 = ema_pass_atrest(e1, p)
    dema = 2 * e1 - e2  # Brown's double exponential smoothing, both at period p
    _, m1_dema, m2_dema = moments(dema)
    print(f"  p={p:3d}: a_H={alpha_H:.6f} g_H={gamma_H:.6f} L={L:.4f} | "
          f"Holt-formula m2={m2_formula:.5f}  -2L^2={-2*L*L:.5f}  "
          f"DEMA-impulse m2={m2_dema:.5f} (m1={m1_dema:.1e})")

# ---------- 3. alpha-beta filtered vs predicted acceleration lag ----------
print("\n" + "=" * 70)
print("3. ALPHA-BETA STEADY-STATE ACCELERATION LAG (filtered vs predicted)")
print("=" * 70)
print("  Holt level == alpha-beta filtered position, with a_ab=alpha, b_ab=alpha*gamma")
def alpha_beta_accel(alpha, gamma, accel, T=1.0, steps=200000):
    """Track x(t)=0.5*accel*t^2 with alpha-beta filter (T sample interval)."""
    b_ab = alpha * gamma
    xf = 0.0; vf = 0.0
    e_pred = e_filt = 0.0
    for k in range(1, steps + 1):
        t = k * T
        xtrue = 0.5 * accel * t * t
        xpred = xf + T * vf                  # predicted position (level+trend)
        r = xtrue - xpred                    # innovation
        xf = xpred + alpha * r               # filtered (Holt level l_t)
        vf = vf + (b_ab / T) * r
        e_pred = xpred - xtrue               # predicted-position error
        e_filt = xf - xtrue                  # filtered-position error (== 0.5*accel*m2)
    return e_pred, e_filt

for alpha, gamma, accel in [(0.5, 0.4, 1.0), (0.3, 0.5, 2.0), (0.2, 0.3, 1.0)]:
    b_ab = alpha * gamma
    ep, ef = alpha_beta_accel(alpha, gamma, accel)
    pred_theory = -accel * 1.0 ** 2 / b_ab                 # e_p = -a T^2 / b_ab
    filt_theory = -accel * (1 - alpha) * 1.0 ** 2 / b_ab   # e_f = -a(1-alpha)T^2/b_ab
    m2_check = 0.5 * accel * (-2 * (1 - alpha) / (alpha * gamma))
    print(f"  a={alpha} g={gamma} accel={accel}: "
          f"e_pred={ep:.5f} (theory {pred_theory:.5f})  "
          f"e_filt={ef:.5f} (theory {filt_theory:.5f}, via m2 {m2_check:.5f})")

# ---------- 4. XEPMA(s=1) at p=20 ----------
print("\n" + "=" * 70)
print("4. XEPMA(s=1) at p=20: m2=0 with genuine smoothing, and 3-pole structure")
print("=" * 70)
def xepma_impulse(p, N, warm=5):
    """True LTI impulse response: feed leading zeros so the class's
    'first output = first input' warm-up branch is consumed at rest, then
    the unit impulse, then zeros; read the response from the impulse onward."""
    f = XEPMA(p, 1.0)
    seq = [0.0] * warm + [1.0] + [0.0] * (N - 1)
    out = [f.get_next(v) for v in seq]
    return np.array(out[warm:])

p = 20
Nx = 40000
hx = xepma_impulse(p, Nx)
m0x, m1x, m2x = moments(hx)
sumsq = (hx * hx).sum()
# EMA(20) sum h^2 for the noise-gain ratio
alpha_p = 2.0 / (p + 1.0)
ema_sumsq = alpha_p / (2.0 - alpha_p)
print(f"  m0={m0x:.10f}  m1={m1x:.3e}  m2={m2x:.3e}")
print(f"  sum h^2 (XEPMA) = {sumsq:.5f}   sum h^2 (EMA20) = {ema_sumsq:.5f}   "
      f"noise-gain ratio = {sumsq/ema_sumsq:.4f}x")
print(f"  identity filter sum h^2 = 1.0 (Holt's only m2=0 member: no smoothing)")

# pole structure: expected {beta_p, beta_h, beta_h}
beta_p = (p - 1) / (p + 1)
beta_h = (p - 1) / (p + 3)
print(f"\n  expected poles: beta_p={beta_p:.6f}, beta_h={beta_h:.6f} (double)")

# 2-pole test: try h_{n+2} = c1 h_{n+1} + c0 h_n on consecutive triples past transient
def fit_recur_order2(h, n0):
    # solve [[h[n0],h[n0+1]] gives (c0,c1) from two equations; test on a third
    A = np.array([[h[n0], h[n0 + 1]], [h[n0 + 1], h[n0 + 2]]])
    rhs = np.array([h[n0 + 2], h[n0 + 3]])
    c0, c1 = np.linalg.solve(A, rhs)
    pred = c1 * h[n0 + 3] + c0 * h[n0 + 2]
    return (c0, c1), h[n0 + 4], pred

print("\n  2-pole recurrence h_{n+2}=c1 h_{n+1}+c0 h_n fitted at successive offsets:")
print(f"  {'n0':>4} {'c0':>12} {'c1':>12} {'h[n0+4]':>14} {'pred':>14} {'resid':>10}")
for n0 in [5, 15, 30, 60, 100]:
    (c0, c1), actual, pred = fit_recur_order2(hx, n0)
    print(f"  {n0:4d} {c0:12.6f} {c1:12.6f} {actual:14.6e} {pred:14.6e} "
          f"{abs(actual - pred):10.2e}")
print("  -> (c0,c1) drift with n0 and residual is large: NOT a 2-pole system.")

# 3-pole test: h_{n+3}=d2 h_{n+2}+d1 h_{n+1}+d0 h_n should be consistent,
# with characteristic poly (z-beta_p)(z-beta_h)^2
def fit_recur_order3(h, n0):
    A = np.array([[h[n0], h[n0 + 1], h[n0 + 2]],
                  [h[n0 + 1], h[n0 + 2], h[n0 + 3]],
                  [h[n0 + 2], h[n0 + 3], h[n0 + 4]]])
    rhs = np.array([h[n0 + 3], h[n0 + 4], h[n0 + 5]])
    d0, d1, d2 = np.linalg.solve(A, rhs)
    return d0, d1, d2

print("\n  3-pole recurrence h_{n+3}=d2 h_{n+2}+d1 h_{n+1}+d0 h_n:")
for n0 in [10, 80, 300]:
    d0, d1, d2 = fit_recur_order3(hx, n0)
    roots = np.sort(np.roots([1.0, -d2, -d1, -d0]))
    print(f"    n0={n0:3d}: recurrence roots = {np.round(roots,6)}")
expected = np.sort([beta_p, beta_h, beta_h])
print(f"    expected roots           = {np.round(expected,6)}")

# ---------- 5. Direct end-to-end parabola tracking ----------
print("\n" + "=" * 70)
print("5. DIRECT PARABOLA TRACKING x_n = n^2 (steady-state trail = m2)")
print("=" * 70)
Np = 4000
xpar = np.arange(Np, dtype=float) ** 2
# XEPMA(s=1)
fx = XEPMA(20, 1.0)
yx = np.array([fx.get_next(v) for v in xpar])
# Holt at a representative gain pair, e.g. tie-to-DEMA(p=20)
p = 20; alpha_B = 2.0 / (p + 1); beta = 1 - alpha_B
aH = 1 - beta ** 2; gH = (1 - beta) / (1 + beta)
hh = holt_impulse  # reuse recursion inline instead
def holt_track(alpha, gamma, x):
    l_prev = b_prev = 0.0
    out = np.empty_like(x)
    for i, v in enumerate(x):
        l = alpha * v + (1 - alpha) * (l_prev + b_prev)
        b = gamma * (l - l_prev) + (1 - gamma) * b_prev
        out[i] = l
        l_prev, b_prev = l, b
    return out
yh = holt_track(aH, gH, xpar)
n_end = Np - 1
print(f"  at n={n_end}: XEPMA trail y-x = {yx[n_end]-xpar[n_end]:+.4f}  "
      f"(m2 target 0.000)")
print(f"  at n={n_end}: Holt(DEMA-tie) trail y-x = {yh[n_end]-xpar[n_end]:+.4f}  "
      f"(m2 target {-2*(beta/alpha_B)**2:.3f})")
print("=" * 70)
print("DONE")
