"""QuadraticXEPMA follow-up verification: gamma sweep of step-response overshoot
vs (s, p), and fractional-s moment checks. From-rest recursions give the true LTI
kernels.

Run: python proofs/verify_qxepma_fractional.py
"""
import os
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np
import math


def ema(x, q):
    a = 2.0 / (q + 1.0)
    y = np.empty(len(x))
    s = 0.0
    for i, v in enumerate(x):
        s += a * (v - s)
        y[i] = s
    return y


def multi_ema(x, p, n):
    q = (p - 1.0) / n + 1.0
    y = x
    for _ in range(n):
        y = ema(y, q)
    return y


def ifema_blend(x, p, s):
    """IFEMA blend: integer -> plain cascade; fractional -> second-moment matched
    blend of bracketing integer cascades (w_floor=(1-f)*fl/s, w_ceil=f*ce/s)."""
    fl, ce = math.floor(s), math.ceil(s)
    if fl == ce:
        return multi_ema(x, p, fl)
    f = s - fl
    return (1 - f) * fl / s * multi_ema(x, p, fl) + f * ce / s * multi_ema(x, p, ce)


def diff(y):
    d = np.empty(len(y))
    d[0] = y[0]  # from rest: previous value 0
    d[1:] = y[1:] - y[:-1]
    return d


def qxepma(x, p, s, gamma):
    L = (p - 1.0) / 2.0
    c = gamma * ((s - 1.0) / (2.0 * s)) * L * L
    y = ifema_blend(x, p, s) + L * diff(ifema_blend(x, p, s + 1))
    if c != 0.0:
        y = y + c * diff(diff(ifema_blend(x, p, s + 2)))
    return y


T = 600
step = np.ones(T)

print('1. Step-overshoot gamma sweep: overshoot = max(y) - 1, from rest')
print('   %-5s %-4s | %-14s %-14s | %-8s %-16s' % ('s', 'p', 'XEPMA (g=0)', 'QXEPMA (g=1)', 'g_min', 'overshoot(g_min)'))
gammas = np.arange(0.0, 1.0001, 0.02)
for s in (1.5, 2, 2.5, 3, 4):
    for p in (10, 20, 40):
        ov = [qxepma(step, p, s, g).max() - 1.0 for g in gammas]
        k = int(np.argmin(ov))
        print('   %-5s %-4d | %-14.4f %-14.4f | %-8.2f %-16.4f' %
              (s, p, ov[0], ov[-1], gammas[k], ov[k]))

print()
print('2. Fractional-s moment check (impulse response, from rest): expect m1 ~ 0, m2 ~ 0 at gamma = 1')
imp = np.zeros(3000)
imp[0] = 1.0
n = np.arange(3000.0)
for s in (1.5, 2.5, 3.25, 3.5):
    for p in (10, 20, 40):
        h = qxepma(imp, p, s, 1.0)
        m0, m1, m2 = h.sum(), (n * h).sum(), (n * n * h).sum()
        print('   s=%-5s p=%-3d: m0-1 = %+.2e  m1 = %+.2e  m2 = %+.2e' % (s, p, m0 - 1, m1, m2))

print()
print('3. Fractional-s XEPMA m2 vs ((1-s)/s)L^2 (confirms the closed form extends to fractional s):')
for s in (1.5, 2.5, 3.5):
    p = 20
    L = (p - 1) / 2.0
    h = qxepma(imp, p, s, 0.0)
    m2 = (n * n * h).sum()
    print('   s=%-4s p=20: measured %+.4f  formula %+.4f' % (s, m2, (1 - s) / s * L * L))

print()
print('4. Exact min-overshoot gamma.')
print('   Step response y_g(t) = 1 + eX(t) + g*c*w(t) is affine in g, so overshoot(g)')
print('   = max_t(...) is convex piecewise-linear in g. The minimum sits where the')
print('   argmax switches between the adjacent times t2 < t1 straddling the mode of')
print('   the s+2 correction cascade (w = Delta(h_G) changes sign there; continuous')
print('   mode = (s+1)/(s+2)*L). Exact value: crossing of the two active branches.')
print('   %-4s %-5s | %-10s %-5s %-5s | %-10s | %-10s' %
      ('s', 'p', 'gamma*', 't2', 't1', 'ovsh(g*)', 'mode_cont'))
for s in (2, 3, 4):
    for p in (10, 20, 40, 100, 200):
        T = max(600, 30 * p)
        stp = np.ones(T)
        L = (p - 1.0) / 2.0
        c = ((s - 1.0) / (2.0 * s)) * L * L
        eX = ifema_blend(stp, p, s) + L * diff(ifema_blend(stp, p, s + 1)) - 1.0
        w = diff(diff(ifema_blend(stp, p, s + 2)))
        k = int(np.argmax(np.cumsum(w)))          # discrete mode of h_G: last w > 0
        t2, t1 = k, k + 1                          # ascending branch, descending branch
        g_star = (eX[t1] - eX[t2]) / (c * (w[t2] - w[t1]))
        y_star = eX + g_star * c * w
        peak = y_star.max()
        assert abs(y_star[t1] - y_star[t2]) < 1e-12 and peak - y_star[t1] < 1e-12
        print('   %-4d %-5d | %-10.6f %-5d %-5d | %-10.6f | %-10.3f' %
              (s, p, g_star, t2, t1, peak, (s + 1.0) / (s + 2.0) * L))

print()
print('5. Closed form for the continuous-limit optimum:')
print('   gamma*(s) = 2s e^(t*)/((s-1)(s+2)^(s+2)) * (s^(s+1) e^(t*) - (s+1)^(s+1)/(s+2)),')
print('   t* = (s+1)/(s+2);  Ov*(s) = g_(s+1)(t*) - Q_s(t*).  Verified two ways below.')


def gdens(t, k):
    return k ** k * t ** (k - 1) * np.exp(-k * t) / math.factorial(k - 1)


def gsurv(t, s):
    return np.exp(-s * t) * sum((s * t) ** j / math.factorial(j) for j in range(s))


def gamma_star_formula(s):
    ts = (s + 1.0) / (s + 2.0)
    return 2 * s * math.exp(ts) / ((s - 1) * (s + 2) ** (s + 2)) * \
        (s ** (s + 1) * math.exp(ts) - (s + 1) ** (s + 1) / (s + 2))


tt = np.arange(1e-6, 8.0, 2e-5)
for s in (2, 3, 4):
    cc = (s - 1) / (2.0 * s)
    eXc = gdens(tt, s + 1) - gsurv(tt, s)
    wc = gdens(tt, s + 2) * ((s + 1) / tt - (s + 2))

    def ov(g):
        return (eXc + g * cc * wc).max()

    lo, hi = 0.0, 1.0
    R = (math.sqrt(5) - 1) / 2
    a, b = hi - R * (hi - lo), lo + R * (hi - lo)
    fa, fb = ov(a), ov(b)
    for _ in range(80):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - R * (hi - lo); fa = ov(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + R * (hi - lo); fb = ov(b)
    g_num = (lo + hi) / 2
    ts = (s + 1.0) / (s + 2.0)
    ov_pred = float(gdens(np.array([ts]), s + 1)[0]) - gsurv(ts, s)
    # discrete convergence check at large p via the crossing formula in step 4
    disc = []
    for p in (201, 1001, 4001):
        T = 4 * p
        stp = np.ones(T)
        L = (p - 1.0) / 2.0
        c = ((s - 1.0) / (2.0 * s)) * L * L
        eX = ifema_blend(stp, p, s) + L * diff(ifema_blend(stp, p, s + 1)) - 1.0
        w = diff(diff(ifema_blend(stp, p, s + 2)))
        k = int(np.argmax(np.cumsum(w)))
        disc.append((eX[k + 1] - eX[k]) / (c * (w[k] - w[k + 1])))
    print('   s=%d: formula %.6f | continuous numeric %.6f (Ov* %.6f vs pred %.6f)' %
          (s, gamma_star_formula(s), g_num, ov(g_num), ov_pred))
    print('        discrete p=201/1001/4001: %.6f / %.6f / %.6f  (O(1/p) approach)' %
          (disc[0], disc[1], disc[2]))

print()
print('6. s = 1 min-overshoot in absolute-coefficient form (gamma degenerates, c = 0;')
print('   this is the min-overshoot coefficient computed at construction):')
for p in (10, 20, 40):
    s = 1
    T = 4 * p + 20
    stp = np.ones(T)
    L = (p - 1.0) / 2.0
    eX = ifema_blend(stp, p, s) + L * diff(ifema_blend(stp, p, s + 1)) - 1.0
    w = diff(diff(ifema_blend(stp, p, s + 2)))
    k = int(np.argmax(np.cumsum(w)))
    d_star = (eX[k + 1] - eX[k]) / (w[k] - w[k + 1])
    print('   p=%-3d: d* = %.6f (continuous limit 0.044321*L^2 = %.6f), overshoot %.6f -> %.6f' %
          (p, d_star, 0.044321 * L * L, eX.max(), (eX + d_star * w).max()))
