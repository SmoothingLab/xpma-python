"""Zero-order MultiEMA shelf filter: closed form, accuracy against the exact
fractional reference, and moment-exact replacement candidates.

MultiEMA(p, n) is the lag-matched integer-order EMA cascade: n passes of EMA at
sub-period (p - 1) / n + 1, so every order n >= 1 has mean lag L = (p - 1) / 2.
There is no finite order-0 member of the lag-matched family; the s -> 0 limit is
an all-pass compound-Poisson filter that degenerates to the identity, so MultiEMA
and IFEMA raise at order 0. The zero-order shelf object is the chain

    EMA(p) -> EMA(p) -> ReverseMultiEMA(p, 2),

with ReverseMultiEMA(p, 2) two ReverseEMA stages of period q = (p + 1) / 2. Since
ReverseEMA(q) is the exact algebraic inverse of EMA(q) (a 2-tap FIR with transfer
1 / EMA(q)), the whole chain is the shelf filter

    H(z) = [EMA(p) / EMA(q)]^2,   q = (p + 1) / 2.

This script verifies the closed form and its moment properties numerically
(A1-A8 below), quantifies the shelf filter's accuracy against EIFEMA (the exact
fractional cascade), evaluates two moment-exact replacements C1 / C2, and prints
the consumer-impact study S1 (TrendMomentum differential). The zero-order shelf
is rebuilt from EMA / ReverseEMA primitives (ZeroOrderShelf); A6 verifies the
IFEMA C1 branch on (0, 1), and A8 verifies the compound-Poisson s -> 0 limit
structure.

Kernels are built analytically (closed geometric / double-pole forms) and
cross-checked against the actual xpma classes fed a unit impulse after a long
zero warm-up (the classes seed on first input, so a zero warm-up puts every
cascade at rest and the subsequent impulse response is the true LTI kernel).

Reference: EIFEMA(p, s), the exact fractional cascade [alpha / (1 - lambda z^-1)]^s
with sub-period e = (p - 1) / s + 1. Its precomputed `weights` array is the
reference kernel; its exact transfer is used for spectral metrics (no truncation).

Figures are written to a temporary directory; the pass/fail certification is the
process exit code (0 = all A1-A8 checks pass).

Run: python proofs/verify_zero_order_multiema.py
"""
import os
import tempfile

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))

import math
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xpma.ema import EMA
from xpma.reverse_ema import ReverseEMA
from xpma.multi_ema import MultiEMA
from xpma.ifema import IFEMA
from xpma.eifema import EIFEMA

# Illustrative figures only; the certification is the process exit code.
FIGDIR = tempfile.mkdtemp(prefix="xpma-zero-order-figs-")

# ----------------------------------------------------------------------------
# Analytic building blocks. Period P: alpha = 2/(P+1), lambda = 1 - alpha,
# mean lag L = (P-1)/2. All kernels below are truncated FIR approximations of
# the true (mostly IIR) impulse response; N is chosen so the residual tail mass
# is far below any reported tolerance.
# ----------------------------------------------------------------------------


def alpha_of(P):
    return 2.0 / (P + 1.0)


def lam_of(P):
    return 1.0 - alpha_of(P)


def ema_kernel(P, N):
    a = alpha_of(P)
    lam = lam_of(P)
    k = np.arange(N)
    return a * lam ** k


def single_shelf_kernel(p, q, N):
    """G = EMA(p) / EMA(q). g0 = ap/aq; gk = (ap/aq)(lp - lq) lp^(k-1), k>=1.

    Non-negative iff p > q (then lp > lq). Collapses to EMA(p) when q = 1."""
    ap, aq = alpha_of(p), alpha_of(q)
    lp, lq = lam_of(p), lam_of(q)
    C = ap / aq
    k = np.arange(N)
    g = np.empty(N)
    g[0] = C
    g[1:] = C * (lp - lq) * lp ** (k[1:] - 1)
    return g


def double_shelf_kernel(p, q, N):
    """H = (EMA(p)/EMA(q))^2 = C^2 (1 - lq z^-1)^2 / (1 - lp z^-1)^2.

    1/(1 - lp z^-1)^2 has impulse response (k+1) lp^k; the numerator shifts it."""
    ap, aq = alpha_of(p), alpha_of(q)
    lp, lq = lam_of(p), lam_of(q)
    C = (ap / aq) ** 2
    k = np.arange(N)
    b = (k + 1.0) * lp ** k
    b1 = np.zeros(N)
    b1[1:] = b[:-1]
    b2 = np.zeros(N)
    b2[2:] = b[:-2]
    return C * (b - 2.0 * lq * b1 + lq * lq * b2)


def ratio_shelf_kernel(p, N):
    return double_shelf_kernel(p, (p + 1.0) / 2.0, N)


# Candidate parameterisations (all keep mean lag exactly L, non-negative kernel).

def c1_params(p, s):
    """C1 1-1 shelf, valid s in (0, 1]. EMA(p1)/EMA(q1)."""
    L = (p - 1.0) / 2.0
    L1 = L * (1.0 + s) / (2.0 * s)
    m1 = L1 - L
    p1 = 2.0 * L1 + 1.0
    q1 = 2.0 * m1 + 1.0
    return p1, q1


def c2_params(p, s):
    """C2 2-2 shelf, valid s in (0, 2]. (EMA(pf)/EMA(qf))^2. Ratio shelf = C2 at s=2/3."""
    L = (p - 1.0) / 2.0
    Lf = L / (2.0 * s) + L / 4.0
    mf = Lf - L / 2.0
    pf = 2.0 * Lf + 1.0
    qf = 2.0 * mf + 1.0
    return pf, qf


def c1_kernel(p, s, N):
    p1, q1 = c1_params(p, s)
    return single_shelf_kernel(p1, q1, N)


def c2_kernel(p, s, N):
    pf, qf = c2_params(p, s)
    return double_shelf_kernel(pf, qf, N)


# ----------------------------------------------------------------------------
# Kernel from the actual classes (impulse after a zero warm-up).
# ----------------------------------------------------------------------------

def class_kernel(make, N, warmup=4000):
    f = make()
    for _ in range(warmup):
        f.get_next(0.0)
    h = np.empty(N)
    h[0] = f.get_next(1.0)
    for k in range(1, N):
        h[k] = f.get_next(0.0)
    return h


class C1Filter:
    """C1 realised from the existing primitives: EMA(p1) then ReverseEMA(q1).

    At s = 1, q1 = 1 (ReverseEMA(1) is the identity) so it is plain EMA(p)."""

    def __init__(self, period, s):
        p1, q1 = c1_params(period, s)
        self._ema = EMA(p1)
        self._rev = None if abs(q1 - 1.0) < 1e-12 else ReverseEMA(q1)

    def get_next(self, x):
        y = self._ema.get_next(x)
        if self._rev is not None:
            y = self._rev.get_next(y)
        return y


class C2Filter:
    """C2 from primitives: EMA(pf) -> EMA(pf) -> ReverseEMA(qf) -> ReverseEMA(qf)."""

    def __init__(self, period, s):
        pf, qf = c2_params(period, s)
        self._f = [EMA(pf), EMA(pf), ReverseEMA(qf), ReverseEMA(qf)]

    def get_next(self, x):
        for f in self._f:
            x = f.get_next(x)
        return x


class ZeroOrderShelf:
    """The zero-order shelf, rebuilt from primitives (MultiEMA order 0 raises):
    EMA(p) -> EMA(p) -> ReverseEMA(q) -> ReverseEMA(q), q = (p + 1) / 2.

    This is C2 at s = 2/3 (there pf = p, qf = (p + 1) / 2), so its impulse response
    equals ratio_shelf_kernel(p) bit-for-bit. Used wherever the zero-order shelf is
    characterised via the class."""

    def __init__(self, period):
        q = (period + 1.0) / 2.0
        self._f = [EMA(period), EMA(period), ReverseEMA(q), ReverseEMA(q)]

    def get_next(self, x):
        for f in self._f:
            x = f.get_next(x)
        return x


# ----------------------------------------------------------------------------
# Moment helpers (kernel-based) and exact transfer functions (spectral).
# ----------------------------------------------------------------------------

def kernel_moments(h):
    """Return (mass, mean, kappa2, kappa3) of a (truncated) kernel."""
    k = np.arange(len(h))
    S = h.sum()
    m1 = (k * h).sum() / S
    m2 = (k * k * h).sum() / S
    m3 = (k ** 3 * h).sum() / S
    kappa2 = m2 - m1 * m1
    kappa3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
    return S, m1, kappa2, kappa3


def effective_order(kappa2, L):
    return L * L / (kappa2 - L)


def ema_H(P, w):
    a = alpha_of(P)
    lam = lam_of(P)
    return a / (1.0 - lam * np.exp(-1j * w))


def eifema_H(p, s, w):
    e = (p - 1.0) / s + 1.0
    return ema_H(e, w) ** s


def kernel_length(p, s):
    """Length so the slowest pole's tail is well below 1e-13."""
    poles = [lam_of(p), lam_of((p + 1.0) / 2.0)]
    for pp, qq in (c1_params(p, s), c2_params(p, s)):
        poles += [lam_of(pp), lam_of(qq)]
    poles.append(lam_of((p - 1.0) / s + 1.0))
    lam_max = max(pl for pl in poles if pl < 1.0)
    n = int(math.ceil(-42.0 / math.log(lam_max))) + 600
    return min(n, 400000)


# ----------------------------------------------------------------------------
# A1 - A8 verification.
# ----------------------------------------------------------------------------

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"   {detail}" if detail else ""))


print("=" * 78)
print("A1-A8 analytical verification")
print("=" * 78)

PGRID = [5, 10, 21, 50, 100, 200]

print("\nA1  ratio_shelf = [EMA(p)/EMA(q)]^2, q=(p+1)/2: unit DC gain, h0=(ap/aq)^2,")
print("    analytic kernel == zero-order shelf class kernel")
for p in PGRID:
    N = kernel_length(p, 0.5)
    ha = ratio_shelf_kernel(p, N)
    hc = class_kernel(lambda: ZeroOrderShelf(p), N)
    q = (p + 1.0) / 2.0
    h0_expected = (alpha_of(p) / alpha_of(q)) ** 2
    max_kernel_err = np.max(np.abs(ha - hc))
    dc = ha.sum()
    check(f"A1 p={p}: analytic==class kernel", max_kernel_err < 1e-11,
          f"max|dh|={max_kernel_err:.2e}")
    check(f"A1 p={p}: unit DC gain", abs(dc - 1.0) < 1e-9, f"sum(h)={dc:.12f}")
    check(f"A1 p={p}: h0=(ap/aq)^2", abs(ha[0] - h0_expected) < 1e-13,
          f"h0={ha[0]:.6e} exp={h0_expected:.6e}")

print("\nA2  mean lag of ratio_shelf == L, and == MultiEMA(p,n) mean lag for n=1,2,3")
for p in PGRID:
    L = (p - 1.0) / 2.0
    N = kernel_length(p, 0.5)
    _, mu_shelf, _, _ = kernel_moments(ratio_shelf_kernel(p, N))
    check(f"A2 p={p}: mean(ratio_shelf)=L", abs(mu_shelf - L) < 1e-6,
          f"mu={mu_shelf:.6f} L={L}")
    for n in (1, 2, 3):
        _, mu_n, _, _ = kernel_moments(class_kernel(lambda: MultiEMA(p, n), N))
        check(f"A2 p={p} n={n}: mean(MultiEMA)=L", abs(mu_n - L) < 1e-6,
              f"mu={mu_n:.6f}")

print("\nA3  kappa2(ratio_shelf)=1.5 L^2 + L; family line kappa2(s)=L^2/s+L; s_eff=2/3")
for p in PGRID:
    L = (p - 1.0) / 2.0
    N = kernel_length(p, 0.5)
    _, _, k2_shelf, _ = kernel_moments(ratio_shelf_kernel(p, N))
    expected = 1.5 * L * L + L
    check(f"A3 p={p}: kappa2(ratio_shelf)=1.5L^2+L", abs(k2_shelf - expected) < 1e-4,
          f"k2={k2_shelf:.6f} exp={expected:.6f}")
    s_eff = effective_order(k2_shelf, L)
    check(f"A3 p={p}: s_eff(ratio_shelf)=2/3", abs(s_eff - 2.0 / 3.0) < 1e-6,
          f"s_eff={s_eff:.8f}")
    # family line against integer MultiEMA orders and fractional EIFEMA
    for n in (1, 2, 3, 4):
        _, _, k2n, _ = kernel_moments(class_kernel(lambda: MultiEMA(p, n), N))
        check(f"A3 p={p} n={n}: kappa2=L^2/n+L", abs(k2n - (L * L / n + L)) < 1e-4,
              f"k2={k2n:.4f} exp={L*L/n+L:.4f}")
    for s in (0.3, 0.7):
        ref = EIFEMA(p, s)
        _, _, k2e, _ = kernel_moments(np.array(ref.weights))
        check(f"A3 p={p} s={s}: kappa2(EIFEMA)=L^2/s+L",
              abs(k2e - (L * L / s + L)) < 1e-3 * (L * L / s + L),
              f"k2={k2e:.3f} exp={L*L/s+L:.3f}")

print("\nA4  ratio_shelf kernel non-negative -> monotone step, zero overshoot, zero ring")
for p in PGRID:
    N = kernel_length(p, 0.5)
    h = ratio_shelf_kernel(p, N)
    step = np.cumsum(h)
    check(f"A4 p={p}: kernel >= 0", h.min() > -1e-15, f"min(h)={h.min():.2e}")
    check(f"A4 p={p}: step monotone", np.diff(step).min() > -1e-15,
          f"min(dstep)={np.diff(step).min():.2e}")
    check(f"A4 p={p}: overshoot ~ 0", step.max() - 1.0 < 1e-9,
          f"overshoot={step.max()-1.0:.2e}")

print("\nA5  linear blend (1-s)ratio_shelf + s EMA(p): kappa2=(1.5-0.5s)L^2+L,")
print("    s_eff=1/(1.5-0.5s), nominal [0,1] -> effective [2/3,1]")
for p in (21, 100):
    L = (p - 1.0) / 2.0
    N = kernel_length(p, 0.5)
    hk = ratio_shelf_kernel(p, N)
    ep = ema_kernel(p, N)
    for s in (0.1, 0.5, 0.9):
        blend = (1.0 - s) * hk + s * ep
        _, mu, k2, _ = kernel_moments(blend)
        exp_k2 = (1.5 - 0.5 * s) * L * L + L
        check(f"A5 p={p} s={s}: mean=L", abs(mu - L) < 1e-6, f"mu={mu:.6f}")
        check(f"A5 p={p} s={s}: kappa2=(1.5-0.5s)L^2+L",
              abs(k2 - exp_k2) < 1e-4, f"k2={k2:.4f} exp={exp_k2:.4f}")
        s_eff = effective_order(k2, L)
        check(f"A5 p={p} s={s}: s_eff=1/(1.5-0.5s)",
              abs(s_eff - 1.0 / (1.5 - 0.5 * s)) < 1e-6, f"s_eff={s_eff:.6f}")
check("A5: nominal 0 -> s_eff 2/3", abs(1.0 / 1.5 - 2.0 / 3.0) < 1e-12)
check("A5: nominal 1 -> s_eff 1", abs(1.0 / 1.0 - 1.0) < 1e-12)

print("\nA6  IFEMA raises at s <= 0 (no order-0 member); on")
print("    (0,1) it is the moment-exact C1 shelf (mean L, kappa2 = L^2/s + L),")
print("    not a plain EMA(p) dead zone")
rng = np.random.default_rng(20260711)
series = np.cumsum(rng.standard_normal(3000)) * 0.5 + 100.0
for p in (10, 21, 50):
    L = (p - 1.0) / 2.0
    ema = EMA(p)
    ema_out = np.array([ema.get_next(float(x)) for x in series])
    # (a) IFEMA raises at s <= 0 (no finite order-0 member).
    for s0 in (0.0, -0.5):
        raised = False
        try:
            IFEMA(p, s0)
        except ValueError:
            raised = True
        check(f"A6 p={p} s={s0}: IFEMA raises (no order-0 member)", raised)
    # (b) On (0,1) IFEMA is the C1 shelf bit-for-bit, and no longer equals EMA(p).
    for s in (0.3, 0.7):
        ig = IFEMA(p, s)
        io = np.array([ig.get_next(float(x)) for x in series])
        c1 = C1Filter(p, s)
        c1o = np.array([c1.get_next(float(x)) for x in series])
        max_dev = np.max(np.abs(io - c1o))
        # C1Filter derives p1/q1 via c1_params (2*L1+1 form); IFEMA uses the
        # algebraically identical L*(1+s)/s+1 form, so they agree to rounding.
        check(f"A6 p={p} s={s}: IFEMA == C1 shelf", max_dev < 1e-11,
              f"max|IFEMA-C1|={max_dev:.2e}")
        departs = np.max(np.abs(io - ema_out))
        check(f"A6 p={p} s={s}: IFEMA != EMA(p) (dead zone gone)", departs > 1e-6,
              f"max|IFEMA-EMA|={departs:.4f}")
        # Moment exactness from the class impulse response.
        N = kernel_length(p, s)
        h = class_kernel(lambda p=p, s=s: IFEMA(p, s), N)
        _, mu, k2, _ = kernel_moments(h)
        check(f"A6 p={p} s={s}: mean(IFEMA)=L", abs(mu - L) < 1e-5,
              f"mu={mu:.6f} L={L}")
        exp_k2 = L * L / s + L
        check(f"A6 p={p} s={s}: kappa2(IFEMA)=L^2/s+L",
              abs(k2 - exp_k2) < 1e-3 * exp_k2, f"k2={k2:.3f} exp={exp_k2:.3f}")

print("\nA7  C1 1-1 shelf and C2 2-2 shelf: mean=L exactly, kappa2=L^2/s+L")
print("    exactly, non-negative kernel; C1(s=1)=EMA(p); C2(s=2/3)=ratio_shelf")
for p in (21, 100):
    L = (p - 1.0) / 2.0
    for s in (0.05, 0.2, 0.5, 0.9):
        N = kernel_length(p, s)
        for name, ker in (("C1", c1_kernel(p, s, N)), ("C2", c2_kernel(p, s, N))):
            _, mu, k2, _ = kernel_moments(ker)
            exp_k2 = L * L / s + L
            check(f"A7 {name} p={p} s={s}: mean=L", abs(mu - L) < 1e-5,
                  f"mu={mu:.5f} L={L}")
            check(f"A7 {name} p={p} s={s}: kappa2=L^2/s+L",
                  abs(k2 - exp_k2) < 1e-3 * exp_k2, f"k2={k2:.3f} exp={exp_k2:.3f}")
            check(f"A7 {name} p={p} s={s}: kernel >= 0", ker.min() > -1e-15,
                  f"min={ker.min():.2e}")
    # continuity limits
    N = kernel_length(p, 1.0)
    c1_at1 = c1_kernel(p, 1.0, N)
    ema_p = ema_kernel(p, N)
    check(f"A7 p={p}: C1(s=1) == EMA(p)", np.max(np.abs(c1_at1 - ema_p)) < 1e-12,
          f"max|d|={np.max(np.abs(c1_at1-ema_p)):.2e}")
    c2_23 = c2_kernel(p, 2.0 / 3.0, N)
    hk = ratio_shelf_kernel(p, N)
    check(f"A7 p={p}: C2(s=2/3) == ratio_shelf", np.max(np.abs(c2_23 - hk)) < 1e-12,
          f"max|d|={np.max(np.abs(c2_23-hk)):.2e}")

print("\nA8  s -> 0 limit structure: the lag-matched member (EIFEMA) is exactly a")
print("    compound-Poisson filter with logarithmic jumps. With alpha(s)=s/(L+s),")
print("    lambda(s)=L/(L+s), r=s ln(1+L/s): w0=e^-r; L1 to identity=2(1-e^-r);")
print("    h_k -> s lambda^k/k as s->0; rate x mean-log-jump = L exactly")
for p in (21, 100):
    L = (p - 1.0) / 2.0
    for s in (0.1, 0.01, 0.001):
        lam = L / (L + s)
        alpha = s / (L + s)
        r = s * math.log(1.0 + L / s)
        ref = EIFEMA(p, s)
        w = np.array(ref.weights)
        S = w.sum()
        # alpha(s) and lambda(s) match the single pole.
        check(f"A8 p={p} s={s}: alpha=s/(L+s)", abs((1.0 - ref.lam) - alpha) < 1e-12,
              f"alpha={1.0-ref.lam:.10f} exp={alpha:.10f}")
        check(f"A8 p={p} s={s}: lambda=L/(L+s)", abs(ref.lam - lam) < 1e-12,
              f"lam={ref.lam:.10f} exp={lam:.10f}")
        # w0 = e^-r exactly (w0 = alpha^s and r = -s ln alpha).
        check(f"A8 p={p} s={s}: w0 = e^-r", abs(w[0] - math.exp(-r)) < 1e-12,
              f"w0={w[0]:.8e} e^-r={math.exp(-r):.8e}")
        # L1 distance to identity = 2(1-e^-r); tail-corrected for the truncation cap
        # (l1_truncated + missing_tail = 2(1-w0) = 2(1-e^-r) exactly).
        delta = np.zeros(len(w))
        delta[0] = 1.0
        l1 = np.abs(w - delta).sum()
        tail = 1.0 - S
        check(f"A8 p={p} s={s}: L1(identity) = 2(1-e^-r)",
              abs(l1 + tail - 2.0 * (1.0 - math.exp(-r))) < 1e-10,
              f"l1+tail={l1+tail:.10f} exp={2*(1-math.exp(-r)):.10f}")
        # rate x mean logarithmic jump = L exactly (this is why the mean is pinned).
        mean_jump = (lam / (1.0 - lam)) / math.log(1.0 / (1.0 - lam))
        check(f"A8 p={p} s={s}: rate x mean-jump = L",
              abs(r * mean_jump - L) < 1e-8 * L,
              f"prod={r*mean_jump:.8f} L={L}")
    # h_k -> s lambda^k/k as s -> 0: the relative error must shrink with s.
    prev = None
    for s in (0.1, 0.01, 0.001):
        lam = L / (L + s)
        w = np.array(EIFEMA(p, s).weights)
        rels = [abs(w[k] - s * lam ** k / k) / (s * lam ** k / k) for k in range(1, 6)]
        m = max(rels)
        if prev is not None:
            check(f"A8 p={p}: h_k -> s lam^k/k rel err shrinks (s={s})", m < prev,
                  f"maxrel={m:.3e} < prev={prev:.3e}")
        prev = m

# ----------------------------------------------------------------------------
# Check EIFEMA truncation cap at the hardest cell (smallest s, largest p).
# ----------------------------------------------------------------------------
print("\nEIFEMA reference truncation check (cap = 100000 terms):")
cap_binds = False
for p in (100, 200):
    for s in (0.05, 0.1):
        ref = EIFEMA(p, s)
        w = np.array(ref.weights)
        binding = len(w) >= 100000
        cap_binds = cap_binds or binding
        print(f"  p={p:3d} s={s:.2f}: terms={len(w):6d} "
              f"tail_mass={1.0 - w.sum():.2e} cap_binds={binding}")

# ----------------------------------------------------------------------------
# Accuracy tables over the grid.
# ----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("Accuracy vs EIFEMA reference (kernel L1, effective-order error)")
print("=" * 78)
print("(the 'ifema' contender is the plain EMA(p) dead-zone baseline on (0,1);")
print(" the IFEMA class equals the 'C1' contender on this interval)")

SGRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
W = np.linspace(1e-4, math.pi, 512)


def contender_kernels(p, s, N):
    L = (p - 1.0) / 2.0
    hk = ratio_shelf_kernel(p, N)
    ep = ema_kernel(p, N)
    return {
        "ratio_shelf": hk,                       # fixed s_eff = 2/3
        "linear": (1.0 - s) * hk + s * ep,
        "ifema": ep.copy(),               # = EMA(p) for s in (0,1)
        "C1": c1_kernel(p, s, N),
        "C2": c2_kernel(p, s, N),
    }


def contender_specmag(p, s, name):
    if name == "ratio_shelf":
        Hc = (ema_H(p, W) / ema_H((p + 1.0) / 2.0, W)) ** 2
    elif name == "linear":
        hk = (ema_H(p, W) / ema_H((p + 1.0) / 2.0, W)) ** 2
        Hc = (1.0 - s) * hk + s * ema_H(p, W)
    elif name == "ifema":
        Hc = ema_H(p, W)
    elif name == "C1":
        p1, q1 = c1_params(p, s)
        Hc = ema_H(p1, W) / ema_H(q1, W)
    elif name == "C2":
        pf, qf = c2_params(p, s)
        Hc = (ema_H(pf, W) / ema_H(qf, W)) ** 2
    return np.abs(Hc)


# Full-grid metric collection.
metric_rows = []
l1_heat = {name: np.full((len(PGRID), len(SGRID)), np.nan)
           for name in ("ratio_shelf", "linear", "ifema", "C1", "C2")}

for ip, p in enumerate(PGRID):
    L = (p - 1.0) / 2.0
    for is_, s in enumerate(SGRID):
        N = kernel_length(p, s)
        ref = EIFEMA(p, s)
        href = np.array(ref.weights)
        Nc = max(N, len(href))
        # pad both to Nc
        rk = np.zeros(Nc)
        rk[:len(href)] = href
        e = (p - 1.0) / s + 1.0
        # analytic reference moments (exact, no truncation noise)
        k2_ref = L * L / s + L
        lam_e = lam_of(e)
        a_e = alpha_of(e)
        k3_ref = s * lam_e * (1.0 + lam_e) / a_e ** 3
        ref_specmag = np.abs(eifema_H(p, s, W))
        ref_noise = (href * href).sum()
        rstep = np.cumsum(rk)

        kers = contender_kernels(p, s, N)
        for name, hc in kers.items():
            ck = np.zeros(Nc)
            ck[:len(hc)] = hc
            _, mu, k2, k3 = kernel_moments(ck)
            l1 = np.abs(ck - rk).sum()
            l2 = math.sqrt(((ck - rk) ** 2).sum())
            step = np.cumsum(ck)
            step_sup = np.max(np.abs(step - rstep))
            noise_ratio = (hc * hc).sum() / ref_noise
            specmag = contender_specmag(p, s, name)
            spec_rmse = math.sqrt(np.mean((specmag - ref_specmag) ** 2))
            s_eff = effective_order(k2, L)
            l1_heat[name][ip, is_] = l1
            metric_rows.append(dict(
                p=p, s=s, name=name, mean_err=mu - L,
                k2_err=k2 - k2_ref, seff_err=s_eff - s, k3=k3, k3_ref=k3_ref,
                l1=l1, l2=l2, step_sup=step_sup, noise_ratio=noise_ratio,
                spec_rmse=spec_rmse, kmin=hc.min()))

# IIR cost (multiplies per sample); EIFEMA cost is its tap count.
COST = {"ratio_shelf": 4, "linear": 7, "ifema": 7, "C1": 2, "C2": 4}


def row(p, s, name):
    for r in metric_rows:
        if r["p"] == p and abs(r["s"] - s) < 1e-9 and r["name"] == name:
            return r
    return None


print("\nHeadline: p=21, kernel L1 error and effective-order error vs reference")
print(f"{'s':>5} {'contender':>8} {'L1_err':>10} {'seff':>8} {'seff_err':>10} "
      f"{'k2_err':>11} {'spec_rmse':>10} {'noise/ref':>10}")
for s in (0.1, 0.5, 0.9):
    for name in ("ratio_shelf", "linear", "ifema", "C1", "C2"):
        r = row(21, s, name)
        L = 10.0
        s_eff = r["seff_err"] + s
        print(f"{s:>5.2f} {name:>8} {r['l1']:>10.3e} {s_eff:>8.4f} "
              f"{r['seff_err']:>10.3e} {r['k2_err']:>11.3e} "
              f"{r['spec_rmse']:>10.3e} {r['noise_ratio']:>10.4f}")

print("\nEIFEMA reference cost (FIR taps) vs candidate IIR cost (mult/sample):")
for p in (21, 100):
    for s in (0.1, 0.5):
        ref = EIFEMA(p, s)
        print(f"  p={p:3d} s={s:.2f}: EIFEMA taps={len(ref.weights):6d}   "
              f"C1={COST['C1']} C2={COST['C2']} ratio_shelf={COST['ratio_shelf']} "
              f"linear={COST['linear']}")

print("\nC1 vs C2 vs ratio_shelf: kernel L1 error to reference at p=21 (lower = closer)")
print(f"{'s':>5} {'shelf_L1':>10} {'linear_L1':>10} {'C1_L1':>10} {'C2_L1':>10}")
for s in SGRID:
    print(f"{s:>5.2f} {row(21,s,'ratio_shelf')['l1']:>10.3e} "
          f"{row(21,s,'linear')['l1']:>10.3e} {row(21,s,'C1')['l1']:>10.3e} "
          f"{row(21,s,'C2')['l1']:>10.3e}")

# ----------------------------------------------------------------------------
# S1: TrendMomentum-style differential fast - slow = order0(p) - EMA(p).
# ----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("S1  TrendMomentum differential: order0(p) - EMA(p) on GBM (p=21)")
print("=" * 78)
P_S1 = 21
rng = np.random.default_rng(424242)
ret = rng.standard_normal(10000) * 0.01
price = 100.0 * np.exp(np.cumsum(ret))


def differential(order0_factory):
    fast = order0_factory()
    slow = EMA(P_S1)
    out = np.empty(len(price))
    for i, x in enumerate(price):
        out[i] = fast.get_next(float(x)) - slow.get_next(float(x))
    return out[500:]  # drop warm-up


configs = [
    ("ratio_shelf (s_eff=2/3)", lambda: ZeroOrderShelf(P_S1)),
    ("C1 s=0.25", lambda: C1Filter(P_S1, 0.25)),
    ("C1 s=0.50", lambda: C1Filter(P_S1, 0.50)),
    ("C1 s=2/3", lambda: C1Filter(P_S1, 2.0 / 3.0)),
]
print(f"{'fast leg':>18} {'std':>12} {'p95-p5 spread':>16}")
for label, fac in configs:
    d = differential(fac)
    spread = np.percentile(d, 95) - np.percentile(d, 5)
    print(f"{label:>18} {d.std():>12.6f} {spread:>16.6f}")

# ----------------------------------------------------------------------------
# Figures.
# ----------------------------------------------------------------------------
print("\nWriting figures to", FIGDIR)
os.makedirs(FIGDIR, exist_ok=True)

# (i) kernel shapes at p=21, s=0.5
p, s = 21, 0.5
N = kernel_length(p, s)
ref = EIFEMA(p, s)
href = np.array(ref.weights)
kers = contender_kernels(p, s, N)
T = 70
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(href[:T], "k-", lw=2.2, label="EIFEMA (exact, s=0.5)")
ax.plot(kers["C1"][:T], "-", color="#1f77b4", label="C1 1-1 shelf")
ax.plot(kers["C2"][:T], "--", color="#2ca02c", label="C2 2-2 shelf")
ax.plot(kers["linear"][:T], "-", color="#d62728", label="linear blend")
ax.plot(kers["ratio_shelf"][:T], ":", color="#ff7f0e", label="ratio_shelf (fixed s_eff=2/3)")
ax.plot(kers["ifema"][:T], "-.", color="#9467bd", label="IFEMA = EMA(p)")
ax.set_xlabel("tap k")
ax.set_ylabel("weight h[k]")
ax.set_title(f"Kernel shapes vs exact reference (p={p}, nominal s={s})")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "zero-order-kernels.png"), dpi=110)
plt.close(fig)

# (ii) effective order vs nominal s
ss = np.linspace(0.001, 1.0, 400)
linear_seff = 1.0 / (1.5 - 0.5 * ss)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ss, ss, "k-", lw=1.5, label="target / EIFEMA / C1 / C2 (exact)")
ax.plot(ss, linear_seff, "-", color="#d62728", lw=2, label="linear blend")
ax.plot(ss, np.ones_like(ss), "-.", color="#9467bd", lw=2,
        label="IFEMA (= EMA, s_eff=1)")
ax.plot([0.0], [2.0 / 3.0], "o", color="#9467bd", ms=8,
        label="IFEMA at s=0 (jump to ratio_shelf, 2/3)")
ax.axhline(2.0 / 3.0, color="gray", ls=":", lw=0.8)
ax.set_xlabel("nominal smoothness s")
ax.set_ylabel("effective order (from kappa2)")
ax.set_title("Effective order vs nominal knob")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "zero-order-effective-order.png"), dpi=110)
plt.close(fig)

# (iii) step responses at p=21, s=0.5
fig, ax = plt.subplots(figsize=(8, 5))
Ts = 90
ax.plot(np.cumsum(href)[:Ts], "k-", lw=2.2, label="EIFEMA (exact)")
ax.plot(np.cumsum(kers["C1"])[:Ts], "-", color="#1f77b4", label="C1")
ax.plot(np.cumsum(kers["C2"])[:Ts], "--", color="#2ca02c", label="C2")
ax.plot(np.cumsum(kers["linear"])[:Ts], "-", color="#d62728", label="linear blend")
ax.plot(np.cumsum(kers["ratio_shelf"])[:Ts], ":", color="#ff7f0e", label="ratio_shelf (2/3)")
ax.axhline(1.0, color="gray", ls=":", lw=0.8)
ax.set_xlabel("sample")
ax.set_ylabel("step response")
ax.set_title(f"Step responses (p={p}, nominal s={s})")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "zero-order-step.png"), dpi=110)
plt.close(fig)

# (iv) L1 kernel error heat maps: linear blend and C1
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, name in zip(axes, ("linear", "C1")):
    data = l1_heat[name]
    im = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(SGRID)))
    ax.set_xticklabels([f"{s:g}" for s in SGRID], rotation=45, fontsize=7)
    ax.set_yticks(range(len(PGRID)))
    ax.set_yticklabels(PGRID)
    ax.set_xlabel("nominal s")
    ax.set_ylabel("period p")
    ax.set_title(f"L1 kernel error: {name}")
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "zero-order-l1-heatmap.png"), dpi=110)
plt.close(fig)

# ----------------------------------------------------------------------------
# Summary.
# ----------------------------------------------------------------------------
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = sum(1 for _, ok, _ in results if not ok)
print("\n" + "=" * 78)
print(f"A1-A8 checks: {n_pass} PASS, {n_fail} FAIL")
if n_fail:
    print("FAILURES:")
    for name, ok, detail in results:
        if not ok:
            print(f"  {name}: {detail}")
print("Figures: zero-order-kernels.png, zero-order-effective-order.png,")
print("         zero-order-step.png, zero-order-l1-heatmap.png")
print("=" * 78)
sys.exit(1 if n_fail else 0)
