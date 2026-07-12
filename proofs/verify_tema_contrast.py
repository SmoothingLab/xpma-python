"""TEMA contrast: exactness is classical, propriety is not (paper Section 6.2).

Reproduces the moment-calculus table behind the paper's honest disposal of the
quadratic-exactness narrative. Three zero-lag constructions are compared at their
kernel level (extracted from at-rest recursions of the xpma EMAs, which seed on
first input, so a long zero warm-up puts every cascade at rest before the impulse):

  EMA(p)          the plain smoother, L = (p - 1) / 2
  DEMA(p)         = 2 EMA - EMA^2            (Brown's linear discounted regression)
  TEMA(p)         = 3 EMA - 3 EMA^2 + EMA^3  (Mulloy; Brown's quadratic form 1 - (1-EMA)^3)
  XEPMA(p, 1)     the two-rate endpoint, EMA(p) + L * Delta MultiEMA(p, 2)

DEMA and TEMA use same-period passes (Mulloy's construction), so their kernels are
exact convolutions of EMA(p) with itself.

Every zero-lag construction is written F = EMA(p) + L * Delta(G), with Delta the
backward difference (Delta G = G - z^{-1} G) and G the integrated implicit slope
filter. The scipt identifies G in closed form and verifies the decomposition
L * Delta(G) = F - EMA(p) directly (to ~1e-14):

  DEMA:       G = EMA^2                    (a proper smoother, but lag 2L)
  TEMA:       G = 2 EMA^2 - EMA^3          (improper: negative variance)
  XEPMA(p,1): G = MultiEMA(p, 2)           (the lag-matched half-period cascade)

The load-bearing facts, all asserted at p = 20 and p = 50:

  * TEMA is quadratic-exact in discrete time (m1 = m2 = 0), m3 = 6 L^3 exactly,
    exactly as DEMA is linear-exact - exactness per se is classical.
  * BUT TEMA's implicit slope filter G has m1(G) = L, m2(G) = L, so
    Var(G) = m2(G) - m1(G)^2 = L - L^2 = -L(L-1) < 0 for every p > 3: a negative
    variance, so G is not the derivative of ANY smoothed price. TEMA's correction
    is improper (its kernel carries materially negative weights).
  * XEPMA's G IS the lag-matched half-period cascade MultiEMA(p, 2) identically
    (recovered to ~1e-15): a proper, lag-matched slope estimator. That, not
    exactness, is the OLS-endpoint property the paper targets.

Run: python proofs/verify_tema_contrast.py
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

from xpma import EMA, MultiEMA, XEPMA

WARMUP = 2000    # zeros consume the seed-on-first-input warm-up; filter at rest
KLEN = 20000     # kernel length: slowest pole (p = 50) decays well inside this
NEG_TOL = 1e-12  # a kernel weight below -NEG_TOL counts as materially negative


class Cascade:
    """n same-period EMA passes: EMA(p)^n (Mulloy's same-period convention)."""

    def __init__(self, p, n):
        self._e = [EMA(p) for _ in range(n)]

    def get_next(self, x):
        for e in self._e:
            x = e.get_next(x)
        return x


class DEMA:
    """Brown's DEMA at a single period p: 2 EMA(p) - EMA(p)^2."""

    def __init__(self, p):
        self._e1 = EMA(p)
        self._e2 = EMA(p)

    def get_next(self, x):
        a = self._e1.get_next(x)
        b = self._e2.get_next(a)
        return 2.0 * a - b


class TEMA:
    """Mulloy's TEMA at a single period p: 3 EMA - 3 EMA^2 + EMA^3."""

    def __init__(self, p):
        self._e1 = EMA(p)
        self._e2 = EMA(p)
        self._e3 = EMA(p)

    def get_next(self, x):
        a = self._e1.get_next(x)
        b = self._e2.get_next(a)
        c = self._e3.get_next(b)
        return 3.0 * a - 3.0 * b + c


def kernel(make):
    """Impulse response of a fresh filter, extracted from an at-rest recursion."""
    f = make()
    for _ in range(WARMUP):
        f.get_next(0.0)
    h = [f.get_next(1.0)]
    for _ in range(KLEN):
        h.append(f.get_next(0.0))
    return np.asarray(h, dtype=float)


def moments(h):
    """Raw kernel moments m0..m3 (m_j = sum_k k^j h_k)."""
    k = np.arange(len(h), dtype=float)
    return h.sum(), (k * h).sum(), (k ** 2 * h).sum(), (k ** 3 * h).sum()


def g_stats(g):
    """m0, lag m1, m2 and Var = m2 - m1^2 of a unit-DC slope filter kernel G."""
    k = np.arange(len(g), dtype=float)
    m0 = g.sum()
    m1 = (k * g).sum()
    m2 = (k ** 2 * g).sum()
    return m0, m1, m2, m2 - m1 ** 2


def backward_diff(g):
    """L-free Delta G = G - z^{-1} G (G taken zero before the first sample)."""
    d = np.empty_like(g)
    d[0] = g[0]
    d[1:] = g[1:] - g[:-1]
    return d


def noise_gain(h):
    return float((h ** 2).sum())


def step_peak(h):
    """Peak of the unit-step response (cumulative kernel)."""
    return float(np.cumsum(h).max())


def run(p):
    L = (p - 1.0) / 2.0
    print("=" * 80)
    print(f"p = {p:g}   L = (p-1)/2 = {L:g}")
    print("=" * 80)

    # -- F-level kernels and moments ------------------------------------------
    h_ema = kernel(lambda: EMA(p))
    h_dema = kernel(lambda: DEMA(p))
    h_tema = kernel(lambda: TEMA(p))
    h_xepma = kernel(lambda: XEPMA(p, 1.0))

    # -- implicit slope filters G (closed form), extracted directly -----------
    h_h2 = kernel(lambda: Cascade(p, 2))          # EMA(p)^2
    h_h3 = kernel(lambda: Cascade(p, 3))          # EMA(p)^3
    g_dema = h_h2                                  # G_DEMA  = EMA^2
    g_tema = 2.0 * h_h2 - h_h3                     # G_TEMA  = 2 EMA^2 - EMA^3
    g_xepma = kernel(lambda: MultiEMA(p, 2))       # G_XEPMA = lag-matched half-period cascade

    ng_ema = noise_gain(h_ema)

    print("\nKernel moments  (m_j = sum_k k^j h_k)")
    print(f"{'F':<14}{'m0':>10}{'m1':>13}{'m2':>16}{'m3':>18}")
    fmoms = {}
    for name, h in [("EMA(p)", h_ema), ("DEMA(p)", h_dema),
                    ("TEMA(p)", h_tema), ("XEPMA(p,1)", h_xepma)]:
        m = moments(h)
        fmoms[name] = m
        print(f"{name:<14}{m[0]:>10.5f}{m[1]:>13.5f}{m[2]:>16.4f}{m[3]:>18.4f}")
    print(f"  closed forms:  m1(EMA)=L={L:g}, m2(EMA)=2L^2+L={2*L*L+L:g}, "
          f"m3(EMA)=6L^3+6L^2+L={6*L**3+6*L*L+L:g};  m3(TEMA)=6L^3={6*L**3:g}")

    # -- the decomposition F = EMA + L*Delta(G), verified directly ------------
    print("\nDecomposition F = EMA(p) + L*Delta(G): max |F - EMA - L*Delta(G)|")
    for name, hF, g in [("DEMA(p)", h_dema, g_dema),
                        ("TEMA(p)", h_tema, g_tema),
                        ("XEPMA(p,1)", h_xepma, g_xepma)]:
        resid = float(np.max(np.abs((hF - h_ema) - L * backward_diff(g))))
        print(f"  {name:<12} {resid:.2e}")
        assert resid < 1e-13, f"{name}: G is not the implicit slope filter ({resid})"

    # -- propriety of G --------------------------------------------------------
    print("\nImplicit slope filter G in  F = EMA(p) + L*Delta(G)")
    print(f"{'F':<14}{'m0(G)':>9}{'m1(G) lag':>12}{'m2(G)':>14}{'Var(G)':>16}{'proper?':>10}")
    gmoms = {}
    for name, g in [("DEMA(p)", g_dema), ("TEMA(p)", g_tema), ("XEPMA(p,1)", g_xepma)]:
        s = g_stats(g)
        gmoms[name] = s
        print(f"{name:<14}{s[0]:>9.5f}{s[1]:>12.5f}{s[2]:>14.4f}{s[3]:>16.4f}"
              f"{('yes' if s[3] > 0 else 'NO'):>10}")
    print(f"  closed forms:  DEMA m1(G)=2L={2*L:g}, Var>0;  "
          f"TEMA m1(G)=L={L:g}, m2(G)=L={L:g}, Var(G)=L-L^2={L-L*L:g}=-L(L-1);  "
          f"XEPMA m1(G)=L, m2(G)=(3/2)L^2+L={1.5*L*L+L:g}")

    n_neg = int((g_tema < -NEG_TOL).sum())
    print(f"  TEMA G materially-negative bars (weight < -{NEG_TOL:g}): {n_neg}")

    # -- exact sign threshold (paper 6.2): g_n = a^2 (n+1) b^n (2 - a(n+2)/2) --
    # -- vanishes at n0 = 4/alpha - 2; every weight beyond n0 is negative ------
    alpha = 2.0 / (p + 1.0)
    n0 = int(round(4.0 / alpha - 2.0))
    assert abs(g_tema[n0]) < 1e-14, f"g_TEMA[{n0}] = {g_tema[n0]:.3e} not ~0"
    assert (g_tema[:n0] > 0.0).all(), "g_TEMA positive segment before n0 broken"
    _tail = g_tema[n0 + 1:]
    assert (_tail[:5000] < 0.0).all(), "g_TEMA negative tail past n0 broken"
    assert float(_tail.max()) < 1e-300, \
        "g_TEMA far tail carries more than denormal rounding dust"
    print(f"  TEMA G sign threshold: g[{n0}] = 0 (n0 = 4/alpha - 2), "
          f"all later weights negative")

    # -- noise gains and step peaks -------------------------------------------
    print("\nNoise gain (sum h^2, ratio vs EMA) and step peak (max cumulative kernel)")
    for name, h in [("EMA(p)", h_ema), ("DEMA(p)", h_dema),
                    ("TEMA(p)", h_tema), ("XEPMA(p,1)", h_xepma)]:
        ng = noise_gain(h)
        print(f"  {name:<12} sum h^2 = {ng:.5f}  ({ng/ng_ema:5.3f}x EMA)   "
              f"step peak = {step_peak(h):.4f}")

    # -- G_XEPMA recovered from the decomposition equals the cascade ----------
    recovered_g_xepma = np.cumsum(h_xepma - h_ema) / L
    dev = float(np.max(np.abs(recovered_g_xepma - g_xepma)))
    print(f"\nmax |recovered G_XEPMA - MultiEMA(p,2)| = {dev:.2e}")

    # -- assertions ------------------------------------------------------------
    _, m1T, m2T, m3T = fmoms["TEMA(p)"]
    assert abs(m1T) < 1e-6, f"m1(TEMA) = {m1T} not ~0"
    assert abs(m2T) < 1e-4, f"m2(TEMA) = {m2T} not ~0"
    assert abs(m3T - 6.0 * L ** 3) < 1e-8 * (6.0 * L ** 3), \
        f"m3(TEMA) = {m3T} != 6L^3 = {6*L**3}"

    m0G, m1G, m2G, varG = gmoms["TEMA(p)"]
    assert abs(m0G - 1.0) < 1e-10, f"m0(G_TEMA) = {m0G} != 1"
    assert abs(m1G - L) < 1e-8 * L, f"m1(G_TEMA) = {m1G} != L = {L}"
    assert abs(m2G - L) < 1e-8 * L, f"m2(G_TEMA) = {m2G} != L = {L}"
    assert abs(varG - (L - L ** 2)) < 1e-8 * (L ** 2), \
        f"Var(G_TEMA) = {varG} != L - L^2 = {L - L**2}"
    assert varG < 0.0, f"Var(G_TEMA) = {varG} is not negative (p > 3 required)"

    assert dev < 1e-12, f"recovered G_XEPMA deviates from the lag-matched cascade by {dev}"

    print(f"\nAll assertions passed at p = {p:g}.\n")


if __name__ == "__main__":
    for p in (20.0, 50.0):
        run(p)
    print("verify_tema_contrast.py: all assertions passed at p = 20 and p = 50.")
