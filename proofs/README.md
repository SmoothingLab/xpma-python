# Machine-checkable proofs

These scripts re-derive and certify the mathematical claims behind the `xpma`
package. Nothing here has to be taken on trust: each script either proves an
identity symbolically (`sympy`), evaluates a bound to tens of significant
figures (`mpmath`), or sweeps an exhaustive numeric grid (`numpy`) and asserts
the result, printing the certified numbers as it goes. Run them yourself. Every
script exits with status 0 when its claim holds and non-zero (a failed
assertion or verdict check) if it ever does not, so a plain `python
proofs/<name>.py` is a self-checking certificate.

Each script is standalone and imports the installed package, so it certifies the
code that actually ships, not a private copy.

## How to run

Install the package together with the proof dependencies (symbolic algebra,
high-precision arithmetic, numeric grids and plotting):

```bash
pip install "xpma[proofs]"
```

Then run any script from the repository root:

```bash
python proofs/rmax_closed_form.py
```

or run the whole set:

```bash
for f in proofs/*.py; do echo "== $f =="; python "$f" || echo "FAILED: $f"; done
```

If you are working from a checkout with the bundled virtual environment, the
equivalent is `.venv/bin/python proofs/<name>.py`.

Runtimes below are for a warm interpreter (the first script you run in a session
pays a few seconds of import cost on top). The heaviest scripts drive
pure-Python IIR recursions over long impulse responses at large periods, which
is why they take minutes rather than seconds.

## What each script certifies

The scripts fall into three groups, matching the paper's appendices.

### Discrete monotonicity: the maximal monotone lag reduction (r_crit^M)

These certify that at `r = r_crit^M(s)` the discrete XPMA kernel is
monotone non-increasing past its mode at every period, so FastEMA and LeadEMA
sit exactly on the maximal-monotone-advance boundary (Section 5.2, Appendix A).

| Script | Claim it certifies | Paper | Method | Typical runtime | Extra deps |
|---|---|---|---|---|---|
| `disc_sympy.py` | The closed reduction `R_n = P rho^n / (n - n0)`, its `P` closed form, the continuous minimiser `x* = n0 + 1/ln(rho)`, and the linear-in-mu form `g = A - mu C`. | 5.2, App. A | symbolic (sympy) | ~2s | sympy |
| `disc_verify1.py` | The discrete monotone boundary equals `r_crit^M(s)` and the `R`-formula reproduces the direct kernel ratio across a grid of `(s, p)`. | 5.2, App. A | high-precision (mpmath) | ~5s | mpmath, numpy |
| `disc_verify2.py` | The master inequality `g(u, s) >= 0` holds over the whole domain `u in (0, 1/s)` for many `s` (no counterexample). | 5.2, App. A | high-precision (mpmath) | ~2s | mpmath |
| `disc_verify3.py` | The clean form `g(x, mu)` equals the exact `g(u, s)`; both endpoint slices and the integer-`s` slices stay non-negative. | 5.2, App. A | high-precision (mpmath) | ~4s | mpmath |
| `disc_verify4.py` | Ground truth: the discrete kernel `h_n(r*) >= 0` for all `n` across `(s, p)`; the reduction-chain identities; `C(x) > 0`; the `s = 1` lemma bounds. | 5.2, App. A | high-precision (mpmath, exhaustive) | ~3.5min | mpmath |
| `disc_final.py` | The two classical bounds (`sinh(z)/z <= e^(z^2/6)`, `z coth(z) >= 1`) and the assembled `s = 1` lemma `I(x) >= x(18-x)/24 > 0` on `(0, ln2)`. | 5.2, App. A | high-precision (mpmath) | ~17s | mpmath |

### Concavity and no-overshoot boundaries (r_crit^C, r_crit^O)

These certify the closed form of the concavity / monotone-rate boundary
`r_crit^C(s)`, its ordering against `r_crit^M` and `r_crit^O`, and its discrete
safety (Section 5.5, Appendix D).

| Script | Claim it certifies | Paper | Method | Typical runtime | Extra deps |
|---|---|---|---|---|---|
| `rmax_symbolic.py` | The kernel-derivative sign function, the binding ratio `R(u)`, the stationarity cubic `P(u)` and its exact anchors (`P(2)`, `P'(2)`, `s = 1` degeneracy, large-`s` limit, discriminant sign). | 5.5, App. D | symbolic (sympy) | ~1s | sympy |
| `rmax_closed_form.py` | The Cardano closed form for `tau_1(s)` and `r_crit^C(s)`, cross-checked against an independent root find; the interior minimum; the `s -> infinity` behaviour. | 5.5, App. D | high-precision (mpmath) | <1s | mpmath, numpy |
| `rmax_asymptotics.py` | The exact rational coefficients of the `s -> infinity` series for `tau_1` and the ratio `r_crit^C / r_crit^M` (no numeric cancellation). | 5.5, App. D | symbolic (sympy) | ~5s | sympy |
| `rmax_premode_proof.py` | The pre-mode zero-set is strictly monotone, so the pre-mode region never births an extra sign-change pair (it never binds). | 5.5, App. D | symbolic (sympy) | <1s | sympy |
| `rmax_sign_structure.py` | The full sign-change structure of `h'(tau)`: same-sign regions, pre-mode monotonicity, post-mode unique interior minimum, and bisection `r_crit^C` equal to the closed form. | 5.5, App. D | numeric grid (numpy) + mpmath | ~25s | numpy, mpmath |
| `rmax_numeric_targets.py` | Numerical ground truth for `r_crit^C(s)` and the tangency `tau_1(s)`, `s = 1..4`, plus the discrete boundaries at `s = 1`. | 5.5, App. D | exhaustive numeric grid | ~20s | numpy |
| `rmax_discrete.py` | Discrete safety: the discrete `r_crit^C(s, p)` exceeds the continuous constant for every tested `(s, p)`, and the margin shrinks as `O(1/p)`. | 5.5, App. D | exhaustive numeric grid | ~90s | numpy |
| `rmax_correspondence.py` | The concavity, monotonicity and no-overshoot boundaries coincide on the FIR line, all vanish on the same-rate exponential line, and fully split (`0 < r_crit^C < r_crit^M < r_crit^O`) on the two-rate family. | 5.5, App. D | numeric grid (numpy) + closed form | <1s | numpy |
| `rmax_fractional.py` | The closed form extends to real `s >= 1`; the cascade-realised discrete boundary is not always safe at fractional `s`; the output-level interpolation stays unimodal in every case tested. | 5.5, App. D | numeric grid + closed form | ~2.5min | numpy |

### Paper-claim verifiers

| Script | Claim it certifies | Paper | Method | Typical runtime | Extra deps |
|---|---|---|---|---|---|
| `verify_discrete_transfer.py` | The printed z-domain transfer `H_{s,p}(z)`, rebuilt from scratch, matches `xpma.XPMA` to about `1e-12`. | 2.5 | numeric parity vs package | <1s | numpy |
| `verify_tema_contrast.py` | TEMA is quadratic-exact but its implicit slope filter has `Var(G) = -L(L-1) < 0` (improper), whereas XEPMA's implicit slope filter is the proper lag-matched half-period cascade. | 6.2 | numeric kernel moments | <1s | numpy |
| `verify_zero_order_multiema.py` | The zero-order shelf closed form `[EMA(p)/EMA(q)]^2`, its moments (mean `L`, effective order 2/3), its accuracy against the exact fractional cascade, and the moment-exact C1 / C2 replacements (checks A1 to A8). | 8(c) | numeric kernel grid (numpy) + figures | ~2s | numpy, matplotlib |
| `verify_holt_quadratic.py` | Holt's level-output `m2 = -2(1-a)/(a g)`; no non-degenerate Holt member is quadratic-exact; XEPMA at `s = 1` is quadratic-exact with genuine smoothing and is three-pole, not two-pole. | 1.4 / related work | numeric kernel moments | <1s | numpy |
| `verify_qxepma.py` | The moment-knob lemma, `XEPMA m2 = ((1-s)/s) L^2`, and `QuadraticXEPMA m2 = 0` at machine precision with genuine smoothing and sub-period freedom. | 6, 8 | numeric kernel moments | ~5s | numpy |
| `verify_qxepma_fractional.py` | The QuadraticXEPMA overshoot gamma sweep, the fractional-`s` moment checks (`m1 ~ 0`, `m2 ~ 0`), and the continuous-limit minimum-overshoot closed form. | 6, 8 | numeric | ~2s | numpy |

All third-party dependencies named above are installed by `pip install
"xpma[proofs]"`. Scripts in the second and third groups also import the `xpma`
package itself, which is the object under test.

## What does not ship

The benchmark confidence-interval reproduction is **not** included here. It
depends on external benchmark harnesses and data that are not part of this
package: the MIT-BIH ECG records (streamed from PhysioNet), the scikit-image
test images, third-party libraries (`pywt`, `wfdb`, `scikit-image`) and the
published benchmark report tables it checks its cell means against. That
reproduction lives with the paper's benchmark materials, not with the library.

One consumer-impact study is also omitted. `verify_zero_order_multiema.py`
retains its A1 to A8 certification (the exit code) and its S1 study (a
fast-minus-slow differential built from the shelf and EMA), but a second study exercising a filter that is
not part of this package's public surface has been removed; it ships with the
paper materials. No A1 to A8 check was removed, so the certification count is
unchanged.
