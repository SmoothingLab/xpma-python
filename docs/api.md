# API reference

Everything here is derived from the code. Class and parameter names are exact,
including capitalisation (`XEPMA`, `FastEMA`, `LeadEMA`, `ConvexFastEMA`).

## The common interface

Every filter is a stateful object with the same two methods:

- **`get_next(value) -> float | None`**: feed one sample, advance the state,
  return the new output.
- **`calc_next(value) -> float | None`**: return what the output would be for
  this next sample, without advancing the state. A repeatable what-if probe.

Shared conventions:

- **Bad input**: if `value` is `None` or not finite (`nan`, `inf`), both methods
  return `None` and leave the state unchanged.
- **Warm-up**: the first `get_next` on fresh data returns the first input value
  (there is no history to average yet).
- **Reset**: there is no `reset()`; construct a new instance to start over.
- **`period`**: a real number greater than 1; larger means more smoothing and
  more lag. Fractional periods are supported. A period of 1 or below is accepted
  by the code but degenerates toward a plain passthrough, and the overshoot
  guarantees no longer hold, so keep to period > 1.
- **`smoothness`**: how many layers of averaging (the filter order). Default `1`
  behaves like one EMA pass; higher removes more noise. Fractional values are
  supported, but the range depends on the filter: `IFEMA`, `EIFEMA` and `XEPMA`
  accept orders below 1, while the lag-reduced filters (`FastEMA`, `LeadEMA`, the
  `Convex` and `Apex` pairs) are designed for `smoothness >= 1` (a value between
  0 and 1 either falls back to order 1 or raises). Each entry below notes its
  range where it is not the default.

The one exception is `SecantSolver`, which is a tool rather than a filter and has
a `solve` method instead. It is noted below.

---

## Core smoothers

### `EMA(period)`

The plain exponential moving average, `alpha = 2 / (period + 1)`. Smooth and
cheap, but lags the data by `(period - 1) / 2` samples in steady state.

### `ReverseEMA(period)`

The algebraic inverse of the EMA's update: given an EMA output stream it
recovers the input. Used inside the fractional smoother and available for
building reversible pipelines.

### `MultiEMA(period, num_smooths)`

`num_smooths` EMA passes in series, each with its period adjusted so the whole
cascade still has mean lag `(period - 1) / 2`. `num_smooths` is a positive
integer (a non-integer is truncated to a whole number rather than rejected);
order 0 does not exist and raises (use `IFEMA` for a sub-unit order). This is the
integer-order backbone of the exponential family.

### `ReverseMultiEMA(period, num_smooths)`

The reverse of `MultiEMA`: a chain of `ReverseEMA` stages. Same integer-order
constraint.

### `IFEMA(period, smoothness)`

A smoother of any fractional smoothness order, with no lag reduction. `smoothness`
must be greater than 0 (it raises at 0 or below):

- `0 < smoothness < 1`: a genuine gentler-than-EMA smoother.
- `smoothness >= 1`: an order between two whole-number cascades, matched so its
  noise behaviour is correct; at whole numbers it equals `MultiEMA`.

Use it when you want a smoothness between whole numbers. It is also the core the
lag-reduced and zero-lag filters are built on.

### `EIFEMA(period, smoothness)`

An exact fractional-order smoother computed directly (rather than by matching two
whole-order cascades as `IFEMA` does). `smoothness > 0`. Slightly more work per
sample; useful as a high-accuracy reference. For everyday use `IFEMA` is the
lighter choice.

---

## Lag-reduced, overshoot-free filters

These spend a carefully chosen amount of lag reduction so they respond faster
than a plain EMA without overshooting. Each takes `(period, smoothness=1)`.

The **`Fast*`** filters keep the nominal period and spend the saving on speed.
The **`Lead*`** siblings inflate the period internally so their lag matches
`EMA(period)`, spending the saving on closer tracking instead. See
[choosing-a-filter.md](choosing-a-filter.md) for when to pick which.

All are designed for `smoothness >= 1`. A value between 0 and 1 falls back to the
order-1 filter for `FastEMA`, `LeadEMA` and the `Apex` pair, and raises for
`ConvexFastEMA` and `ConvexLeadEMA`.

What is proven versus checked numerically for these filters is summarised in
[theory.md](theory.md).

### `FastEMA(period, smoothness=1)`

Same nominal period as `EMA(period)`, less lag. Rises straight to a new level
(monotone), no overshoot. The faster response passes more noise than
`EMA(period)` (quantified in [choosing-a-filter.md](choosing-a-filter.md)).
Exposes `lag_reduction`,
`ma_lag` and `time_lag` attributes; these are exact at whole-number smoothness,
and at fractional smoothness they describe the ideal filter rather than the
blended instance actually run (they differ by a fraction of a sample).

### `LeadEMA(period, smoothness=1)`

The lag-matched sibling: same lag as `EMA(period)`, spent on tracking a move more
closely with less undershoot. No overshoot. The default for reference-level
roles. Exposes `time_lag` (equal to `EMA(period)`'s).

### `ConvexFastEMA(period, smoothness=1)`

Slightly slower than `FastEMA`. At smoothness 1 its approach to a new level only
ever slows down; at higher smoothness the rate rises to a single peak then slows
without re-accelerating. Either way it is the gentlest shape for reading slope or
curvature. No overshoot.

### `ConvexLeadEMA(period, smoothness=1)`

The lag-matched sibling of `ConvexFastEMA`.

### `ApexFastEMA(period, smoothness=1)`

The fastest of the no-overshoot filters at a given period: it climbs closest to
the level, then dips just below before settling (not monotone). It never crosses
above the level.

### `ApexLeadEMA(period, smoothness=1)`

The lag-matched sibling of `ApexFastEMA`: the least undershoot of the lag-matched
filters, at the cost of the same small dip.

---

## The zero-lag endpoint

### `XEPMA(period, smoothness=1)`

The zero-lag endpoint of the family: a level-plus-trend nowcast with essentially
no lag at any smoothness. **It can overshoot after a sharp move; that is the
intrinsic cost of zero lag.** Excellent as an oscillator, poor as a reference
level (use `LeadEMA` there).

### `QuadraticXEPMA(period, smoothness=1)`

A variant of `XEPMA` that reproduces curved (quadratic) trends exactly at every
smoothness order of 1 and above (plain `XEPMA` does this only at `smoothness =
1`), at the cost of more noise. Identical to `XEPMA` at `smoothness = 1`.
Primarily of mathematical interest; reach for it only when exact curvature
tracking matters more than noise.

### `DampedXEPMA(period, smoothness=1)`

`XEPMA` with the smallest overshoot within its correction family while keeping
zero lag: it trades a little of the endpoint's trend exactness for reduced
overshoot at every `(period, smoothness)`. (This is the least overshoot for this
particular curvature correction, not a global optimum over all filters.) The
practical choice if you want `XEPMA`'s speed with less overshoot.

---

## The general family

### `XPMA(period, smoothness, lag_reduction)`

The parameterised family that all the named filters are points on:

- `period`: as usual.
- `smoothness`: `>= 0`. (`0` is only valid together with `lag_reduction=1`,
  where the filter is the identity.)
- `lag_reduction`: how much of the lag to remove, from `0` (a plain, maximally
  smooth cascade with full lag) to `1` (the zero-lag `XEPMA` endpoint).
  Intermediate values interpolate between the two. Values outside `[0, 1]` are
  accepted and extrapolate beyond the family (negative undoes reduction, above 1
  overshoots the endpoint); stay within `[0, 1]` unless you specifically want
  that.

All three arguments are required: the lag reduction is part of the filter's
identity and is fixed at construction. For the safe, boundary-hitting values,
prefer the named filters above, which pass the critical constant for you.

---

## Reversal

### `SecantSolver(indicator, max_iterations=5, max_error=1e-6)`

Inverts any filter that has both `get_next` and `calc_next`: given a wanted
output, it finds the input that produces it, committing the final input to the
filter once it converges (it uses the non-advancing `calc_next` probe while
iterating).

- `indicator`: the filter instance to invert (it must have `get_next` and
  `calc_next`).
- `max_iterations`: secant iterations before giving up (raises `RuntimeError`).
- `max_error`: the output tolerance to stop at.

Method: **`solve(target, estimate) -> float`**, where `target` is the wanted
output and `estimate` is a starting guess for the input.

```python
from xpma import LeadEMA, SecantSolver

filt = LeadEMA(10)
for x in [10.0, 10.2, 10.1]:
    filt.get_next(x)

solver = SecantSolver(filt)
recovered = solver.solve(target=11.0, estimate=10.5)   # input that makes it read 11.0
```

---

## Advanced: fractional-order interpolation

### `FractionalSmoothness(smoothness, make_filter)`

The mechanism the lag-reduced filters use to reach a fractional smoothness order
safely. Given a `make_filter` callable that builds a filter for a whole-number
order (`make_filter(order) -> filter`), it blends the two whole orders on either
side of `smoothness` at the output level. If the whole-order filters it blends
are themselves overshoot-free, the blend is too (a blend of two non-negative
responses stays non-negative); with an arbitrary `make_filter` that property is
only inherited, not created. At a whole-number `smoothness` it just delegates to
the single instance.

You only need this if you are building your own order-parameterised filter and
want the same fractional handling the built-ins get; for ordinary use the named
filters already apply it. It exposes the usual `get_next` / `calc_next`.

## Module-level helpers

You rarely call these directly (the named filters use them internally), but they
are public for analysis and for building your own `XPMA` configurations. Each
takes a smoothness order `s` and returns a lag-reduction fraction in `(0, 1)`.

| Function | Meaning | Valid `s` |
|---|---|---|
| `r_crit_m(s)` | The largest lag reduction that stays overshoot-free and monotone (used by `FastEMA`, `LeadEMA`). | `s >= 1` (where the guarantee is proven); the formula also evaluates for `0 < s < 1`, but the guarantee is not proven there |
| `max_monotone_lag_reduction(s)` | Identical to `r_crit_m`; the descriptive name. | as `r_crit_m` |
| `r_crit_c(s)` | The convex-approach boundary (used by `ConvexFastEMA`, `ConvexLeadEMA`); smaller than `r_crit_m`. | `s >= 1` |
| `r_crit_o(s)` | The no-overshoot boundary (used by the `Apex` filters); larger than `r_crit_m`. `e/4` at `s=1`. | integer `s >= 1` (zero raises) |
| `r_crit_o_effective(s)` | The `r_crit_o` value for any real `s`, interpolated between whole orders. | `s >= 1` |

For each boundary there is also the characteristic step-response time it comes
from, available if you want to reason about the shapes: `tau0_m(s)` (the stall
time behind `r_crit_m`), `tau_p_o(s)` (integer `s`, behind `r_crit_o`) and
`tau1_c(s)` (behind `r_crit_c`).

```python
from xpma import r_crit_m, r_crit_c, r_crit_o

r_crit_c(1), r_crit_m(1), r_crit_o(1)   # 0.4618..., 0.5602..., 0.6796...  (C < M < O)
```


---

**Next:** [theory.md](theory.md) for the ideas behind the family and what is
proven versus checked numerically; [../examples/](../examples/) for runnable
scripts to adapt. If you have not chosen a filter yet, start with
[choosing-a-filter.md](choosing-a-filter.md).
