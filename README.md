# xpma

`xpma` is a small, dependency-free Python library of exponentially weighted
smoothers that give you independent control over two things every smoother
trades off: how much noise it removes (smoothness) and how far behind the data
it runs (lag). It includes a family of lag-reduced filters that are provably
free of overshoot (they never cross beyond the data after a sharp move), a
zero-lag endpoint filter, and the standard building blocks they are built from
(the EMA, cascades of EMAs, and a reverse-EMA that runs the EMA calculation
backwards).

If you have ever wanted an EMA that keeps up with price without lurching past
it, or a smoother whose lag you can dial down to zero when you understand the
cost, this is that toolbox.

## The three ideas, in plain English

- **Lag**: a smoother runs behind the data, because it averages in the past.
  More smoothing means more lag.
- **Overshoot**: some filters buy back the lag by extrapolating the recent
  trend. Push that too far and after a sharp move the filter crosses beyond the
  data and has to come back. That is overshoot, and it feeds downstream rules
  values the data never contained.
- **The trade**: you can have less lag, more smoothness, or no overshoot; you
  cannot freely have all three. This library lets you choose where you sit,
  and several of its filters give up nothing on overshoot at all.

## Install

The package is pure Python (no compiled parts, no required dependencies). Until
it is published to a package index, install it straight from the repository:

```bash
git clone https://github.com/SmoothingLab/xpma-python.git
cd xpma-python
pip install .
```

The plotting in two of the examples needs `matplotlib`, which you can pull in
with the optional `examples` extra (it also installs `numpy`):

```bash
pip install ".[examples]"
```

The library itself never needs either.

## Quickstart

```python
from xpma import LeadEMA

prices = [10.0, 10.4, 10.2, 11.1, 12.3, 11.8, 12.5, 13.0, 12.7, 13.4]

smoother = LeadEMA(period=10)                 # smoothness defaults to 1
smoothed = [smoother.get_next(p) for p in prices]

print(smoothed[-1])                           # the current smoothed value
```

Every filter shares the same tiny interface: `get_next(x)` feeds one new sample
and returns the updated value; `calc_next(x)` asks "what would the output be for
this next sample?" without changing anything. Periods and smoothness accept
fractional values, not just whole ones; the supported ranges are per-filter and
are listed in the API reference.

## Which filter do I want?

Every filter below takes a `period` (roughly, how many samples it averages over)
and, where shown, a `smoothness` (how many layers of averaging: 1 behaves like a
single EMA, higher removes more noise). The rows are grouped by role, not ranked,
so read for the job that matches yours rather than top to bottom.

| Filter | In one sentence |
|---|---|
| `EMA(period)` | The plain exponential moving average: simple and smooth, but lags the data. |
| `LeadEMA(period, smoothness=1)` | Same lag as the EMA, spent on tracking a move more closely (it leads price with less undershoot) rather than on speed. No overshoot. The safe default when the smoothed line is a reference level compared against price. |
| `FastEMA(period, smoothness=1)` | Same nominal period as `EMA(period)` but genuinely less lag: it reaches a new level sooner, rising straight to it with no overshoot. The faster response passes more noise (at period 20, smoothness 1, about 2.9 times the EMA's), so it is not a free lunch. |
| `ConvexFastEMA(period, smoothness=1)` | A touch slower than `FastEMA`; at smoothness 1 its approach to a new level only ever slows down, and at higher smoothness it rises to a single peak rate then slows without re-accelerating. The gentlest choice for reading slope or curvature. No overshoot. |
| `ConvexLeadEMA(period, smoothness=1)` | The lag-matched sibling of `ConvexFastEMA`: the EMA's lag, spent on closer tracking with that same non-re-accelerating approach. |
| `ApexFastEMA(period, smoothness=1)` | The fastest of the no-overshoot filters at a given period: it climbs closest to the level, eases into a small dip just below, then settles. It never crosses above the level, and only just touches it (exactly, only in the idealised continuous limit). |
| `ApexLeadEMA(period, smoothness=1)` | The lag-matched sibling of `ApexFastEMA`: the EMA's lag with the least undershoot of the lag-matched filters, at the cost of a small dip. |
| `XEPMA(period, smoothness=1)` | The zero-lag endpoint: it reads the current level and slope and projects them to now, so it has essentially no lag and is the fastest of all, but it can overshoot after a sharp move (that is the trade). Excellent for oscillators, risky as a reference level. |
| `XPMA(period, smoothness, lag_reduction)` | The general family the Fast, Lead, Convex and Apex filters (and the `XEPMA` endpoint) are points on. Dial `lag_reduction` yourself from 0 (plain cascade, most lag) to 1 (the zero-lag `XEPMA`). |
| `IFEMA(period, smoothness)` | A smoother of any fractional smoothness order, with no lag reduction. Use it for a smoothness between whole numbers; it is also the core the others build on. |

The **Fast** filters run at the period you give and spend their lag reduction on
speed. Their **Lead** siblings inflate the period internally so the lag matches a
plain `EMA(period)`, spending the same reduction on tracking closer instead. The
three flavours (Convex, plain, Apex) are three settings on the same dial, from
the most cautious approach to the fastest.

For a proper walk-through of that choice, see
[docs/choosing-a-filter.md](docs/choosing-a-filter.md).

## Documentation

- [docs/getting-started.md](docs/getting-started.md): install, your first
  smoothing, streaming vs the stateless probe, float periods.
- [docs/choosing-a-filter.md](docs/choosing-a-filter.md): the plain-English
  decision guide, the trade-off triangle, and when overshoot matters.
- [docs/api.md](docs/api.md): each documented class and helper, with its
  parameters and their valid ranges.
- [docs/theory.md](docs/theory.md): a gentle bridge to the mathematics, and what
  is proven versus what is checked numerically.
- [examples/](examples/): three small, runnable scripts.

## What is proven, and what is checked

This library is honest about its guarantees. All of the following holds for
every smoothness order 1 and above, including order 1 itself:

- `FastEMA` and `LeadEMA` are **proven** overshoot-free (their step response is
  monotone), both in the idealised continuous limit and at any finite period.
- `ConvexFastEMA` and `ConvexLeadEMA` have their convex-approach property
  **proven** in the continuous limit. What rests on measurement is only the
  finite-period margin: at every period and order tested the discrete filter
  stays on the safe side, by a gap that shrinks as the period grows.
- `ApexFastEMA` and `ApexLeadEMA` sit at a boundary whose defining constant is
  **exact** (confirmed symbolically for smoothness 1 through 8); their
  no-overshoot property at a finite period is backed by **numerical evidence**
  (tested for smoothness 1 to 4 and periods 5 to 200), not a full proof.

So none of these is "proven only above smoothness 1": the underlying properties
hold at every order including 1, and what is numerical is finite-period safety.
If you need the guarantee to hold absolutely, prefer `FastEMA` or `LeadEMA`.
`XEPMA` overshoots by design: that is the price of its zero lag. The repository
ships a machine-checkable `proofs/` tree (see [proofs/README.md](proofs/README.md))
that verifies the paper's claims.

## Paper

The theory is written up in:

> Marcus Don, *Maximal monotone lag reduction in exponentially weighted
> smoothers: the two-rate discounted-regression family and its critical
> reductions* (2026). Preprint forthcoming; Zenodo DOI: _to be assigned_.

## Licence

Released under the MIT Licence. See [LICENSE](LICENSE).
