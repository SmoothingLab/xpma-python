# A little theory

You do not need this page to use the library. It is here for the curious: a
plain-language bridge to the ideas the filters are built on, with pointers to the
paper and the machine-checkable proofs for the full story. There are a couple of
formulas, kept to the end of each section and safe to skip.

## Smoothing is fitting a line to the recent past

Take the plainest view of a smoother. At each step it looks at the recent
samples and fits a line through them, giving more weight to some samples than
others. The output is the level of that fitted line. Because the fit is anchored
on the weighted middle of the past, that level sits a little behind the present:
that is the lag.

- A **simple moving average** weights the last few samples equally. That is
  ordinary least-squares fitting over a fixed window.
- An **exponential moving average** weights recent samples more and older ones
  progressively less, never quite forgetting. That is the same least-squares fit
  with a *discount* on the past, which is why this family is a "discounted
  regression".

Seen this way, a smoother gives you more than a level. The fitted line also has a
**slope**, and the two together let you ask a sharper question: instead of
reading the line where it sat a few samples back (the lagging level), read it
where the slope says it should be *now*. That is what lag reduction is: advancing
the point at which you read the fitted line, from "the middle of the window"
toward "the present".

## Lag reduction, and where overshoot comes from

Call the advance `r`. At `r = 0` you read the plain, lagging level. As `r` grows
you slide the reading point forward along the trend, and the lag shrinks. At the
far end, the reading point reaches the present and the lag is gone: that is the
zero-lag endpoint, `XEPMA`.

The catch is that the slope is estimated from noisy, past data. Push the advance
too far and the filter starts *extrapolating* a trend that has already ended, so
after a sharp move it sails past the data and has to come back. That is
overshoot. In precise terms, overshoot corresponds to the filter's response to a
one-off spike going negative somewhere; a filter whose spike response never goes
negative can never push its output beyond the range of its input. That
"never-negative" property is the strong form of "no overshoot", and it is the
guarantee the overshoot-free filters are built to keep.

So the central question of the whole family is: **how far can you advance the
reading point before overshoot appears?**

## Why the obvious construction fails, and the two-rate fix

The obvious way to estimate the trend is from an EMA and an EMA-of-an-EMA (this is
the classical DEMA, and the building block of Tillson's T3). It turns out this
**overshoots for every advance greater than zero**: the correction it adds decays
at the same rate as the level it corrects, and that shared slow tail always
produces a small overshoot. This is easy to get wrong: a modest, partial lag
reduction looks safe but is not, for this construction (the overshoot is tiny at
small advances, but never exactly zero).

The fix that this library is built on is to take the trend correction from a
cascade one *order higher*, which is lag-matched to the level but decays strictly
faster. That faster-decaying correction restores a genuine safe advance: a
largest reduction at which the response is still perfectly monotone. That value
has a closed form at each smoothness order `s`:

```
r_crit_m(s) = s^(s+1) / (s+1)^(s+2) * e^((2s+1)/(s+1))
```

`FastEMA` and `LeadEMA` sit exactly here. Just inside and just outside it are two
more boundaries, forming a ladder:

```
r_crit_c(s)   <   r_crit_m(s)   <   r_crit_o(s)
convex           monotone           no-overshoot
approach         (no dip)           (small dip allowed)
```

- Below `r_crit_c` the approach to a new level only ever slows, like a plain
  EMA's (`ConvexFastEMA`).
- At `r_crit_m` the approach is a clean, straight, monotone rise (`FastEMA`,
  `LeadEMA`).
- At `r_crit_o` you have squeezed out the most advance possible with still no
  overshoot (in the idealised continuous limit; at a finite period the safe
  boundary can sit a little higher), at the cost of a small dip below the level
  first (the `Apex` filters).

Beyond `r_crit_o`, overshoot is unavoidable in that same limit, all the way to
the zero-lag endpoint `XEPMA`, which overshoots by design in exchange for having
no lag at all. At smoothness `s = 1` that endpoint has a bonus property: it
reproduces not just straight-line trends but gently curved (quadratic) ones with
no error whatsoever. That is what "quadratic-exact" means, and it holds only at
`s = 1` (the `QuadraticXEPMA` variant restores it at higher orders).

## What is proven, and what is checked numerically

The library is careful to distinguish what is proven from what is measured, and
so is this documentation. All of the following holds for every smoothness order
`s >= 1`, including `s = 1`.

- **`FastEMA`, `LeadEMA` (the `r_crit_m` boundary): fully proven.** The monotone,
  overshoot-free property holds in discrete time, at every period, not just in
  the idealised continuous limit. This is what makes them the filters to reach
  for when the guarantee must hold absolutely.
- **`ConvexFastEMA`, `ConvexLeadEMA` (the `r_crit_c` boundary): property proven,
  margin measured.** The convex-approach property of the boundary is proven in
  the continuous limit (for every real order above 1, and exactly at order 1).
  What rests on measurement is the finite-period margin: at every order and
  period tested, the true discrete boundary sits a little above the continuous
  constant the filter uses, so the filter stays on the safe side, and that gap
  shrinks like one over the period.
- **`ApexFastEMA`, `ApexLeadEMA` (the `r_crit_o` boundary): constant exact,
  finite-period safety measured.** The boundary constant itself is exact (it is
  the root of an exact polynomial reduction, and that root is proven to be the
  relevant one), confirmed symbolically for orders 1 through 8. The no-overshoot
  property at a finite period is supported by numerical evidence only, tested for
  orders 1 to 4 and periods 5 to 200.

So it is not the case that these properties are "proven only above order 1". The
continuous properties are proven (Convex) or exact with symbolic confirmation
(Apex) at every order including 1; what is numerical, for both, is safety at a
finite period. In practice the Convex and Apex filters are very well supported;
prefer `FastEMA` or `LeadEMA` if you need the proven version.

## Where to go for the full story

- The paper: Marcus Don, *Maximal monotone lag reduction in exponentially
  weighted smoothers: the two-rate discounted-regression family and its critical
  reductions* (2026; preprint forthcoming, Zenodo DOI to be assigned). It gives
  the derivations, the connections to the classical smoothers (Brown, Holt,
  Tillson), and the benchmark results.
- The `proofs/` tree in this repository ships machine-checkable verification of
  the paper's key claims, so you can confirm them rather than take them on trust.
  Start at `proofs/README.md`.
