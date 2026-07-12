# Choosing a filter

This is the decision guide. It assumes no mathematics; the reasoning lives in
plain language, and the theory page picks up where this leaves off.

## The trade-off triangle

Every smoother that uses only past and present data (a causal smoother) juggles
three things:

- **Smoothness**: how much of the wiggle it removes. More smoothness means a
  cleaner line.
- **Lag**: how far behind the data it runs. A smoother averages the past, so it
  reacts late. More smoothing means more lag.
- **Overshoot**: whether, after a sharp move, the filter crosses beyond the data
  and has to come back. Filters that fight lag by projecting the recent trend
  can overshoot.

You cannot maximise all three at once. The whole point of this library is to let
you pick which corner you give ground on, and to offer a set of filters that
refuse to give any ground on overshoot at all.

A useful way to hold it: **smoothness is set by `period` and `smoothness`; lag
and overshoot are set by which filter you choose.**

## First question: is overshoot allowed?

This is the decision that matters most, and it comes down to how the smoothed
value is used.

- **The smoothed line is a reference level** compared against the raw data: a
  band centreline, a breakout or pullback level, a trend anchor you ask "is
  price above or below it?". Here overshoot is actively harmful: an overshooting
  filter swings past price and flips the comparison for no real reason. **Use a
  no-overshoot filter** (`LeadEMA` by default, or one of the `Fast`/`Convex`/
  `Apex` family).
- **The smoothed value is an oscillator or an input to a ratio** (a momentum
  reading, a normalised `numerator / (numerator + denominator)` measure). Here a
  little overshoot is often harmless or even cancels out, and the extra speed of
  a zero-lag filter is worth having. **`XEPMA` is in play.**

If you are not sure, treat it as a reference level and stay overshoot-free. It
is the safe direction to be wrong in.

## Nominal period versus lag-matched: Fast\* and Lead\*

Several filters come in two siblings that spend the same lag reduction in
opposite ways:

- **`Fast*` (nominal period)** keeps the period you gave and spends the saved lag
  on **speed**: it reaches a new level sooner than `EMA(period)`.
- **`Lead*` (lag-matched)** inflates the period internally so its lag matches
  `EMA(period)` exactly, and spends the saving on **tracking closer** instead: at
  the same lag it leads price toward each new level with less undershoot.

So `FastEMA(20)` is faster than `EMA(20)`; `LeadEMA(20)` is exactly as laggy as
`EMA(20)` but hugs a rising or falling series more tightly. Pick `Fast*` when you
want responsiveness at a chosen smoothing; pick `Lead*` when you have a lag
budget (often "match my existing EMA") and want the best tracking within it.

The speed is not free: a `Fast*` filter's quicker response passes more of the
noise through. At period 20 and smoothness 1, `FastEMA` passes about 2.9 times
the noise energy of `EMA(20)`; the `Lead*` siblings, running at matched lag, pay
far less.

**`LeadEMA` is the sensible default for reference-level roles.** Matching a
familiar EMA's lag while tracking closer, and never overshooting, is exactly what
a centreline or trend anchor wants.

## The three no-overshoot boundaries: cautious to fast

Within the no-overshoot filters there are three settings, which are really three
answers to "how hard can I push the lag down before the step response misbehaves?"
From most cautious to fastest, at a given period:

| Filter | Approach to a new level | Speed |
|---|---|---|
| `ConvexFastEMA` | At smoothness 1, only ever slows down, like a plain EMA's approach; at higher smoothness it rises to a single peak rate then slows without re-accelerating. The gentlest shape. | Slowest of the three |
| `FastEMA` | Rises straight to the level, no dip, no overshoot (monotone). | Middle |
| `ApexFastEMA` | Climbs closest to the level, then dips just below before settling (it grazes the level exactly only in the continuous limit). | Fastest |

All three refuse to overshoot. They differ in the *shape* of the approach:

- Choose **`ConvexFastEMA`** when you read the smoothed line's slope or curvature
  (momentum, acceleration). Its approach settles without re-accelerating, so a
  slope read does not see a false second burst of momentum as the filter catches
  up.
- Choose **`FastEMA`** as the general-purpose overshoot-free speed-up: a clean,
  straight approach with no surprises.
- Choose **`ApexFastEMA`** when you want the most speed you can get without
  overshooting and can tolerate a small dip below the level before it settles.

Each has a lag-matched sibling (`ConvexLeadEMA`, `LeadEMA`, `ApexLeadEMA`) that
applies the same three shapes at `EMA(period)`'s lag instead of at speed.

A note on confidence, all for smoothness 1 and above: `FastEMA` and `LeadEMA`'s
no-overshoot behaviour is **proven**, at any period. `ConvexFastEMA`'s approach
shape is **proven** in principle (in the continuous limit); only its
finite-period margin rests on measurement. The `Apex` filters sit at an **exact**
boundary, but their no-overshoot property at a finite period rests on numerical
evidence. In short, what is measured rather than proven for the Convex and Apex
filters is finite-period safety, not the underlying property. If your application
depends on the guarantee absolutely, prefer the proven pair. See
[theory.md](theory.md).

## `XEPMA`: the zero-lag endpoint (and its honest caveat)

`XEPMA` is the far end of the same dial: instead of a careful, bounded lag
reduction it removes the lag almost entirely. It does this by reading the current
level and the current trend together and projecting to now, a level-plus-trend
nowcast in the spirit of Holt and Brown.

The honest caveat: because it extrapolates the trend, **`XEPMA` can overshoot**
after a sharp move, crossing beyond the data and settling back. That is not a bug
to be tuned away; it is the intrinsic cost of zero lag. So:

- `XEPMA` shines as an **oscillator** or a fast signal where the overshoot does
  not corrupt a downstream comparison, and its lack of lag is the point.
- `XEPMA` is a **poor reference level**: used as a centreline or a
  "price above the line?" anchor, its overshoot oscillates across price and
  produces false flips. Use `LeadEMA` there instead.

At smoothness 1, `XEPMA` has a special exactness property: it reproduces not just
straight-line trends but gently curved (quadratic) ones with no error at all. At
higher smoothness it is still zero-lag, but it loses that exactness and the
overshoot grows. The theory page explains this.

### The two XEPMA variants

Two siblings adjust the endpoint's trade in opposite directions, and both keep
its zero lag:

- **`DampedXEPMA`** is the one to reach for in practice: it gives back a little
  of the endpoint's trend exactness to reduce the overshoot, at every period and
  smoothness. If you like `XEPMA`'s speed but its overshoot keeps stinging, try
  this before abandoning zero lag altogether.
- **`QuadraticXEPMA`** goes the other way: it restores the curved-trend
  exactness that plain `XEPMA` only has at smoothness 1, making it exact at
  every order, but pays in extra noise. It exists because the mathematics allows
  it; reach for it only when exact curvature tracking matters more than noise.

## Dialling it yourself: `XPMA`

`XPMA(period, smoothness, lag_reduction)` is the general family that the Fast,
Lead, Convex and Apex filters, and the `XEPMA` endpoint, are specific points on
(`QuadraticXEPMA` and `DampedXEPMA` are separate constructions, off this dial).
`lag_reduction` runs from `0` (no reduction: a plain, maximally
smooth cascade with full lag) to `1` (the zero-lag `XEPMA` endpoint); values
outside that range are accepted but extrapolate beyond the family, so stay within
`0` to `1` unless you have a reason not to. The named filters exist because
picking a *safe* value of
`lag_reduction` is the hard part, and each one bakes in the value that hits a
particular boundary. Reach for raw `XPMA` only when you want to explore the dial
directly; otherwise use the named filters.

## Fractional smoothness: `IFEMA`

If you want a smoothness *between* whole numbers (say 2.5), and you do not need
lag reduction, `IFEMA(period, smoothness)` is the smoother to use. It is also the
core that the lag-reduced filters are built on. Smoothness below 1 is meaningful
too: it removes a little less noise than a single EMA (a gentler roll-off),
filling the gap under a single EMA.

## Quick reference

- Reference level, want it safe: **`LeadEMA`**.
- Reference level, want more speed and can accept a small dip: **`ApexLeadEMA`**
  or, at nominal period, **`ApexFastEMA`**.
- Reading slope or curvature: **`ConvexFastEMA`** (or `ConvexLeadEMA`).
- Faster than EMA at the same nominal period (a little more noise), no overshoot: **`FastEMA`**.
- Oscillator or ratio input, want zero lag, overshoot acceptable: **`XEPMA`**.
- Zero lag, but the overshoot hurts: **`DampedXEPMA`**.
- A smoothness between whole numbers: **`IFEMA`**.
- Explore the lag dial yourself: **`XPMA`**.

---

**Next:** [api.md](api.md) has the exact signature, parameters and valid ranges
for whichever filter you have picked; [theory.md](theory.md) explains what is
proven versus checked numerically behind these recommendations.
