#!/usr/bin/env python3
"""Compare the step responses of the three overshoot-free lag-reducing filters.

All three sit at a different "how far can I reduce the lag" boundary, from the
most cautious to the fastest:

  - ConvexFastEMA: at the smoothness 1 used here, the approach to a new level
    only ever slows, like a plain EMA. The gentlest read for a slope or curvature.
  - FastEMA: the fastest response that still rises straight to the level with no
    dip and no overshoot (monotone).
  - ApexFastEMA: the fastest response that never crosses above the level, at the
    cost of a small dip just below it before settling (not monotone).

EMA is shown as the slow baseline. Watch the time each filter takes to reach the
new level: ApexFastEMA < FastEMA < ConvexFastEMA < EMA. This is a numerical
demonstration at the period and smoothness set below: none of the three
overshoots here. (FastEMA and LeadEMA are proven overshoot-free for smoothness
>= 1; the Convex and Apex boundaries rest on numerical evidence.)

    python3 02_step_response.py                # numbers only
    python3 02_step_response.py --save out.png # write a chart
    python3 02_step_response.py --show         # open a window
"""

import argparse

from xpma import EMA, FastEMA, ConvexFastEMA, ApexFastEMA

PERIOD = 20
SMOOTHNESS = 1
WARMUP = 60
LENGTH = 120


def step_response(filt):
    """Settle the filter at 0, then feed a step up to 1 and record the output."""
    for _ in range(WARMUP):
        filt.get_next(0.0)
    return [filt.get_next(1.0) for _ in range(LENGTH)]


def bars_to_reach(resp, frac=0.9):
    return next((i for i, v in enumerate(resp) if v >= frac), None)


def overshoot(resp):
    return max(0.0, max(resp) - 1.0)


def is_monotone(resp):
    return all(resp[i + 1] >= resp[i] - 1e-12 for i in range(len(resp) - 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", metavar="PATH", help="write the chart to PATH")
    ap.add_argument("--show", action="store_true", help="open a chart window")
    args = ap.parse_args()

    responses = {
        "EMA":           step_response(EMA(PERIOD)),
        "ConvexFastEMA": step_response(ConvexFastEMA(PERIOD, SMOOTHNESS)),
        "FastEMA":       step_response(FastEMA(PERIOD, SMOOTHNESS)),
        "ApexFastEMA":   step_response(ApexFastEMA(PERIOD, SMOOTHNESS)),
    }

    print(f"{'filter':15s} {'bars to 90%':>12s} {'overshoot':>10s} {'monotone':>9s}")
    for name, resp in responses.items():
        print(f"{name:15s} {str(bars_to_reach(resp)):>12s} "
              f"{overshoot(resp):>10.4f} {str(is_monotone(resp)):>9s}")
    print("\nApexFastEMA climbs closest to the level, then dips just below "
          "before settling\n(not monotone). FastEMA and ConvexFastEMA rise "
          "straight to it. None overshoot here.")

    if not (args.save or args.show):
        return

    try:
        import matplotlib
        if args.save and not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: install it to use --save/--show)")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(1.0, color="0.6", lw=1.0, ls="--", label="target level")
    for name, resp in responses.items():
        ax.plot(resp, lw=1.7, label=name)
    ax.set_title("Step response: EMA vs ConvexFastEMA vs FastEMA vs ApexFastEMA "
                 f"(period {PERIOD}, s={SMOOTHNESS})")
    ax.set_xlabel("bars after the step")
    ax.set_ylabel("output")
    ax.legend(loc="lower right")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f"\nsaved chart to {args.save}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
