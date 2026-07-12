#!/usr/bin/env python3
"""Smooth a noisy series with EMA, LeadEMA and XEPMA, and show the trade-off.

The point of this example is to make three plain-English ideas visible:

  - lag: a plain EMA runs behind the data. LeadEMA carries the same lag as the
    EMA but leads price toward each new level; XEPMA aims for zero lag.
  - overshoot: XEPMA is a level-plus-trend nowcast, so after a sharp move it can
    cross beyond the data and come back. LeadEMA is built never to do that.
  - the trade: less lag is bought with either more overshoot or more smoothing
    work. You choose where to sit on that scale.

The filtering runs on the standard library alone. matplotlib is only needed if
you pass --save or --show; without it the script still prints a numeric summary.

    python3 01_smoothing_comparison.py                # numbers only
    python3 01_smoothing_comparison.py --save out.png # write a chart
    python3 01_smoothing_comparison.py --show         # open a window
"""

import argparse
import math
import random

from xpma import EMA, LeadEMA, XEPMA

PERIOD = 20


def make_series(n: int = 240, seed: int = 7):
    """A smooth underlying signal (two ramps joined by a step) plus noise."""
    rng = random.Random(seed)
    clean, noisy = [], []
    for t in range(n):
        if t < 80:
            level = 10.0 + 0.05 * t
        elif t < 90:
            level = 14.0 + 1.2 * (t - 80)          # a fast run up
        else:
            level = 26.0 - 0.03 * (t - 90)
        clean.append(level)
        noisy.append(level + rng.gauss(0.0, 0.8))
    return clean, noisy


def run(filt, series):
    return [filt.get_next(x) for x in series]


def mean_lag(make_filter, n=400):
    """Rigorous mean lag: feed a steady ramp y = t, the output settles t - lag
    bars behind, so the shortfall is the mean lag in bars."""
    filt = make_filter()
    out = None
    for t in range(n):
        out = filt.get_next(float(t))
    return (n - 1) - out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", metavar="PATH", help="write the chart to PATH")
    ap.add_argument("--show", action="store_true", help="open a chart window")
    args = ap.parse_args()

    clean, noisy = make_series()
    makers = {
        "EMA(20)":     lambda: EMA(PERIOD),
        "LeadEMA(20)": lambda: LeadEMA(PERIOD),
        "XEPMA(20)":   lambda: XEPMA(PERIOD),
    }
    # Plot on the noisy series; measure on the clean one so the numbers are
    # deterministic and reflect the filter, not the particular noise draw.
    noisy_out = {name: run(make(), noisy) for name, make in makers.items()}
    clean_out = {name: run(make(), clean) for name, make in makers.items()}
    clean_peak = max(clean[80:120])

    print(f"{'filter':13s} {'mean lag':>9s} {'lag behind a fast rise':>23s} "
          f"{'overshoot above peak':>21s}")
    for name, make in makers.items():
        out = clean_out[name]
        shortfall = max(max(0.0, clean[t] - out[t]) for t in range(80, 96))
        overshoot = max(0.0, max(out[80:120]) - clean_peak)
        lag = mean_lag(make)
        if abs(lag) < 1e-6:
            lag = 0.0
        print(f"{name:13s} {lag:>6.2f} bars "
              f"{shortfall:>18.3f} {overshoot:>21.3f}")
    print("\nEMA and LeadEMA carry the same mean lag, but LeadEMA tracks a fast "
          "rise closer\n(smaller shortfall) because it undershoots less. XEPMA "
          "removes the lag almost\nentirely but overshoots the peak, then settles back.")

    filters = noisy_out

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
    ax.plot(noisy, color="0.75", lw=0.8, label="noisy input")
    ax.plot(clean, color="0.3", lw=1.2, ls="--", label="clean signal")
    for name, out in filters.items():
        ax.plot(out, lw=1.6, label=name)
    ax.set_title("EMA vs LeadEMA vs XEPMA on a noisy series")
    ax.set_xlabel("bar")
    ax.legend(loc="lower right")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f"\nsaved chart to {args.save}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
