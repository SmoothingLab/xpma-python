#!/usr/bin/env python3
"""Streaming updates, the stateless probe, float periods, and reversal.

This example uses the standard library only (no numpy, no matplotlib). It shows
the four things you need to drive any filter in the package:

  1. get_next(x): feed one new sample, advance the filter, return the new value.
  2. calc_next(x): ask "what would the output be if the next sample were x?"
     without advancing anything. Safe to call as many times as you like.
  3. float periods: periods do not have to be whole numbers.
  4. reversal: because calc_next is a clean probe, SecantSolver can invert a
     filter, recovering the input that would produce a wanted output.

    python3 03_streaming_and_probe.py
"""

from xpma import EMA, LeadEMA, SecantSolver


def streaming_and_probe():
    print("1 + 2) streaming with get_next, probing with calc_next")
    filt = LeadEMA(period=10)
    for x in [10.0, 10.5, 11.2, 10.9, 11.8]:
        filt.get_next(x)

    # The probe does not advance the filter: call it repeatedly, get the same
    # answer, and the running state is untouched.
    what_if_20 = filt.calc_next(20.0)
    what_if_5 = filt.calc_next(5.0)
    print(f"   if the next sample were 20.0 the output would be {what_if_20:.4f}")
    print(f"   if the next sample were  5.0 the output would be {what_if_5:.4f}")

    committed = filt.get_next(12.0)   # now actually advance with the real sample
    print(f"   committed the real sample 12.0 -> output {committed:.4f}\n")


def float_periods():
    print("3) float periods are fine")
    for p in (10, 12.5, 15.75):
        f = EMA(p)
        out = None
        for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
            out = f.get_next(x)
        print(f"   EMA(period={p:>5}) after 1..5 -> {out:.4f}")
    print()


def reset_by_reconstruction():
    print("4a) there is no reset method: build a fresh instance to start over")
    f = EMA(10)
    for x in [100.0, 101.0, 102.0]:
        f.get_next(x)
    f = EMA(10)   # fresh, no history
    print(f"   fresh EMA(10) first output = {f.get_next(50.0):.4f} "
          f"(equals the first input)\n")


def reversal():
    print("4b) reverse a filter with SecantSolver (uses calc_next internally)")
    filt = LeadEMA(period=10)
    for x in [10.0, 10.2, 10.1, 10.4, 10.3]:
        filt.get_next(x)

    solver = SecantSolver(filt)
    target_output = 11.0
    # Find the input that makes the filter output land on target_output. The
    # solver probes with calc_next while iterating and commits the final input
    # once it converges.
    recovered_input = solver.solve(target=target_output, estimate=10.5)
    print(f"   input needed for the filter to read {target_output:.2f} "
          f"= {recovered_input:.4f}")


def main():
    streaming_and_probe()
    float_periods()
    reset_by_reconstruction()
    reversal()


if __name__ == "__main__":
    main()
