"""Maximal monotone lag-reduction coefficient r_crit^M(s)."""

import math


def max_monotone_lag_reduction(smoothness: float) -> float:
    """r_crit^M(s) = (s^(s+1) / (s+1)^(s+2)) * e^((2s+1)/(s+1)).

    Largest lag reduction whose unit-step response stays monotone (no sub-target dip,
    no overshoot). Standard lag reduction for FastEMA and LeadEMA.
    """
    s = float(smoothness)
    return (s ** (s + 1.0) / (s + 1.0) ** (s + 2.0)) * math.exp((2.0 * s + 1.0) / (s + 1.0))


# Alias of max_monotone_lag_reduction.
lead_ema_max_lag_reduction = max_monotone_lag_reduction
