# Examples

Small, self-contained scripts. Each one runs on the standard library plus
`xpma`; the plotting in the first two is optional and only needs `matplotlib`,
which you can install with the package's `examples` extra
(`pip install ".[examples]"`).

Run them from anywhere once `xpma` is installed (`pip install .` from the
package root), or straight from a checkout by pointing `PYTHONPATH` at the
source:

```bash
PYTHONPATH=src python3 examples/01_smoothing_comparison.py
```

| Script | What it shows |
|---|---|
| `01_smoothing_comparison.py` | EMA vs LeadEMA vs XEPMA on a noisy series: lag, tracking, and overshoot side by side. |
| `02_step_response.py` | Three lag-reducing filters (ConvexFastEMA, FastEMA, ApexFastEMA) reacting to a sudden level change: a numerical demonstration that none overshoots at the period and smoothness shown. |
| `03_streaming_and_probe.py` | Feeding samples with `get_next`, the stateless `calc_next` probe, float periods, and reversing a filter with `SecantSolver`. Pure standard library. |

Scripts 01 and 02 print a numeric summary by default. Pass `--save out.png` to
write a chart, or `--show` to open a window. No images are written unless you
ask for them.
