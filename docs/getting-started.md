# Getting started

This page gets you from an empty environment to smoothing your own data, and
covers the four things you need to drive any filter in the package: the
streaming update, the stateless probe, resetting, and float periods.

## Install

`xpma` is pure Python with no required dependencies. Until it is published to a
package index, install it straight from the repository:

```bash
git clone https://github.com/SmoothingLab/xpma-python.git
cd xpma-python
pip install .
```

That gives you `import xpma`. Python 3.10 or newer is required (the code uses
the `X | None` type syntax). If you also want to run the plotting examples,
install the optional `examples` extra, which adds `matplotlib` and `numpy`:

```bash
pip install ".[examples]"
```

To work straight from a checkout without installing, point `PYTHONPATH` at the
source directory:

```bash
PYTHONPATH=src python3 your_script.py
```

## Your first smoothing

Every filter is a small object you feed one sample at a time. Create it with a
period, then call `get_next` for each new value:

```python
from xpma import EMA

ema = EMA(period=5)
for x in [1, 2, 3, 4, 5]:
    print(ema.get_next(x))
```

```
1
1.3333333333333333
1.8888888888888888
2.5925925925925926
3.3950617283950617
```

Two things to notice. The first output equals the first input: with no history
to average, the filter starts at the value it is given and warms up from there.
And the output lags the input (after feeding 5 it reads about 3.4), which is the
lag every smoother carries. If that lag is a problem, that is what the rest of
this library is about; see [choosing-a-filter.md](choosing-a-filter.md).

Most filters also take a `smoothness` (default 1), which is how many layers of
averaging they stack. Higher smoothness removes more noise:

```python
from xpma import FastEMA

series = [10.0, 10.4, 10.2, 11.1, 12.3, 11.8, 12.5, 13.0, 12.7, 13.4]
filt = FastEMA(period=20, smoothness=2)
smoothed = [filt.get_next(x) for x in series]
```

## Streaming with `get_next`, probing with `calc_next`

Two methods, on every filter:

- **`get_next(x)`** feeds the sample `x`, advances the filter's internal state,
  and returns the new output. This is the normal call.
- **`calc_next(x)`** returns what the output *would* be if the next sample were
  `x`, without advancing anything. It is a what-if probe you can call as many
  times as you like; the filter is unchanged.

```python
from xpma import FastEMA

filt = FastEMA(period=10)
for x in [10.0, 11.0, 12.0]:
    filt.get_next(x)

# Probe two hypothetical next samples. Neither changes the filter.
print(filt.calc_next(20.0))   # 14.8204...
print(filt.calc_next(20.0))   # 14.8204... (identical: no state changed)

# Now commit the real next sample.
print(filt.get_next(13.0))    # 11.8770...
```

The probe is what makes filters reversible: `ReverseFilter` wraps any filter and
recovers its input stream from its output stream (see the Reversal section of
the [API reference](api.md) and
[`examples/03_streaming_and_probe.py`](../examples/03_streaming_and_probe.py)).

## Handling missing or bad values

If you pass `None` or a non-finite number (`nan`, `inf`), the filters return
`None` and leave their state untouched, so a gap in your data does not corrupt
the running value. Check for `None` in warm-up-sensitive code:

```python
value = filt.get_next(sample)
if value is not None:
    use(value)
```

## Resetting

The filters have no `reset()` method. To start fresh, build a new instance;
construction is cheap and carries no history:

```python
from xpma import EMA

filt = EMA(10)
for x in old_data:
    filt.get_next(x)

filt = EMA(10)                 # fresh, no memory of old_data
print(filt.get_next(50.0))     # 50.0 (equals the first input again)
```

## Float periods and smoothness

Periods do not have to be whole numbers, which matters when you want equally
spaced periods on a logarithmic (geometric) scale rather than integer steps. Use
a period greater than 1 (the code accepts a period of 1 or below, but there the
filters degenerate toward a plain passthrough and the overshoot guarantees no
longer hold):

```python
from xpma import EMA, FastEMA

EMA(12.5)                      # fine
FastEMA(20, smoothness=1.5)    # fractional smoothness (>= 1) is fine too
```

Smoothness also takes fractional values, but the supported range depends on the
filter. The lag-reduced filters (`FastEMA`, `LeadEMA`, the `Convex` and `Apex`
pairs) are designed for smoothness `1` and above; a value between 0 and 1 either
falls back to order 1 or is rejected, depending on the filter. `IFEMA` and
`XEPMA` do accept orders below 1. The API reference states the exact range per
filter.

Fractional orders are realised carefully. `FastEMA` and `LeadEMA` keep their
proven no-overshoot guarantee between whole orders; the other lag-reduced filters
are handled the same way and rest on the same numerical evidence as at whole
orders. You do not need to think about how, only that `smoothness=1.5` behaves
sensibly between `1` and `2`.

## Next steps

- [choosing-a-filter.md](choosing-a-filter.md): which filter for which job.
- [api.md](api.md): the full reference.
- [theory.md](theory.md): what is proven versus checked numerically, and where
  the family comes from.
- [examples/](../examples/): runnable scripts you can adapt.
