# Schumann-Resonance Foam-Signature Pipeline

Two programs that take the raw Sierra Nevada ELF data and test the Schumann-
resonance parameters for a spacetime-foam signature, the honest way: remove the
known drivers first, then look only at what is left. The expected outcome is a
measured upper bound, not a detection — the same energy budget that excluded the
seismic and solar channels applies here too. The pipeline is built to make a
real signal, if one exists, impossible to miss, and to report a clean bound if
it does not.

--------------------------------------------------------------------------------
## Contents

| File | What it does |
|------|--------------|
| `elf_to_sr.py`     | Stream raw ELF hour-files into a Schumann-parameter table. Low, constant memory; crash-safe resume. |
| `foam_analyzer.py` | Run the foam / lunar / lightning tests on that table and emit a summary + figures. |
| `requirements.txt` | Python dependencies. |

## Install
```
pip install -r requirements.txt          # numpy, scipy, pandas, matplotlib
```

--------------------------------------------------------------------------------
## Workflow — three steps

### Step 1 — verify the raw binary format on ONE file (do this first)
```
python elf_to_sr.py -i /path/to/raw_root -o out --inspect
```
`--inspect` decodes a single hour-file and prints the `info.txt`, the implied
duration, and the first parameter rows. Two checks confirm we are reading the
bytes correctly:
- implied duration should be **~3600 s**, and
- the modes should land near **7.8 / 14.3 / 20.8 Hz**.

If the duration is off (e.g. ~1800 s or ~7200 s) the samples are not plain
little-endian int16; fix it with one flag — `--dtype` (`>i2`, `<i4`, `<f4`, …)
or `--header-bytes`. Catch this on one file, not after hours on the full archive.

### Step 2 — extract the parameters (resumable)
```
python elf_to_sr.py -i /path/to/raw_root -o out
```
- `-i` may be the **parent of all five unzipped years**; it walks the tree.
- Default cadence is **60 s** → Nyquist 8.3 mHz, which reaches the 2 mHz line.
  Change with `--cadence`.
- **Interrupted?** Re-run the *same command*. It reads `out/progress.log` and
  skips every file already finished — power loss, Ctrl-C, or crash all resume
  cleanly.
- A quick `--limit-files 10` trial first gives a real files/sec rate and ETA.
- Output: **`out/sr_parameters.csv`** — one row per minute per sensor, with
  mode-1/2/3 frequency, amplitude, and width.

The CSV is a few hundred MB over five years; `gzip` it before moving it, or run
one full year first (2014 or 2016 are clean) — already 700+ lunar cycles.

### Step 3 — run the tests
```
python foam_analyzer.py --sr out/sr_parameters.csv --out results
```
Add a lightning catalog to enable the coupling arms:
```
python foam_analyzer.py --sr out/sr_parameters.csv --lightning strokes.csv --out results
```
Output: **`results/summary.txt`** plus four figures (driver fit, lunar band,
2 mHz residual spectrum, ringdown stack).

--------------------------------------------------------------------------------
## What each arm tests (all on the residual after known drivers are removed)

1. **Lunar line (H3)** — a 12.42 h (lunar M2) line, separated from the 12.00 h
   solar line that was removed. Block-bootstrap null.
2. **Foam line (H1)** — 2.005 mHz (c / 1 AU) and its octaves, reachable only at
   ≲1-min cadence. The analyzer auto-skips and says so if the cadence is too
   coarse.
3. **Lightning dose-response** *(needs catalog)* — SR response per unit lightning.
   This is *expected to be real* (ordinary cavity physics); it is the baseline a
   foam signal must beat, not evidence of foam by itself.
4. **λ-sweep** *(needs catalog)* — builds Φ(t) = Σ Iᵢ/Rᵢ·e^(−Rᵢ/λ) for screening
   lengths 500 km → global and compares fit quality. A finite-λ term that
   improves the fit *beyond* ordinary cavity geometry would be the anomaly; the
   cavity-geometry confound is reported, not papered over.
5. **Stroke-triggered ringdown** *(needs catalog)* — superpose the residual after
   the largest strokes, fit a damped 2 mHz sinusoid, test against a random-trigger
   null.

## The lightning catalog (optional)
A CSV with columns for **time, lat, lon**, and optionally **current / peak_kA**.
Column names are auto-detected. WWLLN per-stroke data is ideal; a time-resolved
gridded product also works if reshaped to one row per cell-time with a count or
moment. If no current column is present, strokes are unit-weighted (and the
script says so).

--------------------------------------------------------------------------------
## Honesty notes (these are enforced in the code, not just here)
- Every foam arm is tested on the **residual** after the known drivers are removed.
- The **expected outcome is a bound**, printed up front.
- Framework magnitude predictions are **not** used. In particular the floated
  "Q ~ 10⁶" contradicts the measured Schumann Q of ~4–6, so the analyzer measures
  and bounds rather than assuming an amplitude.
- The **effective number of tests** is reported; any lone p < 0.05 among them is
  flagged as needing held-out confirmation before it means anything.

## Validation
Both programs were checked on synthetic data before use. `elf_to_sr.py`
round-trips int16 hour-files and resumes correctly after interruption.
`foam_analyzer.py` returns bounds when nothing is injected, and recovers an
injected lunar line, lightning dose-response, and stroke-triggered ringdown —
including pulling back the exact screening length that was planted.

--------------------------------------------------------------------------------
## Station
Sierra Nevada ELF station, 37.033°N, 3.317°W (override with
`foam_analyzer.py --station lat,lon`). Data: Rodríguez-Camacho et al. (2022),
raw on Zenodo, processed parameters on the Granada repository, CC-BY-4.0.
