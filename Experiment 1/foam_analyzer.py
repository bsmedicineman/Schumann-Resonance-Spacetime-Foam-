#!/usr/bin/env python3
"""
foam_analyzer.py
================
Tests a Schumann-resonance parameter series for foam signatures, the honest way:
remove the known drivers first, then look only at the residual.

Runs, in order:
  (1) Conditioning + QC of the SR parameter table (output of elf_to_sr.py).
  (2) Known-driver regression — solar tides (24/12/8/6 h), their seasonal
      sidebands, annual/semiannual terms, and a slow trend — removed by least
      squares.  Everything below is tested on the RESIDUAL.
  (3) Lunar test (H3): is there a 12.42 h (lunar M2) line, cleanly separated
      from the 12.00 h solar S2 that was just removed?  Block-bootstrap null.
  (4) Foam-line test (H1): is there a 2.005 mHz line (and its octaves, where the
      cadence allows)?  Needs ~1-min cadence to reach 2 mHz.  Block-bootstrap null.
  (5) Lightning arms (only if --lightning is given):
        a. Dose-response: add the lightning source series as a regressor and
           measure the SR response per unit lightning (this part is EXPECTED to
           be real — ordinary cavity physics).
        b. lambda-sweep: build Phi(t)=sum_i I_i/R_i * exp(-R_i/lambda) for a range
           of screening lengths and compare fit quality.  A finite-lambda term that
           improves the fit BEYOND ordinary cavity geometry would be the anomaly.
        c. Ringdown stack: superpose the residual in the minutes after the
           largest strokes and look for a damped 2 mHz ringing scaling with
           I/R*exp(-R/lambda).  Random-trigger null.

HONESTY NOTES baked in:
  * The expected outcome of every foam arm is a BOUND, not a detection — the
    same energy budget that excluded the seismic and solar channels applies here.
  * No magnitude prediction from the framework is assumed.  In particular the
    "Q ~ 1e6" and LIGO-scaling numbers floated elsewhere are NOT used; the
    measured Schumann Q is ~4-6 and the script simply measures and bounds.
  * Several parameters x several lambda x several lines are tested; any lone "hit"
    among them needs held-out confirmation before it means anything.  The script
    prints the effective number of tests so you can judge look-elsewhere.

USAGE
  python foam_analyzer.py --sr sr_parameters.csv[.gz] --out results/
  python foam_analyzer.py --sr sr_parameters.csv.gz --lightning wwlln.csv --out results/

DEPENDENCIES:  pip install numpy scipy pandas matplotlib
"""
import argparse, os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

FOAM_HZ = 0.0020045          # c / 1 AU  (light-crossing time of 1 AU)
LUNAR_H = 12.4206            # lunar semidiurnal M2 period, hours
SOLAR_S2_H = 12.000          # solar semidiurnal (removed in regression; sanity ref)
STA_LAT, STA_LON = 37.033, -3.317     # Sierra Nevada ELF station (37 02 N, 3 19 W)
NAVY = "#1F3864"; RED = "#b00020"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------------------- #
#  1. load + condition
# ----------------------------------------------------------------------------- #
def load_sr(path, mode=1):
    log(f"Loading {path}")
    df = pd.read_csv(path)
    fcol, acol, wcol = f"m{mode}_freq", f"m{mode}_amp", f"m{mode}_width"
    need = {"timestamp", "sensor", fcol, acol, wcol}
    if not need.issubset(df.columns):
        sys.exit(f"CSV missing columns {need - set(df.columns)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "saturated" in df:
        df = df[df["saturated"] == 0]
    lo_f, hi_f = (5.0, 10.5) if mode == 1 else (11.0, 25.0)
    ok = (df[fcol].between(lo_f, hi_f)) & (df[acol] > 0) & (df[wcol].between(0.1, 4.0))
    df = df[ok]
    # combine the two horizontal sensors: total power = P_NS + P_EW
    piv = df.pivot_table(index="timestamp", columns="sensor", values=acol, aggfunc="mean")
    power = piv.sum(axis=1, min_count=1)             # summed mode power
    amp = np.sqrt(power)                              # total horizontal amplitude
    freq = df.pivot_table(index="timestamp", columns="sensor", values=fcol,
                          aggfunc="mean").mean(axis=1)
    out = pd.DataFrame({"amp": amp, "freq": freq}).dropna(how="all")
    # uniform grid at the native cadence
    dt = pd.Series(out.index).diff().median()
    cad = dt.total_seconds()
    log(f"  native cadence ~{cad:.0f} s  ({len(out)} valid bins, "
        f"{(out.index.max()-out.index.min()).days} d span)")
    grid = pd.date_range(out.index.min(), out.index.max(), freq=dt)
    out = out.reindex(grid)
    return out, cad


# ----------------------------------------------------------------------------- #
#  2. known-driver regression
# ----------------------------------------------------------------------------- #
def design_matrix(t_h):
    """t_h = hours from start. Columns: trend + solar tides + seasonal + sidebands."""
    cols = [np.ones_like(t_h), t_h / t_h.max()]
    names = ["dc", "trend"]
    for P, nm in [(24, "S1"), (12, "S2"), (8, "S3"), (6, "S4")]:
        w = 2 * np.pi / P
        cols += [np.sin(w * t_h), np.cos(w * t_h)]; names += [f"{nm}s", f"{nm}c"]
    for P, nm in [(365.25 * 24, "yr"), (182.6 * 24, "semiyr")]:
        w = 2 * np.pi / P
        cols += [np.sin(w * t_h), np.cos(w * t_h)]; names += [f"{nm}s", f"{nm}c"]
    # seasonal sidebands of the diurnal & semidiurnal tides (strong in SR)
    wy = 2 * np.pi / (365.25 * 24)
    for P, nm in [(24, "S1"), (12, "S2")]:
        w = 2 * np.pi / P
        cols += [np.sin(w * t_h) * np.cos(wy * t_h), np.cos(w * t_h) * np.cos(wy * t_h)]
        names += [f"{nm}xyr_s", f"{nm}xyr_c"]
    return np.array(cols).T, names


def regress(t_h, y, valid):
    X, names = design_matrix(t_h)
    Xv, yv = X[valid], y[valid]
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    model = X @ beta
    resid = y - model
    var_expl = 1 - np.nanvar(resid[valid]) / np.nanvar(yv)
    log(f"  regression removed known drivers (R^2={var_expl:.3f}); testing residual")
    return resid, model, dict(zip(names, beta)), var_expl


# ----------------------------------------------------------------------------- #
#  line test with circular block-bootstrap null
# ----------------------------------------------------------------------------- #
def line_amp(t_h, r, valid, period_h):
    w = 2 * np.pi / period_h
    s, c = np.sin(w * t_h[valid]), np.cos(w * t_h[valid])
    rv = r[valid]
    A = np.array([s, c]).T
    coef, *_ = np.linalg.lstsq(A, rv, rcond=None)
    return float(np.hypot(*coef)), coef


def block_bootstrap_pvalue(t_h, r, valid, period_h, cad_s, nboot=1000, block_h=24, seed=1):
    obs, _ = line_amp(t_h, r, valid, period_h)
    rfull = r.copy()
    n = len(rfull)
    blk = max(2, int(block_h * 3600 / cad_s))
    rng = np.random.default_rng(seed)
    null = np.empty(nboot)
    idx_all = np.arange(n)
    for b in range(nboot):
        # circular block resample of the full grid, then refit on valid points
        out = np.empty(n); pos = 0
        while pos < n:
            start = rng.integers(0, n)
            take = min(blk, n - pos)
            sl = (np.arange(start, start + take) % n)
            out[pos:pos + take] = rfull[sl]; pos += take
        v = valid & np.isfinite(out)
        if v.sum() < 100:
            null[b] = np.nan; continue
        null[b], _ = line_amp(t_h, out, v, period_h)
    null = null[np.isfinite(null)]
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    ub95 = float(np.quantile(null, 0.95))
    return obs, p, ub95, null


def lombscargle_band(t_h, r, valid, f_lo_hz, f_hi_hz, nfreq=300):
    ts = t_h[valid] * 3600.0
    rs = r[valid] - np.nanmean(r[valid])
    fs = np.linspace(f_lo_hz, f_hi_hz, nfreq)
    ang = 2 * np.pi * fs
    P = sps.lombscargle(ts, rs, ang, normalize=True)
    return fs, P


# ----------------------------------------------------------------------------- #
#  lightning
# ----------------------------------------------------------------------------- #
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_lightning(path):
    df = pd.read_csv(path)
    cl = {c.lower(): c for c in df.columns}
    def pick(opts):
        for o in opts:
            for k, orig in cl.items():
                if o in k:
                    return orig
        return None
    tcol = pick(["time", "date", "datetime"])
    latc = pick(["lat"]); lonc = pick(["lon", "long"])
    icol = pick(["current", "peak", "kA", "moment", "amp"])
    if not (tcol and latc and lonc):
        sys.exit("lightning CSV needs time, lat, lon columns")
    t = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    lat = df[latc].astype(float); lon = df[lonc].astype(float)
    cur = df[icol].astype(float).abs() if icol else pd.Series(np.ones(len(df)))
    out = pd.DataFrame({"t": t, "lat": lat, "lon": lon, "I": cur}).dropna(subset=["t"])
    log(f"  lightning: {len(out)} strokes, {out['t'].min()} -> {out['t'].max()}"
        f"  ({'current-weighted' if icol else 'unit-weighted (no current column)'})")
    return out


def phi_foam(strokes, grid, lam_km, rmin_km=10.0):
    """Source series on the SR grid: sum_i I_i/R_i * exp(-R_i/lam) per bin."""
    R = haversine_km(strokes["lat"].values, strokes["lon"].values, STA_LAT, STA_LON)
    R = np.clip(R, rmin_km, None)
    w = strokes["I"].values / R * (np.exp(-R / lam_km) if np.isfinite(lam_km) else 1.0)
    edges = grid.values.astype("datetime64[ns]").astype("int64")
    st = strokes["t"].values.astype("datetime64[ns]").astype("int64")
    bins = np.searchsorted(edges, st) - 1
    ok = (bins >= 0) & (bins < len(grid))
    phi = np.zeros(len(grid))
    np.add.at(phi, bins[ok], w[ok])
    return phi


def dose_response(t_h, y, valid, phi):
    """Add phi as a regressor beside the known drivers; return its coeff + t-stat."""
    X, names = design_matrix(t_h)
    phin = (phi - np.nanmean(phi[valid])) / (np.nanstd(phi[valid]) + 1e-12)
    X = np.column_stack([X, phin]); names = names + ["lightning"]
    Xv, yv = X[valid], y[valid]
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    resid = yv - Xv @ beta
    dof = max(1, Xv.shape[0] - Xv.shape[1])
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(Xv.T @ Xv)
    se = np.sqrt(np.diag(cov))[-1]
    coef = beta[-1]
    tstat = coef / (se + 1e-30)
    r2 = 1 - np.var(resid) / np.var(yv)
    return coef, tstat, r2


def ringdown_stack(grid, resid, valid, strokes, lam_km, cad_s,
                   topfrac=0.01, pre_min=15, post_min=60, seed=3):
    """Superpose the residual after the largest strokes; fit a damped 2 mHz sinusoid."""
    R = np.clip(haversine_km(strokes["lat"].values, strokes["lon"].values, STA_LAT, STA_LON), 10, None)
    wgt = strokes["I"].values / R * np.exp(-R / lam_km)
    k = max(20, int(len(strokes) * topfrac))
    big = strokes.iloc[np.argsort(wgt)[::-1][:k]]
    edges = grid.values.astype("datetime64[ns]").astype("int64")
    g0 = edges[0]; dt_ns = int(cad_s * 1e9); span = int(edges[-1] - edges[0])
    pre, post = int(pre_min * 60 / cad_s), int(post_min * 60 / cad_s)
    L = pre + post + 1
    tax = (np.arange(L) - pre) * cad_s

    def collect(times_ns):
        S = []
        for tn in times_ns:
            i = int((int(tn) - g0) // dt_ns); a, b = i - pre, i + post + 1
            if a < 0 or b > len(resid):
                continue
            seg = resid[a:b].astype(float)
            if np.isfinite(seg).sum() < 0.8 * L:
                continue
            if np.isfinite(seg[:pre]).any():
                seg = seg - np.nanmean(seg[:pre])
            S.append(np.nan_to_num(seg, nan=0.0))
        return np.array(S)

    big_ns = big["t"].values.astype("datetime64[ns]").astype("int64")
    real = collect(big_ns)
    if len(real) < 10:
        return None
    stack = real.mean(0)

    def damped(t, A, tau, ph):
        return A * np.exp(-np.clip(t, 0, None) / tau) * np.cos(2 * np.pi * FOAM_HZ * t + ph)
    seg_post = slice(pre, L)
    try:
        popt, _ = curve_fit(damped, tax[seg_post], stack[seg_post],
                            p0=[np.std(stack), 600, 0], maxfev=8000,
                            bounds=([0, 30, -np.pi], [np.inf, 1e4, np.pi]))
        amp_obs = popt[0]
    except Exception:
        popt, amp_obs = None, np.nan

    rng = np.random.default_rng(seed); null = []
    for _ in range(300):
        rnd_ns = g0 + rng.integers(0, max(span, 1), size=len(big))
        st = collect(rnd_ns)
        if len(st) < 10:
            continue
        sm = st.mean(0)
        try:
            po, _ = curve_fit(damped, tax[seg_post], sm[seg_post], p0=[np.std(sm), 600, 0],
                              maxfev=4000, bounds=([0, 30, -np.pi], [np.inf, 1e4, np.pi]))
            null.append(po[0])
        except Exception:
            pass
    null = np.array(null)
    p = float((np.sum(null >= amp_obs) + 1) / (len(null) + 1)) if null.size else np.nan
    ub = float(np.quantile(null, 0.95)) if null.size else np.nan
    return dict(tax=tax, stack=stack, n=len(real), fit=popt, amp=amp_obs, p=p,
                ub95=ub, pre=pre, damped=damped)


# ----------------------------------------------------------------------------- #
#  main
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Foam-signature analyzer for Schumann parameter series.")
    ap.add_argument("--sr", required=True, help="sr_parameters.csv[.gz] from elf_to_sr.py")
    ap.add_argument("--lightning", default=None, help="optional stroke catalog CSV (time,lat,lon[,current])")
    ap.add_argument("--out", default="foam_results", help="output folder")
    ap.add_argument("--mode", type=int, default=1, help="SR mode to analyze (default 1)")
    ap.add_argument("--nboot", type=int, default=1000, help="bootstrap iterations")
    ap.add_argument("--lambdas", default="500,1000,2000,6371,inf",
                    help="screening lengths km, comma list (inf = no screening)")
    ap.add_argument("--station", default=None, help="override station 'lat,lon'")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global STA_LAT, STA_LON
    if args.station:
        STA_LAT, STA_LON = map(float, args.station.split(","))

    print("=" * 74)
    print(" FOAM ANALYZER — residual test on Schumann parameters")
    print(" Expected outcome of the foam arms is a BOUND, not a detection.")
    print("=" * 74)

    data, cad = load_sr(args.sr, args.mode)
    y = data["amp"].values.astype(float)
    valid = np.isfinite(y)
    t0 = data.index[0]
    t_h = (data.index - t0).total_seconds().values / 3600.0
    span_d = t_h.max() / 24

    resid, model, beta, r2 = regress(t_h, y, valid)
    summary = []

    # (3) lunar M2
    obs, p, ub, _ = block_bootstrap_pvalue(t_h, resid, valid, LUNAR_H, cad, args.nboot)
    frac = obs / np.nanmean(y[valid])
    ubfrac = ub / np.nanmean(y[valid])
    verdict = "DETECTION" if p < 0.05 else "null"
    summary.append(("Lunar M2 (12.42 h)", f"amp={frac*100:.3f}%  p={p:.3f}  "
                    f"95%UB={ubfrac*100:.3f}%  -> {verdict}"))
    log(f"LUNAR M2: fractional amp {frac*100:.3f}%  p={p:.3f}  bound {ubfrac*100:.3f}%  [{verdict}]")

    # (4) foam line at 2.005 mHz (+ octaves within Nyquist)
    nyq = 0.5 / cad
    foam_tests = [("2.005 mHz", FOAM_HZ)]
    if FOAM_HZ * 2 < nyq: foam_tests.append(("4.01 mHz (octave)", FOAM_HZ * 2))
    if FOAM_HZ * 4 < nyq: foam_tests.append(("8.02 mHz (octave)", FOAM_HZ * 4))
    foam_runnable = FOAM_HZ < nyq
    if not foam_runnable:
        log(f"FOAM LINE: cadence {cad:.0f}s -> Nyquist {nyq*1e3:.2f} mHz is below 2 mHz; "
            "need finer cadence. (Lunar test above is unaffected.)")
        summary.append(("Foam line 2 mHz", f"NOT TESTABLE at {cad:.0f}s cadence (Nyquist {nyq*1e3:.2f} mHz)"))
    else:
        for nm, fhz in foam_tests:
            ph = 1 / fhz / 3600.0
            o, pp, u, _ = block_bootstrap_pvalue(t_h, resid, valid, ph, cad, args.nboot)
            fr = o / np.nanmean(y[valid]); ufr = u / np.nanmean(y[valid])
            vv = "DETECTION" if pp < 0.05 else "null"
            summary.append((f"Foam line {nm}", f"amp={fr*100:.3f}%  p={pp:.3f}  "
                            f"95%UB={ufr*100:.3f}%  -> {vv}"))
            log(f"FOAM {nm}: fractional amp {fr*100:.3f}%  p={pp:.3f}  bound {ufr*100:.3f}%  [{vv}]")

    # ---------------- lightning arms ----------------
    ld = None
    if args.lightning:
        log("Lightning arms enabled.")
        strokes = load_lightning(args.lightning)
        lams = [float("inf") if x.strip() == "inf" else float(x) for x in args.lambdas.split(",")]
        # (5a/b) dose-response + lambda sweep
        sweep = []
        for lam in lams:
            phi = phi_foam(strokes, data.index, lam)
            coef, tstat, r2l = dose_response(t_h, y, valid, phi)
            sweep.append((lam, tstat, r2l))
            log(f"  lambda={'inf' if np.isinf(lam) else int(lam)} km: "
                f"dose-response t={tstat:+.1f}  model R^2={r2l:.3f}")
        best = max(sweep, key=lambda s: s[2])
        summary.append(("Lightning dose-response",
                        f"strongest t={max(s[1] for s in sweep):+.1f} "
                        f"(EXPECTED real; ordinary cavity physics)"))
        summary.append(("lambda-sweep best fit",
                        f"lambda={'inf' if np.isinf(best[0]) else int(best[0])} km "
                        f"(R^2={best[2]:.3f}) — better-than-global needed for anomaly; "
                        f"cavity-geometry confound applies"))
        # (5c) ringdown stack (use a mid screening length)
        lam_r = 2000.0
        if foam_runnable:
            ld = ringdown_stack(data.index, resid, valid, strokes, lam_r, cad)
            if ld:
                vv = "DETECTION" if (np.isfinite(ld["p"]) and ld["p"] < 0.05) else "null"
                fr = ld["amp"] / np.nanmean(y[valid])
                ub = ld["ub95"] / np.nanmean(y[valid])
                tau = ld["fit"][1] if ld["fit"] is not None else np.nan
                summary.append(("Stroke-triggered 2 mHz ringdown",
                                f"amp={fr*100:.3f}%  tau={tau:.0f}s  p={ld['p']:.3f}  "
                                f"95%UB={ub*100:.3f}%  (n={ld['n']} epochs) -> {vv}"))
                log(f"RINGDOWN: amp {fr*100:.3f}%  tau={tau:.0f}s  p={ld['p']:.3f}  [{vv}]")
        else:
            summary.append(("Stroke-triggered 2 mHz ringdown", "needs finer cadence (2 mHz unreachable)"))

    # ---------------- figures ----------------
    make_figures(args.out, data, y, valid, model, resid, t_h, cad, foam_runnable, ld)

    # ---------------- summary ----------------
    n_tests = len([s for s in summary if "->" in s[1]])
    print("\n" + "=" * 74)
    print(" RESULTS")
    print("=" * 74)
    for k, v in summary:
        print(f"  {k:34s} {v}")
    print("-" * 74)
    print(f"  span {span_d:.0f} d | cadence {cad:.0f}s | known-driver R^2={r2:.3f}")
    print(f"  effective foam/lunar tests run: {n_tests} "
          f"(look-elsewhere: any lone p<0.05 needs held-out confirmation)")
    print("=" * 74)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        for k, v in summary:
            f.write(f"{k}\t{v}\n")
    log(f"Wrote {args.out}/summary.txt and figures.")


def make_figures(out, data, y, valid, model, resid, t_h, cad, foam_runnable, ld):
    # fit overview (first 14 days)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    m = t_h < 24 * 14
    ax.plot(t_h[m] / 24, y[m], lw=0.4, color="#888", label="SR amplitude")
    ax.plot(t_h[m] / 24, model[m], lw=0.8, color=RED, label="known-driver model")
    ax.set_xlabel("days"); ax.set_ylabel("mode-1 amplitude"); ax.legend(fontsize=8)
    ax.set_title("Known-driver model removed before any foam test (first 14 d)", color=NAVY, fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(out, "fit_overview.png"), dpi=140); plt.close()

    # lunar band periodogram (residual), 8-16 h
    fs, P = lombscargle_band(t_h, resid, valid, 1/(16*3600), 1/(8*3600), 400)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(1/fs/3600, P, lw=0.8, color=NAVY)
    ax.axvline(LUNAR_H, color=RED, ls="--", lw=1, label="lunar M2 12.42 h")
    ax.axvline(SOLAR_S2_H, color="#888", ls=":", lw=1, label="solar S2 12.00 h (removed)")
    ax.set_xlabel("period (h)"); ax.set_ylabel("LS power (residual)"); ax.legend(fontsize=8)
    ax.set_title("Residual near the semidiurnal band — lunar line vs removed solar line", color=NAVY, fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(out, "lunar_band.png"), dpi=140); plt.close()

    # foam line periodogram (residual), around 2 mHz
    if foam_runnable:
        fs, P = lombscargle_band(t_h, resid, valid, 0.0008, 0.0035, 400)
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.plot(fs * 1e3, P, lw=0.8, color=NAVY)
        ax.axvline(FOAM_HZ * 1e3, color=RED, ls="--", lw=1, label="2.005 mHz (c/1AU)")
        ax.set_xlabel("frequency (mHz)"); ax.set_ylabel("LS power (residual)"); ax.legend(fontsize=8)
        ax.set_title("Residual spectrum at the foam frequency (now reachable at 1-min cadence)",
                     color=NAVY, fontsize=10)
        plt.tight_layout(); plt.savefig(os.path.join(out, "foam_line.png"), dpi=140); plt.close()

    # ringdown stack
    if ld:
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(ld["tax"] / 60, ld["stack"], lw=0.9, color=NAVY, label=f"stack of {ld['n']} strokes")
        if ld["fit"] is not None:
            tt = ld["tax"][ld["pre"]:]
            ax.plot(tt / 60, ld["damped"](tt, *ld["fit"]), lw=1.2, color=RED,
                    label="damped 2 mHz fit")
        ax.axvline(0, color="k", lw=0.5)
        if np.isfinite(ld["ub95"]):
            ax.axhspan(-ld["ub95"], ld["ub95"], color="#ddd", alpha=0.5, label="95% null band")
        ax.set_xlabel("minutes from stroke"); ax.set_ylabel("residual"); ax.legend(fontsize=8)
        ax.set_title("Stroke-triggered superposed-epoch — 2 mHz ringdown test", color=NAVY, fontsize=10)
        plt.tight_layout(); plt.savefig(os.path.join(out, "ringdown_stack.png"), dpi=140); plt.close()


if __name__ == "__main__":
    main()
