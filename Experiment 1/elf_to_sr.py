#!/usr/bin/env python3
"""
elf_to_sr.py — Stream Sierra Nevada ELF raw records into a Schumann-resonance
parameter table, with crash-safe resume.

Memory stays low and CONSTANT regardless of dataset size: each hour-file is read
one analysis-window at a time straight from disk, so peak RAM is one window
(a few hundred KB), never the whole archive.

--------------------------------------------------------------------------------
USAGE
  # 1) verify the binary format on a single file FIRST:
  python elf_to_sr.py -i /path/to/raw_root -o /path/to/out_dir --inspect

  # 2) run the full extraction:
  python elf_to_sr.py -i /path/to/raw_root -o /path/to/out_dir

  # if it is ever interrupted (Ctrl-C, crash, power), just run the SAME command
  # again — it skips everything already finished and continues.

The input root may be the parent of all five unzipped years; the script walks
the year/month tree recursively
(e.g. .../2014/1412/smplGRTU1_sensor_0_1412010430 ...).

OUTPUT (written inside -o)
  sr_parameters.csv   the result — one row per window per sensor
  progress.log        the resume ledger of finished source files (do NOT delete)
  run.log             a human-readable log

DEFAULT CADENCE is 60 s, giving a parameter series sampled every minute
(Nyquist 8.3 mHz) — high enough to reach the 2 mHz line. Change with --cadence.

DEPENDENCIES:  pip install numpy scipy
--------------------------------------------------------------------------------
"""
import argparse, os, sys, re, time, signal
import datetime as dt
import numpy as np
from scipy import signal as sps
try:
    from scipy.optimize import curve_fit
    HAVE_FIT = True
except Exception:
    HAVE_FIT = False

# data file basenames look like  smplGRTU1_sensor_0_1412010430  (sensor 0=NS, 1=EW)
DATA_RE = re.compile(r'sensor_([01])_(\d{10})')
SENSOR_NAME = {0: "NS", 1: "EW"}
# (mode number, nominal freq Hz, search band Hz)
MODES = [(1, 7.83, (5.5, 10.5)), (2, 14.3, (11.5, 17.5)), (3, 20.8, (18.0, 24.5))]
COLS = (["timestamp", "sensor"]
        + [f"m{m}_freq" for m, _, _ in MODES]
        + [f"m{m}_amp"  for m, _, _ in MODES]
        + [f"m{m}_width" for m, _, _ in MODES]
        + ["saturated", "n_samples"])


def log(msg, logf=None):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logf:
        logf.write(line + "\n"); logf.flush()


def parse_info(info_path):
    """Best-effort read of a *_info.txt sidecar for the sample rate. Returns {}."""
    out = {}
    if not (info_path and os.path.exists(info_path)):
        return out
    try:
        txt = open(info_path, "r", errors="ignore").read()
    except Exception:
        return out
    out["_raw"] = txt
    for pat in [r'(?:sampl\w*\s*freq\w*|sampling|fs)\D{0,12}(\d{2,5}(?:\.\d+)?)',
                r'(\d{2,4}(?:\.\d+)?)\s*hz']:
        m = re.search(pat, txt, re.I)
        if m:
            try:
                out["fs"] = float(m.group(1)); break
            except Exception:
                pass
    return out


def find_files(root):
    """Walk root; return sorted [(path, sensor_int, 'YYMMDDHHMM'), ...] for data files."""
    found = []
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.endswith("_info.txt") or fn.startswith("ficheros"):
                continue
            m = DATA_RE.search(fn)
            if not m:
                continue
            found.append((os.path.join(dp, fn), int(m.group(1)), m.group(2)))
    found.sort(key=lambda r: (r[2], r[1]))   # by time, then sensor
    return found


def stamp_to_base(s):
    """'YYMMDDHHMM' -> timezone-aware UTC datetime (year assumed 20YY)."""
    yy, mo, dd = int(s[0:2]), int(s[2:4]), int(s[4:6])
    hh, mi = int(s[6:8]), int(s[8:10])
    return dt.datetime(2000 + yy, mo, dd, hh, mi, 0, tzinfo=dt.timezone.utc)


def iter_windows(path, dtype, win_samp, header_bytes):
    """Yield (idx, float64 samples) one window at a time, read directly from disk."""
    d = np.dtype(dtype); nbytes = win_samp * d.itemsize
    with open(path, "rb") as f:
        if header_bytes:
            f.seek(header_bytes)
        idx = 0
        while True:
            b = f.read(nbytes)
            if len(b) < nbytes:        # drop the final short remainder
                break
            yield idx, np.frombuffer(b, dtype=d).astype(np.float64)
            idx += 1


def _lorentz(f, A, f0, g, c):
    return A * (g * g) / ((f - f0) ** 2 + g * g) + c


def fit_fast(fb, Pb):
    """Peak estimate: (freq, amp_above_background, FWHM). Fast, robust."""
    if Pb.size < 5:
        return (np.nan, np.nan, np.nan)
    ipk = int(np.argmax(Pb)); pk = Pb[ipk]; base = float(np.median(Pb))
    if 0 < ipk < len(Pb) - 1:                      # parabolic peak interpolation
        y0, y1, y2 = Pb[ipk - 1], Pb[ipk], Pb[ipk + 1]
        den = (y0 - 2 * y1 + y2)
        d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    else:
        d = 0.0
    df = fb[1] - fb[0]
    f0 = fb[ipk] + d * df
    half = base + 0.5 * (pk - base)

    def cross(side):
        i = ipk
        while 0 <= i < len(Pb) and Pb[i] > half:
            i += side
        if i < 0 or i >= len(Pb):
            return None
        p1, p2 = Pb[i - side], Pb[i]; f1, f2 = fb[i - side], fb[i]
        return f1 + (half - p1) * (f2 - f1) / (p2 - p1) if p2 != p1 else fb[i]

    lf, rf = cross(-1), cross(+1)
    width = (rf - lf) if (lf is not None and rf is not None) else np.nan
    return (f0, pk - base, width)


def fit_lorentz(fb, Pb):
    if not HAVE_FIT or Pb.size < 6:
        return fit_fast(fb, Pb)
    ipk = int(np.argmax(Pb)); pk = Pb[ipk]; base = float(np.median(Pb))
    p0 = [max(pk - base, 1e-12), fb[ipk], 0.8, base]
    lo = [0, fb[0], 0.05, 0]; hi = [np.inf, fb[-1], (fb[-1] - fb[0]), np.inf]
    try:
        popt, _ = curve_fit(_lorentz, fb, Pb, p0=p0, bounds=(lo, hi), maxfev=4000)
        return (popt[1], popt[0], 2 * popt[2])     # freq, amp, FWHM
    except Exception:
        return fit_fast(fb, Pb)


def process_file(path, sensor, stamp, args):
    base = stamp_to_base(stamp)
    win_samp = int(round(args.cadence * args.fs))
    nperseg = int(min(args.nperseg, win_samp))
    fitter = fit_fast if args.method == "fast" else fit_lorentz
    rows = []
    for idx, x in iter_windows(path, args.dtype, win_samp, args.header_bytes):
        sat = int(np.any(np.abs(x) >= args.sat_level))
        if args.gain != 1.0:
            x = x * args.gain
        x = x - x.mean()
        f, P = sps.welch(x, fs=args.fs, nperseg=nperseg,
                         noverlap=nperseg // 2, window="hann")
        vals = []
        for _, _, (lo, hi) in MODES:
            sel = (f >= lo) & (f <= hi)
            vals.append(fitter(f[sel], P[sel]))
        ts = (base + dt.timedelta(seconds=idx * args.cadence)).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append([ts, sensor]
                    + [v[0] for v in vals] + [v[1] for v in vals] + [v[2] for v in vals]
                    + [sat, x.size])
    return rows


def fmt_row(r):
    return ",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r)


def main():
    ap = argparse.ArgumentParser(
        description="Stream Sierra Nevada ELF raw -> Schumann-resonance parameter table (resumable).")
    ap.add_argument("--input", "-i", required=True, help="root folder of unzipped raw data")
    ap.add_argument("--output", "-o", required=True, help="output folder")
    ap.add_argument("--cadence", type=float, default=60.0,
                    help="window length in seconds = parameter sampling interval (default 60)")
    ap.add_argument("--fs", type=float, default=256.0,
                    help="sample rate Hz (default 256; auto-overridden by info.txt if it states one)")
    ap.add_argument("--dtype", default="<i2",
                    help="raw sample dtype (default little-endian int16 '<i2'; e.g. '>i2','<i4','<f4')")
    ap.add_argument("--header-bytes", type=int, default=0,
                    help="bytes to skip at the start of each raw file (default 0)")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="multiply samples by this for calibration (default 1 = raw counts)")
    ap.add_argument("--sat-level", type=float, default=32760.0,
                    help="abs sample value flagged as saturation (default 32760 for int16)")
    ap.add_argument("--nperseg", type=int, default=4096,
                    help="Welch segment length inside each window (default 4096)")
    ap.add_argument("--method", choices=["fast", "lorentzian"], default="fast",
                    help="'fast' peak estimate (default) or 'lorentzian' curve fit (slower)")
    ap.add_argument("--limit-files", type=int, default=0, help="process only the first N files (testing)")
    ap.add_argument("--inspect", action="store_true",
                    help="decode ONE file, print format diagnostics, and exit")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    logf = open(os.path.join(args.output, "run.log"), "a")
    out_csv = os.path.join(args.output, "sr_parameters.csv")
    prog_path = os.path.join(args.output, "progress.log")

    files = find_files(args.input)
    if not files:
        log(f"No data files found under {args.input}", logf); sys.exit(1)
    log(f"Discovered {len(files)} raw data files under {args.input}", logf)

    info = parse_info(files[0][0] + "_info.txt")
    if "fs" in info:
        if abs(info["fs"] - args.fs) > 1e-6:
            log(f"info.txt states sample rate {info['fs']} Hz (using it instead of {args.fs})", logf)
        args.fs = info["fs"]

    if args.inspect:
        p, sensor, stamp = files[0]
        d = np.dtype(args.dtype); sz = os.path.getsize(p)
        log(f"INSPECT  {p}", logf)
        log(f"  size {sz} bytes | dtype {args.dtype} ({d.itemsize} B) -> {sz // d.itemsize} samples", logf)
        log(f"  fs={args.fs} Hz -> implied duration {sz / d.itemsize / args.fs:.1f} s "
            f"({sz / d.itemsize / args.fs / 60:.2f} min)  [expect ~3600 s / 60 min]", logf)
        if os.path.exists(p + "_info.txt"):
            log("  --- info.txt (first 40 lines) ---", logf)
            for ln in open(p + "_info.txt", errors="ignore").read().splitlines()[:40]:
                log("    " + ln, logf)
        x = np.frombuffer(open(p, "rb").read(int(args.fs * 5) * d.itemsize), dtype=d).astype(float)
        log(f"  first 5 s decoded: n={x.size} min={x.min():.1f} max={x.max():.1f} "
            f"mean={x.mean():.2f} std={x.std():.2f}", logf)
        rows = process_file(p, sensor, stamp, args)[:3]
        log("  first parameter rows (look for modes near 7.8 / 14.3 / 20.8 Hz):", logf)
        log("    " + ",".join(COLS), logf)
        for r in rows:
            log("    " + fmt_row(r), logf)
        log("INSPECT done. If duration ~3600 s and the mode freqs look right, "
            "rerun without --inspect.", logf)
        return

    done = set()
    if os.path.exists(prog_path):
        done = set(l.strip() for l in open(prog_path) if l.strip())
        log(f"Resuming — {len(done)} files already complete, skipping them.", logf)

    new = (not os.path.exists(out_csv)) or os.path.getsize(out_csv) == 0
    csv = open(out_csv, "a")
    if new:
        csv.write(",".join(COLS) + "\n"); csv.flush()
    prog = open(prog_path, "a")

    stop = {"flag": False}

    def handler(sig, frm):
        stop["flag"] = True
        log("Interrupt received — finishing the current file, then exiting cleanly.", logf)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    todo = [f for f in files if os.path.basename(f[0]) not in done]
    if args.limit_files:
        todo = todo[:args.limit_files]
    log(f"{len(todo)} files to process this run "
        f"(cadence {args.cadence}s, method {args.method}, dtype {args.dtype}).", logf)

    t0 = time.time(); n = 0
    for k, (path, sensor, stamp) in enumerate(todo):
        try:
            rows = process_file(path, sensor, stamp, args)
        except Exception as e:
            log(f"  ERROR on {os.path.basename(path)}: {e} — skipping", logf)
            continue
        # buffer this file's rows, write them all, THEN mark the file done.
        # interruption mid-file writes nothing -> the file is cleanly reprocessed on resume.
        for r in rows:
            csv.write(fmt_row(r) + "\n")
        csv.flush(); os.fsync(csv.fileno())
        prog.write(os.path.basename(path) + "\n"); prog.flush(); os.fsync(prog.fileno())
        n += 1
        if n % 50 == 0 or k == len(todo) - 1:
            el = time.time() - t0; rate = n / el if el > 0 else 0
            eta = (len(todo) - n) / rate / 3600 if rate > 0 else 0
            log(f"  {n}/{len(todo)} files  ({rate:.1f}/s, ~{eta:.1f} h left)", logf)
        if stop["flag"]:
            log("Stopped cleanly. Re-run the SAME command to resume.", logf)
            break

    csv.close(); prog.close()
    log(f"Run complete: {n} files processed this run.  ->  {out_csv}", logf)


if __name__ == "__main__":
    main()
