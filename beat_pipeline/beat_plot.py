#!/usr/bin/env python3
"""
Beat Pipeline - matplotlib / command-line front-end
===================================================
Same stage-1 preprocessing as the Qt tool (beat_window.py), plotted with
matplotlib instead. No Qt widgets involved: useful over SSH, in a notebook,
for saving figures to disk, or when the Qt window does not fit the screen.

The QC report goes to STDOUT, so it can be scrolled in the terminal, piped
to a file, or grepped - none of which the Qt panel allows.

Examples
--------
    # inspect one recording interactively (opens a window)
    python3 beat_pipeline/beat_plot.py dataset/txt/SKNA_BP_alice_10kHz.txt

    # first 60 s, zoom the view to 20-30 s, save a PNG instead of showing
    python3 beat_pipeline/beat_plot.py dataset/txt/SKNA_BP_Jose.txt \
        --dur 60 --xlim 20 30 --out jose.png

    # QC numbers only, no figure at all
    python3 beat_pipeline/beat_plot.py dataset/txt/*.txt --no-plot

    # one-line-per-recording summary table across the whole corpus
    python3 beat_pipeline/beat_plot.py dataset/txt/*.txt --summary
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beat_processing as bp

MAX_PLOT_POINTS = 400_000


def _decimate(t, y, max_points=MAX_PLOT_POINTS):
    n = len(y)
    if n <= max_points:
        return t, y
    step = int(np.ceil(n / max_points))
    return t[::step], y[::step]


def _decimate_minmax(t, y, max_points=MAX_PLOT_POINTS):
    """Envelope-preserving decimation for the ECG traces.

    Plain stride decimation at 10 kHz throws away the R-peak SAMPLE itself,
    so the drawn trace sits below the true peak. Keeping each bin's min and
    max preserves peak amplitude.

    The two points must be emitted at DIFFERENT x, in the order they occur
    within the bin. Emitting both at the bin's start x (the obvious way)
    draws a vertical bar then a flat jump, which renders a clean ECG as a
    staircase and looks like bad data.
    """
    n = len(y)
    if n <= max_points:
        return t, y
    step = int(np.ceil(n / (max_points // 2)))
    m = (n // step) * step
    if m == 0:
        return t, y
    yy = y[:m].reshape(-1, step)
    tt = t[:m].reshape(-1, step)
    imin = yy.argmin(axis=1)
    imax = yy.argmax(axis=1)
    first = np.minimum(imin, imax)          # keep chronological order within
    second = np.maximum(imin, imax)         # the bin so the line still flows
    rows = np.arange(yy.shape[0])
    out_y = np.empty(yy.shape[0] * 2, dtype=y.dtype)
    out_t = np.empty(yy.shape[0] * 2, dtype=t.dtype)
    out_y[0::2] = yy[rows, first]; out_t[0::2] = tt[rows, first]
    out_y[1::2] = yy[rows, second]; out_t[1::2] = tt[rows, second]
    return out_t, out_y


def make_figure(res, xlim=None, figsize=(13.0, 9.0)):
    """Five stacked, x-linked panels mirroring the Qt tool's layout."""
    import matplotlib.pyplot as plt

    fs = res["fs"]
    raw, filt, rpeaks, st = res["raw"], res["filtered"], res["rpeaks"], res["stages"]
    t = np.arange(len(raw)) / fs

    fig, ax = plt.subplots(5, 1, figsize=figsize, sharex=True)
    fig.suptitle(f"{os.path.basename(res['path'])}  -  {res['channel']}  "
                 f"({fs:.0f} Hz, lead-in {res['leadin']:.2f}s skipped, "
                 f"FIR delay {res['delay']:+.2f}s)", fontsize=11)

    a, b = _decimate_minmax(t, raw)
    ax[0].plot(a, b, lw=0.6, color="0.65", label="raw")
    a, b = _decimate_minmax(t, filt)
    ax[0].plot(a, b, lw=0.6, color="0.1", label="high-passed")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("1. ECG - raw vs high-passed", fontsize=9, loc="left")

    fsd = st["fs_detect"]
    t2 = np.arange(len(st["bandpassed"])) / fsd
    for i, (key, title) in enumerate(
            [("bandpassed", "2. Bandpass 5-15 Hz (QRS isolation)"),
             ("squared", "3. Squared derivative"),
             ("integrated", "4. Moving-window integration + adaptive threshold")], start=1):
        a, b = _decimate(t2, st[key])
        ax[i].plot(a, b, lw=0.6, color="#1e5ac8")
        ax[i].set_title(title, fontsize=9, loc="left")

    # initial THRESHOLD_I1, as a visual "did this beat clear the bar" reference
    integ = st["integrated"]
    from scipy import signal as _sig
    cand, _ = _sig.find_peaks(integ, distance=max(1, int(0.2 * fsd)))
    if len(cand):
        spki, npki = bp.pt_init_thresholds(integ, cand, fsd)
        ax[3].axhline(npki + 0.25 * (spki - npki), color="#c83c3c", ls="--", lw=1)

    a, b = _decimate_minmax(t, filt)
    ax[4].plot(a, b, lw=0.6, color="0.1")
    if len(rpeaks):
        ax[4].plot(rpeaks / fs, filt[rpeaks], "o", ms=6, mfc="#e61e1e",
                   mec="k", mew=0.5, ls="none", zorder=5)
        ax[4].set_title(f"5. High-passed ECG with {len(rpeaks)} detected R-peaks (red)",
                        fontsize=9, loc="left")
    else:
        ax[4].set_title("5. High-passed ECG - NO R-peaks detected", fontsize=9, loc="left")

    for a_ in ax:
        a_.grid(alpha=0.25)
    ax[-1].set_xlabel("time (s)")

    # Default to the first 20 s: at full extent hundreds of markers merge into
    # a solid band and a doubled detection is indistinguishable from a good one.
    if xlim:
        ax[0].set_xlim(*xlim)
    elif len(t):
        ax[0].set_xlim(0, min(20.0, t[-1]))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def summary_row(res):
    s = res["stats"]
    name = os.path.basename(res["path"])[:26]
    if "error" in s:
        return f"{name:26s} {'FAIL':>8s}  {s['error']}"
    n = len(res["raw"])
    rate = s["n_rpeaks"] / (n / res["fs"]) * 60.0
    return (f"{name:26s} {res['delay']:6.2f}s {res['leadin']:7.2f}s "
            f"{s['n_rpeaks']:6d} {s['hr_mean']:7.1f} {rate:6.1f} "
            f"{s['n_rr_nonphysio']:5d} {s['seg_p99']:6.3f}s "
            f"{bp.recommend_L(s['seg_p99'], res['target_fs']):6d}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Beat-segment stage-1 preprocessing with matplotlib plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("files", nargs="+", help="BIOPAC .txt recording(s); globs allowed")
    p.add_argument("--channel", default="CH40", choices=list(bp.ECG_CHANNELS),
                   help="ECG channel (default CH40, the smoother FIR-filtered trace)")
    p.add_argument("--fs", type=float, default=None, help="override header sample rate")
    p.add_argument("--start", type=float, default=None, help="start time (s)")
    p.add_argument("--dur", type=float, default=None, help="duration to analyse (s)")

    g = p.add_argument_group("preprocessing")
    g.add_argument("--no-delay", action="store_true",
                   help="do NOT compensate the channel's FIR delay vs CH1")
    g.add_argument("--leadin", action="store_true",
                   help="skip the settling transient at the start (off by default: "
                        "load-length dependent, shifts the clock)")
    g.add_argument("--leadin-factor", type=float, default=3.0,
                   help="lead-in threshold, x steady-state amplitude (default 3)")
    g.add_argument("--no-hpf", action="store_true", help="disable the high-pass")
    g.add_argument("--fc", type=float, default=0.08, help="high-pass corner Hz (default 0.08)")
    g.add_argument("--order", type=int, default=2, help="high-pass order (default 2)")

    g = p.add_argument_group("Pan-Tompkins")
    g.add_argument("--bp-low", type=float, default=5.0)
    g.add_argument("--bp-high", type=float, default=15.0)
    g.add_argument("--integ-ms", type=float, default=150.0)
    g.add_argument("--refractory-ms", type=float, default=200.0)
    g.add_argument("--twave-ms", type=float, default=360.0)
    g.add_argument("--detect-fs", type=float, default=250.0)
    g.add_argument("--refine-ms", type=float, default=50.0)
    g.add_argument("--artifact-k", type=float, default=5.0,
                   help="reject candidates above k x p99 of peak heights (default 5)")

    g = p.add_argument_group("output")
    g.add_argument("--target-fs", type=float, default=2000.0,
                   help="stage-2 segment resample rate, for the L recommendation")
    g.add_argument("--xlim", nargs=2, type=float, metavar=("T0", "T1"),
                   help="zoom the plots to this time range (s)")
    g.add_argument("--out", default=None,
                   help="save figure here instead of showing it; with multiple "
                        "inputs it is treated as a directory")
    g.add_argument("--no-plot", action="store_true", help="QC report only, no figure")
    g.add_argument("--summary", action="store_true",
                   help="one line per recording instead of a full report")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    paths = []
    for pat in args.files:
        paths.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat])
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        print("no input files matched", file=sys.stderr)
        return 2

    make_plots = not (args.no_plot or args.summary)
    if make_plots:
        import matplotlib
        if args.out:                      # headless-safe when only saving
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    multi = len(paths) > 1
    if args.summary:
        print(f"{'recording':26s} {'delay':>7s} {'lead-in':>8s} {'beats':>6s} "
              f"{'HRmean':>7s} {'rate':>6s} {'badRR':>5s} {'segp99':>7s} {'L':>6s}")

    p99s = []
    for path in paths:
        res = bp.run_pipeline(
            path, channel=args.channel, fs=args.fs, start_sec=args.start,
            dur_sec=args.dur, compensate_delay=not args.no_delay,
            skip_leadin=args.leadin, leadin_factor=args.leadin_factor,
            hpf=not args.no_hpf, fc=args.fc, order=args.order,
            target_fs=args.target_fs,
            bp_low=args.bp_low, bp_high=args.bp_high, integ_ms=args.integ_ms,
            refractory_ms=args.refractory_ms, twave_ms=args.twave_ms,
            detect_fs=args.detect_fs, refine_ms=args.refine_ms,
            artifact_k=args.artifact_k)

        if args.summary:
            print(summary_row(res))
        else:
            if multi:
                print(f"\n########## {os.path.basename(path)} ##########")
            print(bp.format_qc_report(res))

        if "error" not in res["stats"]:
            p99s.append(res["stats"]["seg_p99"])

        if make_plots:
            fig = make_figure(res, xlim=tuple(args.xlim) if args.xlim else None)
            if args.out:
                if multi or os.path.isdir(args.out):
                    os.makedirs(args.out, exist_ok=True)
                    dest = os.path.join(
                        args.out, os.path.splitext(os.path.basename(path))[0] + ".png")
                else:
                    dest = args.out
                fig.savefig(dest, dpi=130)
                plt.close(fig)
                print(f"[saved {dest}]")

    if make_plots and not args.out:
        import matplotlib.pyplot as plt
        plt.show()

    # Across a corpus, L must cover the LONGEST segment anywhere, so the
    # per-recording numbers above are not enough on their own.
    if len(p99s) > 1:
        m = max(p99s)
        print(f"\n{'=' * 52}\nACROSS {len(p99s)} RECORDINGS\n{'=' * 52}")
        print(f"p99 segment: min {min(p99s):.3f}s  max {m:.3f}s")
        print(f"-> corpus-wide L = {bp.recommend_L(m, args.target_fs)} "
              f"@ {args.target_fs:.0f} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
