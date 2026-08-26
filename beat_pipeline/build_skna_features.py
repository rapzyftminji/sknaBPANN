#!/usr/bin/env python3
"""
Build the 11-feature SKNA window table for every subject, with BP labels.
========================================================================

    python3 beat_pipeline/build_skna_features.py --npz beat_pipeline/built/skna_features_30s.npz
    python3 beat_pipeline/build_skna_features.py --out skna_features_30s.csv
    python3 beat_pipeline/build_skna_features.py --subjects s1 s6 --dur 300

Same window/stride grid and same .npz layout as build_ecg_features.py, so the
two tables join on (subject, window_idx) - keep --window/--stride/--dur
identical between the two builds or that guarantee is void. Use an INNER join,
not a positional stack: this table can be one window shorter per recording
(see skna_features.build_recording).

Writes a row per window: Subject_ID, Recording, window_idx, t_start_sec,
t_center_sec, t_raw_center_sec, <11 features>, SBP, DBP, label_valid, usable.
`usable` = labelled AND all 11 features finite; use it as the training mask.

See skna_features.py for the signal chain, why dfSKNA needs a search band,
and which of these features are collinear by construction.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beat_labels as bl
import skna_features as sf


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None,
                   help="tidy CSV output (default: no CSV, npz only)")
    p.add_argument("--npz", default=None,
                   help="pack every subject into ONE .npz for training")
    p.add_argument("--npz-dir", default=None,
                   help="also write one <subject>_skna_features.npz per subject")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--txt-root", default="dataset/txt")
    p.add_argument("--feature-root", default="feature_result/5sWindow")
    p.add_argument("--window", type=float, default=sf.DEFAULT_WINDOW_SEC)
    p.add_argument("--stride", type=float, default=sf.DEFAULT_STRIDE_SEC)
    p.add_argument("--skna-channel", default="CH41")
    p.add_argument("--ecg-channel", default="CH40",
                   help="only used to put SKNA on the ECG's clock")
    p.add_argument("--theta-uv", type=float, default=sf.THETA_UV,
                   help="wampSKNA threshold in uV (absolute)")
    p.add_argument("--df-band", type=float, nargs=2, default=list(sf.DF_BAND),
                   metavar=("LO", "HI"), help="dfSKNA search band in Hz")
    p.add_argument("--amp-norm", default="none", choices=["none", "recording"],
                   help="'recording' scales each recording to a common SD "
                        "(REF_SD_UV) before features, removing the 6x "
                        "between-subject gain spread at source")
    p.add_argument("--detrend", action="store_true",
                   help="cubic-detrend before the bandpass (core_processing's step)")
    p.add_argument("--label-mode", default="interp", choices=["interp", "window"])
    p.add_argument("--dur", type=float, default=None)
    p.add_argument("--skip-excluded", action="store_true",
                   help="skip s5 (excluded from this project's LOSO runs)")
    args = p.parse_args(argv)

    if not (args.out or args.npz or args.npz_dir):
        print("nothing to write: pass --npz (and/or --out, --npz-dir)",
              file=sys.stderr)
        return 2

    subjects = args.subjects or list(bl.SUBJECT_RECORDING)
    if args.skip_excluded:
        subjects = [s for s in subjects if s not in bl.EXCLUDED_SUBJECTS]

    frames, packed, t0 = [], [], time.time()
    for subj in subjects:
        rec = os.path.join(args.txt_root, bl.SUBJECT_RECORDING[subj])
        if not os.path.isfile(rec):
            print(f"{subj:12s} SKIP  missing {rec}", flush=True)
            continue
        t = time.time()
        try:
            # keep_signals=False: the 10 kHz traces are what would make a
            # 16-subject loop hold GBs, and nothing here plots them.
            r = sf.build_recording(
                rec, subject=subj, window_sec=args.window, stride_sec=args.stride,
                skna_channel=args.skna_channel, ecg_channel=args.ecg_channel,
                feature_root=args.feature_root, label_mode=args.label_mode,
                theta_uv=args.theta_uv, df_band=tuple(args.df_band),
                detrend=args.detrend, keep_signals=False, dur_sec=args.dur,
                amp_norm=args.amp_norm)
        except Exception as e:
            print(f"{subj:12s} FAILED  {type(e).__name__}: {e}", flush=True)
            continue

        df = pd.DataFrame(r["X"], columns=sf.FEATURE_NAMES)
        df.insert(0, "Subject_ID", subj)
        df.insert(1, "Recording", os.path.basename(rec))
        df.insert(2, "window_idx", np.arange(len(df)))
        df.insert(3, "t_start_sec", r["t_start_sec"])
        df.insert(4, "t_center_sec", r["t_center_sec"])
        df.insert(5, "t_raw_center_sec", r["t_center_sec"] + r["clock_offset_sec"])
        df["SBP"], df["DBP"] = r["SBP"], r["DBP"]
        df["label_valid"], df["usable"] = r["label_valid"], r["usable"]
        frames.append(df)

        a = sf.window_arrays(r)
        packed.append(a)
        if args.npz_dir:
            os.makedirs(args.npz_dir, exist_ok=True)
            sf.save_npz(os.path.join(args.npz_dir, f"{subj}_skna_features.npz"), a)

        i = sf.FEATURE_NAMES.index
        print(f"{subj:12s} {os.path.basename(rec):26s} windows {len(df):5d}  "
              f"usable {int(r['usable'].sum()):5d}  "
              f"aSKNA {np.nanmedian(r['X'][:, i('aSKNA')]):7.2f} uV  "
              f"dfSKNA {np.nanmedian(r['X'][:, i('dfSKNA')]):5.2f} Hz  "
              f"SBP {np.nanmean(r['SBP']):6.1f}  [{time.time() - t:.0f}s]",
              flush=True)

    if not frames:
        print("nothing built", file=sys.stderr)
        return 1
    out = pd.concat(frames, ignore_index=True)
    wrote = []
    if args.out:
        out.to_csv(args.out, index=False)
        wrote.append(args.out)
    if args.npz:
        d = os.path.dirname(args.npz)
        if d:
            os.makedirs(d, exist_ok=True)
        sf.save_npz(args.npz, packed)
        wrote.append(args.npz)
    if args.npz_dir:
        wrote.append(os.path.join(args.npz_dir, "<subject>_skna_features.npz"))

    print("\n" + "=" * 74)
    print(f"{len(out)} windows from {out.Subject_ID.nunique()} subjects "
          f"in {time.time() - t0:.0f}s  ->  {', '.join(wrote)}")
    print(f"  window {args.window:.0f} s, stride {args.stride:.0f} s, "
          f"native rate (no decimation)")
    print(f"  usable {int(out.usable.sum())} ({100 * out.usable.mean():.1f}%)   "
          f"labelled {int(out.label_valid.sum())}")
    print(f"  amp_norm {args.amp_norm}"
          + (f" (ref SD {sf.REF_SD_UV:g} uV)" if args.amp_norm == "recording" else ""))
    print(f"  wamp theta {args.theta_uv:g} uV   df band "
          f"{args.df_band[0]:g}-{args.df_band[1]:g} Hz")
    miss = [c for c in sf.FEATURE_NAMES if out[c].isna().any()]
    if miss:
        print("  features with any NaN: "
              + ", ".join(f"{c}({out[c].isna().sum()})" for c in miss))
    else:
        print("  no NaNs in any of the 11 features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
