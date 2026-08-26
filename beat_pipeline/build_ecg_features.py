#!/usr/bin/env python3
"""
Build the 19-feature ECG window table for every subject, with BP labels.
=======================================================================

    python3 beat_pipeline/build_ecg_features.py --out ecg_features_30s.csv
    python3 beat_pipeline/build_ecg_features.py --window 30 --stride 5 --out f.csv
    python3 beat_pipeline/build_ecg_features.py --subjects s1 s6 --dur 300

Writes ONE tidy CSV: a row per window, columns

    Subject_ID, Recording, window_idx, t_start_sec, t_center_sec,
    t_raw_center_sec, n_beats, lf_reliable, <19 features>,
    SBP, DBP, label_valid, usable

t_center_sec is on the analysed span; t_raw_center_sec adds clock_offset_sec
so it can be compared with anything derived from the untrimmed recording.
`usable` = labelled AND >=3 beats AND all 19 features finite - use it as the
training mask rather than dropping rows here, so nothing is silently lost.

FOR TRAINING, add --npz: same table as arrays, loadable with
ecg_features.load_npz(). The CSV stays the human-readable copy.

    python3 beat_pipeline/build_ecg_features.py --npz beat_pipeline/built/ecg_features_30s.npz
    python3 beat_pipeline/build_ecg_features.py --npz-dir beat_pipeline/built/   # per subject

    # already have the CSV? pack it without recomputing anything:
    python3 beat_pipeline/build_ecg_features.py --from-csv ecg_features_30s.csv \\
        --npz beat_pipeline/built/ecg_features_30s.npz
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beat_labels as bl
import ecg_features as ef


def arrays_from_csv(csv_path, analysis_fs=ef.ANALYSIS_FS, label_mode="interp"):
    """Re-pack an existing CSV built by this script, no recomputation.

    window_sec and stride_sec are recovered exactly from the time columns.
    analysis_fs and label_mode are NOT in the CSV - they are recorded as
    whatever is passed here, so override them if the CSV was not built with
    the defaults, or the packed file will claim a rate it does not have.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in ef.FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing {len(missing)} feature columns "
                         f"({', '.join(missing[:4])}...) - not a table from this script")
    win = 2.0 * float(np.median(df.t_center_sec - df.t_start_sec))
    d = np.diff(df.t_start_sec.to_numpy())
    stride = float(np.median(d[d > 0])) if (d > 0).any() else win
    out = {
        "X": df[ef.FEATURE_NAMES].to_numpy(np.float32),
        "subject": df.Subject_ID.to_numpy(str),
        "person": df.Subject_ID.map(bl.person_of).to_numpy(str),
        "recording": df.Recording.to_numpy(str),
        "window_idx": df.window_idx.to_numpy(np.int32),
        "window_sec": win, "stride_sec": stride,
        "analysis_fs": float(analysis_fs), "label_mode": label_mode,
    }
    for c in ("t_start_sec", "t_center_sec", "t_raw_center_sec", "SBP", "DBP"):
        out[c] = df[c].to_numpy(np.float64)
    out["n_beats"] = df.n_beats.to_numpy(np.int32)
    for c in ("lf_reliable", "label_valid", "usable"):
        out[c] = df[c].to_numpy(bool)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="ecg_features_30s.csv",
                   help="tidy CSV output ('' or --no-csv to skip)")
    p.add_argument("--no-csv", action="store_true", help="skip the CSV")
    p.add_argument("--npz", default=None,
                   help="also pack every subject into ONE .npz for training")
    p.add_argument("--npz-dir", default=None,
                   help="also write one <subject>_features.npz per subject")
    p.add_argument("--from-csv", default=None,
                   help="pack an existing CSV to --npz without recomputing")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--txt-root", default="dataset/txt")
    p.add_argument("--feature-root", default="feature_result/5sWindow")
    p.add_argument("--window", type=float, default=ef.DEFAULT_WINDOW_SEC)
    p.add_argument("--stride", type=float, default=ef.DEFAULT_STRIDE_SEC)
    p.add_argument("--analysis-fs", type=float, default=ef.ANALYSIS_FS)
    p.add_argument("--ecg-channel", default="CH40")
    p.add_argument("--amp-norm", default="none", choices=["none", "recording"],
                   help="'recording' scales each recording to unit SD before "
                        "features; moves ST_level/R_slope only, timing "
                        "features and R-peak detection are unaffected")
    p.add_argument("--label-mode", default="interp", choices=["interp", "window"])
    p.add_argument("--dur", type=float, default=None)
    p.add_argument("--skip-excluded", action="store_true",
                   help="skip s5 (excluded from this project's LOSO runs)")
    args = p.parse_args(argv)

    if args.from_csv:
        if not args.npz:
            print("--from-csv needs --npz (there is nothing else to write)",
                  file=sys.stderr)
            return 2
        a = arrays_from_csv(args.from_csv, args.analysis_fs, args.label_mode)
        shape = ef.save_npz(args.npz, a)
        print(f"packed {args.from_csv} -> {args.npz}  X{shape}")
        print(f"  window {a['window_sec']:.0f} s, stride {a['stride_sec']:.0f} s, "
              f"analysis {a['analysis_fs']:.0f} Hz (assumed, not stored in the CSV)")
        print(f"  {len(np.unique(a['subject']))} subjects / "
              f"{len(np.unique(a['person']))} people (group LOSO on `person`)   "
              f"usable {int(a['usable'].sum())}/{len(a['usable'])}")
        return 0

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
            r = ef.build_recording(
                rec, subject=subj, window_sec=args.window, stride_sec=args.stride,
                analysis_fs=args.analysis_fs, ecg_channel=args.ecg_channel,
                feature_root=args.feature_root, label_mode=args.label_mode,
                dur_sec=args.dur, amp_norm=args.amp_norm)
        except Exception as e:
            print(f"{subj:12s} FAILED  {type(e).__name__}: {e}", flush=True)
            continue

        df = pd.DataFrame(r["X"], columns=ef.FEATURE_NAMES)
        df.insert(0, "Subject_ID", subj)
        df.insert(1, "Recording", os.path.basename(rec))
        df.insert(2, "window_idx", np.arange(len(df)))
        df.insert(3, "t_start_sec", r["t_start_sec"])
        df.insert(4, "t_center_sec", r["t_center_sec"])
        df.insert(5, "t_raw_center_sec", r["t_center_sec"] + r["clock_offset_sec"])
        df.insert(6, "n_beats", r["n_beats"])
        df.insert(7, "lf_reliable", r["lf_reliable"])
        df["SBP"], df["DBP"] = r["SBP"], r["DBP"]
        df["label_valid"], df["usable"] = r["label_valid"], r["usable"]
        frames.append(df)

        # Convert now and drop `r`: it holds the whole decimated recording plus
        # every beat's delineation, and 16 of those held at once is GBs.
        if args.npz or args.npz_dir:
            a = ef.window_arrays(r)
            packed.append(a)
            if args.npz_dir:
                os.makedirs(args.npz_dir, exist_ok=True)
                ef.save_npz(os.path.join(args.npz_dir, f"{subj}_features.npz"), a)
        del r["ecg_a"], r["rp_a"], r["beats"]

        print(f"{subj:12s} {os.path.basename(rec):26s} windows {len(df):5d}  "
              f"usable {int(r['usable'].sum()):5d}  beats/win "
              f"{np.median(r['n_beats']):3.0f}  "
              f"SBP {np.nanmean(r['SBP']):6.1f}  DBP {np.nanmean(r['DBP']):5.1f}  "
              f"[{time.time() - t:.0f}s]", flush=True)

    if not frames:
        print("nothing built", file=sys.stderr)
        return 1
    out = pd.concat(frames, ignore_index=True)
    wrote = []
    if args.out and not args.no_csv:
        out.to_csv(args.out, index=False)
        wrote.append(args.out)
    if args.npz:
        d = os.path.dirname(args.npz)
        if d:
            os.makedirs(d, exist_ok=True)
        ef.save_npz(args.npz, packed)
        wrote.append(args.npz)
    if args.npz_dir:
        wrote.append(os.path.join(args.npz_dir, "<subject>_features.npz"))

    print("\n" + "=" * 74)
    print(f"{len(out)} windows from {out.Subject_ID.nunique()} subjects "
          f"in {time.time() - t0:.0f}s  ->  {', '.join(wrote) or 'nothing written'}")
    print(f"  window {args.window:.0f} s, stride {args.stride:.0f} s, "
          f"analysis {args.analysis_fs:.0f} Hz")
    print(f"  usable {int(out.usable.sum())} ({100 * out.usable.mean():.1f}%)   "
          f"labelled {int(out.label_valid.sum())}   "
          f"LF-reliable {int(out.lf_reliable.sum())}")
    miss = [c for c in ef.FEATURE_NAMES if out[c].isna().any()]
    if miss:
        print("  features with any NaN: "
              + ", ".join(f"{c}({out[c].isna().sum()})" for c in miss))
    else:
        print("  no NaNs in any of the 19 features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
