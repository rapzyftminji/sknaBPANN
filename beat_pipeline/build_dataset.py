#!/usr/bin/env python3
"""
Build the labelled beat dataset for every subject.
==================================================

Runs stage 1 (preprocessing + R-peaks), stage 2 (segmentation + feature
vectors) and stage 2b (BP labelling) over every subject in
beat_labels.SUBJECT_RECORDING, and writes one .npz per subject.

    python3 beat_pipeline/build_dataset.py --out built/
    python3 beat_pipeline/build_dataset.py --out built/ --subjects s1 s2 s6
    python3 beat_pipeline/build_dataset.py --out built/ --dur 300   # quick pass

Each .npz holds the beat matrix ONCE plus index windows for sequences; see
beat_features.save_npz. Labels are per beat, from the alignment already
resolved in feature_result (see beat_labels).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beat_features as bf
import beat_labels as bl


def build_one(subject, txt_root, feature_root, out_dir, args):
    rec = os.path.join(txt_root, bl.SUBJECT_RECORDING[subject])
    if not os.path.isfile(rec):
        return None, f"missing recording {rec}"
    built = bf.build_recording(
        rec, ecg_channel=args.ecg_channel, skna_channel=args.skna_channel,
        ecg_L=args.ecg_L, skna_L=args.skna_L, M=args.seq_len, stride=args.stride,
        method=args.method, amp_norm=args.amp_norm, dur_sec=args.dur,
        compensate_delay=True, skip_leadin=False)
    bl.attach_labels(built, subject, feature_root=feature_root,
                     mode=args.label_mode, max_gap_sec=args.max_gap)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        dest = os.path.join(out_dir, f"{subject}_beats.npz")
        np.savez_compressed(
            dest, X=built["X"], segments=built["segments"],
            durations=built["durations"], keep=built["keep"],
            seq_idx=built["seq_idx"], beat_time_sec=built["beat_time_sec"],
            SBP=built["SBP"], DBP=built["DBP"],
            label_valid=built["label_valid"], usable=built["usable"],
            fs=built["fs"], ecg_L=built["lengths"]["ecg"],
            skna_L=built["lengths"]["skna"], M=built["M"], stride=built["stride"],
            amp_norm=built["amp_norm"], label_mode=built["label_mode"],
            clock_offset_sec=built["stage1"]["clock_offset_sec"],
            subject=subject, source=os.path.basename(rec))
        built["_dest"] = dest
    return built, None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="output directory for per-subject .npz")
    p.add_argument("--subjects", nargs="*", default=None,
                   help="subset (default: all in SUBJECT_RECORDING)")
    p.add_argument("--txt-root", default="dataset/txt")
    p.add_argument("--feature-root", default="feature_result/5sWindow")
    p.add_argument("--dur", type=float, default=None, help="limit seconds per recording")
    p.add_argument("--ecg-channel", default=bf.DEFAULT_ECG_CHANNEL)
    p.add_argument("--skna-channel", default=bf.DEFAULT_SKNA_CHANNEL)
    p.add_argument("--ecg-L", type=int, default=bf.DEFAULT_ECG_L)
    p.add_argument("--skna-L", type=int, default=bf.DEFAULT_SKNA_L)
    p.add_argument("-M", "--seq-len", type=int, default=12)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--method", default="fft", choices=["fft", "interp"])
    p.add_argument("--amp-norm", default="recording",
                   choices=["recording", "segment", "none"])
    p.add_argument("--label-mode", default="interp", choices=["interp", "window"])
    p.add_argument("--max-gap", type=float, default=10.0)
    p.add_argument("--skip-excluded", action="store_true",
                   help="skip s5 (excluded from this project's LOSO runs)")
    args = p.parse_args(argv)

    subjects = args.subjects or list(bl.SUBJECT_RECORDING)
    if args.skip_excluded:
        subjects = [s for s in subjects if s not in bl.EXCLUDED_SUBJECTS]

    rows, t0 = [], time.time()
    for s in subjects:
        t = time.time()
        try:
            built, err = build_one(s, args.txt_root, args.feature_root, args.out, args)
        except Exception as e:                       # keep going; report at the end
            built, err = None, f"{type(e).__name__}: {e}"
        if err:
            print(f"{s:12s} FAILED  {err}", flush=True)
            rows.append((s, None))
            continue
        print(f"{s:12s} {os.path.basename(built['path']):26s} "
              f"beats {len(built['X']):5d}  labelled {int(built['label_valid'].sum()):5d}  "
              f"seq {len(built['seq_idx']):5d}  "
              f"SBP {np.nanmean(built['SBP']):6.1f}+/-{np.nanstd(built['SBP']):4.1f}  "
              f"DBP {np.nanmean(built['DBP']):5.1f}+/-{np.nanstd(built['DBP']):4.1f}  "
              f"[{time.time() - t:.0f}s]", flush=True)
        rows.append((s, built))

    ok = [(s, b) for s, b in rows if b is not None]
    print("\n" + "=" * 74)
    print(f"BUILT {len(ok)}/{len(subjects)} subjects in {time.time() - t0:.0f}s")
    if ok:
        nb = sum(len(b["X"]) for _, b in ok)
        nl = sum(int(b["label_valid"].sum()) for _, b in ok)
        ns = sum(len(b["seq_idx"]) for _, b in ok)
        mb = sum(b["X"].nbytes for _, b in ok) / 1e6
        print(f"  beats {nb}   labelled {nl} ({100 * nl / max(nb, 1):.1f}%)   sequences {ns}")
        print(f"  beat matrices {mb:.0f} MB   (materialised sequences would be "
              f"{ns * args.seq_len * ok[0][1]['X'].shape[1] * 4 / 1e9:.1f} GB)")
        excl = [s for s, _ in ok if s in bl.EXCLUDED_SUBJECTS]
        if excl:
            print(f"  NOTE: {', '.join(excl)} built but excluded from this "
                  f"project's LOSO runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
