#!/usr/bin/env python3
"""Device-validation grades for a BP predictions file — SBP *and* DBP.

    python3 beat_pipeline/bp_grading.py --preds beat_pipeline/built/<file>.csv

Wraps src/bp_standards.py (BHS, IEEE 1708, AAMI SP10) and reports, per channel:

    n, ME, SD, MAE, BHS grade + the three cumulative percentages,
    IEEE 1708 grade, AAMI SP10 pass/fail

ME and SD are the two AAMI quantities and are NOT interchangeable with MAE:
MAE is the average size of the error, ME its average signed value (bias), SD its
spread. A model can have ME ~ 0 and still be useless if SD is large.

EVERY BASELINE IS GRADED TOO, and that is the point. These standards were
written for a device against a reference, not for a model against a baseline,
so a grade on its own does not say whether the features did anything. If
"predict this person's own mean, using no features" also grades A, then A is
what the protocol yields on this cohort, not what the model earned.

Two caveats that must travel with any grade produced here:
  * BHS/AAMI/ISO assume >= 85 subjects with several readings each. With 13-14
    subjects these are indicative only.
  * Readings from a calibrated model already contain the per-subject
    calibration offset, which flatters every absolute-error statistic.

Arms are auto-detected from the column names, so it works on both
ann_bp_loso.py's predictions file and ann_personalize.py's.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from bp_standards import compute_bp_standards          # noqa: E402

CHANNELS = ("SBP", "DBP")

# label -> column suffix, in report order
ARMS = [
    ("ANN (the model)", "pred"),
    ("population model (before personalizing)", "pre"),
    ("BASELINE zero-delta / train-mean", "zero"),
    ("BASELINE adapt-mean (own avg, NO features)", "amean"),
    ("ORACLE subject-mean (cheats, NO features)", "oracle"),
]


def grade_file(path, subject_col="person"):
    df = pd.read_csv(path)
    if subject_col not in df.columns:
        subject_col = "Subject_ID" if "Subject_ID" in df.columns else None
    subj = df[subject_col] if subject_col else np.zeros(len(df))

    rows = []
    for label, suf in ARMS:
        if not all(f"{c}_{suf}" in df.columns for c in CHANNELS):
            continue
        for ch in CHANNELS:
            st = compute_bp_standards(df[f"{ch}_{suf}"], df[f"{ch}_true"], subj)
            rows.append(dict(
                arm=label, channel=ch, n=st["n"], n_subjects=st["n_subjects"],
                ME=st["me"], SD=st["sd"], MAE=st["mae"],
                BHS=st["bhs"]["grade"],
                pct5=st["bhs"]["pct_within_5"],
                pct10=st["bhs"]["pct_within_10"],
                pct15=st["bhs"]["pct_within_15"],
                IEEE=st["ieee"]["grade"],
                AAMI="PASS" if st["aami"]["passed"] else "FAIL"))
    return pd.DataFrame(rows)


def report(g, title):
    w = 104
    out = ["=" * w, f"BP DEVICE-VALIDATION GRADES — {title}", "=" * w, ""]
    hdr = (f"{'arm':<42}{'ch':<5}{'ME':>7}{'SD':>7}{'MAE':>7}"
           f"{'BHS':>5}{'≤5%':>7}{'≤10%':>7}{'≤15%':>7}{'IEEE':>6}{'AAMI':>6}")
    out += [hdr, "-" * w]
    for arm in g.arm.unique():
        for _, r in g[g.arm == arm].iterrows():
            out.append(f"{arm if r.channel == 'SBP' else '':<42}"
                       f"{r.channel:<5}{r.ME:>7.2f}{r.SD:>7.2f}{r.MAE:>7.2f}"
                       f"{r.BHS:>5}{r.pct5:>7.1f}{r.pct10:>7.1f}"
                       f"{r.pct15:>7.1f}{r.IEEE:>6}{r.AAMI:>6}")
        out.append("")
    n_sub = int(g.n_subjects.iloc[0])
    out += [
        "-" * w,
        "ME = mean signed error (bias).  SD = spread of the error.  "
        "MAE = mean absolute error.",
        "BHS grade needs ALL THREE of >=60/85/95 % within 5/10/15 mmHg for A; "
        "50/75/90 for B; 40/65/85 for C.",
        "IEEE 1708 grades on MAE alone: A <=5, B <=6, C <=7, D >7 mmHg.",
        "AAMI SP10 passes if |ME| <= 5 AND SD <= 8 mmHg.",
        "",
        f"CAVEAT 1: these standards assume >=85 subjects; this cohort has "
        f"{n_sub}. Treat as indicative only.",
        "CAVEAT 2: with calibration on, every arm here already contains the "
        "subject's own calibration",
        "          offset, which flatters all of these statistics.",
        "CAVEAT 3: compare the model's row against the no-feature rows above "
        "it. A grade the baseline also",
        "          reaches is a property of this cohort, not of the model.",
    ]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds", required=True, help="predictions CSV")
    p.add_argument("--title", default="")
    p.add_argument("--out", default="", help="write the tidy CSV here")
    a = p.parse_args()

    g = grade_file(a.preds)
    if not len(g):
        raise SystemExit(f"no gradable arms found in {a.preds}")
    txt = report(g, a.title or os.path.basename(a.preds))
    print(txt)
    out = a.out or os.path.splitext(a.preds)[0] + "_grades.csv"
    g.to_csv(out, index=False)
    with open(os.path.splitext(out)[0] + ".txt", "w") as fh:
        fh.write(txt + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
