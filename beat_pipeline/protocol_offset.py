#!/usr/bin/env python3
"""
Recover each recording's true CPT onset from the SKNA response itself.
======================================================================

`is_cpt` is manufactured, not measured: add_protocol() derives it from
`t_center_sec % 600`, which assumes every recording begins exactly at position
0 of the 10-min protocol cycle. Nothing in the raw exports says otherwise -
they carry three channels (CH1, ECG, SKNA) and no event or trigger track. So if
the operator started recording a few seconds early or late, or cued the
immersion by hand, every label in that recording is shifted by an unknown
amount and nobody would notice.

src/core_processing.align_time_axes solves a DIFFERENT alignment - it puts the
BP monitor's clock onto the ECG clock by cross-correlating the two HR traces.
It says nothing about when the hand went into the water.

This recovers that offset the same way, using the response instead of a second
instrument. SKNA's CPT response is a near-binary box (see figures/
onset_latency.png), so sliding the nominal box template against the out-of-fold
p(CPT) trace and taking the peak gives the recording's true onset.

Why this is not circular
------------------------
The predictions are leave-one-person-out. A held-out person's p(CPT) comes from
a model fit on OTHER people, so it never saw - and cannot have absorbed - that
recording's own mis-timed labels. The model supplies a subject-independent
"what CPT looks like in SKNA"; the cross-correlation asks where in THIS
recording that pattern actually sits. Mis-timing in the training labels only
blunts the template, it cannot manufacture a peak at the wrong place.

Read the output two ways:
  * offsets tight around 0  -> the nominal grid is sound; latency measured
    against it is real, and a finer-grained rebuild is worth doing.
  * offsets scattered       -> the grid is loose. Sub-window latency is not
    measurable without fixing this first, because the reference itself moves.

    python3 beat_pipeline/protocol_offset.py
    python3 beat_pipeline/protocol_offset.py --max-shift 120 --step 5

Outputs
-------
  <out>/protocol_offset.csv / .txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_ablation as ma
from onset_latency import CPT_START, CPT_END, CYCLE_SEC, out_of_fold


def box(t_center, shift):
    """Nominal CPT template evaluated as if the protocol began `shift` seconds
    later than assumed."""
    c = ((t_center - shift) / 60.0) % (CYCLE_SEC / 60.0)
    return ((c >= CPT_START) & (c < CPT_END)).astype(int)


def scan(t_center, p, shifts):
    """Point-biserial correlation between p(CPT) and the shifted template."""
    out = []
    for s in shifts:
        b = box(t_center, s)
        if b.sum() == 0 or b.sum() == len(b):
            out.append(np.nan)
            continue
        out.append(np.corrcoef(p, b)[0, 1])
    return np.array(out)


def main():
    p_ = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_.add_argument("--ecg", default="ecg_features_5s.csv")
    p_.add_argument("--skna", default="skna_features_5s.csv")
    p_.add_argument("--out", default="beat_pipeline/built")
    p_.add_argument("--norm", default="expanding")
    p_.add_argument("--family", default="SKNA")
    p_.add_argument("--fpr", type=float, default=0.05)
    p_.add_argument("--max-shift", type=float, default=120.0)
    p_.add_argument("--step", type=float, default=5.0)
    p_.add_argument("--k", type=int, default=2)
    a = p_.parse_args()

    X, names, y, _, groups, meta = ma.build(
        a.norm, a.ecg, a.skna, a.out, a.k, True)
    X = np.nan_to_num(np.asarray(X, float), nan=0.0, posinf=0.0, neginf=0.0)
    y = y.astype(int)
    cycle_min = np.asarray(meta["cycle_min"], float)
    t_center = np.asarray(meta["t_center_sec"], float)
    rec = np.asarray(meta["recording"])

    cols = [i for i, c in enumerate(names)
            if ma.FAMILY[ma.split_column(c)[0]] in set(a.family.split(","))]
    pr, _thr = out_of_fold(X[:, cols], y, groups, cycle_min, a.fpr)

    shifts = np.arange(-a.max_shift, a.max_shift + a.step, a.step)
    rows = []
    for r in pd.unique(rec):
        m = rec == r
        if m.sum() < 40:
            continue
        c = scan(t_center[m], pr[m], shifts)
        if not np.isfinite(c).any():
            continue
        best = int(np.nanargmax(c))
        zero = int(np.argmin(np.abs(shifts)))
        rows.append(dict(
            recording=r, person=groups[m][0], n=int(m.sum()),
            best_shift=float(shifts[best]), r_at_best=float(c[best]),
            r_at_zero=float(c[zero]),
            gain=float(c[best] - c[zero]),
            auc_nominal=roc_auc_score(y[m], pr[m])
            if len(np.unique(y[m])) > 1 else np.nan,
            auc_shifted=roc_auc_score(box(t_center[m], shifts[best]), pr[m])))
    df = pd.DataFrame(rows).sort_values("best_shift")

    sh = df.best_shift
    lines = [
        f"protocol offset recovery  family={a.family}  norm={a.norm}  "
        f"shifts {shifts[0]:+.0f}..{shifts[-1]:+.0f} s step {a.step:.0f} s",
        "",
        f"recordings              {len(df)}",
        f"best shift              median {sh.median():+.1f} s   "
        f"IQR {sh.quantile(.25):+.1f}..{sh.quantile(.75):+.1f}   "
        f"min {sh.min():+.0f}  max {sh.max():+.0f}",
        f"|shift| <= 5 s          {(sh.abs() <= 5).sum()}/{len(df)}",
        f"|shift| <= 10 s         {(sh.abs() <= 10).sum()}/{len(df)}",
        f"mean AUC nominal        {df.auc_nominal.mean():.3f}",
        f"mean AUC at best shift  {df.auc_shifted.mean():.3f}",
        "",
        df[["recording", "person", "best_shift", "r_at_zero", "r_at_best",
            "auc_nominal", "auc_shifted"]].round(3).to_string(index=False),
    ]

    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "protocol_offset.csv"), index=False)
    txt = os.path.join(a.out, "protocol_offset.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {txt}")


if __name__ == "__main__":
    main()
