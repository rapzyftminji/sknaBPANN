#!/usr/bin/env python3
"""
Which window length should the model use?
=========================================

Runs the identical evaluation at several window lengths and puts the numbers
side by side. The BP arm is the one that decides it - CPT is carried along as
the sanity check that the features still describe autonomic state at all.

Deliberately uses RIDGE, not the ANN, for the BP comparison. The ANN has an
iteration cap it was silently hitting (`NOT CONVERGED` on nearly every fold at
max_iter=1000), so any window ranking drawn from it would partly measure how
far training got rather than what the window contains. Ridge has no such knob,
so a difference between windows is a difference between windows.

Every BP cell is reported against ITS OWN baseline, and the column that matters
is `gap` - MAE minus baseline MAE. A window that lowers MAE while lowering the
baseline by more has not helped. Baselines differ per window because the window
set, and therefore the label set, differs.

Stride convention: pass tables built with stride == window, so the windows are
disjoint and n means what it says. The 30 s cohort table in the repo root is
built at stride 5 (83% overlap) - it is flagged in the output rather than
silently compared against disjoint ones.

    python3 beat_pipeline/window_sweep.py
    python3 beat_pipeline/window_sweep.py --windows 5,10,15 --no-cpt

Outputs
-------
  <out>/window_sweep.csv / .txt
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_ablation as ma
import ann_bp_loso as abp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cpt_auc(X, y, groups):
    """SKNA-only logistic LOSO - the sanity check, not the deliverable."""
    oof = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    ok = ~np.isnan(oof)
    return (roc_auc_score(y[ok], oof[ok]),
            average_precision_score(y[ok], oof[ok]))


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", default="5,10,15,30")
    p.add_argument("--out", default="beat_pipeline/built")
    p.add_argument("--norm", default="expanding")
    p.add_argument("--family", default="SKNA,XSIG,HRV,ECG_MORPH,NONLIN",
                   help="families for the BP model")
    p.add_argument("--calib-min", type=float, default=2.0)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--no-cpt", action="store_true")
    a = p.parse_args()

    fams = set(a.family.split(","))
    rows = []
    for w in [x.strip() for x in a.windows.split(",") if x.strip()]:
        ecg = os.path.join(ROOT, f"ecg_features_{w}s.csv")
        skna = os.path.join(ROOT, f"skna_features_{w}s.csv")
        if not (os.path.isfile(ecg) and os.path.isfile(skna)):
            print(f"[{w}s] SKIP - build it first: "
                  f"build_ecg_features.py --window {w} --stride {w}")
            continue
        t0 = time.time()
        X, names, y_cpt, y_sbp, groups, meta = ma.build(
            a.norm, ecg, skna, a.out, a.k, True)
        X = np.nan_to_num(np.asarray(X, float), nan=0.0, posinf=0.0,
                          neginf=0.0)
        Y = np.column_stack([np.asarray(y_sbp, float),
                             np.asarray(meta["y_dbp"], float)])
        t_center = np.asarray(meta["t_center_sec"], float)

        # stride is recoverable from consecutive centres inside one recording
        rec = np.asarray(meta["recording"])
        first = rec == rec[0]
        d = np.diff(np.sort(t_center[first]))
        stride = float(np.median(d)) if len(d) else float(w)

        cols = [i for i, c in enumerate(names)
                if ma.FAMILY[ma.split_column(c)[0]] in fams]
        Xf = X[:, cols]

        row = dict(window=int(w), stride=stride,
                   disjoint=abs(stride - float(w)) < 1e-6,
                   n_windows=len(Y), n_features=len(cols),
                   n_people=len(np.unique(groups)))
        for mode in ("none", "offset"):
            oof, base, _f = abp.run_loso(Xf, Y, groups, t_center,
                                         lambda: abp.make_ref(0), mode,
                                         a.calib_min)
            r = abp.score(Y, oof, "ridge")
            b = abp.score(Y, base, "base")
            row[f"MAE_{mode}"] = r["MAE_SBP"]
            row[f"base_{mode}"] = b["MAE_SBP"]
            row[f"gap_{mode}"] = r["MAE_SBP"] - b["MAE_SBP"]
            row[f"MAE_dbp_{mode}"] = r["MAE_DBP"]
        if not a.no_cpt:
            sk = [i for i, c in enumerate(names)
                  if ma.FAMILY[ma.split_column(c)[0]] == "SKNA"]
            auc, ap = cpt_auc(X[:, sk], np.asarray(y_cpt).astype(int), groups)
            row["cpt_auc"] = auc
            row["cpt_ap"] = ap
            row["cpt_pos"] = int(np.asarray(y_cpt).sum())
        row["seconds"] = time.time() - t0
        rows.append(row)
        print(f"[{w}s] n={row['n_windows']:<5} feats={row['n_features']:<4} "
              f"SBP MAE none {row['MAE_none']:5.2f} (gap {row['gap_none']:+.2f}) "
              f"| offset {row['MAE_offset']:5.2f} (gap {row['gap_offset']:+.2f})"
              + ("" if a.no_cpt else f" | CPT AUC {row['cpt_auc']:.3f}")
              + f"  [{row['seconds']:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("nothing to compare")
        return
    show = ["window", "stride", "disjoint", "n_windows", "n_features",
            "MAE_none", "base_none", "gap_none",
            "MAE_offset", "base_offset", "gap_offset"]
    if not a.no_cpt:
        show += ["cpt_auc", "cpt_ap"]
    best_bp = df.loc[df.gap_offset.idxmin()]
    lines = [
        f"window sweep   norm={a.norm}  family={a.family}  "
        f"calib_min={a.calib_min}  model=ridge (BP) / logistic-SKNA (CPT)",
        "",
        df[show].round(3).to_string(index=False),
        "",
        "gap = model MAE - baseline MAE, in mmHg. NEGATIVE means the model "
        "beats its baseline;",
        "that is the only column that decides anything. Baselines differ per "
        "window because the",
        "window set, and therefore the label set, differs.",
        "",
        f"Smallest gap (calibration=offset): {int(best_bp.window)} s "
        f"({best_bp.gap_offset:+.2f} mmHg)"
        + ("" if best_bp.gap_offset < 0 else
           " - still ABOVE zero, i.e. no window here beats the baseline."),
    ]
    if (~df.disjoint).any():
        bad = df.loc[~df.disjoint, "window"].tolist()
        lines += ["", f"NOTE: {bad} built at stride < window, so their windows "
                      "overlap and their n overstates independent samples. "
                      "Rebuild with --stride == --window to compare cleanly."]
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "window_sweep.csv"), index=False)
    txt = os.path.join(a.out, "window_sweep.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {txt}")


if __name__ == "__main__":
    main()
