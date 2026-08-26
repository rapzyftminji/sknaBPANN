#!/usr/bin/env python3
"""
The CPT detector: two-layer feedforward ANN, leave-one-subject-out.
===================================================================

Architecture follows Yao et al., IEEE JBHI 26(8) 2022 - a two-layer
feedforward network with sigmoid hidden units and a fixed hidden width of 10,
trained by scaled conjugate gradient. sklearn has no SCG; lbfgs is the closest
quasi-Newton stand-in and is deterministic given the seed, which matters more
here than matching the optimizer name.

Evaluation is leave-one-PERSON-out, not the paper's within-subject split. s5
has two sessions and s13 two cuts of one recording, so 16 Subject_IDs are 14
people; splitting on Subject_ID would put the same person on both sides. The
scaler and the decision threshold are fit inside each training fold - a
held-out person contributes to neither.

The threshold is set at a fixed false-positive rate on the TRAINING folds'
pre-CPT rest windows, so sensitivity/specificity are reported at an operating
point that was chosen without seeing the test subject. Post-CPT recovery is
excluded from that estimate; if sympathetic tone is still settling there it is
not a clean specificity reference.

`--compare` adds a logistic regression on the identical folds. Keep it in the
output: with 14 groups the effective sample size for generalization is 14, not
5281, and the linear reference is what tells you whether the hidden layer is
earning its parameters.

    python3 beat_pipeline/ann_loso.py
    python3 beat_pipeline/ann_loso.py --hidden 5,10,20,50 --compare
    python3 beat_pipeline/ann_loso.py --family SKNA,XSIG --fpr 0.02

Outputs
-------
  <out>/ann_loso_folds.csv     per-subject metrics
  <out>/ann_loso_summary.txt
  <out>/ann_cpt_model.joblib   scaler + net + threshold, fit on ALL people
"""
import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_ablation as ma

CPT_START = 4.5           # minutes into the 10-min cycle


def make_ann(hidden, seed, alpha):
    return MLPClassifier(hidden_layer_sizes=(hidden,), activation="logistic",
                         solver="lbfgs", alpha=alpha, max_iter=1000,
                         random_state=seed)


def make_ref(seed):
    return LogisticRegression(max_iter=2000, class_weight="balanced")


def run_loso(X, y, groups, cycle_min, build_model, fpr, seed):
    """Out-of-fold predictions plus per-fold metrics at an in-fold threshold."""
    oof = np.full(len(y), np.nan)
    thr_all = np.full(len(y), np.nan)
    rows = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = build_model().fit(sc.transform(X[tr]), y[tr])
        p = m.predict_proba(sc.transform(X[te]))[:, 1]
        oof[te] = p

        rest = tr[(y[tr] == 0) & (cycle_min[tr] < CPT_START)]
        thr = np.quantile(m.predict_proba(sc.transform(X[rest]))[:, 1],
                          1.0 - fpr)
        thr_all[te] = thr

        pos, neg = y[te] == 1, y[te] == 0
        rows.append(dict(
            person=groups[te][0], n=len(te), n_pos=int(pos.sum()),
            auc=roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else np.nan,
            ap=average_precision_score(y[te], p) if pos.any() else np.nan,
            sens=float((p[pos] >= thr).mean()) if pos.any() else np.nan,
            spec=float((p[neg] < thr).mean()) if neg.any() else np.nan,
            threshold=float(thr)))
    return oof, thr_all, pd.DataFrame(rows)


def summarize(name, y, oof, folds, elapsed):
    ok = ~np.isnan(oof)
    return dict(
        model=name,
        auc_pooled=roc_auc_score(y[ok], oof[ok]),
        ap_pooled=average_precision_score(y[ok], oof[ok]),
        auc_fold_mean=folds.auc.mean(), auc_fold_std=folds.auc.std(),
        sens_mean=folds.sens.mean(), spec_mean=folds.spec.mean(),
        folds_above_chance=int((folds.auc > 0.5).sum()), n_folds=len(folds),
        seconds=elapsed)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ecg", default="ecg_features_5s.csv")
    p.add_argument("--skna", default="skna_features_5s.csv")
    p.add_argument("--out", default="beat_pipeline/built")
    p.add_argument("--norm", default="expanding")
    p.add_argument("--family", default="SKNA",
                   help="comma-separated FAMILY classes (default SKNA: the "
                        "best set in the ablation)")
    p.add_argument("--hidden", default="10",
                   help="comma-separated hidden widths; 10 is the paper's")
    p.add_argument("--alpha", type=float, default=1e-2, help="L2 penalty")
    p.add_argument("--fpr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--compare", action="store_true",
                   help="also run logistic regression on the identical folds")
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()

    X, names, y, _, groups, meta = ma.build(
        a.norm, a.ecg, a.skna, a.out, a.k, True)
    X = np.nan_to_num(np.asarray(X, float), nan=0.0, posinf=0.0, neginf=0.0)
    y = y.astype(int)
    cycle_min = np.asarray(meta["cycle_min"], float)

    fams = set(a.family.split(","))
    cols = [i for i, c in enumerate(names)
            if ma.FAMILY[ma.split_column(c)[0]] in fams]
    feat = [names[i] for i in cols]
    Xf = X[:, cols]
    print(f"{Xf.shape[1]} features from {sorted(fams)} | {len(y)} windows, "
          f"{y.sum()} CPT ({100.0 * y.mean():.1f}%), "
          f"{len(np.unique(groups))} people\n")

    runs, fold_tables, best = [], {}, None
    for h in [int(x) for x in a.hidden.split(",")]:
        t0 = time.time()
        oof, thr, folds = run_loso(
            Xf, y, groups, cycle_min,
            lambda h=h: make_ann(h, a.seed, a.alpha), a.fpr, a.seed)
        s = summarize(f"ANN({h})", y, oof, folds, time.time() - t0)
        runs.append(s)
        fold_tables[s["model"]] = folds
        if best is None or s["auc_pooled"] > best[1]["auc_pooled"]:
            best = (h, s, thr)
        print(f"  ANN({h:<3}) AUC={s['auc_pooled']:.3f} AP={s['ap_pooled']:.3f} "
              f"fold={s['auc_fold_mean']:.3f}+-{s['auc_fold_std']:.3f} "
              f"sens={s['sens_mean']:.3f} spec={s['spec_mean']:.3f} "
              f"({s['folds_above_chance']}/{s['n_folds']})  [{s['seconds']:.0f}s]")

    if a.compare:
        t0 = time.time()
        oof, _t, folds = run_loso(Xf, y, groups, cycle_min,
                                  lambda: make_ref(a.seed), a.fpr, a.seed)
        s = summarize("logistic", y, oof, folds, time.time() - t0)
        runs.append(s)
        fold_tables["logistic"] = folds
        print(f"  logistic  AUC={s['auc_pooled']:.3f} AP={s['ap_pooled']:.3f} "
              f"fold={s['auc_fold_mean']:.3f}+-{s['auc_fold_std']:.3f} "
              f"sens={s['sens_mean']:.3f} spec={s['spec_mean']:.3f} "
              f"({s['folds_above_chance']}/{s['n_folds']})  [{s['seconds']:.0f}s]")

    res = pd.DataFrame(runs)
    h_best, s_best, thr_best = best
    fb = fold_tables[s_best["model"]]

    lines = [
        f"ANN LOSO  family={a.family}  norm={a.norm}  alpha={a.alpha}  "
        f"fpr={a.fpr}  seed={a.seed}",
        f"{Xf.shape[1]} features, {len(y)} windows, {y.sum()} CPT, "
        f"{len(np.unique(groups))} people (leave-one-person-out)",
        "",
        res.round(3).to_string(index=False),
        "",
        f"per-subject detail, {s_best['model']}:",
        fb.round(3).to_string(index=False),
        "",
        f"prevalence baseline: AUC 0.500, AP {y.mean():.3f}",
    ]
    os.makedirs(a.out, exist_ok=True)
    pd.concat([f.assign(model=m) for m, f in fold_tables.items()]).to_csv(
        os.path.join(a.out, "ann_loso_folds.csv"), index=False)
    txt = os.path.join(a.out, "ann_loso_summary.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[3:]))

    if not a.no_save:
        # Deployment artifact: refit on ALL people. Its threshold is the mean of
        # the fold thresholds - each was estimated without its own test subject,
        # so the average is not tuned to any one person.
        sc = StandardScaler().fit(Xf)
        net = make_ann(h_best, a.seed, a.alpha).fit(sc.transform(Xf), y)
        path = os.path.join(a.out, "ann_cpt_model.joblib")
        joblib.dump(dict(scaler=sc, model=net, feature_names=feat,
                         threshold=float(np.nanmean(thr_best)),
                         hidden=h_best, family=a.family, norm=a.norm,
                         window_sec=5.0, loso_auc=s_best["auc_pooled"]), path)
        print(f"\nsaved {path}  (hidden={h_best}, "
              f"threshold={np.nanmean(thr_best):.3f})")
    print(f"wrote {txt}")


if __name__ == "__main__":
    main()
