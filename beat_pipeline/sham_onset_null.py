#!/usr/bin/env python3
"""
Sham-onset null for the CPT classifier.
=======================================

`is_cpt` is a deterministic function of clock position: the cold pressor runs
at 4.5-5.5 min of every 10-min cycle, always. Two stages of the pipeline have
a time-dependent character of their own - the expanding normalizer's
median/IQR are unstable for the first ~10 windows of a recording and stabilize
later, and the temporal block (_d1/_rm/_rs/_slope) smooths over adjacent
windows. So there is a path by which a model scores well by reading WHERE IN
THE RECORDING WE ARE rather than what the autonomic state is, and it would
look exactly like a real result.

The usual permutation test - shuffle labels within recording - cannot detect
this, because shuffling destroys the very time structure that is the suspected
carrier. A shuffled label has no clock position to read.

The control that does work: move the onset. Relabel "CPT" as a one-minute slot
somewhere else in the cycle that is genuinely rest, rerun the identical
pipeline, and see what the model scores. A model reading physiology collapses
to chance on a sham onset. A model reading clock position does not.

  real  onset 4.5 min  -> the actual cold pressor
  sham  onset 1.0-3.5  -> pre-CPT rest. The primary null: nothing is
                          happening in this window, for anyone.
  sham  onset 6.5-8.5  -> post-CPT recovery. A weaker null - if sympathetic
                          tone is still settling, a sham here is not empty.
                          Reported, but read the pre-CPT shams first.

Real CPT windows (plus a guard band) are dropped from every sham task, so the
genuine response never sits in the sham's negative class.

    python3 beat_pipeline/sham_onset_null.py
    python3 beat_pipeline/sham_onset_null.py --onsets 1.0,2.0,3.0 --models logit

Outputs
-------
  <out>/sham_onset_null.csv / .txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_ablation as ma

# The five features the in-fold MI ranking selected in 14/14 folds on the real
# task. Testing them by name asks a sharper question than FS8 does: do THESE
# encode clock position?
TOP5 = ["aSKNA_x_HR", "rmsSKNA", "wlSKNA", "wamp_over_wl_rm2",
        "aSKNA_per_beat"]

CPT_START, CPT_END = 4.5, 5.5
GUARD_MIN = 0.5          # keep sham negatives clear of the real response


def sham_mask(cycle_min, onset, width=1.0, guard=GUARD_MIN):
    """(keep, y) for a one-minute sham onset at `onset` min into the cycle.

    Drops the real CPT window and a +-guard band around it, so a model that
    genuinely tracks sympathetic activation cannot be penalized for ranking
    real CPT windows above sham ones - they are simply not in the task.
    """
    real = ((cycle_min >= CPT_START - guard) & (cycle_min < CPT_END + guard))
    y = ((cycle_min >= onset) & (cycle_min < onset + width)).astype(int)
    return ~real, y


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ecg", default="ecg_features_30s.csv")
    p.add_argument("--skna", default="skna_features_30s.csv")
    p.add_argument("--out", default="beat_pipeline/built")
    p.add_argument("--norm", default="expanding,none")
    p.add_argument("--models", default="logit")
    p.add_argument("--onsets", default="1.0,2.0,3.0,6.5,7.5,8.5",
                   help="sham onsets, minutes into the 10-min cycle")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    models = a.models.split(",")
    onsets = [float(x) for x in a.onsets.split(",") if x.strip()]
    rows = []

    for method in a.norm.split(","):
        print(f"\n--- normalization: {method} ---")
        X, names, y_real, _, groups, meta = ma.build(
            method, a.ecg, a.skna, a.out, a.k, True)
        X = np.nan_to_num(np.asarray(X, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        y_real = y_real.astype(int)
        cycle_min = np.asarray(meta["cycle_min"], dtype=float)

        sets = {"FS8_ALL": list(range(X.shape[1]))}
        have = [f for f in TOP5 if f in names]
        if have:
            sets["MI_top5_fixed"] = [names.index(f) for f in have]
            if len(have) < len(TOP5):
                print(f"  note: {set(TOP5) - set(have)} absent under "
                      f"{method}, MI_top5_fixed uses {len(have)}")

        for label, onset in [("REAL", None)] + [("sham", o) for o in onsets]:
            if onset is None:
                keep, y = np.ones(len(y_real), bool), y_real
            else:
                keep, y = sham_mask(cycle_min, onset)
            for set_name, cols in sets.items():
                for mname in models:
                    oof, per_fold = ma.evaluate(
                        X[np.ix_(keep, cols)], y[keep], groups[keep],
                        mname, "cpt", seed=a.seed)
                    s = ma.score(y[keep], oof, per_fold, "cpt")
                    rows.append(dict(norm=method, onset=("real" if onset is None
                                                         else f"{onset:.1f}"),
                                     window=("4.5-5.5" if onset is None
                                             else f"{onset:.1f}-{onset + 1:.1f}"),
                                     feature_set=set_name, model=mname,
                                     n=int(keep.sum()), n_pos=int(y[keep].sum()),
                                     **s))
                    r = rows[-1]
                    print(f"  [{method}] {label:<4} {r['window']:<8} "
                          f"{set_name:<14} {mname:<6} "
                          f"AUC={r['auc_pooled']:.3f} AP={r['ap_pooled']:.3f} "
                          f"fold={r['auc_fold_mean']:.3f}"
                          f"+-{r['auc_fold_std']:.3f} "
                          f"({r['folds_above_chance']}/{r['n_folds']})")

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.splitext(os.path.basename(a.ecg))[0].replace(
        "ecg_features_", "")
    csv_path = os.path.join(a.out, f"sham_onset_null_{tag}.csv")
    df.to_csv(csv_path, index=False)

    grid = ""
    for set_name, sub in df.groupby("feature_set", sort=False):
        g = sub.pivot_table(index="window", columns=["norm", "model"],
                            values="auc_pooled", sort=False)
        grid += f"\n=== pooled AUC  |  {set_name} ===\n{g.round(3).to_string()}\n"
    txt_path = os.path.join(a.out, f"sham_onset_null_{tag}.txt")
    with open(txt_path, "w") as f:
        f.write(f"sham-onset null  models={a.models} onsets={a.onsets}\n"
                f"real CPT = 4.5-5.5 min; sham tasks drop the real window "
                f"+-{GUARD_MIN} min\n{grid}")
    print(grid)
    print(f"wrote {csv_path}\n      {txt_path}")


if __name__ == "__main__":
    main()
