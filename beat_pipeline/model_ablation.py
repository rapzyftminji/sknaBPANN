#!/usr/bin/env python3
"""
Feature-set ablation for the 30 s window table, with and without the
within-recording normalization.
==========================================================================

Answers the two questions a single headline number cannot:

  1. WHICH FEATURE FAMILY carries the signal? Does SKNA add anything over
     ECG/HRV alone, or is the classifier just reading heart rate?
  2. HOW MUCH OF IT IS THE NORMALIZER? Within-recording normalization lifted
     CPT LOSO AUC from 0.685 to 0.961. If the family ranking flips between
     the normalized and un-normalized arms, the ablation is measuring the
     normalizer, not the physiology - so every row is run both ways.

Structured after the FS1..FS7 ablation in Yao et al., "Multi-Dimensional
Feature Combination Method for Continuous Blood Pressure Measurement Based
on Wrist PPG Sensor" (IEEE JBHI 26(8), 2022), with two deliberate
departures:

  * They split the first 70% / last 30% of EACH subject's beats, so every
    test subject was also in training and their demographic features (age,
    height, weight, BMI, gender) act as a subject ID - the network can
    memorize each person's BP level. Here the split is LeaveOneGroupOut on
    `person`, so a held-out person is never seen in any form.
  * They ranked features by mutual information on the whole dataset and
    then reported the top-15 subset. That leaks. `--rank-k` here recomputes
    the MI ranking inside each training fold.

Every table carries its trivial baselines. On the CPT task that is the
prevalence line (AP) and chance (AUC); on the BP task it is the constant
train-mean predictor plus an ORACLE per-subject-mean row, which is the
number the paper's demographic features actually reproduce.

    python3 beat_pipeline/model_ablation.py --task cpt
    python3 beat_pipeline/model_ablation.py --task cpt --rank-k 5,10,15,30
    python3 beat_pipeline/model_ablation.py --task sbp --models ridge

Outputs
-------
  <out>/ablation_<task>.csv     one row per (norm, block, feature set, model)
  <out>/ablation_<task>.txt     the same thing as a printable grid
  <out>/ablation_cache_<m>.npz  engineered table per normalization method
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feature_engineering as fe


# --------------------------------------------------------------------------- 1
# Feature families. Keys are BASE feature names (before the _d1/_rm/_rs/_slope
# temporal suffixes). Every engineered column must map onto exactly one key or
# the run aborts - a silently unmapped column would quietly vanish from every
# feature set and make the ablation a lie.
FAMILY = {
    # beat morphology: what one QRS-T complex looks like
    "TpTe": "ECG_MORPH", "QTc_Fridericia": "ECG_MORPH",
    "QRS_duration": "ECG_MORPH", "ST_level": "ECG_MORPH",
    "T_R_ratio": "ECG_MORPH", "R_slope": "ECG_MORPH", "beat_SNR": "ECG_MORPH",
    # rate and rate variability: the classical autonomic readout
    "meanNN": "HRV", "SDNN": "HRV", "RMSSD": "HRV", "HF_power": "HRV",
    "LF_power": "HRV", "LF_HF_ratio": "HRV", "SD1_SD2": "HRV", "HR": "HRV",
    "SDNN_over_RMSSD": "HRV",
    # nonlinear / complexity descriptors of the ECG trace
    "mobility": "NONLIN", "complexity": "NONLIN",
    "fractal_dimension": "NONLIN", "entropy": "NONLIN",
    "autocorrelation": "NONLIN",
    # SKNA burst content, including SKNA-internal ratios
    "aSKNA": "SKNA", "rmsSKNA": "SKNA", "skewSKNA": "SKNA",
    "kurtSKNA": "SKNA", "wlSKNA": "SKNA", "zcSKNA": "SKNA",
    "sscSKNA": "SKNA", "cfSKNA": "SKNA", "dfSKNA": "SKNA",
    "varSKNA": "SKNA", "wampSKNA": "SKNA", "wamp_over_wl": "SKNA",
    "SKNA_crest": "SKNA",
    # genuinely cross-signal: need BOTH the SKNA burst and the ECG rhythm
    "aSKNA_per_beat": "XSIG", "aSKNA_x_HR": "XSIG",
    "aSKNA_over_RMSSD": "XSIG", "aSKNA_over_SDNN": "XSIG",
}

ECG_ALL = ("ECG_MORPH", "HRV", "NONLIN")

# Mirrors the paper's FS1..FS7: each family alone, then the combinations that
# isolate what SKNA contributes on top of a fully-specified ECG model.
FEATURE_SETS = [
    ("FS1_ECG_MORPH", ("ECG_MORPH",)),
    ("FS2_HRV",       ("HRV",)),
    ("FS3_NONLIN",    ("NONLIN",)),
    ("FS4_SKNA",      ("SKNA",)),
    ("FS5_ECG_ALL",   ECG_ALL),
    ("FS6_ECG+SKNA",  ECG_ALL + ("SKNA",)),
    ("FS7_SKNA+XSIG", ("SKNA", "XSIG")),
    ("FS8_ALL",       ECG_ALL + ("SKNA", "XSIG")),
]

TEMPORAL_SUFFIXES = ("_d1", "_slope")          # plus _rm{k} / _rs{k}, matched
                                               # numerically in split_column()


def split_column(col):
    """(base, suffix) for an engineered column name.

    Matches the longest FAMILY key the column starts with, so `R_slope`
    resolves to base `R_slope` (level) and `R_slope_slope` to base `R_slope`
    with suffix `_slope` - a plain right-strip of `_slope` would mangle both.
    """
    best = None
    for base in FAMILY:
        if col == base or col.startswith(base + "_"):
            if best is None or len(base) > len(best):
                best = base
    if best is None:
        raise KeyError(f"column {col!r} maps to no FAMILY entry - add it to "
                       "FAMILY or the ablation would silently drop it")
    return best, col[len(best):]


def is_level(col):
    return split_column(col)[1] == ""


# --------------------------------------------------------------------------- 2
def build(method, ecg_csv, skna_csv, out_dir, k, causal, force=False,
          keep_s13="full"):
    """Engineer the window table under one normalization method, with cache.

    method='expanding' is the deployable causal within-recording normalizer;
    method='none' is the control arm that leaves the raw levels alone, so the
    between-recording offset (electrode placement, gain, skin impedance) is
    still in the features.
    """
    # tag the cache with the source table, or a 5 s run silently reuses the
    # 30 s engineered features
    tag = os.path.splitext(os.path.basename(ecg_csv))[0].replace(
        "ecg_features_", "")
    # keep_s13 changes WHICH windows are in the table, so it has to be part of
    # the cache identity - otherwise asking for the 10-min cut silently returns
    # a cache built from the full recording. Default stays on the old name so
    # existing caches remain valid.
    if keep_s13 != "full":
        tag += f"_s13{keep_s13}"
    cache = os.path.join(out_dir, f"ablation_cache_{tag}_{method}.npz")
    if os.path.exists(cache) and not force:
        d = np.load(cache, allow_pickle=True)
        if "y_dbp" in d.files:                 # older caches lack it: rebuild
            return _unpack(d)

    t0 = time.time()
    df, cols, _ = fe.engineer(ecg_csv, skna_csv, method=method, k=k,
                              causal=causal, keep_s13=keep_s13)
    print(f"  [{method}] engineered {len(cols)} features x {len(df)} windows "
          f"in {time.time() - t0:.1f}s")
    os.makedirs(out_dir, exist_ok=True)
    X = df[cols].to_numpy(np.float32)
    out = dict(X=X, feature_names=np.array(cols),
               y_cpt=df["is_cpt"].to_numpy(np.int8),
               y_sbp=df["SBP"].to_numpy(np.float32),
               y_dbp=df["DBP"].to_numpy(np.float32),
               groups=df["person"].to_numpy(),
               recording=df["Recording"].to_numpy(),
               cycle_min=df["cycle_min"].to_numpy(np.float32),
               t_center_sec=df["t_center_sec"].to_numpy(np.float32))
    np.savez_compressed(cache, **out)
    return _unpack(out)


def _unpack(d):
    """Cache -> the tuple every caller destructures. `meta` carries the columns
    only the timing analyses need, so adding one never changes the arity."""
    meta = {k: np.asarray(d[k]) for k in
            ("recording", "cycle_min", "t_center_sec", "y_dbp")}
    return (np.asarray(d["X"]), [str(c) for c in d["feature_names"]],
            np.asarray(d["y_cpt"]), np.asarray(d["y_sbp"]),
            np.asarray(d["groups"]), meta)


# --------------------------------------------------------------------------- 3
def make_model(name, task, seed=0):
    """The paper's ANN is MATLAB fitnet's default - 10 sigmoid hidden units,
    linear output, scaled conjugate gradient. `mlp` below is that shape;
    lbfgs is the closest sklearn analogue to SCG. `logit`/`ridge` is the
    reference it has to beat, not a formality: with 14 groups the effective
    sample size for generalization is 14, not 5367."""
    if task == "cpt":
        if name == "logit":
            return LogisticRegression(max_iter=2000, C=1.0,
                                      class_weight="balanced")
        if name == "mlp":
            return MLPClassifier(hidden_layer_sizes=(10,),
                                 activation="logistic", solver="lbfgs",
                                 alpha=1e-2, max_iter=800, random_state=seed)
    else:
        if name == "ridge":
            return Ridge(alpha=10.0)
        if name == "mlp":
            return MLPRegressor(hidden_layer_sizes=(10,),
                                activation="logistic", solver="lbfgs",
                                alpha=1e-2, max_iter=800, random_state=seed)
    raise ValueError(f"unknown model {name!r} for task {task!r}")


def fold_mi(X, y, tr, task, seed=0):
    """Univariate MI of every column, computed on the TRAINING rows only.

    Univariate ranking over the full column set restricts correctly to any
    subset, so this is computed once per (norm, fold) and reused by every
    feature set instead of once per (norm, fold, feature set).
    """
    f = mutual_info_classif if task == "cpt" else mutual_info_regression
    return f(X[tr], y[tr], random_state=seed)


def evaluate(X, y, groups, model_name, task, top_k=None, mi_cache=None,
             seed=0):
    """LeaveOneGroupOut on `person`. The scaler and the MI ranking are fit
    inside the fold; the held-out person contributes nothing to either."""
    logo = LeaveOneGroupOut()
    oof = np.full(len(y), np.nan)
    per_fold = []

    for i, (tr, te) in enumerate(logo.split(X, y, groups)):
        cols = slice(None)
        if top_k is not None:
            mi = mi_cache[i] if mi_cache is not None else fold_mi(X, y, tr,
                                                                 task, seed)
            cols = np.argsort(mi)[::-1][:top_k]
        Xtr, Xte = X[tr][:, cols], X[te][:, cols]

        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

        if task == "cpt" and len(np.unique(y[tr])) < 2:
            continue
        m = make_model(model_name, task, seed).fit(Xtr, y[tr])
        pred = (m.predict_proba(Xte)[:, 1] if task == "cpt"
                else m.predict(Xte))
        oof[te] = pred

        if task == "cpt":
            # a person with no CPT window (or all CPT) has no defined AUC
            per_fold.append(roc_auc_score(y[te], pred)
                            if len(np.unique(y[te])) > 1 else np.nan)
        else:
            per_fold.append(np.mean(np.abs(y[te] - pred)))

    return oof, np.array(per_fold, dtype=float)


def score(y, oof, per_fold, task):
    ok = ~np.isnan(oof)
    if task == "cpt":
        return {
            "auc_pooled": roc_auc_score(y[ok], oof[ok]),
            "ap_pooled": average_precision_score(y[ok], oof[ok]),
            "auc_fold_mean": np.nanmean(per_fold),
            "auc_fold_std": np.nanstd(per_fold),
            "folds_above_chance": int(np.nansum(per_fold > 0.5)),
            "n_folds": int(np.sum(~np.isnan(per_fold))),
        }
    err = oof[ok] - y[ok]
    return {
        "mae": np.mean(np.abs(err)),
        "me": np.mean(err),
        "sde": np.std(err, ddof=1),
        "mae_fold_mean": np.nanmean(per_fold),
        "mae_fold_std": np.nanstd(per_fold),
        "n_folds": int(np.sum(~np.isnan(per_fold))),
    }


# --------------------------------------------------------------------------- 4
def baselines(y, groups, task):
    """The rows every table needs so a headline number cannot be read alone."""
    rows = []
    logo = LeaveOneGroupOut()
    Xd = np.zeros((len(y), 1))

    if task == "cpt":
        oof, per_fold = np.full(len(y), np.nan), []
        for tr, te in logo.split(Xd, y, groups):
            d = DummyClassifier(strategy="prior").fit(Xd[tr], y[tr])
            oof[te] = d.predict_proba(Xd[te])[:, 1]
            per_fold.append(np.nan)
        rows.append(dict(feature_set="BASE_prevalence", model="constant",
                         n_features=0,
                         auc_pooled=0.5,
                         ap_pooled=float(np.mean(y)),
                         auc_fold_mean=np.nan, auc_fold_std=np.nan,
                         folds_above_chance=0, n_folds=0))
        return rows

    # BP: the constant predictor a LOSO model actually has to beat ...
    oof, per_fold = np.full(len(y), np.nan), []
    for tr, te in logo.split(Xd, y, groups):
        oof[te] = np.mean(y[tr])
        per_fold.append(np.mean(np.abs(y[te] - np.mean(y[tr]))))
    rows.append(dict(feature_set="BASE_train_mean", model="constant",
                     n_features=0,
                     **score(y, oof, np.array(per_fold), task)))

    # ... and the ORACLE the paper's demographic features quietly reproduce:
    # predict each held-out person's OWN mean BP. Not achievable without
    # having seen that person's cuff readings. Reported for calibration only.
    oof = np.empty(len(y), dtype=float)
    per_fold = []
    for g in np.unique(groups):
        m = groups == g
        oof[m] = np.mean(y[m])
        per_fold.append(np.mean(np.abs(y[m] - np.mean(y[m]))))
    rows.append(dict(feature_set="ORACLE_subject_mean", model="cheating",
                     n_features=0,
                     **score(y, oof, np.array(per_fold), task)))
    return rows


# --------------------------------------------------------------------------- 5
def run_arm(method, X, names, y, groups, task, models, blocks, rank_k, seed):
    fam = {c: split_column(c)[0] for c in names}
    famcls = {c: FAMILY[fam[c]] for c in names}
    idx = {c: i for i, c in enumerate(names)}

    mi_cache = None
    if rank_k:
        t0 = time.time()
        logo = LeaveOneGroupOut()
        mi_cache = [fold_mi(X, y, tr, task, seed)
                    for tr, _ in logo.split(X, y, groups)]
        print(f"  [{method}] in-fold MI: {len(mi_cache)} folds x "
              f"{X.shape[1]} features in {time.time() - t0:.1f}s")

    rows = []
    for block in blocks:
        for fs_name, fams in FEATURE_SETS:
            cols = [c for c in names if famcls[c] in fams
                    and (block == "level+temporal" or is_level(c))]
            if not cols:
                continue
            sub = X[:, [idx[c] for c in cols]]
            for mname in models:
                t0 = time.time()
                oof, per_fold = evaluate(sub, y, groups, mname, task,
                                         seed=seed)
                rows.append(dict(norm=method, block=block, feature_set=fs_name,
                                 model=mname, n_features=len(cols),
                                 **score(y, oof, per_fold, task)))
                print(f"  [{method}/{block}] {fs_name:<15} {mname:<6} "
                      f"n={len(cols):<4} {_headline(rows[-1], task)} "
                      f"({time.time() - t0:.1f}s)")

    # in-fold MI top-k over the full feature set, the honest version of the
    # paper's FS7
    for k in rank_k:
        for mname in models:
            oof, per_fold = evaluate(X, y, groups, mname, task, top_k=k,
                                     mi_cache=mi_cache, seed=seed)
            rows.append(dict(norm=method, block="level+temporal",
                             feature_set=f"MI_top{k}", model=mname,
                             n_features=k,
                             **score(y, oof, per_fold, task)))
            print(f"  [{method}/MI] top{k:<12} {mname:<6} n={k:<4} "
                  f"{_headline(rows[-1], task)}")

    for b in baselines(y, groups, task):
        rows.append(dict(norm=method, block="-", **b))
    return rows


def _headline(row, task):
    if task == "cpt":
        return (f"AUC={row['auc_pooled']:.3f} AP={row['ap_pooled']:.3f} "
                f"fold={row['auc_fold_mean']:.3f}+-{row['auc_fold_std']:.3f} "
                f"({row['folds_above_chance']}/{row['n_folds']})")
    return (f"MAE={row['mae']:.2f} ME={row['me']:+.2f} SDE={row['sde']:.2f}")


def to_grid(df, task):
    """Table-III-shaped: rows = feature set, columns = norm x model."""
    metric = "auc_pooled" if task == "cpt" else "mae"
    out = []
    for block, sub in df[df["block"] != "-"].groupby("block", sort=False):
        g = sub.pivot_table(index="feature_set", columns=["norm", "model"],
                            values=metric, sort=False)
        out.append(f"\n=== {metric}  |  block = {block} ===\n"
                   + g.round(3).to_string())
    base = df[df["block"] == "-"]
    if len(base):
        cols = ([c for c in ("auc_pooled", "ap_pooled") if c in base]
                if task == "cpt" else ["mae", "me", "sde"])
        out.append("\n=== baselines ===\n"
                   + base[["norm", "feature_set", "model"] + cols]
                   .round(3).to_string(index=False))
    return "\n".join(out)


# --------------------------------------------------------------------------- 6
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ecg", default="ecg_features_30s.csv")
    p.add_argument("--skna", default="skna_features_30s.csv")
    p.add_argument("--out", default="beat_pipeline/built")
    p.add_argument("--task", default="cpt", choices=["cpt", "sbp"])
    p.add_argument("--norm", default="expanding,none",
                   help="comma-separated normalization arms to compare")
    p.add_argument("--models", default=None,
                   help="default: logit,mlp (cpt) or ridge,mlp (sbp)")
    p.add_argument("--blocks", default="level,level+temporal",
                   help="level = the 30 s window only; level+temporal also "
                        "gets _d1/_rm/_rs/_slope, i.e. how it is moving")
    p.add_argument("--rank-k", default="",
                   help="e.g. 5,10,15,30 - in-fold MI top-k over all features")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--acausal", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rebuild", action="store_true",
                   help="ignore the engineered-table cache")
    a = p.parse_args()

    models = (a.models.split(",") if a.models
              else (["logit", "mlp"] if a.task == "cpt" else ["ridge", "mlp"]))
    blocks = a.blocks.split(",")
    rank_k = [int(x) for x in a.rank_k.split(",") if x.strip()]

    rows = []
    for method in a.norm.split(","):
        print(f"\n--- normalization: {method} ---")
        X, names, y_cpt, y_sbp, groups, _ = build(
            method, a.ecg, a.skna, a.out, a.k, not a.acausal, a.rebuild)
        y = y_cpt.astype(int) if a.task == "cpt" else y_sbp.astype(float)
        X = np.nan_to_num(np.asarray(X, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        rows += run_arm(method, X, names, y, groups, a.task, models, blocks,
                        rank_k, a.seed)

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.splitext(os.path.basename(a.ecg))[0].replace(
        "ecg_features_", "")
    csv_path = os.path.join(a.out, f"ablation_{a.task}_{tag}.csv")
    txt_path = os.path.join(a.out, f"ablation_{a.task}_{tag}.txt")
    df.to_csv(csv_path, index=False)
    grid = to_grid(df, a.task)
    with open(txt_path, "w") as f:
        f.write(f"feature-set ablation  task={a.task} norm={a.norm} "
                f"models={','.join(models)} k={a.k} "
                f"causal={not a.acausal}\n{grid}\n")
    print(grid)
    print(f"\nwrote {csv_path}\n      {txt_path}")


if __name__ == "__main__":
    main()
