#!/usr/bin/env python3
"""
BP regression with the Yao et al. ANN, leave-one-subject-out.
=============================================================

The project goal: estimate SBP and DBP from the engineered ECG+SKNA window
features. The earlier attempt used a CNN-BiLSTM on raw waveforms (src/model.py);
this is the same target reached with a different architecture - the two-layer
feedforward network from Yao et al., IEEE JBHI 26(8) 2022:

    inputs -> 10 sigmoid hidden units -> 2 LINEAR output neurons (SBP, DBP)

trained by scaled conjugate gradient (lbfgs is the sklearn stand-in). One
network predicts both channels jointly, as in the paper.

Features come from feature_engineering.py - the same merge / log / interaction /
within-recording normalization / temporal / prune pipeline that tab 5 shows, so
whatever you settle on there is what the network is fed here.

Calibration modes
-----------------
  none    Calibration-free: predict absolute mmHg for a person the model has
          never seen. This is what the paper CLAIMS to do. Its baseline is the
          constant train-mean predictor.
  offset  Per-subject calibration: the held-out person's first `--calib-min`
          minutes of reference BP set an offset, the network predicts the
          DEVIATION from it, and the offset is added back. This is what the
          existing pipeline does, and its baseline is ZERO-DELTA - predict the
          calibration value and never move.

Report the matching baseline in both cases. Yao et al.'s headline
(-0.07 +- 4.47 mmHg SBP) is reproduced on this dataset by an oracle that
predicts each person's own mean BP using no features at all, because their
split puts the first 70% of every subject's beats in training and their
demographic inputs act as a subject ID. A BP number without its baseline beside
it says nothing.

Grades come from src/bp_standards.py (BHS / AAMI SP10 / ISO 81060-2 / IEEE
1708), which already knows to print the zero-delta bar next to them.

    python3 beat_pipeline/ann_bp_loso.py
    python3 beat_pipeline/ann_bp_loso.py --calibration offset --hidden 10,20
    python3 beat_pipeline/ann_bp_loso.py --family SKNA,HRV,ECG_MORPH --compare

Outputs
-------
  <out>/ann_bp_predictions.csv   per-window SBP/DBP true + predicted
  <out>/ann_bp_folds.csv         per-subject metrics
  <out>/ann_bp_summary.txt       metrics, baselines and standards grading
  <out>/ann_bp_model.joblib      scaler + net, refit on all people
"""
import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import model_ablation as ma

try:
    import bp_standards as bps
    HAVE_STANDARDS = True
except ImportError:                      # standards reporting is optional
    HAVE_STANDARDS = False

CHANNELS = ("SBP", "DBP")


# Everything the network exposes. The paper pins activation=logistic,
# solver=scaled-conjugate-gradient (lbfgs here) and one hidden layer of 10;
# the rest are sklearn defaults made visible so they can be tuned rather than
# silently inherited.
DEFAULTS = dict(hidden=(10,), activation="logistic", solver="lbfgs",
                alpha=1e-2, max_iter=1000, learning_rate_init=1e-3,
                early_stopping=False, validation_fraction=0.1, tol=1e-4)


def parse_hidden(spec):
    """'10' -> [(10,)];  '10x2' -> [(10, 10)];  '5,10x2,20' -> three
    architectures to sweep. Comma separates ARCHITECTURES, x sets depth."""
    out = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "x" in chunk.lower():
            w, d = chunk.lower().split("x")
            out.append(tuple([int(w)] * int(d)))
        else:
            out.append((int(chunk),))
    if not out:
        raise ValueError("no architecture parsed from " + repr(spec))
    return out


def make_ann(hp, seed):
    """Feedforward net, linear output - the paper's when hp is left at
    DEFAULTS. MLPRegressor's output activation is identity, and fitting a
    2-column y gives the paper's two output neurons in one network.

    early_stopping is only honoured by sgd/adam; with lbfgs sklearn ignores it,
    so it is forced off there rather than silently pretending to hold out data.
    """
    p = dict(DEFAULTS); p.update(hp or {})
    es = bool(p["early_stopping"]) and p["solver"] in ("sgd", "adam")
    return MLPRegressor(
        hidden_layer_sizes=tuple(p["hidden"]), activation=p["activation"],
        solver=p["solver"], alpha=float(p["alpha"]),
        max_iter=int(p["max_iter"]),
        learning_rate_init=float(p["learning_rate_init"]),
        early_stopping=es, validation_fraction=float(p["validation_fraction"]),
        tol=float(p["tol"]), random_state=seed)


def arch_name(hidden):
    h = tuple(hidden)
    return f"ANN({h[0]}x{len(h)})" if len(h) > 1 else f"ANN({h[0]})"


def make_ref(seed, alpha=10.0):
    return Ridge(alpha=alpha)


def calibration_offset(y, t_center, groups, person, calib_min):
    """Mean reference BP over the held-out person's first `calib_min` minutes.

    This is the only place a test subject's labels are touched, and it mirrors
    what a real deployment does - cuff a new user once, then track. It is NOT
    calibration-free, and the summary says so.
    """
    m = groups == person
    t0 = t_center[m].min()
    early = m & (t_center <= t0 + calib_min * 60.0)
    if early.sum() < 3:
        early = m
    return y[early].mean(axis=0)


def inner_select(X, Y, groups, t_center, specs, seed, mode, calib_min,
                 progress=None):
    """Pick an architecture using ONLY the training people of one outer fold.

    The outer LOSO fold gives train/test. This splits that TRAIN block again by
    person - an inner leave-one-person-out - and scores each candidate there,
    so the held-out test subject never influences which model is chosen.

    Without this, sweeping several architectures and reporting the best outer
    score IS selection on the test set: with 14 folds and a handful of
    candidates the winner is partly picked by luck on the very subjects it is
    then reported against. Use --inner-select whenever --hidden names more than
    one architecture and you intend to quote the winner.
    """
    best, best_mae = None, np.inf
    for hp in specs:
        maes = []
        for tr, va in LeaveOneGroupOut().split(X, Y[:, 0], groups):
            sc = StandardScaler().fit(X[tr])
            off_tr, off_va = _offsets(Y, t_center, groups, tr, va,
                                      groups[va][0], mode, calib_min)
            m = make_ann(hp, seed).fit(sc.transform(X[tr]), Y[tr] - off_tr)
            pred = np.asarray(m.predict(sc.transform(X[va])))
            if pred.ndim == 1:
                pred = pred[:, None]
            maes.append(np.mean(np.abs(pred + off_va - Y[va])))
        mae = float(np.mean(maes))
        if progress:
            progress(f"      inner {arch_name(hp['hidden'])}: MAE {mae:.2f}")
        if mae < best_mae:
            best, best_mae = hp, mae
    return best, best_mae


def _offsets(Y, t_center, groups, tr, te, person, mode, calib_min):
    """Per-person calibration offsets for one split, zeros when mode='none'."""
    off_tr = np.zeros((len(tr), Y.shape[1]))
    off_te = np.zeros((len(te), Y.shape[1]))
    if mode == "offset":
        for p in np.unique(groups[tr]):
            off_tr[groups[tr] == p] = calibration_offset(
                Y, t_center, groups, p, calib_min)
        off_te[:] = calibration_offset(Y, t_center, groups, person, calib_min)
    return off_tr, off_te


def run_loso(X, Y, groups, t_center, build_model, mode, calib_min,
             progress=None, specs=None, seed=0):
    """Out-of-fold SBP/DBP predictions in absolute mmHg, plus per-fold rows.

    THE SPLIT, stated plainly: LeaveOneGroupOut on `person` gives 14 outer
    folds - train on 13 people, test on the 1 held out, never a window of that
    person in training. There is NO separate validation set unless `specs` is
    given, in which case each outer fold runs its own inner leave-one-person-out
    over its 13 training people to choose the architecture (see inner_select).
    The scaler and any calibration offset are likewise fit inside the fold.
    """
    oof = np.full(Y.shape, np.nan)
    base = np.full(Y.shape, np.nan)      # the matching baseline prediction
    rows = []
    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)
    for i, (tr, te) in enumerate(logo.split(X, Y[:, 0], groups), 1):
        person = groups[te][0]
        if progress:
            progress(f"  fold {i}/{n_folds}  hold out {person} "
                     f"({len(te)} windows, train {len(tr)})")
        chosen = None
        if specs:
            chosen, imae = inner_select(X[tr], Y[tr], groups[tr], t_center[tr],
                                        specs, seed, mode, calib_min, progress)
            if progress:
                progress(f"    -> inner pick {arch_name(chosen['hidden'])} "
                         f"(inner MAE {imae:.2f})")
        sc = StandardScaler().fit(X[tr])
        Ytr = Y[tr]
        # every TRAINING person is centred on their own calibration too, so the
        # network learns deviation->deviation, not absolute->absolute
        off_tr, off_te = _offsets(Y, t_center, groups, tr, te, person, mode,
                                  calib_min)
        base[te] = (off_te[0] if mode == "offset"   # zero-delta: never move
                    else Ytr.mean(axis=0))          # constant train mean

        m = ((lambda: make_ann(chosen, seed)) if chosen else build_model)()
        m = m.fit(sc.transform(X[tr]), Ytr - off_tr)
        pred = np.asarray(m.predict(sc.transform(X[te])))
        if pred.ndim == 1:
            pred = pred[:, None]
        oof[te] = pred + off_te

        err = oof[te] - Y[te]
        berr = base[te] - Y[te]
        row = dict(person=person, n=len(te))
        if chosen is not None:
            row["picked"] = arch_name(chosen["hidden"])
        for j, ch in enumerate(CHANNELS):
            row[f"mae_{ch}"] = float(np.mean(np.abs(err[:, j])))
            row[f"me_{ch}"] = float(np.mean(err[:, j]))
            row[f"sde_{ch}"] = float(np.std(err[:, j], ddof=1))
            row[f"base_mae_{ch}"] = float(np.mean(np.abs(berr[:, j])))
        # Ridge's default solver is CLOSED FORM: it has no n_iter_ and no
        # max_iter, so the old `int(None or 0)` printed "iters 0" for every
        # ridge fold - which reads as "the optimizer did nothing" when in fact
        # there is no iteration to count. Distinguish the two cases.
        raw_iter = getattr(m, "n_iter_", None)
        if raw_iter is not None:                 # sag/saga return an array
            raw_iter = int(np.max(np.atleast_1d(raw_iter)))
        cap = getattr(m, "max_iter", None)
        iterative = raw_iter is not None and cap is not None
        row["n_iter"] = raw_iter if iterative else np.nan
        row["converged"] = bool(raw_iter < cap) if iterative else True
        rows.append(row)
        if progress:
            it_txt = (f"iters {raw_iter}" if iterative
                      else "iters n/a (closed-form solver)")
            progress(f"    {person}: SBP MAE {row['mae_SBP']:.2f} "
                     f"(baseline {row['base_mae_SBP']:.2f}) | "
                     f"DBP {row['mae_DBP']:.2f} | {it_txt}"
                     + ("" if row["converged"] else "  NOT CONVERGED"))
    return oof, base, pd.DataFrame(rows)


def score(Y, pred, name):
    err = pred - Y
    out = dict(model=name)
    for j, ch in enumerate(CHANNELS):
        out[f"MAE_{ch}"] = float(np.mean(np.abs(err[:, j])))
        out[f"ME_{ch}"] = float(np.mean(err[:, j]))
        out[f"SDE_{ch}"] = float(np.std(err[:, j], ddof=1))
    return out


def oracle_subject_mean(Y, groups):
    """Predict each person's OWN mean BP. Uses no features and is not
    achievable without their cuff readings - the number Yao et al.'s
    demographic inputs quietly reproduce. Calibration for the eye, not a
    model."""
    out = np.empty_like(Y)
    for p in np.unique(groups):
        m = groups == p
        out[m] = Y[m].mean(axis=0)
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ecg", default="ecg_features_30s.csv")
    p.add_argument("--skna", default="skna_features_30s.csv")
    p.add_argument("--out", default="beat_pipeline/built")
    p.add_argument("--norm", default="expanding")
    p.add_argument("--family", default="SKNA,XSIG,HRV,ECG_MORPH,NONLIN")
    p.add_argument("--hidden", default="10",
                   help="architectures: '10', '10x2' (two layers of 10), "
                        "'5,10,20' to sweep")
    p.add_argument("--alpha", type=float, default=1e-2, help="L2 penalty")
    p.add_argument("--activation", default="logistic",
                   choices=["logistic", "tanh", "relu", "identity"])
    p.add_argument("--solver", default="lbfgs", choices=["lbfgs", "adam",
                                                         "sgd"])
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="learning_rate_init (adam/sgd only)")
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--early-stopping", action="store_true",
                   help="adam/sgd only; sklearn holds out a RANDOM slice of "
                        "training rows, which with overlapping windows leaks "
                        "across the split - prefer --inner-select")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--inner-select", action="store_true",
                   help="nested CV: choose the architecture by an inner "
                        "leave-one-person-out inside each outer fold, so the "
                        "test subject never influences the choice. Use this "
                        "whenever --hidden names more than one architecture")
    p.add_argument("--calibration", default="none", choices=["none", "offset"])
    p.add_argument("--calib-min", type=float, default=2.0,
                   help="minutes of the held-out person's own BP used as the "
                        "calibration offset (calibration=offset only)")
    p.add_argument("--level-only", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--compare", action="store_true",
                   help="also run ridge regression on the identical folds")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--drop-recording", default="",
                   help="comma-separated recordings to exclude, matched "
                        "exactly or as a substring of the Recording filename "
                        "(e.g. 'cindy,SKNA_BP_Tseng2.txt'). Every drop is "
                        "printed and written to the summary - a subject "
                        "removed silently is a subject removed dishonestly")
    p.add_argument("--keep-s13", default="full", choices=["full", "cut"],
                   help="Tseng2 is recorded twice (s13_full = 31 min, "
                        "s13 = its clean 10-min cut). Keeping both would enter "
                        "one person twice, so one is dropped; this picks which")
    a = p.parse_args()

    X, names, _ycpt, y_sbp, groups, meta = ma.build(
        a.norm, a.ecg, a.skna, a.out, a.k, True, keep_s13=a.keep_s13)
    X = np.nan_to_num(np.asarray(X, float), nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.column_stack([np.asarray(y_sbp, float),
                         np.asarray(meta["y_dbp"], float)])
    t_center = np.asarray(meta["t_center_sec"], float)
    rec = np.asarray(meta["recording"]).astype(str)

    drop_note = []
    if a.drop_recording:
        toks = [t.strip() for t in a.drop_recording.split(",") if t.strip()]
        keep = np.ones(len(rec), bool)
        for t in toks:
            hit = np.array([t == r or t in r for r in rec])
            if not hit.any():
                raise SystemExit(f"--drop-recording {t!r} matched no "
                                 f"recording; have {sorted(set(rec))}")
            for r in sorted(set(rec[hit])):
                n = int((rec == r).sum())
                drop_note.append(f"{r} ({n} windows)")
            keep &= ~hit
        X, Y, groups, t_center, rec = (X[keep], Y[keep], groups[keep],
                                       t_center[keep], rec[keep])
        for d in drop_note:
            print(f"DROPPED {d}")

    fams = set(a.family.split(","))
    cols = [i for i, c in enumerate(names)
            if ma.FAMILY[ma.split_column(c)[0]] in fams
            and (not a.level_only or ma.is_level(c))]
    Xf, feat = X[:, cols], [names[i] for i in cols]
    print(f"{Xf.shape[1]} features from {sorted(fams)} | {len(Y)} windows, "
          f"{len(np.unique(groups))} people | calibration={a.calibration}"
          + (f" ({a.calib_min:.0f} min)" if a.calibration == "offset" else ""))

    os.makedirs(a.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    logpath = os.path.join(a.out, f"ann_bp_log_{stamp}.txt")
    logfh = open(logpath, "w")

    def log(msg):
        print(msg, flush=True)
        logfh.write(msg + "\n")
        logfh.flush()          # so `tail -f` shows progress during the run

    hp_base = dict(activation=a.activation, solver=a.solver, alpha=a.alpha,
                   max_iter=a.max_iter, learning_rate_init=a.lr,
                   early_stopping=a.early_stopping,
                   validation_fraction=a.val_fraction, tol=a.tol)
    specs = [dict(hp_base, hidden=h) for h in parse_hidden(a.hidden)]
    log(f"[{stamp}] {Xf.shape[1]} features | {len(Y)} windows, "
        f"{len(np.unique(groups))} people | calibration={a.calibration} | "
        f"solver={a.solver} activation={a.activation} alpha={a.alpha} "
        f"max_iter={a.max_iter} inner_select={a.inner_select} "
        f"keep_s13={a.keep_s13}")
    for d in drop_note:
        log(f"  DROPPED {d}")

    runs, folds, preds, best = [], {}, {}, None
    todo = [None] if a.inner_select else specs
    for hp in todo:
        name = ("ANN(nested)" if hp is None else arch_name(hp["hidden"]))
        log(f"{name}:")
        t0 = time.time()
        oof, base, f = run_loso(
            Xf, Y, groups, t_center,
            (None if hp is None else (lambda hp=hp: make_ann(hp, a.seed))),
            a.calibration, a.calib_min, progress=log,
            specs=(specs if hp is None else None), seed=a.seed)
        s = score(Y, oof, name); s["seconds"] = time.time() - t0
        runs.append(s); folds[name] = f; preds[name] = oof
        if best is None or s["MAE_SBP"] < best[1]["MAE_SBP"]:
            best = (hp, s, oof)
        log(f"  {name} SBP MAE={s['MAE_SBP']:6.2f} ME={s['ME_SBP']:+6.2f} "
            f"SDE={s['SDE_SBP']:5.2f} | DBP MAE={s['MAE_DBP']:6.2f} "
            f"[{s['seconds']:.0f}s]")

    if a.compare:
        t0 = time.time()
        log("ridge:")
        oof, base, f = run_loso(Xf, Y, groups, t_center,
                                lambda: make_ref(a.seed), a.calibration,
                                a.calib_min, progress=log)
        s = score(Y, oof, "ridge"); s["seconds"] = time.time() - t0
        runs.append(s); folds[s["model"]] = f; preds[s["model"]] = oof
        log(f"  ridge     SBP MAE={s['MAE_SBP']:6.2f} ME={s['ME_SBP']:+6.2f} "
            f"SDE={s['SDE_SBP']:5.2f} | DBP MAE={s['MAE_DBP']:6.2f} "
            f"[{s['seconds']:.0f}s]")

    base_name = ("zero-delta (calibration, never moves)" if a.calibration ==
                 "offset" else "constant train mean")
    runs.append(score(Y, base, f"BASELINE {base_name}"))
    runs.append(score(Y, oracle_subject_mean(Y, groups),
                      "ORACLE subject mean (cheating)"))

    res = pd.DataFrame(runs)
    h_best, s_best, oof_best = best
    beat = s_best["MAE_SBP"] < runs[-2]["MAE_SBP"]

    lines = [
        f"BP regression, LOSO on person   norm={a.norm}  family={a.family}  "
        f"calibration={a.calibration}  alpha={a.alpha}  seed={a.seed}",
        f"{Xf.shape[1]} features, {len(Y)} windows, "
        f"{len(np.unique(groups))} people",
        ("DROPPED: " + "; ".join(drop_note)) if drop_note else "",
        "",
        res.round(2).to_string(index=False),
        "",
        f"VERDICT: best ANN SBP MAE {s_best['MAE_SBP']:.2f} vs baseline "
        f"{runs[-2]['MAE_SBP']:.2f} -> "
        + ("BEATS the baseline." if beat else
           "DOES NOT beat the baseline. The model is not adding information "
           "about BP over predicting the baseline value; do not report the MAE "
           "on its own."),
        "",
        f"per-subject, {s_best['model']}:",
        folds[s_best["model"]].round(2).to_string(index=False),
    ]

    # The baseline arms ride along in the predictions file so the device
    # standards (BHS/AAMI/IEEE) can be computed for them on EXACTLY the same
    # readings. A grade for the model alone cannot be interpreted: if the
    # no-feature arm grades the same, the grade belongs to the cohort.
    orc = oracle_subject_mean(Y, groups)
    pred_df = pd.DataFrame({
        "Subject_ID": groups, "SBP_true": Y[:, 0], "DBP_true": Y[:, 1],
        "SBP_pred": oof_best[:, 0], "DBP_pred": oof_best[:, 1],
        "SBP_zero": base[:, 0], "DBP_zero": base[:, 1],
        "SBP_oracle": orc[:, 0], "DBP_oracle": orc[:, 1]})
    if HAVE_STANDARDS:
        std = bps.compute_bp_standards_from_df(pred_df)
        zd = {ch: float(np.mean(np.abs(base[:, j] - Y[:, j])))
              for j, ch in enumerate(CHANNELS)}
        lines += ["", bps.format_standards_report(std, zero_delta_mae=zd)]
    else:
        lines += ["", "(src/bp_standards.py not importable - grading skipped)"]

    pred_df.to_csv(os.path.join(a.out, f"ann_bp_predictions_{stamp}.csv"),
                   index=False)
    pd.concat([f.assign(model=m) for m, f in folds.items()]).to_csv(
        os.path.join(a.out, f"ann_bp_folds_{stamp}.csv"), index=False)
    txt = os.path.join(a.out, f"ann_bp_summary_{stamp}.txt")
    with open(txt, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log("\n" + "\n".join(lines[2:]))

    # one row per run, appended forever - the history across sessions
    hist = os.path.join(a.out, "ann_bp_history.csv")
    hrows = pd.DataFrame(runs).assign(
        stamp=stamp, norm=a.norm, family=a.family, calibration=a.calibration,
        features=Xf.shape[1], solver=a.solver, activation=a.activation,
        alpha=a.alpha, max_iter=a.max_iter, hidden=a.hidden,
        inner_select=a.inner_select, seed=a.seed, ecg=os.path.basename(a.ecg),
        dropped=("; ".join(drop_note) if drop_note else ""),
        keep_s13=a.keep_s13)
    hrows.to_csv(hist, mode="a", header=not os.path.exists(hist), index=False)

    if not a.no_save:
        sc = StandardScaler().fit(Xf)
        off = np.zeros_like(Y)
        if a.calibration == "offset":
            for p_ in np.unique(groups):
                off[groups == p_] = calibration_offset(Y, t_center, groups, p_,
                                                       a.calib_min)
        hp_final = h_best if h_best is not None else specs[0]
        net = make_ann(hp_final, a.seed).fit(sc.transform(Xf), Y - off)
        path = os.path.join(a.out, f"ann_bp_model_{stamp}.joblib")
        joblib.dump(dict(scaler=sc, model=net, feature_names=feat,
                         hyperparams=hp_final, family=a.family, norm=a.norm,
                         calibration=a.calibration, calib_min=a.calib_min,
                         channels=CHANNELS, mae_sbp=s_best["MAE_SBP"],
                         stamp=stamp), path)
        log(f"\nsaved {path}")
    log(f"wrote {txt}\n      {logpath}\n      {hist}")
    logfh.close()


if __name__ == "__main__":
    main()
