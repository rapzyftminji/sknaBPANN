#!/usr/bin/env python3
"""Why personalization gains what it gains — SBP *and* DBP, one LOSO pass.

    python3 beat_pipeline/ann_ft_diagnostics.py --alpha 0.01

Replaces two earlier single-channel scratch scripts. The expensive part is
fitting the population ANN once per fold, so both analyses and both channels
share it.

Analysis 1 — CEILING, in-sample (written for completeness, NOT quotable):
    test_mean   predict the test slice's own mean. No features; cheats on level.
    amean       predict the ADAPT slice's mean (what the fine-tune re-anchors to)
    orc_head    best linear head on the frozen hidden layer, fit ON the test slice
    orc_feat    ridge on the raw features, fit ON the test slice
  orc_* are fit and scored on the same rows (151 features on ~150 windows, of
  which only ~1 in 6 is independent at 30 s / 5 s stride), so they measure
  memorisation. They are reported only to be contrasted with analysis 2.

Analysis 2 — the honest version, entirely INSIDE one person's test slice:
    blocked CV over contiguous blocks with a GAP either side of every test
    block, so no training window shares samples with a test window. Removes
    cross-subject transfer AND the level drift between slices from the question:
    all that is left is "does this signal predict this person's BP right now?"

Also reports DRIFT: how far the person's mean BP moves between the adapt half
and the test half, against how much it varies inside the test half.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_ablation as ma
from ann_bp_loso import DEFAULTS, make_ann, parse_hidden, calibration_offset, CHANNELS
import ann_personalize as apz


def blocked_cv(Xs, y, alpha, n_block, gap):
    """Contiguous-block CV with a gap. Returns out-of-fold predictions and the
    matching train-block-mean predictions, scored only where a fold ran."""
    n = len(y)
    edges = np.linspace(0, n, n_block + 1).astype(int)
    oof, cst = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n_block):
        lo, hi = edges[i], edges[i + 1]
        if hi - lo < 3:
            continue
        te = np.zeros(n, bool); te[lo:hi] = True
        tr = np.ones(n, bool); tr[max(0, lo - gap):min(n, hi + gap)] = False
        if tr.sum() < 10:
            continue
        oof[te] = Ridge(alpha=alpha).fit(Xs[tr], y[tr]).predict(Xs[te])
        cst[te] = y[tr].mean()
    ok = ~np.isnan(oof)
    return oof, cst, ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ecg", default="ecg_features_30s.csv")
    p.add_argument("--skna", default="skna_features_30s.csv")
    p.add_argument("--out", default="beat_pipeline/built")
    p.add_argument("--norm", default="expanding")
    p.add_argument("--hidden", default="10")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--activation", default="logistic")
    p.add_argument("--max-iter", type=int, default=3000)
    p.add_argument("--calib-min", type=float, default=2.0)
    p.add_argument("--adapt-end", type=float, default=0.5)
    p.add_argument("--val-end", type=float, default=0.6)
    p.add_argument("--keep-s13", default="full", choices=["full", "cut"])
    p.add_argument("--drop-recording", default="")
    p.add_argument("--n-block", type=int, default=5)
    p.add_argument("--gap", type=int, default=6,
                   help="windows excluded either side of a CV block; 6 fully "
                        "de-overlaps 30 s windows at 5 s stride")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=2)
    a = p.parse_args()

    X, names, _y, y_sbp, groups, meta = ma.build(
        a.norm, a.ecg, a.skna, a.out, a.k, True, keep_s13=a.keep_s13)
    X = np.nan_to_num(np.asarray(X, float), nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.column_stack([np.asarray(y_sbp, float),
                         np.asarray(meta["y_dbp"], float)])
    t_center = np.asarray(meta["t_center_sec"], float)
    rec = np.asarray(meta["recording"]).astype(str)
    groups = np.asarray(groups)

    if a.drop_recording:
        keep = np.ones(len(rec), bool)
        for t in [s.strip() for s in a.drop_recording.split(",") if s.strip()]:
            hit = np.array([t == r or t in r for r in rec])
            if not hit.any():
                raise SystemExit(f"--drop-recording {t!r} matched nothing")
            keep &= ~hit
        X, Y, groups, t_center, rec = (X[keep], Y[keep], groups[keep],
                                       t_center[keep], rec[keep])

    hp = dict(DEFAULTS, hidden=parse_hidden(a.hidden)[0], alpha=a.alpha,
              activation=a.activation, max_iter=a.max_iter)
    print(f"{X.shape[1]} features | {len(Y)} windows, "
          f"{len(np.unique(groups))} people | alpha={a.alpha} | "
          f"blocked CV: {a.n_block} blocks, gap {a.gap} windows", flush=True)

    ceil_rows, cv_rows = [], []
    t0 = time.time()
    for tr, te in LeaveOneGroupOut().split(X, Y[:, 0], groups):
        person = groups[te][0]
        adapt, val, test = apz.three_way_time_split(
            t_center[te], rec[te], a.adapt_end, a.val_end)
        if min(adapt.sum(), val.sum(), test.sum()) < 3:
            print(f"  {person:8s} SKIP (split too small)", flush=True)
            continue

        off = calibration_offset(Y, t_center, groups, person, a.calib_min)
        off_tr = np.zeros((len(tr), Y.shape[1]))
        for q in np.unique(groups[tr]):
            off_tr[groups[tr] == q] = calibration_offset(
                Y, t_center, groups, q, a.calib_min)

        sc = StandardScaler().fit(X[tr])
        net = make_ann(hp, a.seed).fit(sc.transform(X[tr]), Y[tr] - off_tr)
        Xte_s = sc.transform(X[te])
        Yd = Y[te] - off
        H = apz.hidden_activations(net, Xte_s)
        W0, b0 = net.coefs_[-1], net.intercepts_[-1]
        W, b, _, _ = apz.finetune_head(H[adapt], Yd[adapt], H[val], Yd[val],
                                       W0, b0, 1e-2, 6000, 300, hp["alpha"])

        Ht, Yt, Xt_s = H[test], Yd[test], Xte_s[test]
        oh = Ridge(alpha=1e-6).fit(Ht, Yt)
        of = Ridge(alpha=1.0).fit(Xt_s, Yt)
        post = Ht @ W + b

        order = np.argsort(t_center[te][test])          # chronological
        for j, ch in enumerate(CHANNELS):
            mae = lambda pr: float(np.abs(pr - Yt[:, j]).mean())
            ceil_rows.append(dict(
                person=person, channel=ch, n_adapt=int(adapt.sum()),
                n_test=int(test.sum()),
                test_mean=mae(Yt[:, j].mean()),
                amean=mae(Yd[adapt].mean(axis=0)[j]),
                post=mae(post[:, j]),
                orc_head=mae(oh.predict(Ht)[:, j]),
                orc_feat=mae(of.predict(Xt_s)[:, j]),
                drift=float(Yd[test].mean(axis=0)[j]
                            - Yd[adapt].mean(axis=0)[j]),
                sd_test=float(Yt[:, j].std())))

            yt = Yt[order][:, j]
            ofe, cfe, ok1 = blocked_cv(Xt_s[order], yt, 10.0, a.n_block, a.gap)
            ohe, _, ok2 = blocked_cv(Ht[order], yt, 1.0, a.n_block, a.gap)
            m = lambda pr, msk: float(np.abs(pr[msk] - yt[msk]).mean())
            cv_rows.append(dict(
                person=person, channel=ch, n_test=int(test.sum()),
                cv_mean=m(cfe, ok1), cv_feat=m(ofe, ok1), cv_head=m(ohe, ok2),
                r_feat=float(np.corrcoef(ofe[ok1], yt[ok1])[0, 1]),
                r_head=float(np.corrcoef(ohe[ok2], yt[ok2])[0, 1]),
                sd_test=float(yt.std())))
        print(f"  {person:8s} done", flush=True)

    ceil = pd.DataFrame(ceil_rows)
    cv = pd.DataFrame(cv_rows)
    os.makedirs(a.out, exist_ok=True)
    ceil.to_csv(os.path.join(a.out, "ann_ft_ceiling_insample.csv"), index=False)
    cv.to_csv(os.path.join(a.out, "ann_ft_within_slice_cv.csv"), index=False)

    pd.set_option("display.width", 250)
    for ch in CHANNELS:
        c, v = ceil[ceil.channel == ch], cv[cv.channel == ch]
        print(f"\n{'=' * 78}\n{ch}\n{'=' * 78}")
        print(v.drop(columns="channel").round(2).to_string(index=False))
        print(f"\n  in-sample (memorising, not quotable): "
              f"orc_feat {c.orc_feat.mean():.2f}  orc_head {c.orc_head.mean():.2f}")
        print(f"  gap-protected blocked CV            : "
              f"features {v.cv_feat.mean():.2f}  representation "
              f"{v.cv_head.mean():.2f}  constant {v.cv_mean.mean():.2f}")
        print(f"  beats the constant                  : "
              f"features {int((v.cv_feat < v.cv_mean).sum())}/{len(v)}  "
              f"representation {int((v.cv_head < v.cv_mean).sum())}/{len(v)}")
        print(f"  median r                            : "
              f"features {v.r_feat.median():+.3f}  representation "
              f"{v.r_head.median():+.3f}")
        print(f"  drift adapt->test                   : "
              f"{c.drift.abs().mean():.2f} mmHg vs within-slice SD "
              f"{c.sd_test.mean():.2f} mmHg")
    print(f"\n[{time.time() - t0:.0f}s]  wrote {a.out}/"
          "ann_ft_{ceiling_insample,within_slice_cv}.csv")


if __name__ == "__main__":
    main()
