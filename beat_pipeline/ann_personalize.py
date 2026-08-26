#!/usr/bin/env python3
"""
Per-subject fine-tuning (personalization) for the Yao et al. ANN.
=================================================================

The ANN analogue of `src/personalize_finetune.py`, following that file's
protocol step for step so the two are comparable:

  1. Split the HELD-OUT person's own recording by TIME into three disjoint
     buckets (per recording, so a two-session person never leaks across
     sessions):
        adapt : [0.0, adapt_end)  -- the calibration slice a device would see
        val   : [adapt_end, val_end) -- early-stopping signal for the fine-tune
        test  : [val_end, 1.0]    -- held-out evaluation, never seen while adapting
  2. Base model = the LOSO ANN trained on the OTHER people only. Evaluate it on
     the test slice -> PRE-FT.
  3. Freeze the input->hidden layer and fine-tune ONLY the hidden->output head,
     early-stopping on the val slice. This is the ANN's version of McBP-Net
     "hybrid calibration" (freeze the CNN + earlier LSTM layers, update the last
     LSTM + attention pool + head): the frozen layer keeps the population's
     learned feature map, and the fine-tune can only re-map those features onto
     this one subject's BP scale, not re-learn features from a few minutes of
     calibration data. With 151 inputs the frozen layer holds ~1520 weights and
     the head 22 - fine-tuning the whole net on ~30 adapt windows would be
     fitting 1500 parameters to 30 points.
  4. Evaluate the fine-tuned model on the SAME test slice -> POST-FT.
  5. Compare BOTH against the ZERO-DELTA baseline on that slice (predict the
     calibration reading and never move). That is the bar, not PRE-FT: a
     fine-tune that beats the un-personalized model while still losing to
     "don't move" has not earned anything.

NOT COMPARABLE to ann_bp_loso.py's headline MAE: that scores every window of
the held-out person, this scores only the last (1 - val_end) of the recording.
Different denominators. The zero-delta column is recomputed on this slice, so
compare within this table only.

    python3 beat_pipeline/ann_personalize.py --keep-s13 cut \
        --drop-recording cindy --alpha 10 --max-iter 3000
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_ablation as ma
from ann_bp_loso import (DEFAULTS, make_ann, parse_hidden, calibration_offset,
                         CHANNELS)

ACT = {"logistic": lambda z: 1.0 / (1.0 + np.exp(-z)),
       "tanh": np.tanh,
       "relu": lambda z: np.maximum(z, 0.0),
       "identity": lambda z: z}


# --------------------------------------------------------------------------- 1
def three_way_time_split(t_center, recording, adapt_end=0.5, val_end=0.6):
    """Temporally disjoint adapt / val / test masks over ONE person's windows.

    Fractional position is computed per RECORDING (mirroring the CNN path's
    per-(Subject_ID, Session_ID) fractions), so a person with two sessions is
    split inside each session rather than having session 1 become "adapt" and
    session 2 become "test".

    Fallback for short recordings, same as skna_dataset.three_way_time_split:
    if the middle band lands empty, carve val from the tail of adapt so the
    fine-tune still has an early-stopping signal that precedes test.
    """
    frac = np.empty(len(t_center), float)
    for r in np.unique(recording):
        m = recording == r
        order = np.argsort(t_center[m])
        pos = np.empty(m.sum(), float)
        pos[order] = np.arange(m.sum()) / max(m.sum(), 1)
        frac[m] = pos
    adapt = frac < adapt_end
    val = (frac >= adapt_end) & (frac < val_end)
    test = frac >= val_end
    if val.sum() == 0 and adapt.sum() >= 4 and test.sum() > 0:
        cut = np.quantile(frac[adapt], 0.85)
        val = adapt & (frac >= cut)
        adapt = adapt & (frac < cut)
    return adapt, val, test


# --------------------------------------------------------------------------- 2
def hidden_activations(net, Xs):
    """Forward pass up to the last hidden layer - the FROZEN feature map."""
    h = Xs
    act = ACT[net.activation]
    for W, b in zip(net.coefs_[:-1], net.intercepts_[:-1]):
        h = act(h @ W + b)
    return h


def finetune_head(H_ad, Y_ad, H_va, Y_va, W0, b0, lr, epochs, patience,
                  l2, min_delta=1e-3, bias_only=False):
    """Adam on the output head only, early-stopped on the val slice.

    Mirrors the CNN path's loop (train_one_epoch -> evaluate -> EarlyStopper
    on MAE_SBP + MAE_DBP, restore best weights). Full-batch: the adapt slice is
    tens of windows, so a mini-batch loop would add noise and nothing else.

    bias_only=True freezes W at the population value and adapts ONLY the
    intercept. That is the control arm: it can re-anchor this subject's BP
    level but cannot re-weight a single feature, so whatever the full head
    earns OVER it is the part attributable to the features rather than to
    recalibration.

    Returns (W, b, history_df, best_epoch).
    """
    W, b = W0.copy(), b0.copy()
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = len(H_ad)

    def mae(H, Y, W, b):
        e = np.abs(H @ W + b - Y)
        return e.mean(axis=0)

    best = (mae(H_va, Y_va, W, b).sum(), W.copy(), b.copy(), -1)
    hist = [dict(epoch=-1,                       # -1 = the base model, pre-FT
                 train_MAE_SBP=mae(H_ad, Y_ad, W, b)[0],
                 train_MAE_DBP=mae(H_ad, Y_ad, W, b)[1],
                 val_MAE_SBP=mae(H_va, Y_va, W, b)[0],
                 val_MAE_DBP=mae(H_va, Y_va, W, b)[1])]
    bad = 0
    for ep in range(epochs):
        resid = (H_ad @ W + b) - Y_ad                     # (n, 2)
        gW = H_ad.T @ resid * (2.0 / n) + 2.0 * l2 * W    # MSE + L2, as sklearn
        gb = resid.mean(axis=0) * 2.0
        steps = ((b, gb, mb, vb),) if bias_only else ((W, gW, mW, vW),
                                                      (b, gb, mb, vb))
        for p, g, m, v in steps:
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            mh = m / (1 - b1 ** (ep + 1))
            vh = v / (1 - b2 ** (ep + 1))
            p -= lr * mh / (np.sqrt(vh) + eps)
        tr, va = mae(H_ad, Y_ad, W, b), mae(H_va, Y_va, W, b)
        hist.append(dict(epoch=ep, train_MAE_SBP=tr[0], train_MAE_DBP=tr[1],
                         val_MAE_SBP=va[0], val_MAE_DBP=va[1]))
        sel = va.sum()
        if sel < best[0] - min_delta:
            best, bad = (sel, W.copy(), b.copy(), ep), 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best[1], best[2], pd.DataFrame(hist), best[3]


# --------------------------------------------------------------------------- 3
def run(X, Y, groups, t_center, recording, hp, seed, calib_min, adapt_end,
        val_end, lr, epochs, patience, progress=print):
    rows, curves, preds = [], [], []
    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)
    for i, (tr, te) in enumerate(logo.split(X, Y[:, 0], groups), 1):
        person = groups[te][0]
        adapt, val, test = three_way_time_split(
            t_center[te], recording[te], adapt_end, val_end)
        if min(adapt.sum(), val.sum(), test.sum()) < 3:
            progress(f"  [{person}] SKIP: split too small "
                     f"(adapt={adapt.sum()}, val={val.sum()}, "
                     f"test={test.sum()})")
            continue

        # --- calibration offset from THIS person's own early windows only ----
        off = calibration_offset(Y, t_center, groups, person, calib_min)
        # every training person is centred on their own calibration too, so the
        # base net learns deviation -> deviation (same as ann_bp_loso offset)
        off_tr = np.zeros((len(tr), Y.shape[1]))
        for p in np.unique(groups[tr]):
            off_tr[groups[tr] == p] = calibration_offset(
                Y, t_center, groups, p, calib_min)

        sc = StandardScaler().fit(X[tr])
        base_net = make_ann(hp, seed).fit(sc.transform(X[tr]), Y[tr] - off_tr)

        Xte_s = sc.transform(X[te])
        Yte_d = Y[te] - off                       # deviations from calibration
        H = hidden_activations(base_net, Xte_s)
        W0, b0 = base_net.coefs_[-1], base_net.intercepts_[-1]

        # --- PRE-FT and the bar, both on the TEST slice ----------------------
        pre = np.abs(H[test] @ W0 + b0 - Yte_d[test]).mean(axis=0)
        zero = np.abs(Yte_d[test]).mean(axis=0)   # predict the calibration value

        # --- fine-tune the head only ----------------------------------------
        W, b, hist, best_ep = finetune_head(
            H[adapt], Yte_d[adapt], H[val], Yte_d[val], W0, b0,
            lr, epochs, patience, hp["alpha"])
        post = np.abs(H[test] @ W + b - Yte_d[test]).mean(axis=0)

        # ---- CONTROLS: how much of `post` is just re-anchoring? -------------
        # (a) no features at all - predict the adapt slice's own mean BP
        amean = np.abs(Yte_d[adapt].mean(axis=0) - Yte_d[test]).mean(axis=0)
        # (b) same fine-tune loop, but only the intercept may move
        _, b_only, _, _ = finetune_head(
            H[adapt], Yte_d[adapt], H[val], Yte_d[val], W0, b0,
            lr, epochs, patience, hp["alpha"], bias_only=True)
        bias = np.abs(H[test] @ W0 + b_only - Yte_d[test]).mean(axis=0)

        # ---- is the fine-tuned head still a FUNCTION of the features? -------
        # If it has collapsed to a constant, pred_sd ~ 0 and r is undefined /
        # noise: it has learned this subject's mean BP and nothing else.
        pred = H[test] @ W + b
        # per-window ABSOLUTE predictions on the test slice, for every arm, so
        # ME / SD and the BHS-AAMI-IEEE grades can be computed downstream. The
        # summary rows only carry MAEs, which is not enough for any of those.
        preds.append(pd.DataFrame({
            "person": person,
            "SBP_true": Y[te][test][:, 0], "DBP_true": Y[te][test][:, 1],
            "SBP_pred": pred[:, 0] + off[0], "DBP_pred": pred[:, 1] + off[1],
            "SBP_pre": (H[test] @ W0 + b0)[:, 0] + off[0],
            "DBP_pre": (H[test] @ W0 + b0)[:, 1] + off[1],
            "SBP_amean": Yte_d[adapt].mean(axis=0)[0] + off[0],
            "DBP_amean": Yte_d[adapt].mean(axis=0)[1] + off[1],
            "SBP_zero": off[0], "DBP_zero": off[1],
        }))
        curves.append(hist.assign(person=person))
        row = dict(person=person, n_adapt=int(adapt.sum()),
                   n_val=int(val.sum()), n_test=int(test.sum()),
                   best_epoch=best_ep)
        for j, ch in enumerate(CHANNELS):
            sd_p = float(np.std(pred[:, j]))
            sd_t = float(np.std(Yte_d[test][:, j]))
            row[f"predsd_{ch}"] = sd_p
            row[f"truesd_{ch}"] = sd_t
            row[f"r_{ch}"] = (float(np.corrcoef(pred[:, j],
                                                Yte_d[test][:, j])[0, 1])
                              if sd_p > 1e-9 and sd_t > 1e-9 else np.nan)
        for j, ch in enumerate(CHANNELS):
            row[f"pre_{ch}"] = float(pre[j])
            row[f"post_{ch}"] = float(post[j])
            row[f"zero_{ch}"] = float(zero[j])
            row[f"amean_{ch}"] = float(amean[j])
            row[f"bias_{ch}"] = float(bias[j])
        row["post_beats_pre"] = bool(post[0] < pre[0] and post[1] < pre[1])
        row["post_beats_zero"] = bool(post[0] < zero[0] and post[1] < zero[1])
        row["post_beats_amean"] = bool(post[0] < amean[0] and post[1] < amean[1])
        row["post_beats_bias"] = bool(post[0] < bias[0] and post[1] < bias[1])
        rows.append(row)
        progress(f"  fold {i}/{n_folds} {person:8s} "
                 f"adapt/val/test {adapt.sum():3d}/{val.sum():3d}/{test.sum():3d}"
                 f" | SBP pre {pre[0]:5.2f} post {post[0]:5.2f} "
                 f"zero {zero[0]:5.2f} | DBP pre {pre[1]:5.2f} "
                 f"post {post[1]:5.2f} zero {zero[1]:5.2f} "
                 f"| best_ep {best_ep}"
                 + ("  BEATS ZERO-DELTA" if row["post_beats_zero"] else ""))
    return (pd.DataFrame(rows), pd.concat(curves, ignore_index=True),
            pd.concat(preds, ignore_index=True))


def summarize(df):
    lines = ["", "Personalization (head-only fine-tune), per subject:",
             df.round(2).to_string(index=False), ""]
    for ch in CHANNELS:
        lines.append(
            f"{ch}: PRE {df[f'pre_{ch}'].mean():5.2f}  "
            f"POST {df[f'post_{ch}'].mean():5.2f}  |  bars: "
            f"ZERO-DELTA {df[f'zero_{ch}'].mean():5.2f}  "
            f"ADAPT-MEAN {df[f'amean_{ch}'].mean():5.2f}  "
            f"BIAS-ONLY {df[f'bias_{ch}'].mean():5.2f}   (mean over subjects)")
    lines += [
        "",
        f"post beats pre         : {int(df.post_beats_pre.sum())}/{len(df)}"
        "   (personalization helps at all)",
        f"post beats ZERO-DELTA  : {int(df.post_beats_zero.sum())}/{len(df)}"
        "   (better than never moving)",
        f"post beats ADAPT-MEAN  : {int(df.post_beats_amean.sum())}/{len(df)}"
        "   (better than a no-feature re-anchor)",
        f"post beats BIAS-ONLY   : {int(df.post_beats_bias.sum())}/{len(df)}"
        "   (weak control - see note)",
        "",
        "ADAPT-MEAN is the bar that matters: predict the adapt slice's own "
        "mean BP,",
        "using no features whatsoever. It is exactly the recalibration the "
        "fine-tune",
        "gets for free from seeing that slice, so anything above it is what "
        "the",
        "FEATURES earned.",
        "",
        "BIAS-ONLY flatters the model and should NOT be quoted as the bar: it "
        "is stuck",
        "with the population W, so it keeps injecting feature-driven noise "
        "that its lone",
        "intercept cannot cancel. The full head can beat it just by driving W "
        "toward",
        "zero - i.e. by SUPPRESSING the features, not by using them.",
        "",
        f"predicted-vs-true deviation r on the test slice: median "
        f"{df['r_SBP'].median():.3f} (SBP), {df['r_DBP'].median():.3f} (DBP)",
        f"prediction SD / true SD: median "
        f"{(df.predsd_SBP / df.truesd_SBP).median():.3f} (SBP) - near 0 means "
        f"the head collapsed to a constant",
        "",
        "Scored on the test slice only, so these MAEs are NOT comparable to",
        "ann_bp_loso.py's full-recording numbers. Compare within this table.",
    ]
    if df.post_beats_amean.sum() <= len(df) / 2:
        lines += ["",
                  "VERDICT: the fine-tune does NOT beat ADAPT-MEAN on most "
                  "subjects, i.e. it does no",
                  "better than predicting this person's average BP with no "
                  "features at all. The gain",
                  "over PRE is re-anchoring the BP level, not tracking BP. "
                  "Report it as calibration,",
                  "not as estimation."]
    elif df.post_beats_zero.sum() <= len(df) / 2:
        lines += ["",
                  "VERDICT: personalization does NOT beat the zero-delta "
                  "baseline on most subjects. Do not report POST alone."]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ecg", default="ecg_features_30s.csv")
    p.add_argument("--skna", default="skna_features_30s.csv")
    p.add_argument("--out", default="beat_pipeline/built")
    p.add_argument("--norm", default="expanding")
    p.add_argument("--family", default="SKNA,XSIG,HRV,ECG_MORPH,NONLIN")
    p.add_argument("--hidden", default="10")
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--activation", default="logistic")
    p.add_argument("--max-iter", type=int, default=3000)
    p.add_argument("--calib-min", type=float, default=2.0)
    p.add_argument("--adapt-end", type=float, default=0.5)
    p.add_argument("--val-end", type=float, default=0.6)
    p.add_argument("--ft-lr", type=float, default=1e-2)
    p.add_argument("--ft-epochs", type=int, default=400)
    p.add_argument("--ft-patience", type=int, default=40)
    p.add_argument("--keep-s13", default="full", choices=["full", "cut"])
    p.add_argument("--drop-recording", default="")
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

    if a.drop_recording:
        keep = np.ones(len(rec), bool)
        for t in [s.strip() for s in a.drop_recording.split(",") if s.strip()]:
            hit = np.array([t == r or t in r for r in rec])
            if not hit.any():
                raise SystemExit(f"--drop-recording {t!r} matched nothing")
            for r in sorted(set(rec[hit])):
                print(f"DROPPED {r} ({int((rec == r).sum())} windows)")
            keep &= ~hit
        X, Y, groups, t_center, rec = (X[keep], Y[keep], groups[keep],
                                       t_center[keep], rec[keep])

    fams = set(a.family.split(","))
    cols = [i for i, c in enumerate(names)
            if ma.FAMILY[ma.split_column(c)[0]] in fams]
    Xf = X[:, cols]
    hp = dict(DEFAULTS, hidden=parse_hidden(a.hidden)[0], alpha=a.alpha,
              activation=a.activation, max_iter=a.max_iter)
    print(f"{Xf.shape[1]} features | {len(Y)} windows, "
          f"{len(np.unique(groups))} people | adapt<{a.adapt_end} "
          f"val<{a.val_end} | head-only fine-tune, lr={a.ft_lr}", flush=True)

    # Record the configuration IN the saved output. Without this a result file
    # cannot be told apart from one produced under different regularization or
    # a dropped recording, and the two are not comparable.
    cfg = ("CONFIG  " + "  ".join(
        f"{k}={v}" for k, v in [
            ("norm", a.norm), ("features", Xf.shape[1]),
            ("people", len(np.unique(groups))), ("hidden", a.hidden),
            ("alpha", a.alpha), ("activation", a.activation),
            ("max_iter", a.max_iter), ("calib_min", a.calib_min),
            ("adapt_end", a.adapt_end), ("val_end", a.val_end),
            ("ft_lr", a.ft_lr), ("ft_epochs", a.ft_epochs),
            ("ft_patience", a.ft_patience), ("keep_s13", a.keep_s13),
            ("drop_recording", a.drop_recording or "none"),
            ("seed", a.seed)]))

    t0 = time.time()
    df, curves, preds = run(Xf, Y, groups, t_center, rec, hp, a.seed,
                            a.calib_min, a.adapt_end, a.val_end, a.ft_lr,
                            a.ft_epochs, a.ft_patience)
    txt = cfg + "\n" + summarize(df)
    print(txt)
    print(f"\n[{time.time() - t0:.0f}s]")

    os.makedirs(a.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    df.to_csv(os.path.join(a.out, f"ann_personalize_{stamp}.csv"), index=False)
    curves.to_csv(os.path.join(a.out, f"ann_personalize_curves_{stamp}.csv"),
                  index=False)
    preds.to_csv(os.path.join(a.out, f"ann_personalize_preds_{stamp}.csv"),
                 index=False)
    with open(os.path.join(a.out, f"ann_personalize_{stamp}.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(f"wrote {a.out}/ann_personalize_{stamp}.{{csv,txt}} and _curves_.csv")


if __name__ == "__main__":
    main()
