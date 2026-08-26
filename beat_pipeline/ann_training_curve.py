#!/usr/bin/env python3
"""
Learning curves for the Yao et al. ANN, LOSO, SBP *and* DBP.
============================================================

Why this file exists: `solver='lbfgs'` exposes NO `loss_curve_` (sklearn only
records one for sgd/adam) and no validation curve at all, so there is nothing to
read out of a fitted MLPRegressor. The curve here is generated instead - fit with
`warm_start=True` and step `max_iter` in increments of `--step`, evaluating the
train block and the held-out person at every checkpoint.

CAVEAT, stated because it changes what the picture means: lbfgs keeps a limited-
memory curvature history, and every warm-start `fit` call restarts that history.
So this is "lbfgs restarted every `--step` iterations", not one uninterrupted
run. Bigger --step is closer to the real trajectory but coarser. It is the same
optimizer on the same objective, so the overfitting story it tells is real; do
not read the exact iteration counts as identical to a single long fit.

The dashed line on the left panels is the baseline run_loso() scores against:
ZERO-DELTA when calibrated (predict the calibration reading and never move),
the constant train mean when not. A curve that never crosses it never learned
anything about BP - see ann_bp_loso.py's verdict text.

`curves()` and `draw()` are also what tab 6's "Training curve" checkbox calls,
so the GUI plot and this figure are the same computation.

    python3 beat_pipeline/ann_training_curve.py --keep-s13 cut \
        --drop-recording cindy --alpha 10 --max-iter 2000 --step 50
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_ablation as ma
from ann_bp_loso import DEFAULTS, make_ann, parse_hidden, _offsets, CHANNELS

import matplotlib
if __name__ == "__main__":       # only when run as a script - model_window.py
    matplotlib.use("Agg")        # imports this for curves()/draw() and must
import matplotlib.pyplot as plt  # keep the Qt backend it already selected

# validated 2-slot categorical palette (dataviz reference instance, light mode:
# all-pairs CVD dE 24.7, normal-vision 33.6, contrast >=3:1)
C_TRAIN, C_HELD = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def curves(X, Y, groups, t_center, hp, mode, calib_min, max_iter, step, seed,
           progress=None):
    """One warm-started LOSO run; returns tidy rows of (fold, iter, split,
    channel, mae).

    `progress` is called with one line per finished fold; it defaults to
    printing, so the CLI is unchanged and the GUI can route it to its log.
    """
    say = progress or (lambda m: print(m, flush=True))
    rows = []
    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)
    for i, (tr, te) in enumerate(logo.split(X, Y[:, 0], groups), 1):
        person = groups[te][0]
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        off_tr, off_te = _offsets(Y, t_center, groups, tr, te, person, mode,
                                  calib_min)
        Ytr_c = Y[tr] - off_tr
        # the same baseline run_loso() scores against, per channel: zero-delta
        # when calibrated, the constant train mean when not. Using off_te for
        # mode='none' would compare against predicting 0 mmHg.
        bpred = off_te[0] if mode == "offset" else Y[tr].mean(axis=0)
        base = np.abs(bpred - Y[te]).mean(axis=0)

        m = make_ann(hp, seed)
        m.set_params(warm_start=True, max_iter=step)
        t0 = time.time()
        for it in range(step, max_iter + 1, step):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                m.fit(Xtr, Ytr_c)
            ptr = np.asarray(m.predict(Xtr)) + off_tr
            pte = np.asarray(m.predict(Xte)) + off_te
            for j, ch in enumerate(CHANNELS):
                rows.append(dict(fold=person, iter=it, split="train",
                                 channel=ch,
                                 mae=float(np.abs(ptr[:, j] - Y[tr][:, j]).mean())))
                rows.append(dict(fold=person, iter=it, split="held-out",
                                 channel=ch,
                                 mae=float(np.abs(pte[:, j] - Y[te][:, j]).mean()),
                                 base=float(base[j])))
        say(f"  fold {i}/{n_folds} {person:8s} n={len(te):4d} "
            f"[{time.time() - t0:.0f}s]")
    return pd.DataFrame(rows)


def _place(ax, items, xpos, pad_frac=.06, size=9.5):
    """Direct labels at the curve ends, pushed apart so they never overprint.

    items: [(y, text, colour)]. Two subjects whose curves end 0.02 mmHg apart
    would otherwise render as one unreadable smear ('betdkyr +4.1'). Labels sit
    inside the axes' right margin, so they never collide with the next panel.
    """
    lo, hi = ax.get_ylim()
    minsep = (hi - lo) * pad_frac
    items = sorted(items, key=lambda t: t[0])
    ys = [y for y, _, _ in items]
    for i in range(1, len(ys)):                  # single upward pass is enough
        if ys[i] - ys[i - 1] < minsep:           # for the <=4 labels we place
            ys[i] = ys[i - 1] + minsep
    for (yd, txt, col), y in zip(items, ys):
        # label at the DE-OVERLAPPED y, dot at the curve's true end, so a
        # nudged label still points at the right line
        ax.annotate(txt, xy=(xpos, y), xytext=(9, 0),
                    textcoords="offset points", color=col, fontsize=size,
                    fontweight="bold", va="center")
        ax.plot([xpos], [yd], marker="o", ms=3.5, color=col, zorder=5)


def baseline_label(mode):
    return "zero-delta baseline" if mode == "offset" else "train-mean baseline"


def draw(fig, df, title_bits, label_size=9.5, base_label="zero-delta baseline",
         solver="lbfgs"):
    """Render the whole figure onto `fig` - a saved Agg figure from plot(), or
    the live canvas of the GUI's Training-curve tab. Everything here is
    figure-local; nothing touches pyplot state."""
    chans = list(CHANNELS)
    fig.clear()
    # the GUI canvas is built with tight_layout=True; that engine would re-run
    # on every redraw and undo the rect below, dropping the suptitle onto the
    # top row of titles. One explicit tight_layout at the end instead.
    fig.set_layout_engine("none")
    axes = fig.subplots(len(chans), 2, sharex=True, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")

    for r, ch in enumerate(chans):
        d = df[df.channel == ch]
        base = d.loc[d.split == "held-out", "base"].mean()

        # ---- left: mean across folds, train vs held-out --------------------
        ax = axes[r, 0]
        ax.set_facecolor("#fcfcfb")
        labs = []
        for split, col in (("train", C_TRAIN), ("held-out", C_HELD)):
            g = d[d.split == split].groupby("iter")["mae"]
            mu, q1, q3 = g.mean(), g.quantile(.25), g.quantile(.75)
            ax.fill_between(mu.index, q1, q3, color=col, alpha=.13, lw=0)
            ax.plot(mu.index, mu.values, color=col, lw=2, solid_capstyle="round")
            labs.append((mu.iloc[-1], f"{split}  {mu.iloc[-1]:.2f}", col))
        ax.axhline(base, color=INK_MUTED, ls=(0, (5, 4)), lw=1.4)
        ax.annotate(f"{base_label}  {base:.2f}", (d["iter"].min(), base),
                    color=INK_MUTED, fontsize=label_size, va="bottom",
                    xytext=(2, 4), textcoords="offset points")
        ax.set_title(f"{ch} — mean over folds (band = IQR)", color=INK,
                     fontsize=label_size + 2.5, fontweight="bold", loc="left",
                     pad=10)
        ax.set_ylabel(f"{ch} MAE (mmHg)", color=INK_MUTED,
                      fontsize=label_size + .5)
        _place(ax, labs, d['iter'].max(), size=label_size)

        # ---- right: per-subject gap to that subject's own baseline ---------
        ax = axes[r, 1]
        ax.set_facecolor("#fcfcfb")
        h = d[d.split == "held-out"]
        ends = {}
        for f, g in h.groupby("fold"):
            g = g.sort_values("iter")
            gap = g.mae.values - g.base.values
            ax.plot(g.iter.values, gap, color=INK_MUTED, lw=1, alpha=.45)
            ends[f] = (g.iter.values[-1], gap[-1])
        ax.axhline(0, color=INK, lw=1.4)
        ax.annotate("0 = ties the baseline; below = beats it",
                    (h["iter"].min(), 0), color=INK, fontsize=label_size,
                    va="bottom", xytext=(2, 4), textcoords="offset points")
        # direct-label only the extremes, so 13 lines never need 13 hues
        ranked = sorted(ends.items(), key=lambda kv: kv[1][1])
        won = sum(1 for _, (_, y) in ends.items() if y < 0)
        _place(ax, [(y, f"{f} {y:+.1f}", "#008300" if y < 0 else INK)
                    for f, (_, y) in ranked[:2] + ranked[-2:]],
               h["iter"].max(), size=label_size)
        ax.set_title(f"{ch} — per subject vs own baseline · "
                     f"{won}/{len(ends)} beat it",
                     color=INK, fontsize=label_size + 2.5, fontweight="bold",
                     loc="left", pad=10)
        ax.set_ylabel(f"{ch} MAE − baseline (mmHg)", color=INK_MUTED,
                      fontsize=label_size + .5)

    for ax in axes.ravel():
        ax.grid(True, color=GRID, lw=.7, alpha=.9)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=label_size)
        ax.margins(x=.02)
    for ax in axes[-1]:
        ax.set_xlabel(f"{solver} iterations (warm-started in steps)",
                      color=INK_MUTED, fontsize=label_size + .5)
    # once, not per-axis: sharex makes a per-axis multiplier compound (1.2^4)
    x0, x1 = df["iter"].min(), df["iter"].max()
    axes[0, 0].set_xlim(x0 - (x1 - x0) * .03, x1 + (x1 - x0) * .17)

    fig.suptitle("ANN training curves — LOSO, " + title_bits, color=INK,
                 fontsize=label_size + 4.5, fontweight="bold", x=.008,
                 ha="left", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .975))
    return fig


def plot(df, out_png, title_bits, base_label="zero-delta baseline"):
    fig = plt.figure(figsize=(13.5, 4.3 * len(CHANNELS)))
    draw(fig, df, title_bits, base_label=base_label)
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nwrote {out_png}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ecg", default="ecg_features_30s.csv")
    p.add_argument("--skna", default="skna_features_30s.csv")
    p.add_argument("--out", default="beat_pipeline/built")
    p.add_argument("--fig", default="figures/ann_training_curves.png")
    p.add_argument("--norm", default="expanding")
    p.add_argument("--family", default="SKNA,XSIG,HRV,ECG_MORPH,NONLIN")
    p.add_argument("--hidden", default="10")
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--activation", default="logistic")
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--step", type=int, default=50)
    p.add_argument("--calibration", default="offset", choices=["none", "offset"])
    p.add_argument("--calib-min", type=float, default=2.0)
    p.add_argument("--keep-s13", default="full", choices=["full", "cut"])
    p.add_argument("--drop-recording", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--replot-from", default="",
                   help="redraw the figure from a curve CSV this script "
                        "already wrote, without refitting anything")
    a = p.parse_args()

    if a.replot_from:
        df = pd.read_csv(a.replot_from)
        n_people = df.fold.nunique()
        plot(df, a.fig, f"{n_people} people, calibration={a.calibration}, "
                        f"alpha={a.alpha}", baseline_label(a.calibration))
        return

    X, names, _y, y_sbp, groups, meta = ma.build(
        a.norm, a.ecg, a.skna, a.out, a.k, True, keep_s13=a.keep_s13)
    X = np.nan_to_num(np.asarray(X, float), nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.column_stack([np.asarray(y_sbp, float),
                         np.asarray(meta["y_dbp"], float)])
    t_center = np.asarray(meta["t_center_sec"], float)
    rec = np.asarray(meta["recording"]).astype(str)

    if a.drop_recording:
        keep = np.ones(len(rec), bool)
        for t in [t.strip() for t in a.drop_recording.split(",") if t.strip()]:
            hit = np.array([t == r or t in r for r in rec])
            if not hit.any():
                raise SystemExit(f"--drop-recording {t!r} matched nothing")
            for r in sorted(set(rec[hit])):
                print(f"DROPPED {r} ({int((rec == r).sum())} windows)")
            keep &= ~hit
        X, Y, groups, t_center = X[keep], Y[keep], groups[keep], t_center[keep]

    fams = set(a.family.split(","))
    cols = [i for i, c in enumerate(names)
            if ma.FAMILY[ma.split_column(c)[0]] in fams]
    Xf = X[:, cols]
    hp = dict(DEFAULTS, hidden=parse_hidden(a.hidden)[0], alpha=a.alpha,
              activation=a.activation, max_iter=a.step)
    print(f"{Xf.shape[1]} features | {len(Y)} windows, "
          f"{len(np.unique(groups))} people | calibration={a.calibration} | "
          f"alpha={a.alpha} | curve to {a.max_iter} in steps of {a.step}",
          flush=True)

    df = curves(Xf, Y, groups, t_center, hp, a.calibration, a.calib_min,
                a.max_iter, a.step, a.seed)
    os.makedirs(os.path.dirname(a.fig) or ".", exist_ok=True)
    csv = os.path.splitext(a.fig)[0] + ".csv"
    df.to_csv(csv, index=False)
    print(f"wrote {csv}")
    plot(df, a.fig, f"{len(np.unique(groups))} people, calibration="
                    f"{a.calibration}, alpha={a.alpha}, {Xf.shape[1]} features",
         baseline_label(a.calibration))


if __name__ == "__main__":
    main()
