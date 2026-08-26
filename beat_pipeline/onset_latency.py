#!/usr/bin/env python3
"""
Onset latency: how fast does SKNA detect the cold pressor?
==========================================================

The ablation says SKNA separates CPT from rest (LOSO AUC 0.961 at 5 s). That is
a static question - given a window, was it CPT. The operational question is
different and, at 5 s resolution, finally answerable: *starting from immersion,
how long until the detector fires?*

Method
------
  1. LOSO on `person`, SKNA features only, logistic. Every window gets an
     out-of-fold p(CPT) from a model that never saw that person.
  2. The decision threshold is set INSIDE the training folds, at a fixed
     false-positive rate on pre-CPT rest windows only (`cycle_min < 4.5`).
     Post-CPT recovery is excluded from threshold estimation - if sympathetic
     tone is still elevated there, it is not a clean specificity reference.
  3. An episode is one 60 s immersion (one 10-min cycle of one recording).
     Alarm = the first run of `--persist` consecutive windows at or above
     threshold. Latency is reported as the END of the last window in that run:
     the moment a live detector requiring that much persistence would actually
     fire, not the optimistic centre of the first window.
  4. Offset latency applies the mirror rule after the immersion ends.

Reporting the offset matters beyond symmetry. A previous 30 s analysis found
p(CPT) looked like a box rather than a decay, and attributed the apparent decay
to 30 s windows straddling the immersion boundary. At 5 s with a 5 s stride the
windows are disjoint and every CPT window lies wholly inside the immersion, so
that confound is gone and the recovery shape can be read directly.

    python3 beat_pipeline/onset_latency.py
    python3 beat_pipeline/onset_latency.py --fpr 0.02 --persist 3

Outputs
-------
  <out>/onset_latency_episodes.csv   one row per immersion
  <out>/onset_latency_summary.txt
  figures/onset_latency.png          trajectory + cumulative detection
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_ablation as ma

CPT_START, CPT_END = 4.5, 5.5          # minutes into the 10-min cycle
CYCLE_SEC = 600.0
WIN_SEC = 5.0

# Reference palette, light mode (papers print light): categorical slot 1 blue,
# slot 2 orange. Text and grid wear ink tokens, never the series colour.
C_SERIES = "#2a78d6"
C_ALT = "#eb6834"
C_INK = "#0b0b0b"
C_MUTED = "#52514e"
C_GRID = "#dcdcd8"


def out_of_fold(X, y, groups, cycle_min, fpr):
    """Out-of-fold p(CPT) plus the per-fold threshold, both LOSO-clean."""
    p = np.full(len(y), np.nan)
    thr = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        # specificity reference: pre-CPT rest of the TRAINING people only
        rest = tr[(y[tr] == 0) & (cycle_min[tr] < CPT_START)]
        thr[te] = np.quantile(m.predict_proba(sc.transform(X[rest]))[:, 1],
                              1.0 - fpr)
    return p, thr


def first_run(flags, times, persist):
    """End time of the first run of `persist` consecutive True flags."""
    run = 0
    for f, t in zip(flags, times):
        run = run + 1 if f else 0
        if run >= persist:
            return t
    return np.nan


def episodes(df, persist):
    """One row per immersion: onset latency, offset latency, peak p."""
    rows = []
    for (rec, cyc), g in df.groupby(["recording", "cycle"], sort=False):
        g = g.sort_values("t_rel")
        during = g[(g.t_rel >= 0) & (g.t_rel < 60)]
        after = g[g.t_rel >= 60]
        if len(during) < persist:
            continue                     # truncated immersion, cannot judge
        rows.append(dict(
            recording=rec, cycle=int(cyc), person=g.person.iloc[0],
            n_during=len(during),
            # window centres are at t_rel = 2.5, 7.5, ...; the alarm is the END
            # of the confirming window, hence +WIN_SEC/2
            onset_latency=first_run(during.p >= during.thr,
                                    during.t_rel + WIN_SEC / 2, persist),
            offset_latency=(first_run(after.p < after.thr,
                                      after.t_rel + WIN_SEC / 2, persist) - 60.0
                            if len(after) >= persist else np.nan),
            p_max_during=during.p.max(),
            p_mean_during=during.p.mean(),
            p_mean_before=g.loc[g.t_rel < 0, "p"].mean(),
        ))
    return pd.DataFrame(rows)


def trajectory(df):
    """Mean p(CPT) vs time-from-onset, averaged per subject FIRST so a person
    with more usable cycles does not dominate the cohort curve."""
    per = (df.groupby(["person", "t_rel"])["p"].mean()
           .reset_index())
    g = per.groupby("t_rel")["p"]
    return pd.DataFrame({"t_rel": g.mean().index, "mean": g.mean().values,
                         "sem": (g.std() / np.sqrt(g.count())).values,
                         "n": g.count().values})


def figure(traj, ep, thr_mean, path, persist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=C_GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=9)

    # --- (a) response trajectory -------------------------------------------
    t = traj.t_rel.to_numpy()
    m, se = traj["mean"].to_numpy(), traj["sem"].to_numpy()
    ax1.axvspan(0, 60, color=C_ALT, alpha=0.10, lw=0)
    ax1.fill_between(t, m - se, m + se, color=C_SERIES, alpha=0.20, lw=0)
    ax1.plot(t, m, color=C_SERIES, lw=2.0)
    ax1.axhline(thr_mean, color=C_MUTED, lw=1.2, ls=(0, (4, 3)))
    ax1.text(t.min() + 6, thr_mean + 0.015,
             f"decision threshold (mean)", color=C_MUTED, fontsize=8.5)
    ax1.text(30, ax1.get_ylim()[1] * 0.97, "immersion", color=C_ALT,
             fontsize=9, ha="center", va="top")
    ax1.set_xlabel("time from immersion onset (s)", color=C_INK, fontsize=10)
    ax1.set_ylabel("p(CPT), out-of-fold", color=C_INK, fontsize=10)
    ax1.set_title("SKNA response, LOSO across 14 subjects",
                  color=C_INK, fontsize=11, loc="left")
    ax1.set_xlim(t.min(), t.max())

    # --- (b) cumulative detection ------------------------------------------
    lat = ep.onset_latency.to_numpy(float)
    grid = np.arange(0, 65, 1.0)
    cum = [(np.nan_to_num(lat, nan=np.inf) <= x).mean() * 100 for x in grid]
    ax2.plot(grid, cum, color=C_SERIES, lw=2.0)
    ax2.axvspan(0, 60, color=C_ALT, alpha=0.10, lw=0)
    med = np.nanmedian(lat)
    if np.isfinite(med):
        ax2.axvline(med, color=C_MUTED, lw=1.2, ls=(0, (4, 3)))
        ax2.text(med + 1.5, 8, f"median {med:.0f} s", color=C_MUTED,
                 fontsize=8.5)
    ax2.set_ylim(0, 100)
    ax2.set_xlim(0, 60)
    ax2.set_xlabel("time from immersion onset (s)", color=C_INK, fontsize=10)
    ax2.set_ylabel("episodes detected (%)", color=C_INK, fontsize=10)
    ax2.set_title(f"Cumulative detection ({len(ep)} immersions, "
                  f"{persist}-window persistence)",
                  color=C_INK, fontsize=11, loc="left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ecg", default="ecg_features_5s.csv")
    p.add_argument("--skna", default="skna_features_5s.csv")
    p.add_argument("--out", default="beat_pipeline/built")
    p.add_argument("--fig", default="figures/onset_latency.png")
    p.add_argument("--norm", default="expanding")
    p.add_argument("--family", default="SKNA",
                   help="comma-separated FAMILY classes, default SKNA only "
                        "(the best model in the ablation)")
    p.add_argument("--fpr", type=float, default=0.05,
                   help="false-positive rate on training pre-CPT rest")
    p.add_argument("--persist", type=int, default=2,
                   help="consecutive windows required to declare a detection")
    p.add_argument("--k", type=int, default=2)
    a = p.parse_args()

    X, names, y, _, groups, meta = ma.build(
        a.norm, a.ecg, a.skna, a.out, a.k, True)
    X = np.nan_to_num(np.asarray(X, float), nan=0.0, posinf=0.0, neginf=0.0)
    y = y.astype(int)
    cycle_min = np.asarray(meta["cycle_min"], float)

    fams = set(a.family.split(","))
    cols = [i for i, c in enumerate(names)
            if ma.FAMILY[ma.split_column(c)[0]] in fams]
    print(f"model: {len(cols)} features from {sorted(fams)}, "
          f"{len(y)} windows, {y.sum()} CPT, {len(np.unique(groups))} people")

    pr, thr = out_of_fold(X[:, cols], y, groups, cycle_min, a.fpr)

    # An episode is one immersion: the cycle index within its own recording.
    # Both columns ride along in the cache, so they are aligned with X by
    # construction rather than by re-running the pipeline and trusting order.
    df = pd.DataFrame(dict(
        person=groups, p=pr, thr=thr, y=y, cycle_min=cycle_min,
        recording=np.asarray(meta["recording"]),
        t_rel=(cycle_min - CPT_START) * 60.0,
        cycle=np.floor(np.asarray(meta["t_center_sec"], float)
                       / CYCLE_SEC).astype(int)))

    ep = episodes(df, a.persist)
    traj = trajectory(df)

    det = np.isfinite(ep.onset_latency)
    lat = ep.loc[det, "onset_latency"]
    off = ep.offset_latency.dropna()
    per_subj = ep.loc[det].groupby("person")["onset_latency"].median()

    lines = [
        f"onset latency  family={a.family}  norm={a.norm}  "
        f"fpr={a.fpr}  persist={a.persist} windows ({a.persist * WIN_SEC:.0f} s)",
        "",
        f"episodes                {len(ep)} immersions, "
        f"{ep.person.nunique()} people",
        f"detected within 60 s    {det.sum()}/{len(ep)} "
        f"({100.0 * det.mean():.0f}%)",
        f"onset latency           median {lat.median():.1f} s   "
        f"IQR {lat.quantile(.25):.1f}-{lat.quantile(.75):.1f} s   "
        f"min {lat.min():.1f}  max {lat.max():.1f}",
        f"per-subject median      median {per_subj.median():.1f} s   "
        f"range {per_subj.min():.1f}-{per_subj.max():.1f} s "
        f"({len(per_subj)} people)",
        f"offset latency          median {off.median():.1f} s   "
        f"IQR {off.quantile(.25):.1f}-{off.quantile(.75):.1f} s "
        f"(n={len(off)})",
        f"mean threshold          {np.nanmean(thr):.3f}",
        "",
        "per-subject onset latency (median s):",
    ] + [f"  {k:<10} {v:5.1f}" for k, v in per_subj.sort_values().items()]

    os.makedirs(a.out, exist_ok=True)
    ep.to_csv(os.path.join(a.out, "onset_latency_episodes.csv"), index=False)
    txt = os.path.join(a.out, "onset_latency_summary.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    figure(traj, ep, float(np.nanmean(thr)), a.fig, a.persist)
    print(f"\nwrote {txt}\n      {a.fig}")


if __name__ == "__main__":
    main()
