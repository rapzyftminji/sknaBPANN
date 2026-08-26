#!/usr/bin/env python3
"""
Feature engineering for the 30 s ECG + SKNA window tables.
==========================================================

Sits between feature EXTRACTION (build_ecg_features.py / build_skna_features.py,
which write ecg_features_30s.csv + skna_features_30s.csv) and the learning
model. Extraction answers "what is in this 30 s window"; this answers "what does
that mean for THIS person, relative to THEIR own baseline, and how is it
moving".

    python3 beat_pipeline/feature_engineering.py --out beat_pipeline/built/

Stages, in order:

  1. merge      ECG + SKNA on (Subject_ID, Recording, window_idx)
  2. protocol   10-min cycle position -> phase (Before / During / After), is_cpt
  3. quality    keep usable & label_valid; drop LF band (unreliable at 30 s)
  4. transform  log1p the heavy-tailed features (skew > 2) so scaling is not
                dominated by a handful of windows
  5. normalize  WITHIN each recording (default: causal expanding median/IQR,
                past windows only; also robust / baseline / zscore)
  6. temporal   first difference, rolling mean/std, local slope over +-k windows
  7. interact   cross-signal ratios (SKNA burden per beat, sympatho-vagal, ...)
  8. prune      near-zero variance + collinear (|r| > 0.95) cluster pruning

Why step 5 is the important one
-------------------------------
Pooled across recordings, SKNA<->BP correlation in this dataset is mostly a
recording-batch artifact: different sessions sit at different absolute levels
(electrode placement, gain, skin impedance), and a pooled model learns "which
recording is this" instead of "what is the autonomic state". Normalizing inside
each recording removes the between-recording offset and leaves only the
within-recording dynamics, which is the part that carries the CPT response.

The normalization uses NO labels (medians/IQRs of the feature columns only), so
it can be fit on a held-out subject's own recording without leaking the target -
the same thing you would do at inference time on a new person's first minutes.

Outputs
-------
  <out>/features_30s_engineered.csv   engineered table (one row per window)
  <out>/features_30s_engineered.npz   X, y_sbp, y_dbp, y_cpt, groups, names
  <out>/feature_engineering_report.txt  what was kept, dropped and why
"""
import argparse
import os

import numpy as np
import pandas as pd

# s5 has two sessions and s13 has a 10-min cut plus the full 32-min recording:
# 16 Subject_IDs are 14 people. Grouped CV must split on the PERSON or the same
# person appears on both sides of the split.
PERSON = {
    "s1": "biopac", "s2": "arthur", "s3": "becky", "s4": "eric",
    "s5_session1": "cindy", "s5_session2": "cindy", "s6": "alice",
    "s7": "jose", "s8": "rizmi", "s9": "nhu", "s10": "andreas",
    "s11": "tsai", "s12": "hk", "s13": "tseng2", "s13_full": "tseng2",
    "s14": "eva",
}

META = ["Subject_ID", "Recording", "window_idx", "t_start_sec",
        "t_center_sec", "t_raw_center_sec", "person", "cycle_min", "phase",
        "is_cpt", "n_beats", "lf_reliable"]
TARGETS = ["SBP", "DBP"]
FLAGS = ["label_valid", "usable"]

# LF band needs >= 60 s of NN series to be meaningful; every window in this
# table has lf_reliable == False. LF_HF_ratio inherits the problem.
UNRELIABLE_30S = ["LF_power", "LF_HF_ratio"]

# skew > 2 in the raw table -> log-compress before any scaling, so a handful of
# high-amplitude windows cannot dominate the normalized range.
LOG_FEATURES = ["complexity", "kurtSKNA", "skewSKNA", "cfSKNA",
                "LF_HF_ratio", "LF_power", "HF_power", "varSKNA"]

CYCLE_MIN = 10.0          # protocol: 4.5 rest / 1.0 CPT / 4.5 recovery
CPT_START, CPT_END = 4.5, 5.5


# --------------------------------------------------------------------------- 0
def load_table(path):
    """Read one cohort feature table - the tidy CSV from build_*_features.py
    --out, or the packed NPZ from --npz (single combined file) / --npz-dir
    (per-subject; pass the one you want). Both carry the same information;
    NPZ is unpacked into the same tidy shape here so load_merged() and
    everything downstream of it never needs to know which one was loaded.

    Needs, at minimum: Subject_ID/Recording/window_idx (to join ECG with
    SKNA row-for-row) and SBP/DBP/label_valid/usable (to label and filter).
    See build_ecg_features.py / build_skna_features.py for exactly what each
    format packs - ecg_features.save_npz / skna_features.save_npz write the
    NPZ side."""
    if str(path).lower().endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        names = [str(n) for n in d["feature_names"]]
        df = pd.DataFrame(np.asarray(d["X"], dtype=float), columns=names)
        df.insert(0, "Subject_ID", np.asarray(d["subject"]).astype(str))
        df.insert(1, "Recording", np.asarray(d["recording"]).astype(str))
        df.insert(2, "window_idx", np.asarray(d["window_idx"]).astype(int))
        # n_beats/lf_reliable only exist on the ECG table (see ecg_features.
        # EXTRA_KEYS vs skna_features.EXTRA_KEYS=()); everything else is on
        # both.
        for c in ("t_start_sec", "t_center_sec", "t_raw_center_sec",
                  "n_beats", "lf_reliable", "SBP", "DBP", "label_valid",
                  "usable"):
            if c in d.files:
                df[c] = d[c]
        missing = [c for c in ("SBP", "DBP", "label_valid", "usable")
                  if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing {missing} - this NPZ was "
                             "packed without a subject (no BP labels), so "
                             "there is nothing to engineer against")
        return df
    return pd.read_csv(path)


# --------------------------------------------------------------------------- 1
def load_merged(ecg_csv, skna_csv):
    ecg = load_table(ecg_csv)
    skna = load_table(skna_csv)
    key = ["Subject_ID", "Recording", "window_idx"]
    dup = [c for c in skna.columns if c in ecg.columns and c not in key]
    df = ecg.merge(skna.drop(columns=dup), on=key, how="inner")
    return df.sort_values(["Subject_ID", "window_idx"]).reset_index(drop=True)


# --------------------------------------------------------------------------- 2
def add_protocol(df):
    """Position inside the 10-min CPT cycle, from the window centre time."""
    df = df.copy()
    df["person"] = df["Subject_ID"].map(PERSON)
    if df["person"].isna().any():
        missing = sorted(df.loc[df["person"].isna(), "Subject_ID"].unique())
        raise KeyError(f"no person mapping for {missing}")
    df["cycle_min"] = (df["t_center_sec"] / 60.0) % CYCLE_MIN
    df["phase"] = np.where(df["cycle_min"] < CPT_START, "Before",
                  np.where(df["cycle_min"] < CPT_END, "During", "After"))
    df["is_cpt"] = (df["phase"] == "During").astype(int)
    return df


# --------------------------------------------------------------------------- 3
def quality_filter(df, drop_unreliable=True, keep_s13="full"):
    """keep_s13 picks which Tseng2 recording survives deduplication.

    s13 (Tseng2_cut) is the first 10 minutes of s13_full (Tseng2); keeping both
    would enter one person twice and double-count them in every pooled metric,
    so exactly one has to go. 'full' keeps the 31-minute recording (more
    windows), 'cut' keeps the 10-minute segment (cleaner signal). Neither is
    obviously right, which is why it is an argument and not a constant.
    """
    if keep_s13 not in ("full", "cut"):
        raise ValueError(f"keep_s13 must be 'full' or 'cut', got {keep_s13!r}")
    n0 = len(df)
    notes = []
    df = df[df["usable"] & df["label_valid"]].copy()
    notes.append(f"quality mask (usable & label_valid): {n0} -> {len(df)}")

    if {"s13", "s13_full"} <= set(df["Subject_ID"]):
        n = len(df)
        loser = "s13" if keep_s13 == "full" else "s13_full"
        df = df[df["Subject_ID"] != loser].copy()
        notes.append(f"Tseng2 dedup: kept s13{'_full' if keep_s13 == 'full' else ' (10-min cut)'}"
                     f", dropped {loser}: {n} -> {len(df)}")

    dropped = []
    if drop_unreliable:
        dropped = [c for c in UNRELIABLE_30S if c in df.columns]
        df = df.drop(columns=dropped)
        notes.append(f"dropped LF-band features (lf_reliable all False): {dropped}")
    return df, notes


def feature_columns(df):
    skip = set(META + TARGETS + FLAGS)
    return [c for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


# --------------------------------------------------------------------------- 4
def log_transform(df, feats):
    """log1p the heavy-tailed features; shift first if a column can go <= -1."""
    df = df.copy()
    done = []
    for c in LOG_FEATURES:
        if c not in feats:
            continue
        lo = df[c].min()
        shift = 0.0 if lo > -1 else (-lo + 1e-6)
        df[c] = np.log1p(df[c] + shift)
        done.append(c)
    return df, done


# --------------------------------------------------------------------------- 7
def add_interactions(df, feats):
    """Cross-signal ratios that are physiologically motivated, not exhaustive."""
    df = df.copy()
    eps = 1e-9
    new = []

    def put(name, series):
        df[name] = series.replace([np.inf, -np.inf], np.nan)
        new.append(name)

    hr = 60000.0 / df["meanNN"].clip(lower=eps)      # bpm from mean NN (ms)
    put("HR", hr)
    if "aSKNA" in feats:
        # burst burden normalized by how many beats it was spread over, and by
        # vagal tone - the same window means something different at HR 50 vs 100
        put("aSKNA_per_beat", df["aSKNA"] / df["n_beats"].clip(lower=1))
        put("aSKNA_x_HR", df["aSKNA"] * hr / 60.0)
        if "RMSSD" in feats:
            put("aSKNA_over_RMSSD", df["aSKNA"] / (df["RMSSD"] + eps))
        if "SDNN" in feats:
            put("aSKNA_over_SDNN", df["aSKNA"] / (df["SDNN"] + eps))
    if {"wampSKNA", "wlSKNA"} <= set(feats):
        put("wamp_over_wl", df["wampSKNA"] / (df["wlSKNA"] + eps))
    if {"SDNN", "RMSSD"} <= set(feats):
        put("SDNN_over_RMSSD", df["SDNN"] / (df["RMSSD"] + eps))
    if {"rmsSKNA", "aSKNA"} <= set(feats):
        put("SKNA_crest", df["rmsSKNA"] / (df["aSKNA"] + eps))
    return df, new


# --------------------------------------------------------------------------- 5
def normalize_within_recording(df, feats, method="robust", suffix=""):
    """
    Remove the per-recording offset/scale. Label-free by construction.

      robust    (x - median_rec) / IQR_rec          whole recording: label-free
                                                    but TRANSDUCTIVE - it sees
                                                    every window of the test
                                                    recording, including future
      expanding (x - median_<=t) / IQR_<=t          past windows only: strictly
                                                    causal, deployable live
      baseline  (x - median_rest1_rec) / IQR_rec    pre-CPT rest only. Needs to
                                                    KNOW those windows are rest,
                                                    so on the CPT task it peeks
                                                    at class-0 test labels -
                                                    read it as a calibrated
                                                    upper bound, not a score
      zscore    (x - mean_rec) / std_rec
      none      identity - leave the raw levels alone. The control arm: keeps
                the between-recording offset that every other method removes,
                so a model fit on it can still learn "which recording is this".
    """
    df = df.copy()
    g = df.groupby("Recording", sort=False)

    if method == "none":
        out = df[feats].copy()
        if suffix:
            out.columns = [c + suffix for c in feats]
            df = pd.concat([df, out], axis=1)
        return df, list(out.columns)

    if method == "expanding":
        warm = 10                                  # 5 min before it stabilises
        centre = g[feats].transform(
            lambda s: s.expanding(min_periods=1).median().shift(1)
                       .fillna(s.iloc[:warm].median()))
        scale = g[feats].transform(
            lambda s: (s.expanding(min_periods=4).quantile(0.75)
                       - s.expanding(min_periods=4).quantile(0.25)).shift(1)
                      .bfill())
    elif method == "zscore":
        centre = g[feats].transform("mean")
        scale = g[feats].transform("std")
    else:
        scale = (g[feats].transform(lambda s: s.quantile(0.75))
                 - g[feats].transform(lambda s: s.quantile(0.25)))
        if method == "robust":
            centre = g[feats].transform("median")
        elif method == "baseline":
            rest1 = df["t_center_sec"] < CPT_START * 60.0
            base = (df[rest1].groupby("Recording")[feats].median()
                    .reindex(df["Recording"]).set_index(df.index))
            # a recording with no pre-CPT rest falls back to its own median
            centre = base.fillna(g[feats].transform("median"))
        else:
            raise ValueError(f"unknown method {method!r}")

    scale = scale.replace(0, np.nan)
    out = (df[feats] - centre) / scale
    fallback = g[feats].transform("std").replace(0, np.nan)
    out = out.fillna((df[feats] - centre) / fallback).fillna(0.0)

    if suffix:
        out.columns = [c + suffix for c in feats]
        df = pd.concat([df, out], axis=1)
    else:
        df[feats] = out
    return df, list(out.columns)


# --------------------------------------------------------------------------- 6
def add_temporal(df, feats, k=2, causal=True):
    """
    Windows are a time series, not a bag. Add per-recording:
      _d1     change since the previous window (30 s rate of change)
      _rm{k}  rolling mean over k+1 windows   (smooths window-level noise)
      _rs{k}  rolling std  over k+1 windows   (local variability / instability)
      _slope  least-squares slope over the same span, per window

    causal=True uses only past windows, so the features stay deployable on a
    live stream and cannot see the future of the recording.
    """
    df = df.copy()
    g = df.groupby("Recording", sort=False)
    win = k + 1 if causal else 2 * k + 1
    centre = not causal
    t = np.arange(win, dtype=float)
    t -= t.mean()
    denom = (t ** 2).sum()

    def slope(s):
        return s.rolling(win, min_periods=2, center=centre).apply(
            lambda v: np.dot(v - v.mean(), t[-len(v):] - t[-len(v):].mean())
            / max(denom, 1e-9), raw=True)

    blocks, new = [], []
    for name, fn in (("_d1", lambda s: s.diff()),
                     (f"_rm{k}", lambda s: s.rolling(win, min_periods=1,
                                                     center=centre).mean()),
                     (f"_rs{k}", lambda s: s.rolling(win, min_periods=2,
                                                     center=centre).std()),
                     ("_slope", slope)):
        blk = g[feats].transform(fn)
        blk.columns = [c + name for c in feats]
        blocks.append(blk)
        new += list(blk.columns)

    df = pd.concat([df] + blocks, axis=1)
    df[new] = df[new].fillna(0.0)
    return df, new


# --------------------------------------------------------------------------- 8
def prune(df, feats, var_thresh=1e-8, corr_thresh=0.95):
    """Drop constant columns, then one of every |r| > thresh pair."""
    notes = []
    X = df[feats]

    nzv = [c for c in feats if X[c].var(ddof=0) <= var_thresh]
    keep = [c for c in feats if c not in nzv]
    if nzv:
        notes.append(f"near-zero variance dropped ({len(nzv)}): {nzv}")

    corr = np.asarray(X[keep].corr().abs().values, dtype=float).copy()
    corr[np.isnan(corr)] = 0.0
    np.fill_diagonal(corr, 0.0)
    order = np.argsort(-X[keep].var(ddof=0).values)   # keep the more variable
    dropped, kept_idx = [], []
    for i in order:
        if any(corr[i, j] > corr_thresh for j in kept_idx):
            dropped.append(keep[i])
        else:
            kept_idx.append(i)
    keep = [c for c in keep if c not in dropped]
    if dropped:
        notes.append(f"collinear |r|>{corr_thresh} dropped ({len(dropped)}): "
                     f"{sorted(dropped)}")
    return keep, notes


# --------------------------------------------------------------------------- 3.5
def winsorize_within_recording(df, feats, lower=0.01, upper=0.99):
    """Clip each feature to its OWN recording's [lower, upper] percentile -
    caps a genuine artifact's leverage without deleting the window.

    Computed per recording and blind to phase/is_cpt, so it cannot single
    out CPT windows even by accident: a real CPT responder only gets
    clipped if it is more extreme than the given percentile of windows in
    ITS OWN recording - artifact or physiological, the rule doesn't know
    which. Off by default: the default normalization (median/IQR) is
    already fairly resistant to a handful of extreme windows on its own;
    turn this on only if you can see a specific artifact spike survive
    normalization in the time-series plot.
    """
    df = df.copy()
    g = df.groupby("Recording", sort=False)
    lo = g[feats].transform(lambda s: s.quantile(lower))
    hi = g[feats].transform(lambda s: s.quantile(upper))
    n_clipped = int(((df[feats] < lo) | (df[feats] > hi)).to_numpy().sum())
    df[feats] = df[feats].clip(lower=lo, upper=hi)
    return df, n_clipped


# --------------------------------------------------------------------------- run
def engineer(ecg_csv, skna_csv, method="expanding", k=2, causal=True,
             corr_thresh=0.95, keep_raw=False, do_prune=True,
             winsorize=False, wins_lower=0.01, wins_upper=0.99,
             keep_s13="full"):
    notes = []
    df = load_merged(ecg_csv, skna_csv)
    notes.append(f"merged ECG+SKNA: {len(df)} windows, "
                 f"{df['Subject_ID'].nunique()} recordings")

    df = add_protocol(df)
    df, n = quality_filter(df, keep_s13=keep_s13); notes += n
    notes.append(f"phase counts: {df['phase'].value_counts().to_dict()}")

    feats = feature_columns(df)
    df, logged = log_transform(df, feats)
    notes.append(f"log1p applied ({len(logged)}): {logged}")

    df, inter = add_interactions(df, feats)
    feats = feats + inter
    notes.append(f"interaction features ({len(inter)}): {inter}")

    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    df[feats] = df.groupby("Recording")[feats].transform(
        lambda s: s.fillna(s.median()))
    df[feats] = df[feats].fillna(df[feats].median())

    if winsorize:
        df, n_clip = winsorize_within_recording(df, feats, wins_lower, wins_upper)
        pct = 100.0 * n_clip / max(len(df) * len(feats), 1)
        notes.append(f"winsorized within recording [{wins_lower:.1%}, "
                     f"{wins_upper:.1%}], blind to phase: {n_clip} cell(s) "
                     f"clipped ({pct:.2f}% of {len(feats)} features x "
                     f"{len(df)} windows)")

    suffix = "_n" if keep_raw else ""
    df, level = normalize_within_recording(df, feats, method=method,
                                           suffix=suffix)
    notes.append(f"within-recording normalization: method={method}, "
                 f"{len(level)} level features")

    df, temporal = add_temporal(df, level, k=k, causal=causal)
    notes.append(f"temporal features ({len(temporal)}): k={k}, "
                 f"causal={causal}, 4 x {len(level)} base")

    cols = level + temporal
    if do_prune:
        cols, n = prune(df, cols, corr_thresh=corr_thresh); notes += n
    notes.append(f"FINAL: {len(cols)} features x {len(df)} windows, "
                 f"{df['person'].nunique()} people")
    return df, cols, notes


# --------------------------------------------------------------------------- io
def write_outputs(df, cols, notes, out_dir, method, k, causal):
    """Write the engineered CSV + NPZ + report, exactly as the CLI does.

    Shared by main() and the GUI (feature_engineering_window.py) so both
    paths produce byte-identical output layouts. Returns the three paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    keep = [c for c in META if c in df.columns] + TARGETS + cols
    csv_path = os.path.join(out_dir, "features_30s_engineered.csv")
    df[keep].to_csv(csv_path, index=False)

    npz_path = os.path.join(out_dir, "features_30s_engineered.npz")
    np.savez_compressed(
        npz_path,
        X=df[cols].to_numpy(np.float32),
        feature_names=np.array(cols),
        y_sbp=df["SBP"].to_numpy(np.float32),
        y_dbp=df["DBP"].to_numpy(np.float32),
        y_cpt=df["is_cpt"].to_numpy(np.int8),
        phase=df["phase"].to_numpy(),
        groups=df["person"].to_numpy(),
        subject=df["Subject_ID"].to_numpy(),
        recording=df["Recording"].to_numpy(),
        t_center_sec=df["t_center_sec"].to_numpy(np.float32),
    )

    report = "\n".join(f"[{i + 1}] {n}" for i, n in enumerate(notes))
    rep_path = os.path.join(out_dir, "feature_engineering_report.txt")
    with open(rep_path, "w") as f:
        f.write(f"feature engineering  method={method} k={k} "
                f"causal={causal}\n\n{report}\n\nfeatures:\n"
                + "\n".join(f"  {c}" for c in cols) + "\n")
    return csv_path, npz_path, rep_path


def run(ecg_csv, skna_csv, out_dir, method="expanding", k=2, causal=True,
        corr_thresh=0.95, keep_raw=False, do_prune=True,
        winsorize=False, wins_lower=0.01, wins_upper=0.99):
    """One call that engineers + writes, for callers (CLI, GUI) that just
    want the finished files. Returns (df, cols, notes, csv_path, npz_path,
    rep_path)."""
    df, cols, notes = engineer(ecg_csv, skna_csv, method=method, k=k,
                               causal=causal, corr_thresh=corr_thresh,
                               keep_raw=keep_raw, do_prune=do_prune,
                               winsorize=winsorize, wins_lower=wins_lower,
                               wins_upper=wins_upper)
    csv_path, npz_path, rep_path = write_outputs(df, cols, notes, out_dir,
                                                 method, k, causal)
    return df, cols, notes, csv_path, npz_path, rep_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ecg", default="ecg_features_30s.csv")
    p.add_argument("--skna", default="skna_features_30s.csv")
    p.add_argument("--out", default="beat_pipeline/built")
    p.add_argument("--method", default="expanding",
                   choices=["robust", "expanding", "baseline", "zscore", "none"],
                   help="within-recording normalization (default expanding: "
                        "causal, uses only past windows of the recording)")
    p.add_argument("--k", type=int, default=2, help="temporal span in windows")
    p.add_argument("--acausal", action="store_true",
                   help="centred rolling windows (uses future windows)")
    p.add_argument("--corr-thresh", type=float, default=0.95)
    p.add_argument("--keep-raw", action="store_true",
                   help="keep un-normalized columns alongside normalized ones")
    p.add_argument("--no-prune", action="store_true")
    p.add_argument("--winsorize", action="store_true",
                   help="clip each feature to its own recording's "
                        "[wins-lower, wins-upper] percentile before "
                        "normalizing - caps artifact leverage, computed "
                        "blind to phase so it cannot target CPT windows")
    p.add_argument("--wins-lower", type=float, default=0.01)
    p.add_argument("--wins-upper", type=float, default=0.99)
    a = p.parse_args()

    df, cols, notes, csv_path, npz_path, rep_path = run(
        a.ecg, a.skna, a.out, method=a.method, k=a.k, causal=not a.acausal,
        corr_thresh=a.corr_thresh, keep_raw=a.keep_raw,
        do_prune=not a.no_prune, winsorize=a.winsorize,
        wins_lower=a.wins_lower, wins_upper=a.wins_upper)

    print("\n".join(f"[{i + 1}] {n}" for i, n in enumerate(notes)))
    print(f"\nwrote {csv_path}\n      {npz_path}\n      {rep_path}")


if __name__ == "__main__":
    main()