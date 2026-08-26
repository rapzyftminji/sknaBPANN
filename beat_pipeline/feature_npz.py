"""
Shared .npz packing for the window-feature tables.
==================================================
One format for every feature family (ecg_features, skna_features) so a
training script needs ONE loader, and so two tables built on the same
window/stride grid join row-for-row on (subject, window_idx).

    X                 (N, F) float32, columns in `feature_names` order
    feature_names     (F,)   the column meaning, stored so a stale file
                             cannot be silently mis-columned on load
    subject           (N,)   per RECORDING id
    person            (N,)   per PERSON id - the LOSO grouping key, and NOT
                             the same thing as subject (beat_labels.person_of)
    recording         (N,)   source .txt basename
    window_idx        (N,)   position within that recording
    t_start_sec       (N,)   window start on the ANALYSED span
    t_center_sec      (N,)   window centre on the analysed span
    t_raw_center_sec  (N,)   + clock_offset_sec, i.e. on the raw file clock
    SBP DBP           (N,)   mmHg, NaN where unlabelled
    label_valid       (N,)   a BP label was resolved for this window
    usable            (N,)   the training mask (see each family's builder)
    <extra_keys>      (N,)   family-specific QC columns
    window_sec stride_sec analysis_fs label_mode    scalars

Everything is per-window, flat and concatenated: a 16-subject table of 19
features is ~0.4 MB, so there is nothing to gain from the index-window
tricks the beat matrices need.
"""
import os

import numpy as np

BASE_ARRAY_KEYS = ("X", "subject", "person", "recording", "window_idx",
                   "t_start_sec", "t_center_sec", "t_raw_center_sec",
                   "SBP", "DBP", "label_valid", "usable")
SCALAR_KEYS = ("window_sec", "stride_sec", "analysis_fs", "label_mode")


def _scalar(v):
    """A packed scalar as a plain str/float, whether it came from a build
    result (python value) or from load() (0-d array)."""
    v = np.asarray(v)
    return str(v) if v.dtype.kind in "US" else float(v)


def window_arrays(r, extra_keys=()):
    """Per-window arrays from a build_recording() result, WITHOUT the
    recording-length signal arrays. Keeping those while packing a cohort is
    what would make this run out of memory.

    Recordings built with subject=None carry no labels, so SBP/DBP come back
    NaN and label_valid/usable all-False rather than absent - a packed file
    then has the same keys whatever it was built from.
    """
    if "recording" in r:      # already converted; keep the pack paths idempotent
        return r
    import beat_labels as bl
    n = len(r["X"])
    subj = r.get("subject") or ""
    # np.full, not np.array([v] * n): at n = 0 the list form collapses to
    # float64 and the string columns then refuse to concatenate.
    out = {
        "X": np.asarray(r["X"], dtype=np.float32),
        "subject": np.full(n, subj),
        "person": np.full(n, bl.person_of(subj) if subj else ""),
        "recording": np.full(n, os.path.basename(r["path"])),
        "window_idx": np.arange(n, dtype=np.int32),
        "t_start_sec": np.asarray(r["t_start_sec"], dtype=np.float64),
        "t_center_sec": np.asarray(r["t_center_sec"], dtype=np.float64),
        "t_raw_center_sec": np.asarray(r["t_center_sec"], dtype=np.float64)
        + r["clock_offset_sec"],
        "SBP": np.asarray(r.get("SBP", np.full(n, np.nan)), dtype=np.float64),
        "DBP": np.asarray(r.get("DBP", np.full(n, np.nan)), dtype=np.float64),
        "label_valid": np.asarray(r.get("label_valid", np.zeros(n, bool)), dtype=bool),
        "usable": np.asarray(r.get("usable", np.zeros(n, bool)), dtype=bool),
    }
    for k in extra_keys:
        a = np.asarray(r[k])
        out[k] = a.astype(np.int32) if a.dtype.kind in "iu" else a
    for k in SCALAR_KEYS:
        out[k] = r["analysis_fs"] if k == "analysis_fs" else r.get(k)
    return out


def save(out_path, results, feature_names, extra_keys=()):
    """Pack one build_recording() result, or a list of them, into one .npz.

    Refuses to mix windowing or analysis rates. Feature families are
    rate-dependent (Hjorth/FD for ECG; every count/length feature for SKNA)
    and a different window length makes a different variable, so a silently
    mixed file would train a model on two incompatible feature definitions -
    the failure would surface as unexplained variance, not as an error.
    """
    keys = BASE_ARRAY_KEYS + tuple(extra_keys)
    parts = [results] if isinstance(results, dict) else list(results)
    if not parts:
        raise ValueError("nothing to pack")
    parts = [window_arrays(p, extra_keys) for p in parts]

    meta = {}
    for k in SCALAR_KEYS:
        # _scalar(), not the raw value: parts may come straight from load(),
        # where every scalar is a 0-d array - unhashable, so the set below
        # would raise instead of comparing them.
        vals = {_scalar(p[k]) for p in parts if p[k] is not None}
        if len(vals) > 1:
            raise ValueError(
                f"cannot pack: recordings disagree on {k} ({sorted(vals)}). "
                f"Features are only comparable within one windowing/analysis "
                f"rate - build them the same way or pack separate files.")
        meta[k] = vals.pop() if vals else np.nan

    arrays = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    np.savez_compressed(out_path, feature_names=np.array(feature_names),
                        **arrays, **meta)
    return arrays["X"].shape


def load(path, feature_names, extra_keys=(), usable_only=False):
    """Load a packed feature table for training.

    `usable_only=True` applies the `usable` mask up front, which is what a
    training run wants. Group LOSO folds on `person`, NOT on `subject`.
    Standardise X with statistics fitted on the TRAINING fold only - these
    features have wildly different units, so an unscaled fit is dominated by
    whichever columns happen to be numerically largest.
    """
    keys = BASE_ARRAY_KEYS + tuple(extra_keys)
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    names = [str(s) for s in out["feature_names"]]
    if names != list(feature_names):
        raise ValueError(
            f"{os.path.basename(path)} was built with a different feature set "
            f"or column order ({len(names)} columns, expected "
            f"{len(feature_names)}). Rebuild it - the X columns would not mean "
            f"what the loader says they mean.")
    out["feature_names"] = names
    for k in SCALAR_KEYS:
        if k in out:                      # 0-d arrays -> plain str/float
            out[k] = _scalar(out[k])
    for k in ("subject", "person", "recording"):
        out[k] = out[k].astype(str)
    if usable_only:
        m = out["usable"]
        for k in keys:
            out[k] = out[k][m]
    return out
