"""
BP labelling for beat segments (stage 2b)
=========================================

Attaches SBP/DBP to each 3-R-peak segment.

WHERE THE LABELS COME FROM
    feature_result/<window>/<subject>/windowing_labeling/bp_labels.csv,
    which holds SBP/DBP per 5 s window with Time_start_min / Time_end_min
    ALREADY ON THE ECG CLOCK. That file is the output of the main app's
    alignment step (src/core_processing.align_bp_to_ecg, HR cross-
    correlation), and it is what produced the datasets behind training
    runs 13-21.

    Reusing it - rather than re-deriving a lag here - is deliberate. It
    keeps beat-level labels consistent with the existing 5 s results, so a
    difference in model performance is attributable to the representation
    rather than to two different alignments. It also avoids re-running an
    alignment that is known to be fragile on these recordings.

    The consequence: these labels are only as good as that alignment, and
    the alignment inherits the finger-NIBP problems already documented for
    this dataset. Beat-level labels do NOT make the reference better; they
    only stop the representation from being the limiting factor.

TIME BASES - THE THING THAT MUST LINE UP
    bp_labels.csv is on the RAW file clock (window 0 starts at t=0 of the
    .txt). Beat times from build_recording() are relative to the start of
    the ANALYSED span, which has had the FIR delay, any --start, and the
    lead-in transient trimmed off the front. run_pipeline() reports the sum
    of those as clock_offset_sec, so:

        raw_file_time = clock_offset_sec + beat_time

RESOLUTION MISMATCH
    BP is one value per 5 s; a beat is ~0.8 s. Several consecutive beats
    therefore share a label. That is inherent to the reference device (the
    finger cuff reports at ~0.8 s and was averaged into 5 s windows), not
    something the beat representation introduces - but it does mean the
    effective number of INDEPENDENT labels is ~1/6 of the beat count, which
    matters when reading any beat-level score.
"""
import os
import re

import numpy as np
import pandas as pd

# Subject -> source recording. VERIFIED, not assumed: each subject's
# raw_signal.npz window 0 was correlated against the first 5 s of every
# .txt in dataset/txt. Every subject matched its recording at r >= 0.95
# (s7/Jose at 0.58, still 2.9x its runner-up), and the result agrees with
# the mapping recorded in the project notes.
SUBJECT_RECORDING = {
    "s1":  "BP_SKNA_biopac_10kHz.txt",
    "s2":  "SKNA_BP_arthur_10kHz.txt",
    "s3":  "SKNA_BP_becky_10kHz.txt",
    "s4":  "SKNA_BP_eric_10kHz.txt",
    "s5_session1": "SKNA_BP_cindy_10kHz.txt",
    "s5_session2": "SKNA_BP_cindy2_10kHz.txt",
    "s6":  "SKNA_BP_alice_10kHz.txt",
    "s7":  "SKNA_BP_Jose.txt",
    "s8":  "SKNA_BP_Rizmi.txt",
    "s9":  "SKNA_BP_Nhu.txt",
    "s10": "SKNA_BP_Andreas.txt",
    "s11": "SKNA_BP_Tsai.txt",
    "s12": "SKNA_BP_HK.txt",
    # s13 is the 10-min CUT of Tseng2 (120 BP windows); s13_full is the
    # 32-min original (385 windows). Both start at the same instant -
    # window 0 of each correlates 1.0000 with both files.
    "s13": "SKNA_BP_Tseng2_cut.txt",
    "s13_full": "SKNA_BP_Tseng2.txt",
    "s14": "SKNA_BP_Eva.txt",
}

# s5 is excluded from the LOSO runs in this project (see the training logs:
# "Excluded subjects ['s5']"). It is still built here so the decision stays
# the caller's, but it is flagged.
EXCLUDED_SUBJECTS = ("s5_session1", "s5_session2")


def person_of(subject):
    """Person behind a subject ID - the correct grouping key for LOSO.

    Subject IDs are per RECORDING, and two of them are not distinct people:
    s13 is a 10-min cut of s13_full (same person, same instant, so the cut's
    windows are literally a subset of the full one's), and s5_session1/2 are
    two sessions of one person. Splitting on subject ID would put the same
    person in train and test - for s13/s13_full, the same windows - and the
    held-out score would be an inflated within-person score.
    """
    s = str(subject)
    for suf in ("_full", "_cut"):
        if s.endswith(suf):
            s = s[:-len(suf)]
    s = re.sub(r"_session\d+$", "", s)
    return s


def load_bp_labels(subject, feature_root="feature_result/5sWindow"):
    """Read one subject's bp_labels.csv and return window CENTRE times (s)
    on the raw-file clock, with SBP/DBP."""
    f = os.path.join(feature_root, subject, "windowing_labeling", "bp_labels.csv")
    if not os.path.isfile(f):
        raise FileNotFoundError(f)
    b = pd.read_csv(f)
    t = (b["Time_start_min"].to_numpy() + b["Time_end_min"].to_numpy()) / 2.0 * 60.0
    return t, b["SBP"].to_numpy(dtype=float), b["DBP"].to_numpy(dtype=float)


def labels_for_beats(beat_time_sec, clock_offset_sec, t_bp, sbp, dbp,
                     mode="interp", max_gap_sec=10.0):
    """SBP/DBP per beat.

    mode='interp'  linear interpolation between 5 s window centres. Smoother
                   and closer to the device's own ~0.8 s cadence than a step.
    mode='window'  the value of the 5 s window the beat falls in - exactly
                   what the existing 5 s dataset would have assigned.

    Beats further than `max_gap_sec` from any valid BP sample are marked
    invalid rather than extrapolated: bp_labels.csv carries NaN runs where
    the cuff dropped out (up to 37 windows = 3 min for s4), and
    interpolating across those would invent readings.
    """
    t_beat = np.asarray(beat_time_sec, dtype=float) + float(clock_offset_sec)
    ok = np.isfinite(sbp) & np.isfinite(dbp)
    if not ok.any():
        n = len(t_beat)
        return (np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=bool))
    tv, sv, dv = t_bp[ok], sbp[ok], dbp[ok]

    if mode == "window":
        idx = np.searchsorted(tv, t_beat)
        idx = np.clip(idx, 0, len(tv) - 1)
        left = np.clip(idx - 1, 0, len(tv) - 1)
        pick = np.where(np.abs(tv[left] - t_beat) <= np.abs(tv[idx] - t_beat), left, idx)
        s_out, d_out = sv[pick], dv[pick]
        gap = np.abs(tv[pick] - t_beat)
    else:
        s_out = np.interp(t_beat, tv, sv)
        d_out = np.interp(t_beat, tv, dv)
        nearest = np.clip(np.searchsorted(tv, t_beat), 0, len(tv) - 1)
        left = np.clip(nearest - 1, 0, len(tv) - 1)
        gap = np.minimum(np.abs(tv[nearest] - t_beat), np.abs(tv[left] - t_beat))

    valid = (gap <= max_gap_sec) & (t_beat >= tv[0] - max_gap_sec) \
        & (t_beat <= tv[-1] + max_gap_sec)
    s_out = np.where(valid, s_out, np.nan)
    d_out = np.where(valid, d_out, np.nan)
    return s_out, d_out, valid


def attach_labels(built, subject, feature_root="feature_result/5sWindow",
                  mode="interp", max_gap_sec=10.0):
    """Add per-beat SBP/DBP to a build_recording() result, in place.

    Sequences containing any unlabelled beat are dropped from seq_idx - a
    sequence with a missing target cannot be trained on.
    """
    t_bp, sbp, dbp = load_bp_labels(subject, feature_root)
    s, d, valid = labels_for_beats(built["beat_time_sec"],
                                   built["stage1"]["clock_offset_sec"],
                                   t_bp, sbp, dbp, mode=mode,
                                   max_gap_sec=max_gap_sec)
    built["SBP"], built["DBP"] = s, d
    built["label_valid"] = valid
    built["subject"] = subject
    built["label_mode"] = mode

    usable = valid & built["keep"]
    built["usable"] = usable
    if len(built["seq_idx"]):
        built["seq_idx"] = built["seq_idx"][usable[built["seq_idx"]].all(axis=1)]
    return built


def format_label_report(built):
    s, d = built["SBP"], built["DBP"]
    v = built["label_valid"]
    n = len(s)
    L = [f"labels          bp_labels.csv ({built['label_mode']}), "
         f"clock offset {built['stage1']['clock_offset_sec']:.2f} s",
         f"  labelled      {int(v.sum())}/{n} beats"
         + (f"   ({n - int(v.sum())} outside BP coverage)" if v.sum() < n else "")]
    if v.any():
        L += [f"  SBP           {np.nanmin(s):.1f} - {np.nanmax(s):.1f} mmHg "
              f"(mean {np.nanmean(s):.1f}, sd {np.nanstd(s):.1f})",
              f"  DBP           {np.nanmin(d):.1f} - {np.nanmax(d):.1f} mmHg "
              f"(mean {np.nanmean(d):.1f}, sd {np.nanstd(d):.1f})"]
    L.append(f"  usable beats  {int(built['usable'].sum())} "
             f"(labelled AND segment-duration valid)")
    L.append(f"  sequences     {len(built['seq_idx'])} fully-labelled of M={built['M']}")
    return "\n".join(L)
