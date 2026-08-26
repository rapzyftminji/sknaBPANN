"""
19 handcrafted ECG features on sliding windows, with BP labels.
==============================================================

Feature set as specified, grouped by source:

  complexity / Simjanoska et al. (Inf. Fusion 58:24-39 2020; Sensors 18:1160 2018)
     1 mobility            Hjorth mobility
     2 complexity          Hjorth complexity
     3 fractal_dimension   Higuchi FD
     4 entropy             Shannon entropy of the amplitude histogram
     5 autocorrelation     normalised autocorrelation at a 50 ms lag

  HRV / Task Force ESC-NASPE (Circulation 93:1043-1065 1996)
     6 meanNN  (ms)        7 SDNN (ms)          8 RMSSD (ms)
     9 LF_power (ms^2)    10 HF_power (ms^2)   11 LF_HF_ratio
    12 SD1_SD2             Poincare, geometric method

  repolarisation
    13 TpTe (ms)           Bombelli et al., J Hypertens 34:1823-1830 2016
    14 QTc_Fridericia (ms) Palatini et al., Arch Intern Med 166:909-915 2006

  morphology / Mousavi et al. (Physiol. Meas. 2025)
    15 QRS_duration (ms)  16 ST_level (mV)     17 T_R_ratio
    18 R_slope (mV/s)     19 beat_SNR (dB)     <- also the quality screen

TWO DECISIONS THAT AFFECT EVERY FEATURE
---------------------------------------
FIXED ANALYSIS RATE. Hjorth mobility/complexity and Higuchi FD all scale with
the sample rate - on one 30 s window mobility measured 0.0077 at 10 kHz and
0.385 at 125 Hz. Everything here is therefore computed at ANALYSIS_FS after
anti-aliased decimation, so features are comparable across recordings and
between the 2 kHz and 10 kHz copies of the same recording. R-peaks are still
detected and refined at native rate, then mapped onto the analysis grid.

LF ON A 30 s WINDOW IS MARGINAL. The Task Force specifies >=2 min for LF
(0.04-0.15 Hz); the slowest LF component completes only 1.2 cycles in 30 s.
LF_power and LF_HF_ratio are computed, but their low-frequency edge rests on
very few degrees of freedom. HF (0.15-0.40 Hz, 4.5-12 cycles) is fine.
`lf_reliable` is reported per window so this can be filtered rather than
silently trusted.

PER-BEAT THEN AGGREGATE. Features 13-19 need beat delineation (Q onset,
S offset/J point, T peak, T end). Single-lead delineation is noisy per beat,
so every beat in the recording is delineated ONCE and each window takes the
MEDIAN over the beats inside it - robust to the occasional mis-delineation,
and much cheaper than re-delineating for every overlapping window.
"""
import os

import numpy as np
from scipy import interpolate, signal

import beat_processing as bp
import feature_npz as fnpz

ANALYSIS_FS = 500.0          # Hz, fixed - see module docstring
DEFAULT_WINDOW_SEC = 30.0
DEFAULT_STRIDE_SEC = 5.0

FEATURE_NAMES = [
    "mobility", "complexity", "fractal_dimension", "entropy", "autocorrelation",
    "meanNN", "SDNN", "RMSSD", "LF_power", "HF_power", "LF_HF_ratio", "SD1_SD2",
    "TpTe", "QTc_Fridericia",
    "QRS_duration", "ST_level", "T_R_ratio", "R_slope", "beat_SNR",
]
assert len(FEATURE_NAMES) == 19

# Band-limited features that need a longer window than the LF band gets here.
# They are governed by the per-window lf_reliable flag, not by `usable` - see
# build_recording().
LF_FEATURES = ("LF_power", "LF_HF_ratio")
_LF_COLS = [FEATURE_NAMES.index(n) for n in LF_FEATURES]


# ---------------------------------------------------------------------------
# 1-5  signal complexity
# ---------------------------------------------------------------------------
def hjorth(x):
    """(mobility, complexity). Mobility = sqrt(var(x')/var(x));
    complexity = mobility(x')/mobility(x)."""
    x = np.asarray(x, dtype=np.float64)
    d1 = np.diff(x)
    d2 = np.diff(d1)
    v0, v1, v2 = x.var(), d1.var(), d2.var()
    if v0 <= 0 or v1 <= 0:
        return np.nan, np.nan
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / mob if v2 > 0 else np.nan
    return float(mob), float(comp)


def higuchi_fd(x, kmax=10):
    """Higuchi fractal dimension: slope of log(L(k)) against log(1/k)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2 * kmax:
        return np.nan
    lnL, lnk = [], []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            # NOTE the k**2: normalising by (len-1)*k alone yields the slope
            # of log L vs log(1/k) as FD-1, so a straight line scored 0.0 and
            # white noise 1.0. With k**2 the definition returns FD directly -
            # validated at 1.00 (line), 2.00 (white noise), 1.50 (Brownian).
            Lmk = np.abs(np.diff(x[idx])).sum() * (n - 1) / ((len(idx) - 1) * k * k)
            Lk.append(Lmk)
        if Lk:
            mean_Lk = np.mean(Lk)
            if mean_Lk > 0:
                lnL.append(np.log(mean_Lk))
                lnk.append(np.log(1.0 / k))
    if len(lnL) < 2:
        return np.nan
    return float(np.polyfit(lnk, lnL, 1)[0])


def shannon_entropy(x, bins=64):
    """Shannon entropy (bits) of the amplitude histogram."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2 or not np.isfinite(x).all() or np.ptp(x) <= 0:
        return np.nan
    p, _ = np.histogram(x, bins=bins)
    p = p[p > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def autocorrelation(x, fs=ANALYSIS_FS, lag_ms=50.0):
    """Normalised autocorrelation at `lag_ms` milliseconds.

    Specified in TIME, not samples, for two reasons. A lag of one SAMPLE is
    degenerate on oversampled ECG - at 500 Hz it measured 0.9885 with a
    standard deviation of 0.0012 across windows, i.e. no usable variance -
    and a sample lag would also make the feature depend on the analysis rate.
    At 50 ms the same windows spread over sd 0.10.
    """
    x = np.asarray(x, dtype=np.float64)
    lag = max(1, int(round(lag_ms / 1000.0 * fs)))
    if len(x) <= lag:
        return np.nan
    x = x - x.mean()
    d = (x * x).sum()
    if d <= 0:
        return np.nan
    return float((x[:-lag] * x[lag:]).sum() / d)


# ---------------------------------------------------------------------------
# 6-12  HRV
# ---------------------------------------------------------------------------
def hrv_time(nn_ms):
    """meanNN, SDNN, RMSSD in ms (Task Force definitions)."""
    nn = np.asarray(nn_ms, dtype=np.float64)
    if len(nn) < 2:
        return np.nan, np.nan, np.nan
    return (float(nn.mean()), float(nn.std(ddof=1)),
            float(np.sqrt(np.mean(np.diff(nn) ** 2))))


def hrv_freq(nn_ms, t_beat_sec, resample_fs=4.0,
             lf=(0.04, 0.15), hf=(0.15, 0.40)):
    """LF power, HF power, LF/HF (ms^2) from the NN tachogram.

    The tachogram is irregularly sampled, so it is cubic-interpolated onto a
    regular `resample_fs` grid, linearly detrended, then Welch-averaged -
    the conventional route. Returns (LF, HF, LF/HF, reliable) where
    `reliable` is False when the window is too short to resolve the LF band's
    lower edge (needs ~2 min; see module docstring).
    """
    nn = np.asarray(nn_ms, dtype=np.float64)
    t = np.asarray(t_beat_sec, dtype=np.float64)
    if len(nn) < 4 or len(t) != len(nn):
        return np.nan, np.nan, np.nan, False
    dur = t[-1] - t[0]
    if dur <= 0:
        return np.nan, np.nan, np.nan, False

    grid = np.arange(t[0], t[-1], 1.0 / resample_fs)
    if len(grid) < 8:
        return np.nan, np.nan, np.nan, False
    kind = "cubic" if len(nn) >= 4 else "linear"
    y = interpolate.interp1d(t, nn, kind=kind, bounds_error=False,
                             fill_value=(nn[0], nn[-1]))(grid)
    y = signal.detrend(y, type="linear")

    nperseg = min(len(y), int(round(resample_fs * dur)))
    f, P = signal.welch(y, fs=resample_fs, nperseg=nperseg,
                        noverlap=nperseg // 2, scaling="density")
    band = lambda a, b: float(np.trapezoid(P[(f >= a) & (f < b)],
                                           f[(f >= a) & (f < b)]))
    lf_p, hf_p = band(*lf), band(*hf)
    ratio = lf_p / hf_p if hf_p > 0 else np.nan
    reliable = dur >= 2.0 / lf[0]          # >=2 cycles of the slowest LF term
    return lf_p, hf_p, ratio, bool(reliable)


def poincare(nn_ms):
    """(SD1, SD2, SD1/SD2). SD1 = sqrt(0.5)*SD(dNN); SD2 from SDNN and SD1."""
    nn = np.asarray(nn_ms, dtype=np.float64)
    if len(nn) < 3:
        return np.nan, np.nan, np.nan
    sd1 = np.sqrt(0.5) * np.std(np.diff(nn), ddof=1)
    var2 = 2.0 * np.var(nn, ddof=1) - sd1 ** 2
    sd2 = np.sqrt(var2) if var2 > 0 else np.nan
    ratio = sd1 / sd2 if (sd2 and np.isfinite(sd2) and sd2 > 0) else np.nan
    return float(sd1), float(sd2), float(ratio) if np.isfinite(ratio) else np.nan


# ---------------------------------------------------------------------------
# 13-19  per-beat delineation and morphology
# ---------------------------------------------------------------------------
def delineate_beats(ecg, fs, rpeaks, upright=True, slope_frac=0.08):
    """Delineate every beat once. Returns a dict of per-beat arrays (NaN where
    a landmark could not be found).

    Rule-based, single lead, with RR-scaled search windows:
      baseline   median of the PR segment, 100-40 ms before R
      Q onset    back from R: Q trough, then to where the slope flattens
      S offset   forward from R: S trough, then to where the slope flattens
                 (the J point)
      ST level   amplitude at J+60 ms relative to baseline (standard)
      T peak     largest |deflection| in [J+40 ms, R+min(0.6*RR, 500 ms)]
      T end      TANGENT method: steepest point on the T downslope, tangent
                 extended to baseline. More stable than a fixed threshold.

    Per-beat delineation on one lead is noisy; callers should aggregate with a
    median over a window rather than trusting individual beats.
    """
    n = len(ecg)
    out = {k: np.full(len(rpeaks), np.nan) for k in
           ("baseline", "q_onset", "s_offset", "t_peak", "t_end",
            "qrs_dur", "st_level", "t_r_ratio", "r_amp", "tpte", "qt", "snr")}
    if len(rpeaks) < 2:
        return out

    rr = np.diff(rpeaks) / fs
    rr = np.concatenate([[rr[0]], rr])          # RR preceding each beat
    s_pr0, s_pr1 = int(0.100 * fs), int(0.040 * fs)
    s_q, s_s = int(0.080 * fs), int(0.120 * fs)
    s_st = int(0.060 * fs)
    flat = max(1, int(0.004 * fs))

    for i, r in enumerate(rpeaks):
        if r - s_pr0 < 0 or r + int(0.6 * fs) >= n:
            continue
        base = float(np.median(ecg[r - s_pr0:r - s_pr1]))
        out["baseline"][i] = base
        sgn = 1.0 if upright else -1.0
        out["r_amp"][i] = (ecg[r] - base) * sgn

        # --- QRS onset / offset by SLOPE, not by return-to-baseline.
        # Walking out to where the trace re-crosses the baseline puts the J
        # point well into the ST segment and gave a median QRS of 133 ms
        # (normal 80-120). The standard criterion is where the slope decays
        # to a small fraction of the peak QRS slope, which is what is used
        # here (slope_frac of the largest |dECG/dt| inside the complex).
        lo = max(0, r - s_q)
        hi = min(n - 1, r + s_s)
        d = np.abs(np.diff(ecg[lo:hi]))
        if d.size < 3:
            continue
        thr = slope_frac * d.max()

        q_trough = lo + int(np.argmin(sgn * (ecg[lo:r] - base))) if r > lo else r
        j = q_trough - lo
        while j > 1 and d[j - 1] > thr:
            j -= 1
        q_on = lo + j
        out["q_onset"][i] = q_on

        s_trough = r + int(np.argmin(sgn * (ecg[r:hi] - base))) if hi > r else r
        j = s_trough - lo
        while j < len(d) - 1 and d[j] > thr:
            j += 1
        j_pt = lo + j
        out["s_offset"][i] = j_pt
        out["qrs_dur"][i] = (j_pt - q_on) / fs * 1000.0

        # --- ST level at J+60 ms
        st_i = j_pt + s_st
        if st_i < n:
            out["st_level"][i] = (ecg[st_i] - base) * sgn

        # --- T peak
        t0 = j_pt + int(0.040 * fs)
        t1 = min(n - 1, r + int(min(0.6 * rr[i], 0.5) * fs))
        if t1 - t0 > int(0.02 * fs):
            seg = (ecg[t0:t1] - base) * sgn
            tp = t0 + int(np.argmax(np.abs(seg)))
            out["t_peak"][i] = tp
            t_amp = (ecg[tp] - base) * sgn
            if out["r_amp"][i] not in (0.0,) and np.isfinite(out["r_amp"][i]) \
                    and out["r_amp"][i] != 0:
                out["t_r_ratio"][i] = t_amp / out["r_amp"][i]

            # --- T end by the tangent method
            d1 = np.diff(ecg[tp:t1])
            if len(d1) > 2:
                k = int(np.argmax(-sgn * d1))          # steepest downslope
                slope = d1[k] / (1.0 / fs)
                if slope != 0:
                    x0, y0 = tp + k, ecg[tp + k]
                    t_end = x0 + (base - y0) / slope * fs
                    if tp < t_end < t1 + int(0.1 * fs):
                        out["t_end"][i] = t_end
                        out["tpte"][i] = (t_end - tp) / fs * 1000.0
                        out["qt"][i] = (t_end - q_on) / fs * 1000.0

        # --- beat SNR: QRS amplitude against isoelectric PR-segment noise
        pr = ecg[r - s_pr0:r - s_pr1]
        nz = float(np.std(pr))
        qrs_amp = float(np.ptp(ecg[max(0, r - s_q):min(n, r + s_s)]))
        if nz > 0 and qrs_amp > 0:
            out["snr"][i] = 20.0 * np.log10(qrs_amp / nz)
    return out


def qtc_fridericia(qt_ms, rr_sec):
    """QT / RR^(1/3) - Fridericia correction."""
    qt = np.asarray(qt_ms, dtype=np.float64)
    rr = np.asarray(rr_sec, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = qt / np.cbrt(rr)
    return np.where(np.isfinite(out) & (rr > 0), out, np.nan)


# ---------------------------------------------------------------------------
# window assembly
# ---------------------------------------------------------------------------
def _nanmedian(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def window_features(ecg_a, fs_a, rp_a, beats, win_i0, win_i1,
                    hr_min=30.0, hr_max=200.0):
    """The 19-feature vector for one window, plus diagnostics."""
    seg = ecg_a[win_i0:win_i1]
    feats = dict.fromkeys(FEATURE_NAMES, np.nan)

    mob, comp = hjorth(seg)
    feats["mobility"], feats["complexity"] = mob, comp
    feats["fractal_dimension"] = higuchi_fd(seg)
    feats["entropy"] = shannon_entropy(seg)
    feats["autocorrelation"] = autocorrelation(seg, fs_a)

    # beats fully inside the window
    sel = np.nonzero((rp_a >= win_i0) & (rp_a < win_i1))[0]
    n_beats = len(sel)
    lf_ok = False
    if n_beats >= 3:
        rp_w = rp_a[sel]
        nn = np.diff(rp_w) / fs_a * 1000.0
        hr = 60000.0 / nn
        good = (hr >= hr_min) & (hr <= hr_max)
        nn = nn[good]
        t_nn = rp_w[1:][good] / fs_a
        if len(nn) >= 2:
            feats["meanNN"], feats["SDNN"], feats["RMSSD"] = hrv_time(nn)
            lfp, hfp, ratio, lf_ok = hrv_freq(nn, t_nn)
            feats["LF_power"], feats["HF_power"], feats["LF_HF_ratio"] = lfp, hfp, ratio
            _, _, feats["SD1_SD2"] = poincare(nn)

        # morphology: median over the beats in this window
        feats["TpTe"] = _nanmedian(beats["tpte"][sel])
        feats["QRS_duration"] = _nanmedian(beats["qrs_dur"][sel])
        feats["ST_level"] = _nanmedian(beats["st_level"][sel])
        feats["T_R_ratio"] = _nanmedian(beats["t_r_ratio"][sel])
        feats["beat_SNR"] = _nanmedian(beats["snr"][sel])

        rr_prev = np.concatenate([[np.nan], np.diff(rp_w) / fs_a])
        feats["QTc_Fridericia"] = _nanmedian(
            qtc_fridericia(beats["qt"][sel], rr_prev))

        # R-peak amplitude SLOPE: trend of R amplitude across the window
        amp = beats["r_amp"][sel]
        tt = rp_w / fs_a
        ok = np.isfinite(amp)
        if ok.sum() >= 3:
            feats["R_slope"] = float(np.polyfit(tt[ok] - tt[ok][0], amp[ok], 1)[0])

    return feats, n_beats, lf_ok


def build_recording(path, subject=None, window_sec=DEFAULT_WINDOW_SEC,
                    stride_sec=DEFAULT_STRIDE_SEC, analysis_fs=ANALYSIS_FS,
                    ecg_channel="CH40", feature_root="feature_result/5sWindow",
                    label_mode="interp", max_gap_sec=10.0, amp_norm="none",
                    **pipe_kwargs):
    """Sliding-window features for one recording, with BP labels if `subject`
    is given.

    amp_norm='recording' scales the whole recording to a unit-SD ECG before the
    features are taken, so the amplitude-dependent ones (ST_level, R_slope) stop
    carrying each recording's gain. Applied AFTER run_pipeline, so the R-peaks
    were detected on the un-normalised signal and beat detection is bit-identical
    either way - the only thing that moves is the amplitude features. Timing
    features (meanNN, SDNN, RMSSD, QTc, ...) are scale-free and never move.
    """
    res = bp.run_pipeline(path, channel=ecg_channel, **pipe_kwargs)
    fs, ecg, rp = res["fs"], res["filtered"], res["rpeaks"]

    # one anti-aliased decimation to the fixed analysis grid
    q = max(1, int(round(fs / analysis_fs)))
    ecg_a = signal.resample_poly(ecg, 1, q) if q > 1 else ecg.copy()
    fs_a = fs / q

    amp_scale = 1.0
    if amp_norm == "recording":
        sd = float(np.std(ecg_a))
        if sd > 0:
            amp_scale = 1.0 / sd
            ecg_a = ecg_a * amp_scale
    elif amp_norm != "none":
        raise ValueError(f"unknown amp_norm {amp_norm!r}")
    rp_a = np.clip((rp / q).round().astype(np.int64), 0, len(ecg_a) - 1)

    beats = delineate_beats(ecg_a, fs_a, rp_a,
                            upright=res["stages"].get("upright", True))

    wl, st = int(round(window_sec * fs_a)), int(round(stride_sec * fs_a))
    starts = np.arange(0, max(0, len(ecg_a) - wl + 1), st, dtype=np.int64)

    X = np.full((len(starts), 19), np.nan, dtype=np.float64)
    n_beats = np.zeros(len(starts), dtype=np.int32)
    lf_ok = np.zeros(len(starts), dtype=bool)
    for k, i0 in enumerate(starts):
        f, nb, ok = window_features(ecg_a, fs_a, rp_a, beats, i0, i0 + wl)
        X[k] = [f[n] for n in FEATURE_NAMES]
        n_beats[k], lf_ok[k] = nb, ok

    t_center = (starts + wl / 2.0) / fs_a          # window centre, analysed span
    out = {
        "path": path, "subject": subject, "fs": fs, "analysis_fs": fs_a,
        "X": X, "feature_names": FEATURE_NAMES,
        "window_sec": window_sec, "stride_sec": stride_sec,
        "t_start_sec": starts / fs_a, "t_center_sec": t_center,
        "n_beats": n_beats, "lf_reliable": lf_ok,
        "amp_norm": amp_norm, "amp_scale": amp_scale,
        "analysis_fs_nominal": analysis_fs, "label_mode": label_mode,
        "clock_offset_sec": res["clock_offset_sec"],
        "n_rpeaks": len(rp), "beats": beats,
        # kept so a front-end can redraw the delineation without recomputing
        "ecg_a": ecg_a, "rp_a": rp_a,
    }

    if subject is not None:
        import beat_labels as bl
        t_bp, sbp, dbp = bl.load_bp_labels(subject, feature_root)
        s, d, valid = bl.labels_for_beats(t_center, res["clock_offset_sec"],
                                          t_bp, sbp, dbp, mode=label_mode,
                                          max_gap_sec=max_gap_sec)
        out["SBP"], out["DBP"], out["label_valid"] = s, d, valid
        # The LF band carries its own lf_reliable flag and is dropped outright
        # by feature_engineering.quality_filter, so it must not ALSO veto
        # usability: a window is not bad data merely because it is too short to
        # resolve 0.04-0.15 Hz. At 30 s this changes nothing (LF_power and
        # LF_HF_ratio are finite in all 5495 windows); at 5 s LF_HF_ratio is
        # NaN everywhere and would otherwise mark every single window unusable.
        core = np.delete(X, _LF_COLS, axis=1) if _LF_COLS else X
        out["usable"] = valid & (n_beats >= 3) & np.isfinite(core).all(axis=1)
    return out


# ---------------------------------------------------------------------------
# packing for training  (format and guards live in feature_npz, shared with
# skna_features so both tables load the same way and join row-for-row)
# ---------------------------------------------------------------------------
EXTRA_KEYS = ("n_beats", "lf_reliable")
ARRAY_KEYS = fnpz.BASE_ARRAY_KEYS + EXTRA_KEYS
SCALAR_KEYS = fnpz.SCALAR_KEYS


def window_arrays(r):
    """Per-window arrays only - drops ecg_a/rp_a/beats, which are the
    recording-length arrays."""
    return fnpz.window_arrays(r, EXTRA_KEYS)


def save_npz(out_path, results):
    """Pack one build_recording() result, or a list of them, into one .npz."""
    return fnpz.save(out_path, results, FEATURE_NAMES, EXTRA_KEYS)


def load_npz(path, usable_only=False):
    """Load a packed 19-feature table. `usable` = labelled AND >=3 beats AND
    all 19 features finite. Group LOSO on `person`, not `subject`."""
    return fnpz.load(path, FEATURE_NAMES, EXTRA_KEYS, usable_only)


def format_report(r):
    X, nb = r["X"], r["n_beats"]
    L = ["=" * 66, " ECG WINDOW FEATURES", "=" * 66,
         f"file            {os.path.basename(r['path'])}"
         + (f"   subject {r['subject']}" if r.get("subject") else ""),
         f"native fs       {r['fs']:.0f} Hz -> analysis {r['analysis_fs']:.0f} Hz "
         f"(fixed; Hjorth/FD are rate-dependent)",
         f"R-peaks         {r['n_rpeaks']}",
         f"windows         {len(X)}  ({r['window_sec']:.0f} s, stride "
         f"{r['stride_sec']:.0f} s)",
         f"beats/window    median {np.median(nb):.0f}  min {nb.min()}  max {nb.max()}",
         f"LF reliable     {int(r['lf_reliable'].sum())}/{len(X)} windows "
         f"(needs >=50 s for the LF lower edge)",
         "", f"{'#':>3s} {'feature':20s} {'valid':>7s} {'median':>11s} "
         f"{'p5':>11s} {'p95':>11s}"]
    for i, nm in enumerate(r["feature_names"]):
        v = X[:, i]
        f = v[np.isfinite(v)]
        if f.size:
            L.append(f"{i + 1:3d} {nm:20s} {100 * f.size / len(v):6.1f}% "
                     f"{np.median(f):11.4g} {np.percentile(f, 5):11.4g} "
                     f"{np.percentile(f, 95):11.4g}")
        else:
            L.append(f"{i + 1:3d} {nm:20s} {'0.0%':>7s} {'ALL NaN':>11s}")
    if "SBP" in r:
        v = r["label_valid"]
        L += ["", f"labels          {int(v.sum())}/{len(v)} windows labelled",
              f"  SBP           {np.nanmean(r['SBP']):.1f} +/- {np.nanstd(r['SBP']):.1f} mmHg",
              f"  DBP           {np.nanmean(r['DBP']):.1f} +/- {np.nanstd(r['DBP']):.1f} mmHg",
              f"  usable        {int(r['usable'].sum())} windows "
              f"(labelled AND >=3 beats AND all 19 features finite)"]
    return "\n".join(L)
