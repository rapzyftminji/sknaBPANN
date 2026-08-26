"""
11 handcrafted SKNA features on sliding windows, with BP labels.
===============================================================
Same window/stride grid and same packed .npz format as ecg_features.py, so
the two tables join row-for-row on (subject, window_idx).

    (a) aSKNA    (uV)    mean of the RECTIFIED SKNA - "average SKNA"
    (b) varSKNA  (uV^2)  variance, dispersion about the mean
    (c) rmsSKNA  (uV)    root mean square amplitude
    (d) skewSKNA         skewness of the amplitude distribution
    (e) kurtSKNA         excess kurtosis (Fisher; 0 = Gaussian), peakedness
                         (d) and (e) are taken on the RECTIFIED signal - see
                         "which signal" below; on the bipolar one skewness is
                         identically zero
    (f) wlSKNA   (uV/s)  waveform length, cumulative path length
    (g) zcSKNA   (1/s)   zero crossings
    (h) sscSKNA  (1/s)   slope sign changes
    (i) wampSKNA (uV/s)  Willison amplitude, sum of |dx| exceeding THETA_UV
    (j) cfSKNA           crest factor, peak / rms
    (k) dfSKNA   (Hz)    dominant frequency of the INTEGRATED SKNA

SIGNAL CHAIN - matches src/core_processing.preprocess_skna, deliberately, so
these features describe the same signal the 5 s pipeline and its aSKNA were
built on:

    raw CH41 * 1000        -> uV  (the .txt is in mV)
    Butterworth-3 500-999 Hz bandpass, filtfilt   -> skna   (bipolar)
    |skna|                                        -> rskna  (rectified)
    single-pole IIR, tau = 10 ms, on rskna        -> iskna  (integrated)

The one deliberate difference: core_processing min-max normalises all four
outputs to [0, 1] before returning them. Nothing here does, because seven of
these features are defined in uV and a per-recording rescale would destroy
the unit AND make every amplitude feature relative to that recording's own
maximum - i.e. it would encode recording identity, which is the exact trap
documented in ecg_features.py.

WHICH SIGNAL EACH FEATURE USES, AND WHY IT MATTERS
    The bandpassed signal is bipolar and zero-mean, so its mean is ~0 - aSKNA
    is therefore the mean of the RECTIFIED signal (as in core_processing).
    Conversely zcSKNA and sscSKNA are meaningless on a rectified signal, which
    never crosses zero, so they and every other shape/amplitude feature are
    computed on the bipolar signal. dfSKNA is specified on the integrated
    signal and is taken there.

    NO ANALYSIS-RATE DECIMATION. ecg_features drops to 500 Hz; SKNA lives at
    500-1000 Hz and would be aliased away entirely. Everything here runs at
    the native rate (10 kHz for every recording in dataset/txt), and
    analysis_fs is recorded in the packed file so a mixed-rate pack is
    refused rather than silently averaged - wl/zc/ssc/wamp all scale with the
    sample rate.

    RATES, NOT COUNTS. wl/zc/ssc/wamp are divided by the window duration, so
    they are per-second and a change of window length or sample rate does not
    silently rescale them. The classic sEMG forms are raw sums over the
    window; that is this value times window_sec.

DOMINANT FREQUENCY NEEDS A BAND, NOT AN ARGMAX
    Taking the naked argmax of the iSKNA periodogram returns 120.03 Hz on s13
    and 37.2 Hz on s1 - 60 Hz mains and its harmonics, not nerve activity.
    The integrator is a single pole at 1/(2*pi*0.01) = 15.9 Hz, so iSKNA
    carries no information above ~16 Hz and any peak up there is interference.
    dfSKNA is therefore the argmax within DF_BAND = (0.1, 16] Hz: DC excluded
    (a rectified signal's largest bin is always DC), the 0.1 Hz floor keeps
    residual baseline drift from winning, and the 16 Hz ceiling is the
    integrator's own corner. Widen it with df_band= if you want to look
    higher, but read the peak against the mains harmonics before believing it.

    The periodogram is taken on iSKNA decimated to DF_FS = 100 Hz. Nyquist is
    then 50 Hz, still well above the 16 Hz search band, and it makes the
    per-window FFT ~100x cheaper at identical resolution in-band.

WILLISON THRESHOLD IS ABSOLUTE, AND THE COHORT IS NOT UNIFORM
    THETA_UV is a fixed uV threshold, as the definition requires. Measured
    band-passed SKNA std across this cohort spans ~6x (2.3 uV on s13, 8.9 on
    s1, 14.5 on s6), so at a fixed threshold wampSKNA partly encodes each
    recording's gain/electrode contact rather than its physiology. That is
    inherent to an absolute-threshold feature, not a bug here - but do not
    read a wampSKNA-driven result as physiology without checking it survives
    within-subject.

    THETA_UV = 1 uV is the default because it is the only value measured that
    is non-degenerate across the whole cohort - but at that threshold wampSKNA
    is nearly wlSKNA (sum ratio 0.99 on s6, 0.98 on s1, 0.64 on s13; the sum
    is dominated by the large |dx| that any low threshold keeps). Raising it to
    5 uV separates them (0.77 / 0.54) but zeroes s13 outright (0.0002). No
    single absolute threshold does both, because the cohort's gains differ 6x.

FOUR OF THESE ELEVEN ARE REDUNDANT BY CONSTRUCTION
    varSKNA == rmsSKNA^2 exactly - the bipolar signal is zero-mean, so the
    variance and the mean square are the same number. That dependency is
    NONLINEAR, so it does not cost the matrix any linear rank (measured 11 of
    11 on the built cohort), but one of the two columns carries no information
    the other lacks. Measured on the cohort, corr(wlSKNA, wampSKNA) = +0.9998
    at the default threshold and corr(zcSKNA, sscSKNA) = +0.963 - a narrowband
    signal has one slope reversal per half cycle. All eleven are computed as
    specified and all are kept, but do not read them as eleven independent
    pieces of evidence: the matrix is ill-conditioned rather than singular, so
    prefer a model that tolerates near-collinearity (ridge, trees) over one
    that does not (plain OLS).

THE POOLED CORRELATIONS AGAINST BP ARE NOT PHYSIOLOGY
    Measured on the built 16-subject table: aSKNA, rmsSKNA, wlSKNA, wampSKNA
    and dfSKNA all score pooled r = -0.42 to -0.43 against SBP, while their
    WITHIN-subject median r is -0.02 to +0.00. They are separating recordings,
    not tracking blood pressure - the same trap ecg_features.py documents for
    QTc. Only zcSKNA/sscSKNA retain any within-subject signal (median -0.18 /
    -0.14, |r| > 0.3 in 6 of 14 people). Judge any SKNA feature by the
    within-subject column.

    WHAT THEY ARE ACTUALLY SEPARATING IS THE RECORDING BATCH. dataset/txt
    splits into "*_10kHz.txt" (s1-s6) and the rest (s7-s14), and the medians
    differ by far more than physiology explains:

        aSKNA 8.45 vs 3.49 uV (2.4x)    wlSKNA  2.7x    wampSKNA 3.0x
        rmsSKNA 2.4x   varSKNA 5.9x     dfSKNA 10.7 vs 0.30 Hz (36x)

    |r| between batch and feature reaches 0.77 (wampSKNA), 0.76 (aSKNA),
    0.73 (dfSKNA); |r| between batch and SBP is 0.50. The product,
    0.77 * 0.50 = 0.39, accounts for essentially all of the -0.42 pooled
    correlation. A 36x shift in dfSKNA is an acquisition difference (gain,
    electrode, amplifier setting), not sympathetic activity. Any pooled fit on
    these columns can reach a good score by learning the batch, so group LOSO
    on `person` is a floor, not a safeguard - the batch straddles people.

    PARTLY FIXED by amp_norm="recording" (measured 2026-08-03, rebuild with
    --amp-norm recording). Scaling each recording to REF_SD_UV removes the gain
    component of the batch effect outright:

                  batch ratio          |r| vs batch
        aSKNA     2.43 -> 0.99         0.76 -> 0.06
        rmsSKNA   2.43 -> 0.99         0.75 -> 0.03
        varSKNA   5.92 -> 0.98         0.63 -> 0.00
        wampSKNA  3.01 -> 1.09         0.77 -> 0.25
        wlSKNA    2.73 -> 1.08         0.77 -> 0.24
        mean over the 11 features:     0.50 -> 0.22

    STILL BROKEN, and amplitude normalisation CANNOT fix these - they are
    scale-invariant by construction, so any per-recording gain cancels out of
    them and the residual difference is a genuine acquisition difference in
    the signal's TIMING, not its size:

        dfSKNA    35.7x unchanged,     |r| 0.73 unchanged
        zcSKNA    1.10x unchanged,     |r| 0.60 unchanged
        sscSKNA   1.08x unchanged,     |r| 0.48 unchanged

    Treat dfSKNA in particular as a batch label, not a feature. Dropping all
    three costs 0.004 AUC on the CPT task (0.966 -> 0.962), so nothing that
    matters depends on them.

    Within-subject correlations are IDENTICAL before and after to 3 decimals -
    the transform is one constant per recording, so it cannot and does not
    touch within-recording dynamics. That is the point: it removes only the
    between-recording offset.
"""
import os

import numpy as np
from scipy import signal, stats

import beat_features as bf
import beat_processing as bp
import feature_npz as fnpz

UV_PER_UNIT = 1000.0          # .txt is in mV; core_processing does the same
SKNA_BAND = (500.0, 999.0)    # Hz, Butterworth-3 filtfilt
SKNA_ORDER = 3
INTEGRATE_TAU = 0.01          # s, single-pole IIR on the rectified signal
DEFAULT_WINDOW_SEC = 30.0
DEFAULT_STRIDE_SEC = 5.0
THETA_UV = 1.0                # Willison threshold, absolute uV
DF_FS = 100.0                 # Hz, iSKNA decimated for the periodogram
DF_BAND = (0.1, 16.0)         # Hz, dfSKNA search band - see the docstring

# Per-recording amplitude normalisation (amp_norm="recording").
# Measured cohort SD of the bandpassed SKNA over 16 recordings: 2.59 - 15.48 uV,
# median 4.73 -> the 6x gain spread this module's docstring warns about. Scaling
# each recording to a COMMON SD makes THETA_UV a relative threshold in practice
# while leaving every value in uV, so the default 1 uV stays at the same
# operating point it was chosen for. Gain only, no mean subtraction: the
# bandpassed signal is already zero-mean, and subtracting a mean would not
# survive the rectification that rskna/iskna_d depend on.
REF_SD_UV = 4.73

FEATURE_NAMES = [
    "aSKNA", "varSKNA", "rmsSKNA", "skewSKNA", "kurtSKNA", "wlSKNA",
    "zcSKNA", "sscSKNA", "wampSKNA", "cfSKNA", "dfSKNA",
]
assert len(FEATURE_NAMES) == 11

EXTRA_KEYS = ()               # no family-specific QC columns
ARRAY_KEYS = fnpz.BASE_ARRAY_KEYS
SCALAR_KEYS = fnpz.SCALAR_KEYS


# ---------------------------------------------------------------------------
# signal chain
# ---------------------------------------------------------------------------
def bandpass(x_uv, fs, band=SKNA_BAND, order=SKNA_ORDER):
    """500-999 Hz Butterworth, zero-phase. Zero-phase matters: the features
    below are amplitude/shape statistics and a phase-distorting filter would
    change wl/ssc/cf without changing the physiology."""
    ny = fs / 2.0
    hi = min(band[1], ny - 1.0)
    if band[0] >= hi:
        raise ValueError(f"fs={fs:.0f} Hz cannot carry the {band[0]:.0f}-"
                         f"{band[1]:.0f} Hz SKNA band (Nyquist {ny:.0f} Hz)")
    b, a = signal.butter(order, [band[0] / ny, hi / ny], btype="bandpass")
    return signal.filtfilt(b, a, x_uv)


def integrate(rskna, fs, tau=INTEGRATE_TAU):
    """Single-pole IIR low-pass on the rectified signal, tau = 10 ms."""
    dt = 1.0 / fs
    alpha = dt / (tau + dt)
    return signal.lfilter([alpha], [1.0, -(1.0 - alpha)], rskna)


def preprocess(raw, fs, detrend=False):
    """raw channel -> (skna, rskna, iskna_d, df_fs), all in uV.

    iskna is returned DECIMATED to DF_FS - it is only used for dfSKNA, and
    keeping a second full-rate 10 kHz copy of a 32-minute recording costs
    ~150 MB for nothing. Use integrate() on a slice of rskna when the
    full-rate integrated trace is wanted (e.g. to plot one window).

    `detrend` reproduces core_processing's whole-recording cubic detrend. Off
    by default: the 500 Hz high edge of the bandpass already removes anything
    a cubic could describe, as core_processing's own comment notes.
    """
    x = np.asarray(raw, dtype=np.float64) * UV_PER_UNIT
    if detrend:
        t = np.arange(len(x)) / fs
        x = (x - np.polyval(np.polyfit(t, x, 3), t)) + x.mean()
    skna = bandpass(x, fs).astype(np.float32)
    del x
    rskna = np.abs(skna)
    q = max(1, int(round(fs / DF_FS)))
    iskna_d = signal.resample_poly(integrate(rskna.astype(np.float64), fs), 1, q)
    return skna, rskna, iskna_d, fs / q


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def dominant_frequency(x, fs, band=DF_BAND):
    """Periodogram peak within `band`. The mean is removed first and DC is
    excluded by the band's lower edge, without which a rectified/integrated
    signal always peaks at 0 Hz."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 8:
        return np.nan
    f, P = signal.periodogram(x - x.mean(), fs=fs)
    m = (f > band[0]) & (f <= band[1])
    if not m.any() or not np.isfinite(P[m]).any():
        return np.nan
    return float(f[m][np.argmax(P[m])])


def window_features(skna, rskna, iskna_d, fs, df_fs, i0, i1,
                    theta_uv=THETA_UV, df_band=DF_BAND):
    """The 11 features for one window, given the whole-recording chain and the
    window's sample bounds. float64 for the moment/rate maths - the stored
    signals are float32, which is plenty for uV amplitudes but not for a sum
    of 300 000 terms."""
    w = np.asarray(skna[i0:i1], dtype=np.float64)
    n = len(w)
    if n < 4:
        return {k: np.nan for k in FEATURE_NAMES}
    dur = n / fs
    d = np.diff(w)
    ad = np.abs(d)
    rms = float(np.sqrt(np.mean(w ** 2)))
    peak = float(np.max(np.abs(w)))
    aw = np.abs(w)                      # the AMPLITUDE distribution - see below

    j0 = int(round(i0 * df_fs / fs))
    j1 = int(round(i1 * df_fs / fs))
    return {
        "aSKNA": float(np.mean(rskna[i0:i1], dtype=np.float64)),
        "varSKNA": float(np.var(w)),
        "rmsSKNA": rms,
        # On the AMPLITUDE distribution (rectified), not the bipolar signal:
        # a zero-phase bandpassed signal is symmetric by construction, so its
        # skewness is numerically zero - measured +-0.0002 across s1/s6/s13,
        # i.e. a constant column. Rectified it is 0.80-1.05 with real spread.
        # Both moments are taken there so the pair stays consistent.
        "skewSKNA": float(stats.skew(aw)),
        "kurtSKNA": float(stats.kurtosis(aw)),          # Fisher: 0 = Gaussian
        "wlSKNA": float(ad.sum()) / dur,
        # signbit, not sign: it splits at 0 with no third "zero" state, so a
        # sample sitting exactly on 0.0 cannot be counted as two crossings.
        "zcSKNA": float(np.count_nonzero(np.diff(np.signbit(w)))) / dur,
        "sscSKNA": float(np.count_nonzero(d[:-1] * d[1:] < 0)) / dur,
        "wampSKNA": float(ad[ad > theta_uv].sum()) / dur,
        "cfSKNA": peak / rms if rms > 0 else np.nan,
        "dfSKNA": dominant_frequency(iskna_d[j0:j1], df_fs, df_band),
    }


# ---------------------------------------------------------------------------
# whole-recording driver
# ---------------------------------------------------------------------------
def build_recording(path, subject=None, window_sec=DEFAULT_WINDOW_SEC,
                    stride_sec=DEFAULT_STRIDE_SEC, skna_channel="CH41",
                    ecg_channel="CH40", feature_root="feature_result/5sWindow",
                    label_mode="interp", max_gap_sec=10.0, theta_uv=THETA_UV,
                    df_band=DF_BAND, detrend=False, keep_signals=True,
                    amp_norm="none", **pipe_kwargs):
    """Sliding-window SKNA features for one recording, with BP labels if
    `subject` is given.

    The SKNA channel is put on the ECG's clock by beat_features.
    load_aligned_pair (each channel's own FIR delay measured and removed), and
    skip_leadin defaults to False here to match ecg_features.build_recording -
    that is what makes the two tables share a window grid and therefore join
    on (subject, window_idx). Pass the same start_sec/dur_sec to both.

    The SKNA table can be ONE WINDOW SHORTER than the ECG one for the same
    recording (measured: 376 vs 377 on s5_session1). load_aligned_pair
    truncates the pair to a common length, so the analysed span here is
    min(len(ecg), len(skna)) and occasionally loses the last window. The
    windows that do exist start at identical instants, so join the two tables
    on (subject, window_idx) - an INNER join - rather than assuming equal
    row counts and stacking them positionally.

    keep_signals=False drops the recording-length arrays from the result. The
    GUI wants them for plotting; a cohort build does not, and 16 of them held
    at once is GBs.

    amp_norm:
      'none'       raw uV, as recorded (default - the historical behaviour).
      'recording'  divide the whole recording by its own SD and rescale to
                   REF_SD_UV, removing the 6x between-subject gain spread AT
                   SOURCE. One constant for the entire recording, so every
                   window keeps its amplitude relative to the others and the
                   within-recording dynamics are untouched; only the
                   between-recording offset goes. This is what makes the
                   absolute-threshold features (wampSKNA above theta_uv) stop
                   encoding each subject's amplifier gain.

                   Note the SD is taken AFTER the bandpass, not on the raw
                   channel: raw SD is dominated by DC drift and motion, so
                   scaling by it would set the gain from the artefact rather
                   than from the SKNA band.
    """
    pipe_kwargs.setdefault("skip_leadin", False)
    res = bf.load_aligned_pair(path, ecg_channel=ecg_channel,
                               skna_channel=skna_channel, **pipe_kwargs)
    fs = res["fs"]
    skna, rskna, iskna_d, df_fs = preprocess(res["skna"], fs, detrend=detrend)

    amp_scale = 1.0
    if amp_norm == "recording":
        sd = float(np.std(skna))
        if sd > 0:
            amp_scale = REF_SD_UV / sd
            # everything downstream is linear in the signal, so one multiply
            # each keeps rskna == |skna| and iskna_d == integrate(rskna) exact
            skna = (skna * amp_scale).astype(np.float32)
            rskna = rskna * amp_scale
            iskna_d = iskna_d * amp_scale
    elif amp_norm != "none":
        raise ValueError(f"unknown amp_norm {amp_norm!r}")

    wl, st = int(round(window_sec * fs)), int(round(stride_sec * fs))
    starts = np.arange(0, max(0, len(skna) - wl + 1), st, dtype=np.int64)

    X = np.full((len(starts), len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    for k, i0 in enumerate(starts):
        f = window_features(skna, rskna, iskna_d, fs, df_fs, i0, i0 + wl,
                            theta_uv=theta_uv, df_band=df_band)
        X[k] = [f[n] for n in FEATURE_NAMES]

    t_center = (starts + wl / 2.0) / fs
    out = {
        "path": path, "subject": subject, "fs": fs, "analysis_fs": float(fs),
        "X": X, "feature_names": FEATURE_NAMES,
        "window_sec": window_sec, "stride_sec": stride_sec,
        "label_mode": label_mode, "theta_uv": theta_uv, "df_band": df_band,
        "t_start_sec": starts / fs, "t_center_sec": t_center,
        "clock_offset_sec": res["clock_offset_sec"],
        "amp_norm": amp_norm, "amp_scale": amp_scale,
        "skna_channel": skna_channel, "skna_delay": res["skna_delay"],
        "skna_delay_r": res["skna_delay_r"], "detrend": detrend,
        "n_samples_win": wl, "df_fs": df_fs,
    }
    if keep_signals:
        # rskna is a view-cheap np.abs of skna; recompute it rather than store
        # a second 77 MB copy. iskna_d is small and kept for the df plot.
        out["skna"] = skna
        out["iskna_d"] = iskna_d

    if subject is not None:
        import beat_labels as bl
        t_bp, sbp, dbp = bl.load_bp_labels(subject, feature_root)
        s, d, valid = bl.labels_for_beats(t_center, res["clock_offset_sec"],
                                          t_bp, sbp, dbp, mode=label_mode,
                                          max_gap_sec=max_gap_sec)
        out["SBP"], out["DBP"], out["label_valid"] = s, d, valid
        out["usable"] = valid & np.isfinite(X).all(axis=1)
    return out


# ---------------------------------------------------------------------------
# packing (shared format with ecg_features)
# ---------------------------------------------------------------------------
def window_arrays(r):
    """Per-window arrays only - drops skna/iskna_d, the recording-length ones."""
    return fnpz.window_arrays(r, EXTRA_KEYS)


def save_npz(out_path, results):
    """Pack one build_recording() result, or a list of them, into one .npz."""
    return fnpz.save(out_path, results, FEATURE_NAMES, EXTRA_KEYS)


def load_npz(path, usable_only=False):
    """Load a packed 11-feature SKNA table. `usable` = labelled AND all 11
    features finite. Group LOSO on `person`, not `subject`."""
    return fnpz.load(path, FEATURE_NAMES, EXTRA_KEYS, usable_only)


def format_report(r):
    X = r["X"]
    lo, hi = r["df_band"]
    L = ["=" * 66, " SKNA WINDOW FEATURES", "=" * 66,
         f"file            {os.path.basename(r['path'])}"
         + (f"   subject {r['subject']}" if r.get("subject") else ""),
         f"SKNA channel    {r['skna_channel']} (FIR delay "
         f"{r['skna_delay']:+.2f} s, r={r['skna_delay_r']})",
         f"rate            {r['fs']:.0f} Hz, NOT decimated "
         f"(the {SKNA_BAND[0]:.0f}-{SKNA_BAND[1]:.0f} Hz band needs it)",
         f"chain           bandpass -> rectify -> integrate (tau "
         f"{INTEGRATE_TAU * 1000:.0f} ms); uV, not normalised",
         f"windows         {len(X)}  ({r['window_sec']:.0f} s, stride "
         f"{r['stride_sec']:.0f} s, {r['n_samples_win']} samples)",
         f"wamp threshold  {r['theta_uv']:.3g} uV (absolute)",
         f"df search band  {lo:g}-{hi:g} Hz on iSKNA at {r['df_fs']:.0f} Hz",
         "", f"{'#':>3s} {'feature':12s} {'valid':>7s} {'median':>12s} "
         f"{'p5':>12s} {'p95':>12s}"]
    for i, nm in enumerate(FEATURE_NAMES):
        v = X[:, i]
        f = v[np.isfinite(v)]
        if f.size:
            L.append(f"{i + 1:3d} {nm:12s} {100 * f.size / len(v):6.1f}% "
                     f"{np.median(f):12.4g} {np.percentile(f, 5):12.4g} "
                     f"{np.percentile(f, 95):12.4g}")
        else:
            L.append(f"{i + 1:3d} {nm:12s} {'0.0%':>7s} {'ALL NaN':>12s}")
    if "SBP" in r:
        v = r["label_valid"]
        L += ["", f"labels          {int(v.sum())}/{len(v)} windows labelled",
              f"  SBP           {np.nanmean(r['SBP']):.1f} +/- "
              f"{np.nanstd(r['SBP']):.1f} mmHg",
              f"  DBP           {np.nanmean(r['DBP']):.1f} +/- "
              f"{np.nanstd(r['DBP']):.1f} mmHg",
              f"  usable        {int(r['usable'].sum())} windows "
              f"(labelled AND all 11 features finite)"]
    return "\n".join(L)
