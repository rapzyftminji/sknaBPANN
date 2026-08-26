"""
Beat-level preprocessing for the ANN-LSTM beat-segment architecture.
====================================================================

Pure DSP - no Qt in here, so this module can be imported by a training
script, a notebook, or the GUI in beat_window.py.

Pipeline implemented in this file (stage 1: PREPROCESSING):

    .txt (BIOPAC export)
      -> load_recording()         parse header for fs, read CH1/CH40/CH41
      -> highpass()               2nd-order zero-phase Butterworth @ 0.08 Hz
      -> pan_tompkins()           R-peak detection (full published algorithm)
      -> beat_stats()             RR / HR / 2-cycle segment durations

Stage 2 (segmentation into 3-R-peak windows + feature-vector assembly) will
build on `pan_tompkins`'s output; `beat_stats` already reports the numbers
needed to choose the fixed resample length L for that stage.

WHY CH40: the BIOPAC export carries CH1 (raw) plus CH40/CH41, which the
header describes as FIR-filtered derivatives ("C1 - ... - FIR"). CH40 is the
smoother ECG trace and is the default here, but the channel is a parameter -
nothing below assumes a particular column.

WHY A 0.08 Hz HIGH-PASS: baseline wander sits below ~0.5 Hz and would drag
the Pan-Tompkins integration threshold around, causing missed beats during
drifty stretches. 0.08 Hz is the standard diagnostic-ECG corner - low enough
to leave the ST segment undistorted. It is applied with filtfilt (zero-phase)
so R-peak TIMING is not shifted, which matters because those timings become
segment boundaries downstream.
"""
import re

import numpy as np
import pandas as pd
from scipy import signal

# --- file-format constants, matched to the rest of the pipeline -------------
# src/core_processing.py load_txt_signal / recording_cutter TXT_SKIPROWS
TXT_SKIPROWS = 14
TXT_COLUMNS = ("Time_raw", "CH1", "CH40", "CH41")
ECG_CHANNELS = ("CH40", "CH1", "CH41")   # CH40 first = default (smoother ECG)

DEFAULT_FS = 10000.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def detect_fs(path, default=DEFAULT_FS):
    """Parse 'X ms/sample' out of the 14-line BIOPAC header.

    The raw time COLUMN is not trustworthy (its unit flips between
    'microSec' and 'milliSec' across recording batches - see
    src/diagnostics_alignment.parse_txt_header), so like the rest of the
    pipeline we take fs from the header and use sample_index / fs as the
    real time axis.
    """
    try:
        with open(path, encoding="latin1") as f:
            head = [f.readline() for _ in range(TXT_SKIPROWS)]
    except OSError:
        return default
    for line in head:
        m = re.search(r"([0-9.]+)\s*ms", line)
        if m:
            dt_ms = float(m.group(1))
            if dt_ms > 0:
                return 1000.0 / dt_ms
    return default


def _qrs_envelope(x, fs, lo, hi, out_fs=100.0):
    """Band-limited amplitude envelope, decimated to out_fs. Used for delay
    measurement: CH1 and CH40 carry the same cardiac events but with very
    different spectra, so their raw waveforms correlate poorly while their
    QRS-band envelopes correlate strongly."""
    nyq = fs / 2.0
    sos = signal.butter(2, [lo / nyq, min(hi, nyq * 0.95) / nyq],
                        btype="band", output="sos")
    y = signal.sosfiltfilt(sos, x - x.mean())
    q = max(1, int(round(fs / 1000.0)))
    e = np.abs(signal.hilbert(y[::q]))
    e_fs = fs / q
    q2 = max(1, int(round(e_fs / out_fs)))
    return e[::q2], e_fs / q2


def measure_fir_delay(path, channel="CH40", ref="CH1", fs=None,
                      max_lag_sec=60.0, probe_sec=300.0):
    """Measure `channel`'s FIR group delay relative to `ref`, in seconds.

    WHY THIS EXISTS: CH40/CH41 are FIR-filtered channels in the BIOPAC
    export, and a causal FIR delays its output by half its length. That
    delay is NOT consistent across recording batches - measured across
    dataset/txt it ranges from 0.68 s to 20.4 s for CH40, while CH1 and
    CH41 show none. Pairing R-peaks taken from CH40 with SKNA taken from
    CH41 without compensating would misalign them by SECONDS (tens of
    beats), silently.

    Returns (delay_sec, r). delay_sec > 0 means `channel` LAGS `ref`, so
    dropping the first delay_sec of `channel` puts it on `ref`'s clock.
    A low r means the estimate is untrustworthy - do not apply it.
    """
    if fs is None:
        fs = detect_fs(path)
    i_ch, i_ref = TXT_COLUMNS.index(channel), TXT_COLUMNS.index(ref)
    nrows = int(round(probe_sec * fs))
    df = pd.read_csv(path, skiprows=TXT_SKIPROWS, usecols=[i_ref, i_ch],
                     names=[ref, channel], encoding="latin1", nrows=nrows)
    a = np.nan_to_num(df[ref].to_numpy(dtype=np.float64))
    b = np.nan_to_num(df[channel].to_numpy(dtype=np.float64))
    if min(len(a), len(b)) < int(10 * fs):
        return 0.0, 0.0

    ea, e_fs = _qrs_envelope(a, fs, 5.0, 15.0)
    eb, _ = _qrs_envelope(b, fs, 5.0, 15.0)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n] - ea[:n].mean(), eb[:n] - eb[:n].mean()

    c = signal.correlate(ea, eb, mode="full")
    lags = signal.correlation_lags(n, n, mode="full")
    m = np.abs(lags) <= int(round(max_lag_sec * e_fs))
    c, lags = c[m], lags[m]
    if not len(c):
        return 0.0, 0.0
    k = int(np.argmax(c))
    denom = np.sqrt(np.sum(ea ** 2) * np.sum(eb ** 2))
    r = float(c[k] / denom) if denom > 0 else 0.0
    return float(-lags[k] / e_fs), r


def load_recording(path, channel="CH40", start_sec=None, dur_sec=None, fs=None,
                   delay_sec=0.0):
    """Read one ECG channel from a BIOPAC .txt export.

    `delay_sec` > 0 drops that many seconds off the FRONT of the channel,
    putting a delayed FIR channel (see measure_fir_delay) back onto CH1's
    clock. Do this before slicing, so start_sec always means "seconds into
    the recording" on the common clock rather than on the channel's own.

    Returns (x, fs, n_total) where x is float64 with NaNs zero-filled and
    n_total is the recording's full length in samples (before any
    start/dur slicing), so callers can show how much was trimmed.
    """
    if fs is None:
        fs = detect_fs(path)
    if channel not in TXT_COLUMNS:
        raise ValueError(f"unknown channel {channel!r}; expected one of {ECG_CHANNELS}")
    col = TXT_COLUMNS.index(channel)

    shift = int(round(delay_sec * fs)) if delay_sec else 0
    i0 = 0 if start_sec is None else max(0, int(round(start_sec * fs)))

    # Only parse as far as we actually need. These files run to ~19M rows, so
    # reading all of them to then keep 300 s dominates the runtime.
    nrows = None
    if dur_sec is not None:
        need = i0 + int(round(dur_sec * fs)) + max(shift, 0)
        nrows = need + 1
    df = pd.read_csv(path, skiprows=TXT_SKIPROWS, usecols=[col],
                     names=[channel], encoding="latin1", nrows=nrows)
    x = np.nan_to_num(df[channel].to_numpy(dtype=np.float64))
    n_total = len(x) if nrows is None else _count_rows(path)

    if shift:
        x = x[shift:] if shift > 0 else x[: len(x) + shift]

    i1 = len(x) if dur_sec is None else min(len(x), i0 + int(round(dur_sec * fs)))
    return x[i0:i1], float(fs), n_total


def _count_rows(path):
    """Total data rows, without parsing them. Only needed so the QC report
    can say how much of the recording was analysed."""
    try:
        with open(path, "rb") as f:
            return max(0, sum(1 for _ in f) - TXT_SKIPROWS)
    except OSError:
        return 0


def detect_leadin(x, fs, factor=3.0, max_leadin_sec=60.0, smooth_sec=0.2,
                  settle_sec=2.0):
    """Seconds of settling transient at the START of a recording.

    OFF BY DEFAULT, and that is deliberate. pan_tompkins() already handles
    startup transients through its percentile threshold initialisation and
    its artifact rejection - those two alone detect 24/24 recordings here.
    This trim is therefore redundant, and it is actively harmful for
    labelling: the amplitude statistics of these recordings shift a lot
    over time (the cold-pressor stage), so any adaptive threshold gives a
    different answer depending on how much data was loaded, which moves
    clock_offset_sec and silently shifts every BP label. Enable it only to
    inspect a specific recording, never for a labelled build.

    Returns the end of the INITIAL contiguous disturbance: the first time
    after which the smoothed amplitude stays below `factor` x steady state
    for at least `settle_sec`. 0.0 means the recording starts clean.

    Taking the LAST over-threshold sample within the window instead (the
    obvious implementation) is wrong: any later artefact - a movement
    spike, the cold-pressor onset - drags the trim forward to it, and on
    two subjects that silently discarded 60 s of good data by saturating
    at max_leadin_sec. A lead-in is by definition the disturbance at the
    START, so stop at the first sustained quiet stretch.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < int(round(2 * fs)):
        return 0.0
    w = max(1, int(round(smooth_sec * fs)))
    env = np.convolve(np.abs(x - np.median(x)), np.ones(w) / w, mode="same")

    # Steady-state reference over a FIXED absolute window - the 60 s right
    # after the largest lead-in we would ever trim. Using a fraction of the
    # loaded signal instead (e.g. the back three-quarters) makes the answer
    # depend on how much data the caller happened to load: the same
    # recording gave 7.5 s at --dur 300 and 0 s at full length, because a
    # later high-amplitude stretch (the cold-pressor) moved the median.
    # The clock offset must be a property of the recording, not of the read.
    r0 = min(int(round(max_leadin_sec * fs)), max(0, len(env) - int(round(5 * fs))))
    r1 = min(len(env), r0 + int(round(60 * fs)))
    ref = env[r0:r1]
    if ref.size < int(round(5 * fs)):
        ref = env[len(env) // 2:]
    ss = float(np.median(ref)) if ref.size else 0.0
    if ss <= 0:
        return 0.0

    limit = min(len(env), int(round(max_leadin_sec * fs)))
    bad = env[:limit] > factor * ss
    if not bad.any() or not bad[0]:
        return 0.0                      # starts clean - nothing to trim

    settle = max(1, int(round(settle_sec * fs)))
    # cumulative count of bad samples, so a window sum is O(1)
    c = np.concatenate([[0], np.cumsum(bad)])
    for i in np.nonzero(~bad)[0]:
        j = min(i + settle, limit)
        if c[j] - c[i] == 0:            # quiet for the whole settle window
            return float(min(i + w, limit) / fs)
    return float(limit / fs)


# ---------------------------------------------------------------------------
# High-pass (baseline wander removal)
# ---------------------------------------------------------------------------
def highpass(x, fs, fc=0.08, order=2):
    """2nd-order zero-phase Butterworth high-pass. Same defaults as
    src/core_processing.highpass_filter, reimplemented here so this tool
    has no import dependency on the main app.

    filtfilt doubles the effective order (-> 4th order magnitude response)
    but keeps phase linear at zero, which is what protects R-peak timing.
    """
    x = np.asarray(x, dtype=np.float64)
    nyq = fs / 2.0
    wn = fc / nyq
    if not (0 < wn < 1):
        raise ValueError(f"fc={fc} Hz invalid for fs={fs} Hz")
    b, a = signal.butter(order, wn, btype="highpass")
    # padlen must fit inside the signal; default padlen can exceed short slices
    padlen = min(3 * max(len(a), len(b)), max(0, len(x) - 1))
    return signal.filtfilt(b, a, x, padlen=padlen)


# ---------------------------------------------------------------------------
# Pan-Tompkins R-peak detection
# ---------------------------------------------------------------------------
def _resample_for_detection(x, fs, detect_fs):
    """Anti-aliased decimation to the detection rate.

    Pan-Tompkins was designed around ~200 Hz and its fixed time constants
    (150 ms integration, 200 ms refractory) assume QRS energy dominates.
    Running it directly at 10 kHz is both slow and needlessly sensitive to
    high-frequency content; we detect on a decimated copy and then refine
    each peak back at full rate so the final sample indices stay exact.
    """
    if detect_fs >= fs:
        return x, fs
    g = np.gcd(int(round(fs)), int(round(detect_fs)))
    up, down = int(round(detect_fs)) // g, int(round(fs)) // g
    return signal.resample_poly(x, up, down), float(detect_fs)


def _pt_filter_stages(x, fs, bp_low=5.0, bp_high=15.0, integ_ms=150.0):
    """The four fixed Pan-Tompkins stages: bandpass -> derivative -> square
    -> moving-window integration. Returns each stage for GUI inspection."""
    nyq = fs / 2.0
    hi = min(bp_high, nyq * 0.95)
    b, a = signal.butter(2, [bp_low / nyq, hi / nyq], btype="band")
    bandpassed = signal.filtfilt(b, a, x)

    # 5-point derivative from the paper: (−x[n−2] − 2x[n−1] + 2x[n+1] + x[n+2]) / 8
    deriv = np.convolve(bandpassed, np.array([1, 2, 0, -2, -1]) / 8.0, mode="same")
    squared = deriv ** 2

    win = max(1, int(round(integ_ms / 1000.0 * fs)))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")
    return bandpassed, deriv, squared, integrated, win


def pt_init_thresholds(integrated, cand, fs, init_sec=10.0, spki_pct=75.0):
    """Robust initial (SPKI, NPKI) for the adaptive threshold.

    The published rule is SPKI = max(first 2 s), NPKI = mean(first 2 s).
    That assumes a clean lead-in, which these recordings do not have:
    several carry an amplifier/FIR startup transient in the first seconds
    that is up to ~300x a real QRS. Seeding SPKI from that max puts
    THRESHOLD_I1 far above every genuine beat, and because NPKI decays at
    only 0.875 per candidate the detector never recovers - it returns
    ~1 peak for the whole recording.

    Using a high PERCENTILE of candidate heights instead of the max is
    immune to a handful of outliers while still landing in the QRS
    population, and the normal adaptive updates take over from there.
    """
    n_init = int(round(init_sec * fs))
    in_init = cand[cand < n_init]
    heights = integrated[in_init] if len(in_init) >= 5 else integrated[cand]
    spki = float(np.percentile(heights, spki_pct)) if heights.size else 0.0
    seg = integrated[:n_init] if n_init < len(integrated) else integrated
    npki = float(np.median(seg)) if seg.size else 0.0
    return spki, npki


def _pt_decide(integrated, fs, refractory_ms=200.0, twave_ms=360.0, init_sec=10.0,
               artifact_k=5.0, searchback=False):
    """Adaptive thresholding + searchback + T-wave rejection.

    Follows the published rules: two running estimates (SPKI for signal
    peaks, NPKI for noise peaks) define THRESHOLD_I1; a halved THRESHOLD_I2
    is used when searching back through a gap longer than 1.66 * RR_AVG2.
    A candidate arriving sooner than `twave_ms` after the last QRS is
    accepted only if its slope is at least half the previous QRS slope -
    that is the T-wave discriminator.

    Only the INITIALISATION departs from the paper - see pt_init_thresholds.
    """
    refractory = max(1, int(round(refractory_ms / 1000.0 * fs)))
    cand, _ = signal.find_peaks(integrated, distance=refractory)
    if len(cand) == 0:
        return np.array([], dtype=int)

    # --- artifact rejection, BEFORE the adaptive loop sees these candidates.
    # SPKI only decays when a beat is accepted, so accepting one enormous
    # amplifier/startup transient raises THRESHOLD_I1 above every real beat
    # and the detector deadlocks: nothing is accepted, so nothing brings SPKI
    # back down. One recording (Jose) had a transient 78x the QRS population
    # and yielded 2 detections in 240 s. A genuine QRS is never orders of
    # magnitude larger than the surrounding QRS population, so drop those
    # candidates up front rather than trying to recover afterwards.
    heights = integrated[cand]
    if heights.size:
        cutoff = artifact_k * float(np.percentile(heights, 99))
        keep = heights <= cutoff
        if keep.any():
            cand = cand[keep]

    spki, npki = pt_init_thresholds(integrated, cand, fs, init_sec)

    qrs, rr1, rr2 = [], [], []
    rr_avg2 = None
    last_slope = 0.0
    slope_win = max(1, int(round(0.075 * fs)))   # local slope over ~75 ms

    def local_slope(i):
        lo, hi = max(0, i - slope_win), min(len(integrated), i + 1)
        seg = integrated[lo:hi]
        return float(np.max(np.abs(np.diff(seg)))) if seg.size > 1 else 0.0

    def accept(i, slope):
        nonlocal spki, last_slope, rr_avg2
        if qrs:
            rr = i - qrs[-1]
            rr1.append(rr)
            del rr1[:-8]
            lo_ok = 0.92 * (rr_avg2 if rr_avg2 else rr)
            hi_ok = 1.16 * (rr_avg2 if rr_avg2 else rr)
            if lo_ok <= rr <= hi_ok:
                rr2.append(rr)
                del rr2[:-8]
            rr_avg2 = float(np.mean(rr2)) if rr2 else float(np.mean(rr1))
            rr_avg2 = float(np.clip(rr_avg2, 0.30 * fs, 2.0 * fs))
        qrs.append(i)
        spki = 0.125 * integrated[i] + 0.875 * spki
        last_slope = slope

    for i in cand:
        peak = integrated[i]
        thr_i1 = npki + 0.25 * (spki - npki)
        thr_i2 = 0.5 * thr_i1

        # --- searchback: a gap longer than 1.66*RR_AVG2 means we missed one.
        # OFF BY DEFAULT. Measured across all 24 recordings it never recovered
        # a beat and only ever inserted them: total non-physiological RR
        # intervals were 5 with it off and 13 with it on (even guarded). These
        # recordings are clean and regular with no dropout, which is the
        # condition searchback exists for. Enable it for noisier data.
        # Search only among CANDIDATE peaks, and keep a full refractory clear
        # of `i` at both ends. Taking a raw argmax over [last+refractory, i)
        # instead lets the search land on the rising edge one sample before
        # `i` - which is then accepted too, giving a 4 ms "RR interval" that
        # corrupts RR_AVERAGE1/2 and therefore every later threshold and
        # searchback decision.
        if searchback and qrs and rr_avg2 and (i - qrs[-1]) > 1.66 * rr_avg2:
            lo, hi = qrs[-1] + refractory, i - refractory
            in_gap = cand[(cand >= lo) & (cand < hi)]
            if len(in_gap):
                j = int(in_gap[np.argmax(integrated[in_gap])])
                sl = local_slope(j)
                # A searchback candidate must clear the SAME T-wave test as a
                # normal one. Without it, an under-estimated rr_avg2 early in
                # the record makes 1.66*rr_avg2 shorter than a real RR, so the
                # searchback fires on every normal gap and inserts the T wave
                # (~270 ms after R) as a beat - 38 of them on one recording.
                is_twave = ((j - qrs[-1]) < (twave_ms / 1000.0 * fs)
                            and sl < 0.5 * last_slope)
                if integrated[j] > thr_i2 and not is_twave:
                    accept(j, sl)
                    thr_i1 = npki + 0.25 * (spki - npki)

        if peak > thr_i1:
            slope = local_slope(i)
            # --- T-wave discrimination
            if qrs and (i - qrs[-1]) < (twave_ms / 1000.0 * fs) and slope < 0.5 * last_slope:
                npki = 0.125 * peak + 0.875 * npki      # classify as T wave -> noise
                continue
            accept(i, slope)
        else:
            npki = 0.125 * peak + 0.875 * npki

    return np.asarray(qrs, dtype=int)


def _refine_rpeaks(idx_detect, x_full, fs_full, fs_detect, refine_ms=50.0):
    """Map coarse detections back to full rate and snap each to the true
    R-peak extremum within +/- refine_ms.

    Polarity is decided globally (not per beat): if the mean positive
    excursion around the detections exceeds the mean negative one, R is
    upright, else inverted. Deciding per beat would let noise flip
    individual peaks and inject spurious RR jitter.
    """
    if len(idx_detect) == 0:
        return np.array([], dtype=int), True
    scale = fs_full / fs_detect
    coarse = np.clip((idx_detect * scale).round().astype(int), 0, len(x_full) - 1)
    half = max(1, int(round(refine_ms / 1000.0 * fs_full)))

    los = np.maximum(coarse - half, 0)
    his = np.minimum(coarse + half + 1, len(x_full))
    ups = np.array([x_full[lo:hi].max() for lo, hi in zip(los, his)])
    downs = np.array([x_full[lo:hi].min() for lo, hi in zip(los, his)])
    med = float(np.median(x_full))
    upright = np.mean(ups - med) >= np.mean(med - downs)

    pick = np.argmax if upright else np.argmin
    refined = np.array([lo + int(pick(x_full[lo:hi]))
                        for lo, hi in zip(los, his)], dtype=int)
    refined = np.unique(refined)
    return refined, bool(upright)


def pan_tompkins(x, fs, bp_low=5.0, bp_high=15.0, integ_ms=150.0,
                 refractory_ms=200.0, twave_ms=360.0, detect_fs=250.0,
                 refine_ms=50.0, init_sec=10.0, artifact_k=5.0, searchback=False,
                 return_stages=False):
    """Full Pan-Tompkins R-peak detection.

    `x` should already be high-passed. Returns R-peak sample indices on the
    ORIGINAL fs grid. With return_stages=True also returns a dict of the
    intermediate signals (on the detection grid) for plotting.
    """
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    if len(x) < 2:
        empty = np.array([], dtype=int)
        return (empty, {}) if return_stages else empty

    xd, fsd = _resample_for_detection(x, fs, detect_fs)
    bandpassed, deriv, squared, integrated, win = _pt_filter_stages(
        xd, fsd, bp_low, bp_high, integ_ms)
    idx_d = _pt_decide(integrated, fsd, refractory_ms, twave_ms, init_sec,
                       artifact_k, searchback)
    rpeaks, upright = _refine_rpeaks(idx_d, x, fs, fsd, refine_ms)

    if not return_stages:
        return rpeaks
    stages = {
        "fs_detect": fsd, "bandpassed": bandpassed, "derivative": deriv,
        "squared": squared, "integrated": integrated, "integ_win": win,
        "peaks_detect": idx_d, "upright": upright,
    }
    return rpeaks, stages


# ---------------------------------------------------------------------------
# Beat statistics -> the numbers that set the segmentation length L
# ---------------------------------------------------------------------------
def beat_stats(rpeaks, fs, hr_min=30.0, hr_max=200.0):
    """RR / HR / 2-cycle-segment statistics.

    The 2-cycle numbers are the ones that matter for the next stage: a
    segment spans 3 consecutive R-peaks (peaks k, k+1, k+2), so its duration
    is RR[k] + RR[k+1]. To resample every segment to a fixed length L
    WITHOUT ever decimating one, L must cover the longest segment at the
    target rate - hence the p99/max reported here rather than the mean.
    """
    out = {"n_rpeaks": int(len(rpeaks))}
    if len(rpeaks) < 3:
        out["error"] = "need at least 3 R-peaks for a 2-cycle segment"
        return out

    rr = np.diff(rpeaks) / fs
    hr = 60.0 / rr
    physio = (hr >= hr_min) & (hr <= hr_max)

    seg = rr[:-1] + rr[1:]                       # duration of each 3-peak segment
    seg_ok = physio[:-1] & physio[1:]

    out.update({
        "rr_sec": rr, "hr_bpm": hr, "seg_sec": seg,
        "n_segments": int(len(seg)),
        "n_rr_nonphysio": int((~physio).sum()),
        "hr_mean": float(np.mean(hr[physio])) if physio.any() else float("nan"),
        "hr_sd": float(np.std(hr[physio])) if physio.any() else float("nan"),
        "hr_min_obs": float(np.min(hr[physio])) if physio.any() else float("nan"),
        "hr_max_obs": float(np.max(hr[physio])) if physio.any() else float("nan"),
        "seg_median": float(np.median(seg[seg_ok])) if seg_ok.any() else float("nan"),
        "seg_p99": float(np.percentile(seg[seg_ok], 99)) if seg_ok.any() else float("nan"),
        "seg_max": float(np.max(seg[seg_ok])) if seg_ok.any() else float("nan"),
    })
    return out


def recommend_L(seg_p99, target_fs=2000.0, round_to=256):
    """Fixed resample length L for a 3-R-peak segment at `target_fs`.

    L = ceil(longest_segment * target_fs), rounded up to a multiple of
    `round_to`. Sizing off p99 (not the max) keeps one artefact-stretched
    outlier from inflating L for the whole dataset; segments longer than the
    p99 are better dropped as non-physiological than accommodated.
    """
    if not np.isfinite(seg_p99) or seg_p99 <= 0:
        return None
    L = int(np.ceil(seg_p99 * target_fs))
    return int(np.ceil(L / round_to) * round_to)


# ---------------------------------------------------------------------------
# One-call pipeline + report, shared by both front-ends
# ---------------------------------------------------------------------------
def run_pipeline(path, channel="CH40", fs=None, start_sec=None, dur_sec=None,
                 compensate_delay=True, skip_leadin=False, leadin_factor=3.0,
                 hpf=True, fc=0.08, order=2, target_fs=2000.0, **pt_kwargs):
    """Load -> FIR-delay compensation -> lead-in skip -> high-pass ->
    Pan-Tompkins -> beat statistics, in the order those steps must happen.

    Returns a dict holding every intermediate a front-end needs to plot or
    report. Both the Qt tool and the matplotlib script call this, so the two
    cannot drift apart in what they actually compute.
    """
    if fs is None:
        fs = detect_fs(path)

    delay, delay_r = 0.0, None
    if compensate_delay and channel != "CH1":
        delay, delay_r = measure_fir_delay(path, channel, fs=fs)
        if delay_r is not None and delay_r < 0.5:
            delay = 0.0          # too weak to trust - see measure_fir_delay

    raw, fs, n_total = load_recording(path, channel=channel, start_sec=start_sec,
                                      dur_sec=dur_sec, fs=fs, delay_sec=delay)

    leadin = detect_leadin(raw, fs, factor=leadin_factor) if skip_leadin else 0.0
    if leadin > 0:
        raw = raw[int(round(leadin * fs)):]

    filtered = highpass(raw, fs, fc=fc, order=order) if hpf else raw.copy()
    rpeaks, stages = pan_tompkins(filtered, fs, return_stages=True, **pt_kwargs)
    stats = beat_stats(rpeaks, fs)

    # Where sample 0 of `raw` sits on the ORIGINAL file's clock. Everything
    # trimmed off the front has to be added back before a beat time can be
    # compared against anything derived from the untrimmed recording (BP
    # labels, event marks). delay/leadin were dropped from the front, and
    # start_sec was skipped, in that order.
    clock_offset_sec = float(delay) + float(start_sec or 0.0) + float(leadin)

    return {
        "path": path, "channel": channel, "fs": fs, "n_total": n_total,
        "delay": delay, "delay_r": delay_r, "leadin": leadin,
        "clock_offset_sec": clock_offset_sec,
        "hpf": hpf, "fc": fc, "order": order,
        "raw": raw, "filtered": filtered, "rpeaks": rpeaks,
        "stages": stages, "stats": stats, "target_fs": target_fs,
    }


def format_qc_report(res):
    """Human-readable QC report for a run_pipeline() result."""
    import os as _os
    fs, s = res["fs"], res["stats"]
    n = len(res["raw"])
    st = res["stages"]
    hpf_txt = (f"{res['order']}nd-order Butterworth @ {res['fc']:g} Hz (zero-phase)"
               if res["hpf"] else "DISABLED")
    dtxt = (f"{res['delay']:+.2f} s vs CH1"
            + (f" (r={res['delay_r']:.2f})" if res["delay_r"] is not None else " (not measured)")
            + ("  <- compensated" if res["delay"] else ""))

    L = ["=" * 52, " PREPROCESSING", "=" * 52,
         f"file          {_os.path.basename(res['path'])}",
         f"channel       {res['channel']}",
         f"fs            {fs:.1f} Hz",
         f"analysed      {n} samples ({n / fs:.1f} s) of {res['n_total']} total",
         f"lead-in       {res['leadin']:.2f} s skipped (settling transient)",
         f"FIR delay     {dtxt}",
         f"high-pass     {hpf_txt}",
         f"R polarity    {'upright' if st.get('upright', True) else 'inverted'}",
         f"detection at  {st['fs_detect']:.0f} Hz "
         f"(integration window {st['integ_win']} samples)",
         "", "=" * 52, " R-PEAKS", "=" * 52,
         f"detected      {s['n_rpeaks']} R-peaks"]

    if "error" in s:
        L.append(f"  !! {s['error']}")
        return "\n".join(L)

    rate = s["n_rpeaks"] / (n / fs) * 60.0
    L += [f"mean HR       {s['hr_mean']:.1f} +/- {s['hr_sd']:.1f} bpm",
          f"HR range      {s['hr_min_obs']:.1f} - {s['hr_max_obs']:.1f} bpm",
          f"actual rate   {rate:.1f} beats/min over the analysed span"]
    if s["n_rr_nonphysio"]:
        L.append(f"  !! {s['n_rr_nonphysio']} RR intervals outside 30-200 bpm "
                 "- check the integration panel for missed/doubled beats")
    else:
        L.append("  all RR intervals physiological (30-200 bpm)")

    L += ["", "=" * 52, " 3-R-PEAK SEGMENTS (stage 2)", "=" * 52,
          f"segments      {s['n_segments']}  (sliding by one beat)",
          f"duration      median {s['seg_median']:.3f} s   "
          f"p99 {s['seg_p99']:.3f} s   max {s['seg_max']:.3f} s"]
    tfs = res["target_fs"]
    rec = recommend_L(s["seg_p99"], target_fs=tfs)
    if rec:
        eff = rec / s["seg_p99"]
        L += ["", f"at {tfs:.0f} Hz the p99 segment is {s['seg_p99'] * tfs:.0f} samples",
              f"-> recommended L = {rec}",
              f"   (effective rate {eff:.0f} Hz at the p99 segment, "
              f"Nyquist {eff / 2:.0f} Hz)"]
        if eff / 2 < 1000:
            L.append("   !! Nyquist < 1000 Hz - fine for ECG, but would clip the "
                     "500-1000 Hz SKNA band if the same L is used for it")
    return "\n".join(L)
