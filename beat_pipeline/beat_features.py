"""
Beat segmentation & feature-vector assembly (stage 2)
=====================================================

Implements the ANN-LSTM input preparation, adapted from the 125 Hz
ECG+PPG paper to this 10 kHz ECG+SKNA recording setup.

WHAT THE PAPER SPECIFIES
    - a segment spans three consecutive peaks (two cardiac cycles),
      sliding by ONE peak: 0-1-2, 1-2-3, 2-3-4, ...
    - each channel in the segment is resampled to a FIXED length
    - the normalised segment length is appended as one extra feature
    - M consecutive feature vectors form one sequence, with the
      corresponding SBP/DBP as the target

WHAT CHANGES HERE, AND WHY
    The paper samples at 125 Hz, so two cycles is ~200 samples and its
    "resample to 256" is LENGTH NORMALISATION at roughly native
    resolution - it discards nothing. Reusing 256 at 10 kHz would be a
    ~40x decimation. That is harmless for ECG but destroys SKNA, whose
    information lives at 500-1000 Hz: 256 samples over a 2 s segment is
    an effective 128 Hz, i.e. a Nyquist of 64 Hz.

    So the fixed length is scaled per channel to each one's bandwidth:

        ECG  (< ~100 Hz)      L=512   -> >=236 Hz effective, Nyquist >=118 Hz
        SKNA (500-1000 Hz)    L=4352  -> >=2005 Hz effective, Nyquist >=1002 Hz

    Those L values come from the MEASURED worst-case segment duration
    across dataset/txt (p99 = 2.170 s, see beat_processing.beat_stats and
    recommend_L), not from an assumed heart-rate range - two subjects run
    at 63-67 bpm, below the usual 60-100 assumption.

CHANNEL PAIRING - THE THING THAT SILENTLY BREAKS
    ECG comes from CH40 and SKNA from CH41. CH40 is FIR-filtered and LAGS
    CH1 by 0.68-7.23 s depending on the recording batch; CH41 has no
    delay. Pairing them naively misaligns ECG against SKNA by tens of
    beats. load_aligned_pair() below compensates each channel's own
    measured delay so both end up on CH1's clock, then applies the SAME
    lead-in trim to both so they stay index-aligned.

MEMORY
    Feature vectors are stored ONCE PER BEAT; sequences are index windows
    into that matrix, not copies. Materialising sequences would multiply
    the data by M (12x here) and is what makes this dataset OOM.
"""
import os

import numpy as np
from scipy import signal

import beat_processing as bp

# Defaults derived from the measured corpus - see the module docstring.
DEFAULT_ECG_L = 512
DEFAULT_SKNA_L = 4352
DEFAULT_ECG_CHANNEL = "CH40"
DEFAULT_SKNA_CHANNEL = "CH41"


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def segment_beats(rpeaks, fs, min_sec=0.5, max_sec=2.5):
    """Three-consecutive-R-peak segments, sliding by one beat.

    Returns (segments, durations, keep) where `segments` is an (N, 3) int
    array of [start, mid, end] sample indices, `durations` is the segment
    length in seconds, and `keep` marks segments whose duration is inside
    [min_sec, max_sec].

    A segment spanning a missed or doubled beat has a duration far from
    the population; those are flagged rather than dropped here so the
    caller can see how many were rejected.
    """
    rpeaks = np.asarray(rpeaks, dtype=np.int64)
    if len(rpeaks) < 3:
        return (np.zeros((0, 3), dtype=np.int64),
                np.zeros(0), np.zeros(0, dtype=bool))
    seg = np.stack([rpeaks[:-2], rpeaks[1:-1], rpeaks[2:]], axis=1)
    dur = (seg[:, 2] - seg[:, 0]) / fs
    keep = (dur >= min_sec) & (dur <= max_sec)
    return seg, dur, keep


def resample_segment(x, i0, i1, L, method="fft"):
    """Resample x[i0:i1] to exactly L samples.

    method='fft' (scipy.signal.resample) is band-limited, so DOWN-sampling
    is anti-aliased. method='interp' is linear interpolation: faster, but
    it aliases when decimating - only safe when the source is already
    band-limited below L/(2*duration).
    """
    seg = x[i0:i1]
    if len(seg) < 2:
        return np.zeros(L, dtype=np.float64)
    if len(seg) == L:
        return seg.astype(np.float64, copy=True)
    if method == "interp":
        return np.interp(np.linspace(0, len(seg) - 1, L),
                         np.arange(len(seg)), seg)
    return signal.resample(seg.astype(np.float64), L)


# ---------------------------------------------------------------------------
# Feature vectors
# ---------------------------------------------------------------------------
def build_beat_features(channels, fs, segments, lengths, method="fft",
                        amp_norm="recording", with_duration=True):
    """Assemble the per-beat feature matrix.

    `channels` maps name -> full-length signal; `lengths` maps the same
    names -> fixed resample length L. Output columns are the channels in
    `lengths` order, then the segment duration in seconds if requested.

    amp_norm:
      'recording'  z-score each channel by ITS OWN recording statistics
                   (default). Beat-to-beat amplitude differences survive,
                   which matters because SKNA burst AMPLITUDE is the
                   physiological signal - normalising per segment would
                   divide exactly that away.
      'segment'    z-score each segment independently. Appropriate for
                   morphology-only work; destroys SKNA amplitude.
      'none'       raw units.
    """
    names = list(lengths.keys())
    stats = {}
    if amp_norm == "recording":
        for n in names:
            v = np.asarray(channels[n], dtype=np.float64)
            sd = float(v.std())
            stats[n] = (float(v.mean()), sd if sd > 0 else 1.0)

    n_seg = len(segments)
    width = sum(lengths[n] for n in names) + (1 if with_duration else 0)
    X = np.empty((n_seg, width), dtype=np.float32)

    for k, (i0, _mid, i2) in enumerate(segments):
        col = 0
        for n in names:
            L = lengths[n]
            v = resample_segment(channels[n], int(i0), int(i2), L, method)
            if amp_norm == "recording":
                mu, sd = stats[n]
                v = (v - mu) / sd
            elif amp_norm == "segment":
                sd = v.std()
                v = (v - v.mean()) / (sd if sd > 0 else 1.0)
            X[k, col:col + L] = v
            col += L
        if with_duration:
            X[k, col] = (i2 - i0) / fs
    return X


def feature_layout(lengths, with_duration=True):
    """Column ranges per channel, so downstream code can slice the matrix
    without re-deriving offsets."""
    layout, col = {}, 0
    for n, L in lengths.items():
        layout[n] = (col, col + L)
        col += L
    if with_duration:
        layout["duration_sec"] = (col, col + 1)
        col += 1
    layout["_width"] = col
    return layout


# ---------------------------------------------------------------------------
# Sequences (index windows - never materialised)
# ---------------------------------------------------------------------------
def build_sequences(n_beats, M=12, stride=1):
    """(n_seq, M) index array of M consecutive beats.

    Returns indices INTO the beat matrix rather than a copy of it: at
    M=12 a materialised tensor is 12x the memory for no extra information.
    """
    if n_beats < M:
        return np.zeros((0, M), dtype=np.int64)
    starts = np.arange(0, n_beats - M + 1, stride, dtype=np.int64)
    return starts[:, None] + np.arange(M, dtype=np.int64)[None, :]


def contiguous_sequences(seq_idx, keep):
    """Drop sequences containing any rejected beat.

    A sequence that straddles a bad segment is not M *consecutive* beats
    in the physiological sense, even though its indices are consecutive.
    """
    if len(seq_idx) == 0:
        return seq_idx
    return seq_idx[keep[seq_idx].all(axis=1)]


# ---------------------------------------------------------------------------
# Whole-recording driver
# ---------------------------------------------------------------------------
def load_aligned_pair(path, ecg_channel=DEFAULT_ECG_CHANNEL,
                      skna_channel=DEFAULT_SKNA_CHANNEL, fs=None,
                      start_sec=None, dur_sec=None, compensate_delay=True,
                      skip_leadin=True, leadin_factor=3.0, **pipe_kwargs):
    """Run stage-1 on the ECG channel and return the SKNA channel on the
    SAME sample grid.

    Each channel's own FIR delay is measured and removed, which puts both
    on CH1's clock; the lead-in trim detected on the ECG is then applied
    to both, and the pair is truncated to a common length.
    """
    res = bp.run_pipeline(path, channel=ecg_channel, fs=fs, start_sec=start_sec,
                          dur_sec=dur_sec, compensate_delay=compensate_delay,
                          skip_leadin=skip_leadin, leadin_factor=leadin_factor,
                          **pipe_kwargs)
    fs = res["fs"]

    skna_delay, skna_r = 0.0, None
    if compensate_delay and skna_channel != "CH1":
        skna_delay, skna_r = bp.measure_fir_delay(path, skna_channel, fs=fs)
        if skna_r is not None and skna_r < 0.5:
            skna_delay = 0.0
    skna, _, _ = bp.load_recording(path, channel=skna_channel, start_sec=start_sec,
                                   dur_sec=dur_sec, fs=fs, delay_sec=skna_delay)
    if res["leadin"] > 0:
        skna = skna[int(round(res["leadin"] * fs)):]

    ecg = res["filtered"]
    n = min(len(ecg), len(skna))
    res["ecg"] = ecg[:n]
    res["skna"] = skna[:n]
    res["skna_delay"] = skna_delay
    res["skna_delay_r"] = skna_r
    res["skna_channel"] = skna_channel
    return res


def build_recording(path, ecg_channel=DEFAULT_ECG_CHANNEL,
                    skna_channel=DEFAULT_SKNA_CHANNEL,
                    ecg_L=DEFAULT_ECG_L, skna_L=DEFAULT_SKNA_L,
                    M=12, stride=1, min_sec=0.5, max_sec=2.5,
                    method="fft", amp_norm="recording", **load_kwargs):
    """Full stage-2 build for one recording."""
    res = load_aligned_pair(path, ecg_channel=ecg_channel,
                            skna_channel=skna_channel, **load_kwargs)
    fs = res["fs"]
    segments, dur, keep = segment_beats(res["rpeaks"], fs, min_sec, max_sec)

    lengths = {"ecg": ecg_L, "skna": skna_L}
    channels = {"ecg": res["ecg"], "skna": res["skna"]}
    X = build_beat_features(channels, fs, segments, lengths,
                            method=method, amp_norm=amp_norm)

    seq = build_sequences(len(segments), M=M, stride=stride)
    seq = contiguous_sequences(seq, keep)

    return {
        "path": path, "fs": fs, "X": X, "segments": segments,
        "durations": dur, "keep": keep, "seq_idx": seq,
        "layout": feature_layout(lengths), "lengths": lengths,
        "M": M, "stride": stride, "amp_norm": amp_norm,
        "beat_time_sec": segments[:, 0] / fs if len(segments) else np.zeros(0),
        "stage1": res,
    }


def save_npz(out_path, built):
    """Persist a build_recording() result. The beat matrix is stored once;
    seq_idx indexes into it."""
    np.savez_compressed(
        out_path,
        X=built["X"], segments=built["segments"], durations=built["durations"],
        keep=built["keep"], seq_idx=built["seq_idx"],
        beat_time_sec=built["beat_time_sec"], fs=built["fs"],
        ecg_L=built["lengths"]["ecg"], skna_L=built["lengths"]["skna"],
        M=built["M"], stride=built["stride"], amp_norm=built["amp_norm"],
        source=os.path.basename(built["path"]))


def format_report(built):
    """Human-readable summary of a build."""
    X, seg, keep, seq = built["X"], built["segments"], built["keep"], built["seq_idx"]
    dur, lay = built["durations"], built["layout"]
    s1 = built["stage1"]
    mb = X.nbytes / 1e6
    L = ["=" * 58, " STAGE 2 - SEGMENTS & FEATURE VECTORS", "=" * 58,
         f"file            {os.path.basename(built['path'])}",
         f"ECG channel     {s1['channel']} (FIR delay {s1['delay']:+.2f} s, compensated)",
         f"SKNA channel    {s1['skna_channel']} (FIR delay {s1['skna_delay']:+.2f} s)",
         f"lead-in         {s1['leadin']:.2f} s skipped",
         f"R-peaks         {len(s1['rpeaks'])}",
         "",
         f"segments        {len(seg)} (3 R-peaks, sliding by 1)",
         f"  kept          {int(keep.sum())}   rejected {int((~keep).sum())} "
         f"(duration outside bounds)"]
    if len(dur):
        L.append(f"  duration      median {np.median(dur):.3f} s  "
                 f"min {dur.min():.3f} s  max {dur.max():.3f} s")
    L += ["", f"feature vector  {X.shape[1]} values per beat"]
    for name, rng in lay.items():
        if name == "_width":
            continue
        lo, hi = rng
        if hi - lo > 1 and len(dur):
            eff = (hi - lo) / np.percentile(dur[keep] if keep.any() else dur, 99)
            L.append(f"  {name:12s} cols {lo:5d}-{hi - 1:<5d} (L={hi - lo}, "
                     f"eff {eff:.0f} Hz, Nyquist {eff / 2:.0f} Hz)")
        else:
            L.append(f"  {name:12s} col  {lo}")
    L += ["",
          f"beat matrix     {X.shape[0]} x {X.shape[1]} float32 = {mb:.1f} MB",
          f"sequences       {len(seq)} of M={built['M']} (stride {built['stride']})",
          f"  as indices    {seq.nbytes / 1e6:.2f} MB",
          f"  if copied     {len(seq) * built['M'] * X.shape[1] * 4 / 1e6:.0f} MB "
          f"({built['M']}x - not materialised)",
          f"amplitude norm  {built['amp_norm']}"]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Stage 2: beat segmentation and feature-vector assembly.",
        epilog="Example:\n"
               "  python3 beat_pipeline/beat_features.py dataset/txt/SKNA_BP_alice_10kHz.txt "
               "--dur 300 --out built/\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="BIOPAC .txt recording(s); globs allowed")
    p.add_argument("--ecg-channel", default=DEFAULT_ECG_CHANNEL, choices=list(bp.ECG_CHANNELS))
    p.add_argument("--skna-channel", default=DEFAULT_SKNA_CHANNEL, choices=list(bp.ECG_CHANNELS))
    p.add_argument("--ecg-L", type=int, default=DEFAULT_ECG_L)
    p.add_argument("--skna-L", type=int, default=DEFAULT_SKNA_L)
    p.add_argument("-M", "--seq-len", type=int, default=12, help="beats per sequence")
    p.add_argument("--stride", type=int, default=1, help="sequence stride, in beats")
    p.add_argument("--min-sec", type=float, default=0.5)
    p.add_argument("--max-sec", type=float, default=2.5)
    p.add_argument("--method", default="fft", choices=["fft", "interp"])
    p.add_argument("--amp-norm", default="recording",
                   choices=["recording", "segment", "none"])
    p.add_argument("--fs", type=float, default=None)
    p.add_argument("--start", type=float, default=None)
    p.add_argument("--dur", type=float, default=None)
    p.add_argument("--no-delay", action="store_true")
    p.add_argument("--leadin", action="store_true")
    p.add_argument("--out", default=None, help="directory to write per-recording .npz")
    return p


def main(argv=None):
    import glob
    import sys
    args = build_parser().parse_args(argv)

    paths = []
    for pat in args.files:
        paths.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat])
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        print("no input files matched", file=sys.stderr)
        return 2
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    tot_beats = tot_seq = 0
    for path in paths:
        built = build_recording(
            path, ecg_channel=args.ecg_channel, skna_channel=args.skna_channel,
            ecg_L=args.ecg_L, skna_L=args.skna_L, M=args.seq_len, stride=args.stride,
            min_sec=args.min_sec, max_sec=args.max_sec, method=args.method,
            amp_norm=args.amp_norm, fs=args.fs, start_sec=args.start,
            dur_sec=args.dur, compensate_delay=not args.no_delay,
            skip_leadin=args.leadin)
        print(format_report(built))
        tot_beats += len(built["X"])
        tot_seq += len(built["seq_idx"])
        if args.out:
            dest = os.path.join(args.out,
                                os.path.splitext(os.path.basename(path))[0] + "_beats.npz")
            save_npz(dest, built)
            print(f"[saved {dest}]")
        print()

    if len(paths) > 1:
        print("=" * 58)
        print(f"TOTAL across {len(paths)} recordings: {tot_beats} beats, "
              f"{tot_seq} sequences")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
