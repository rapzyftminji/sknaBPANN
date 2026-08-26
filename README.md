# Beat Pipeline — Yao et al. ANN, SKNA + ECG → BP

Self-contained snapshot of the **feature-table** arm of the project: handcrafted
ECG and SKNA window features, a feature-engineering stage, and the two-layer
feedforward ANN from Yao et al. (IEEE JBHI 26(8) 2022) evaluated
leave-one-subject-out.

This is the sibling of `cnn_bilstm_training_code`. Same target — SBP and DBP —
reached a different way:

| | `cnn_bilstm_training_code` | **this package** |
| --- | --- | --- |
| Input | raw waveforms | 19 ECG + 11 SKNA handcrafted window features |
| Model | CNN → BiLSTM → attention | 2-layer feedforward ANN (10 sigmoid hidden units) |
| Framework | PyTorch | scikit-learn (`MLPRegressor` / `MLPClassifier`) |
| Weight | heavy, GPU helps | light, CPU is fine |

Every module the pipeline imports is included. The `src/` folder here holds
**only** `bp_standards.py`, because `beat_pipeline/ann_bp_loso.py` and
`bp_grading.py` reach up one level for it — keep the two-folder layout intact or
those imports break.

## Install & run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 beat_pipeline/main.py          # the 6-tab GUI
```

No torch required. Every script also runs standalone from the command line —
run them **from the package root**, not from inside `beat_pipeline/`.

## What you need to supply

This package is **code only**; no data ships with it. The scripts expect, at the
package root:

```
dataset/txt/…                  raw BIOPAC .txt recordings
feature_result/5sWindow/<subject>/windowing_labeling/bp_labels.csv
beat_pipeline/built/           output dir the build_* scripts write into
```

`beat_pipeline/built/` in the original repo is ~650 MB (the per-subject
`s*_beats.npz` beat arrays dominate), which is why it is not bundled. The small
derived tables — `ablation_cache_*.npz`, `features_30s_engineered.npz` — are
~16 MB total and can be added if you want the ANN runnable without re-extracting
beats from raw recordings.

## Pipeline order

```
raw BIOPAC .txt  +  feature_result/…/bp_labels.csv
   │
   ├─ STAGE 1-2  beat_processing.py → beat_features.py → beat_labels.py
   │     └─ build_dataset.py            → built/<subject>_beats.npz
   │
   └─ WINDOW FEATURES (the arm the ANN actually uses)
         ecg_features.py  (19 features) ─┐
         skna_features.py (11 features) ─┤ same window/stride grid,
                                         │ join row-for-row
         build_ecg_features.py  --out ecg_features_30s.csv
         build_skna_features.py --out skna_features_30s.csv
              └─ feature_engineering.py
                   merge → log → interactions → within-recording norm
                   → temporal → prune        → built/features_*_engineered.npz
                        └─ model_ablation.py   (shared eval harness + caches)
                             ├─ ann_bp_loso.py      ← THE HEADLINE: SBP/DBP LOSO
                             │    ├─ ann_personalize.py    per-subject fine-tune
                             │    ├─ ann_training_curve.py learning curves
                             │    └─ ann_ft_diagnostics.py why personalization helps
                             ├─ ann_loso.py         CPT detector (sanity check)
                             ├─ window_sweep.py     which window length?
                             ├─ onset_latency.py    how fast is detection?
                             ├─ protocol_offset.py  recover true CPT onset
                             └─ sham_onset_null.py  null control for the above
                        └─ bp_grading.py  → BHS / AAMI / ISO / IEEE grades
```

## The GUI — `beat_pipeline/main.py`

`BeatPipelineApp(QTabWidget)`, six tabs in the order you use them:

1. **Preprocessing & R-peak QC** (`beat_window.py`) — the only place a recording
   is loaded
2. **ECG Feature Explorer** (`feature_window.py`)
3. **SKNA Feature Explorer** (`skna_feature_window.py`)
4. **Feature Distributions** (`feature_boxplot_window.py`)
5. **Feature Engineering Explorer** (`feature_engineering_window.py`)
6. **BP model — ANN LOSO** (`model_window.py`) — the front-end for `ann_bp_loso.py`

---

# Function reference

## Stage 1 — preprocessing & R-peaks

### `beat_pipeline/beat_processing.py`

Pure DSP, no Qt, so a script, notebook or the GUI can all import it.

- `detect_fs`, `_count_rows` — read the sampling rate and length from the file
- `load_recording` — load one channel, optional start/duration slice
- `measure_fir_delay`, `_qrs_envelope` — measure the SKNA filter's group delay
  against the ECG so the two channels can be aligned
- `detect_leadin` — find and drop the amplifier lead-in transient
- `highpass` — baseline-wander removal (fc = 0.08 Hz)
- **Pan-Tompkins**: `_resample_for_detection`, `_pt_filter_stages` (bandpass →
  derivative → square → moving-window integrate), `pt_init_thresholds`,
  `_pt_decide` (adaptive thresholds, 200 ms refractory, 360 ms T-wave
  discrimination), `_refine_rpeaks` (snap back to the full-rate peak),
  `pan_tompkins` — the wrapper that runs all of it
- `beat_stats` — RR/HR statistics with physiological bounds (30–200 bpm)
- `recommend_L` — suggested resample length from the segment-duration p99
- `run_pipeline`, `format_qc_report` — the whole stage plus its QC printout

### `beat_pipeline/beat_plot.py`

Matplotlib/CLI front-end for the same stage-1 output — useful over SSH or when
the Qt window will not fit the screen. `_decimate`, `_decimate_minmax`,
`make_figure`, `summary_row`, `build_parser`, `main`.

## Stage 2 — beat segmentation

### `beat_pipeline/beat_features.py`

- `segment_beats` — three-consecutive-peak segments (two cardiac cycles),
  bounded to 0.5–2.5 s
- `resample_segment` — resample each segment to a fixed length `L` (FFT method)
- `build_beat_features`, `feature_layout` — assemble the per-beat vector and
  record what each column means
- `build_sequences`, `contiguous_sequences` — group beats into `M`-beat
  sequences, keeping only runs with no dropped beats
- `load_aligned_pair` — ECG + SKNA with the FIR delay compensated
- `build_recording`, `save_npz`, `format_report`, `build_parser`, `main`

### `beat_pipeline/beat_labels.py` (stage 2b)

- `person_of` — recording id → person (several recordings map to one person; use
  this for grouping, never the recording id)
- `load_bp_labels` — read `bp_labels.csv` (SBP/DBP per 5 s window)
- `labels_for_beats` — interpolate BP onto beat times, honouring the clock offset
- `attach_labels`, `format_label_report`
- `SUBJECT_RECORDING` — the subject → recording-file map the build scripts iterate

## Window features — the arm the ANN uses

### `beat_pipeline/ecg_features.py` — 19 features

- Complexity (Simjanoska et al.): `hjorth` (mobility, complexity),
  `higuchi_fd`, `shannon_entropy`, `autocorrelation`
- HRV: `hrv_time` (SDNN/RMSSD/pNN50), `hrv_freq` (LF/HF via 4 Hz resampling),
  `poincare` (SD1/SD2)
- Morphology: `delineate_beats` — P/Q/R/S/T onsets and offsets by slope
  threshold; `qtc_fridericia` — rate-corrected QT; `_nanmedian`
- `window_features` — all 19 for one window; `build_recording` — sliding windows
  over a recording; `window_arrays`, `save_npz`, `load_npz`, `format_report`

### `beat_pipeline/skna_features.py` — 11 features

- `bandpass` (500–999 Hz SKNA band, Butterworth-3 `filtfilt`), `integrate` (single-pole IIR on the rectified signal, tau = 10 ms),
  `preprocess` — raw → SKNA / rSKNA (rectified) / iSKNA (integrated)
- `dominant_frequency`
- `window_features` — `aSKNA`, `varSKNA`, `rmsSKNA`, `dfSKNA` (0.1–16 Hz search band),
  `zcSKNA` (zero crossings), `sscSKNA` (slope-sign changes) and the rest; `build_recording`, `window_arrays`, `save_npz`,
  `load_npz`, `format_report`

Same window/stride grid as `ecg_features.py`, so the two tables join
row-for-row on `(subject, window_idx)`.

### `beat_pipeline/feature_npz.py`

One packed format for every feature family, so a training script needs one
loader. `_scalar`, `window_arrays`, `save`, `load` — stores `X (N, F)` next to
`feature_names`, so a stale file is detected rather than silently misread.

### Build scripts

- `build_dataset.py` — `build_one`, `main`: stages 1, 2 and 2b for every subject
  in `SUBJECT_RECORDING` → one `.npz` per subject
- `build_ecg_features.py` — `arrays_from_csv`, `main`: the 19-feature table →
  tidy CSV and/or packed npz
- `build_skna_features.py` — `main`: the 11-feature table, same grid and layout

## Feature engineering — `beat_pipeline/feature_engineering.py`

Sits between extraction and the model. Extraction answers *what is in this
window*; this answers *what does that mean for **this** person relative to their
own baseline*.

- `load_table`, `load_merged` — merge the ECG and SKNA CSVs
- `add_protocol` — derive `is_cpt` from clock position within the 10-min cycle
- `quality_filter` — drop unreliable windows; `keep_s13` handles that recording's
  special case
- `feature_columns` — which columns are features rather than metadata
- `log_transform` — for the heavy-tailed spectral features
- `add_interactions` — cross-family products/ratios
- **`normalize_within_recording`** — the important one. `method="expanding"` is a
  **causal** expanding-window normalization (each window uses only its own past),
  which is what lifted CPT LOSO AUC from 0.685 to 0.961. The `baseline` method
  peeks at labels — do not use it for a reported number.
- `winsorize_within_recording` — clip per-recording tails
- `add_temporal` — lagged/derivative terms, `causal=True` by default
- `prune` — drop near-zero-variance and |r| > 0.95 duplicates
- `engineer` — runs the whole chain; `write_outputs`, `run`, `main`

## Modelling

### `beat_pipeline/model_ablation.py` — the shared harness

Everything downstream imports this; it owns the cache, the CV and the baselines.

- `build` — build (or reuse) the engineered matrix, cached as
  `built/ablation_cache_<window>_<method>.npz`
- `split_column`, `is_level` — classify a feature column by family and by whether
  it is a level or a change
- `make_model` — the model zoo: `mlp` (the ANN, `MLPRegressor`/`MLPClassifier`)
  plus the `ridge` / `logistic` linear references
- `fold_mi`, `evaluate` — **in-fold** mutual-information ranking and grouped
  LOSO evaluation. The ranking is refit inside each fold; ranking on all data
  first would leak.
- `score` — pooled out-of-fold metrics, `baselines` — the bars each task must
  clear, `run_arm` — one feature-block × model arm
- `_headline`, `to_grid`, `main`

### `beat_pipeline/ann_bp_loso.py` — the headline BP result

The Yao et al. network: inputs → 10 sigmoid hidden units → 2 **linear** outputs
(SBP, DBP), one network predicting both jointly. `lbfgs` stands in for the
paper's scaled conjugate gradient.

- `parse_hidden`, `arch_name`, `make_ann`, `make_ref` (ridge reference)
- `calibration_offset`, `_offsets` — the per-subject offset from the first
  `--calib-min` minutes of reference BP
- `inner_select` — nested inner-loop hyperparameter selection
- `run_loso` — the grouped LOSO loop (grouped on **person**, not recording)
- `score`, `oracle_subject_mean` — the oracle that reproduces the paper's
  headline using no features at all, and so shows what the headline really
  measures
- `main`

**Two calibration modes**, each with its own baseline:
- `none` — calibration-free absolute mmHg for an unseen person. What the paper
  claims. Baseline: the constant train-mean predictor.
- `offset` — the held-out person's first minutes set an offset, the net predicts
  the deviation, the offset is added back. Baseline: **zero-delta** — predict the
  calibration value and never move.

### `beat_pipeline/ann_personalize.py`

The ANN analogue of the CNN package's `personalize_finetune.py`, following it
step for step so the two are comparable.

- `three_way_time_split` — adapt / val / test by time, disjoint
- `hidden_activations` — freeze the trunk, read out the hidden layer
- `finetune_head` — fit only the output layer on the adapt slice, early-stopping
  on val (the sklearn equivalent of freezing all but the head)
- `run`, `summarize`, `main`

### Supporting analyses

- `ann_loso.py` — the CPT detector (`make_ann`, `make_ref`, `run_loso`,
  `summarize`, `main`). Not the goal; the sanity check that the features
  describe autonomic state at all.
- `ann_training_curve.py` — `curves`, `draw`, `plot`, `baseline_label`, `_place`,
  `main`. `lbfgs` exposes no `loss_curve_`, so curves are generated by
  `warm_start=True` plus stepped `max_iter`.
- `ann_ft_diagnostics.py` — `blocked_cv`, `main`: why personalization gains what
  it gains, both channels sharing one expensive LOSO pass.
- `window_sweep.py` — `cpt_auc`, `main`: identical evaluation at several window
  lengths, side by side. The BP arm decides; CPT rides along as the check.
- `onset_latency.py` — `out_of_fold`, `first_run`, `episodes`, `trajectory`,
  `figure`, `main`: how fast SKNA detects the cold pressor.
- `protocol_offset.py` — `box`, `scan`, `main`: `is_cpt` is *manufactured* from
  `t_center_sec % 600`, which assumes every recording starts at cycle position 0.
  This recovers the true onset from the SKNA response itself.
- `sham_onset_null.py` — `sham_mask`, `main`: the null control for that — score
  the classifier against fake onsets to see what the real one is worth.
- `bp_grading.py` — `grade_file`, `report`, `main`: wraps `src/bp_standards.py`
  for BHS / IEEE 1708 / AAMI SP10 grades on a predictions CSV, grouped by
  `person`.

### `src/bp_standards.py`

Included only because the two files above import it; see the
`cnn_bilstm_training_code` README for its full function list. `bhs_grade`,
`aami_sp10`, `iso_81060_2`, `ieee_grade`, `compute_bp_standards`,
`compute_bp_standards_from_df`, `format_standards_report`.

## GUI modules

- `main.py` — `BeatPipelineApp(QTabWidget)`, `main()`
- `beat_window.py` — `BeatPipelineWindow`, `_decimate`, `_decimate_minmax`.
  Tab 1, and the only place a recording is loaded.
- `feature_window.py` — `FeatureWindow`. Tab 2, the 19 ECG features on sliding
  windows for one recording.
- `skna_feature_window.py` — `SknaFeatureWindow`, `_decimate_minmax`. Tab 3.
- `feature_boxplot_window.py` — `FeatureBoxplotWindow`, `_FamilyPanel`,
  `MplCanvas`. Tab 4, one box plot per feature across the cohort npz tables.
- `feature_engineering_window.py` — `FeatureEngineeringWindow`,
  `describe_feature`, `MplCanvas`. Tab 5, the interactive front-end for
  `feature_engineering.py`; whatever you settle on here is what the ANN is fed.
- `model_window.py` — `ModelWindow`, `RunWorker(QThread)`, `MplCanvas`. Tab 6,
  the front-end for `ann_bp_loso.py`.

## Things that are easy to get wrong

1. **Group on `person`, not on the recording id.** Several recordings belong to
   the same person; `beat_labels.person_of` exists for this. Grouping on the
   recording id leaks a subject across the LOSO split.
2. **Always quote the matching baseline.** `offset` mode is measured against
   zero-delta, `none` mode against the train-mean predictor. `oracle_subject_mean`
   in `ann_bp_loso.py` shows the paper's headline being reproduced with no
   features at all — the number means little without its bar.
3. **`is_cpt` is derived from clock position, not measured.** It assumes each
   recording starts at cycle position 0. `protocol_offset.py` tests that
   assumption and `sham_onset_null.py` gives the null.
4. **Feature ranking must happen inside the fold.** `model_ablation.fold_mi` does
   it per fold on purpose; ranking on the full table first leaks.
5. **`normalize_within_recording(method="baseline")` peeks at labels.** Use
   `"expanding"` (causal) for anything you intend to report.
