"""
Feature Engineering Explorer - Qt front-end for feature_engineering.py
========================================================================
Tab 5 of the Beat Pipeline app. feature_engineering.py sits between feature
EXTRACTION (build_ecg_features.py / build_skna_features.py, which write the
cohort ecg_features_30s.csv / skna_features_30s.csv) and the model: it merges
the two tables, normalizes WITHIN each recording, adds temporal and
cross-signal features, and prunes collinear ones. That step was previously
only runnable from the command line - this tab runs the same `engineer()`
function and lets you inspect the result before it goes into training:

  * a table of every engineered feature with %finite, median/p5/p95, and
    both the POOLED and WITHIN-PERSON |r| against SBP - read the within-
    person column, the same lesson as tab 2's ECG explorer (a feature can
    look predictive purely by encoding subject identity)
  * the selected feature over time for one recording, with SBP/DBP overlaid
    and the CPT segment of each 10-min cycle shaded, so you can see whether
    the engineering (normalization + temporal terms) actually pulls the
    signal out of the per-recording offset
  * the selected feature split by protocol phase (Before/During/After),
    pooled across the cohort - the direct check on whether it tracks the
    cold-pressor response at all
  * a correlation heatmap of the final (pruned) feature set, to confirm the
    prune step actually left an uncorrelated set rather than trusting the
    report text alone

This tab is independent of tabs 1-3, the same way tab 4 is: it reads the
cohort CSVs that build_ecg_features.py / build_skna_features.py already
wrote, not the currently-loaded recording.
"""
import os
import re
import traceback

import numpy as np
import pyqtgraph as pg
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
    QWidget,
)

import feature_engineering as fe

try:
    from sklearn.feature_selection import mutual_info_regression
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

ROOT = os.path.dirname(os.path.abspath(__file__))
PEN_FEAT = pg.mkPen((30, 90, 200), width=2)
PEN_SBP = pg.mkPen((200, 60, 60), width=2)
PEN_DBP = pg.mkPen((230, 140, 60), width=2, style=Qt.DashLine)
CPT_BRUSH = pg.mkBrush(220, 60, 60, 40)
PHASE_ORDER = ["Before", "During", "After"]
PHASE_COLOR = {"Before": "#3c78c8", "During": "#c83c3c", "After": "#3ca050"}

METHOD_ITEMS = [
    ("expanding", "Expanding median/IQR (causal, default)"),
    ("robust", "Robust median/IQR (whole recording)"),
    ("baseline", "Pre-CPT rest baseline"),
    ("zscore", "Mean/SD (z-score)"),
    ("none", "No normalization (raw, pruned)"),
]

# -- feature-name -> plain-English meaning ------------------------------
# Mirrors feature_engineering.py's naming exactly: a bare name is that
# feature normalized within its own recording; everything else is a suffix
# stacked on top by add_temporal() (_d1/_rmK/_rsK/_slope) or a cross-signal
# ratio from add_interactions() (X_over_Y / X_x_Y / HR). Kept here rather
# than in feature_engineering.py since it's purely a display concern.
INTERACTION_MEANING = {
    "HR": "heart rate from mean NN interval (60000/meanNN), bpm",
    "aSKNA_per_beat": "aSKNA burst burden divided by beats in the window",
    "aSKNA_x_HR": "aSKNA scaled by heart rate",
    "aSKNA_over_RMSSD": "sympathetic burden (aSKNA) relative to vagal tone (RMSSD)",
    "aSKNA_over_SDNN": "sympathetic burden (aSKNA) relative to overall HRV (SDNN)",
    "wamp_over_wl": "SKNA waveform amplitude relative to waveform length",
    "SDNN_over_RMSSD": "overall HRV relative to short-term/vagal HRV",
    "SKNA_crest": "SKNA crest factor: rms amplitude relative to mean burst amplitude",
}
_TEMPORAL_RE = re.compile(r"^(.*)_(d1|rm(\d+)|rs(\d+)|slope)$")


def describe_feature(name):
    """One-line plain-English meaning of an engineered column name."""
    m = _TEMPORAL_RE.match(name)
    if m:
        base, kind = m.group(1), m.group(2)
        base_txt = INTERACTION_MEANING.get(base, f"{base} (normalized within recording)")
        if kind == "d1":
            return f"change vs. the PREVIOUS window, of: {base_txt}"
        if kind.startswith("rm"):
            n = int(kind[2:]) + 1
            return f"rolling MEAN over last {n} windows (~{n * 30}s), of: {base_txt}"
        if kind.startswith("rs"):
            n = int(kind[2:]) + 1
            return f"rolling STD (local variability) over last {n} windows (~{n * 30}s), of: {base_txt}"
        if kind == "slope":
            return f"local trend (slope) over the rolling span, of: {base_txt}"
    return INTERACTION_MEANING.get(name, f"{name}, normalized within its own recording")



class MplCanvas(FigureCanvas):
    def __init__(self, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.fig)


class FeatureEngineeringWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feature Engineering Explorer")
        self.resize(1280, 820)
        self.df = None       # engineered table (merged + normalized + temporal)
        self.cols_full = None  # ALL pruned features from engineer()
        self.cols = None     # current working set (== cols_full unless top-K filtered)
        self.notes = None
        self.feature_stats = {}   # name -> {pct, median, p5, p95, r_pool_sbp, ...}

        ctrl = QScrollArea()
        ctrl.setWidget(self._build_controls())
        ctrl.setWidgetResizable(True)
        ctrl.setMinimumWidth(320)

        left = QSplitter(Qt.Vertical)
        left.addWidget(ctrl)
        left.addWidget(self._build_table())
        left.addWidget(self._build_log())
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 4)
        left.setStretchFactor(2, 2)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(self._build_plots())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([420, 860])

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.addWidget(split)

    # -- controls ------------------------------------------------------
    def _build_controls(self):
        panel = QWidget()
        v = QVBoxLayout(panel)

        gb = QGroupBox("Inputs (cohort tables from build_*_features.py)")
        f = QFormLayout(gb)
        self.le_ecg = QLineEdit(os.path.join(ROOT, "ecg_features_30s.csv"))
        b1 = QPushButton("Browse...")
        b1.clicked.connect(lambda: self._browse(self.le_ecg, "ECG features CSV"))
        self.le_skna = QLineEdit(os.path.join(ROOT, "skna_features_30s.csv"))
        b2 = QPushButton("Browse...")
        b2.clicked.connect(lambda: self._browse(self.le_skna, "SKNA features CSV"))
        f.addRow("ECG csv", self.le_ecg); f.addRow(b1)
        f.addRow("SKNA csv", self.le_skna); f.addRow(b2)
        v.addWidget(gb)

        gb = QGroupBox("Engineering (matches feature_engineering.py flags)")
        f = QFormLayout(gb)
        self.cb_method = QComboBox()
        for key, label in METHOD_ITEMS:
            self.cb_method.addItem(label, key)
        self.sb_k = QSpinBox(); self.sb_k.setRange(1, 10); self.sb_k.setValue(2)
        self.sb_k.setSuffix(" windows")
        self.ck_causal = QCheckBox("causal (past windows only, deployable live)")
        self.ck_causal.setChecked(True)
        self.sb_corr = QDoubleSpinBox()
        self.sb_corr.setRange(0.50, 1.00); self.sb_corr.setSingleStep(0.01)
        self.sb_corr.setValue(0.95)
        self.ck_prune = QCheckBox("prune near-zero-variance + collinear")
        self.ck_prune.setChecked(True)
        self.ck_keep_raw = QCheckBox("keep un-normalized columns alongside (_n)")
        f.addRow("Normalization", self.cb_method)
        f.addRow("Temporal span (k)", self.sb_k)
        f.addRow(self.ck_causal)
        f.addRow("Collinearity |r| >", self.sb_corr)
        f.addRow(self.ck_prune)
        f.addRow(self.ck_keep_raw)
        v.addWidget(gb)

        gb = QGroupBox("Outlier handling (optional, off by default)")
        f = QFormLayout(gb)
        self.ck_wins = QCheckBox("winsorize before normalizing")
        self.sb_wins_lo = QDoubleSpinBox()
        self.sb_wins_lo.setRange(0.0, 20.0); self.sb_wins_lo.setValue(1.0)
        self.sb_wins_lo.setSuffix(" %"); self.sb_wins_lo.setDecimals(1)
        self.sb_wins_hi = QDoubleSpinBox()
        self.sb_wins_hi.setRange(80.0, 100.0); self.sb_wins_hi.setValue(99.0)
        self.sb_wins_hi.setSuffix(" %"); self.sb_wins_hi.setDecimals(1)
        f.addRow(self.ck_wins)
        f.addRow("Lower percentile", self.sb_wins_lo)
        f.addRow("Upper percentile", self.sb_wins_hi)
        f.addRow(QLabel("<i>Clips each feature to its OWN recording's<br>"
                        "percentile range - computed BLIND to phase, so a<br>"
                        "real CPT response only gets clipped if it's more<br>"
                        "extreme than an actual artifact would be. The<br>"
                        "default median/IQR normalization is already fairly<br>"
                        "outlier-resistant; turn this on only if you spot a<br>"
                        "specific spike surviving it in the time-series plot.</i>"))
        v.addWidget(gb)

        self.btn_run = QPushButton("Run feature engineering")
        self.btn_run.setMinimumHeight(34)
        self.btn_run.clicked.connect(self.on_run)
        v.addWidget(self.btn_run)

        row = QHBoxLayout()
        self.le_out = QLineEdit(os.path.join(ROOT, "built"))
        b3 = QPushButton("...")
        b3.setMaximumWidth(28)
        b3.clicked.connect(lambda: self._browse_dir(self.le_out))
        row.addWidget(self.le_out, 1); row.addWidget(b3)
        v.addLayout(row)
        self.btn_save = QPushButton("Save outputs (csv + npz + report)...")
        self.btn_save.clicked.connect(self.on_save)
        self.btn_save.setEnabled(False)
        v.addWidget(self.btn_save)
        v.addWidget(QLabel("<i>Run first to inspect in this tab; Save writes "
                           "features_30s_engineered.csv/.npz + the report to "
                           "the folder above, same as the CLI.</i>"))
        v.addStretch(1)
        return panel

    def _build_table(self):
        box = QWidget(); lv = QVBoxLayout(box)
        lv.setContentsMargins(0, 0, 0, 0)
        self.lbl_final = QLabel("<b>Final features</b> - run first")
        lv.addWidget(self.lbl_final)

        rank_row = QHBoxLayout()
        rank_row.addWidget(QLabel("Rank by"))
        self.cb_rank = QComboBox()
        for label, key in [
            ("Pearson |r| within-person, SBP (recommended)", "r_within_sbp"),
            ("Pearson |r| pooled, SBP", "r_pool_sbp"),
            ("Mutual information, SBP" + ("" if HAVE_SKLEARN else " (needs scikit-learn)"), "mi_sbp"),
            ("Pearson |r| within-person, DBP", "r_within_dbp"),
            ("Pearson |r| pooled, DBP", "r_pool_dbp"),
            ("Mutual information, DBP" + ("" if HAVE_SKLEARN else " (needs scikit-learn)"), "mi_dbp"),
        ]:
            self.cb_rank.addItem(label, key)
        rank_row.addWidget(self.cb_rank, 1)
        lv.addLayout(rank_row)

        topk_row = QHBoxLayout()
        topk_row.addWidget(QLabel("Keep top"))
        self.sb_topk = QSpinBox()
        self.sb_topk.setRange(1, 999); self.sb_topk.setValue(15)
        topk_row.addWidget(self.sb_topk)
        self.btn_topk = QPushButton("Apply filter")
        self.btn_topk.clicked.connect(self.on_filter_topk)
        topk_row.addWidget(self.btn_topk)
        self.btn_reset_filter = QPushButton("Reset (use all pruned)")
        self.btn_reset_filter.clicked.connect(self.on_reset_filter)
        topk_row.addWidget(self.btn_reset_filter)
        topk_row.addStretch(1)
        lv.addLayout(topk_row)
        lv.addWidget(QLabel("<i>Filtering trims the feature set used for the "
                            "table/plots below AND for Save - like the "
                            "paper's FS7 (top-15 by Pearson). Reset restores "
                            "the full pruned set from the last Run.</i>"))

        lv.addWidget(QLabel("Click a row to plot it; click a column header to sort."))
        self.tbl = QTableWidget(0, 8)
        self.tbl.setHorizontalHeaderLabels(
            ["feature", "%finite", "median", "p5", "p95",
             "r|SBP pooled / within", "MI SBP / DBP", "what it means"])
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl.horizontalHeader().setSectionResizeMode(7, QHeaderView.Stretch)
        self.tbl.setSortingEnabled(True)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.currentCellChanged.connect(lambda *_: self._draw_all())
        lv.addWidget(self.tbl)
        return box

    def _build_log(self):
        box = QWidget(); lv = QVBoxLayout(box)
        lv.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout(); row.addWidget(QLabel("<b>Report</b>")); row.addStretch(1)
        b = QPushButton("Save..."); b.clicked.connect(self.on_save_log)
        row.addWidget(b); lv.addLayout(row)
        self.txt = QPlainTextEdit(); self.txt.setReadOnly(True)
        self.txt.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.txt.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.txt.setMinimumHeight(100)
        lv.addWidget(self.txt)
        return box

    def _build_plots(self):
        outer = QWidget(); ov = QVBoxLayout(outer)
        ov.setContentsMargins(0, 0, 0, 0)
        self.lbl_selected = QLabel("<i>no feature selected</i>")
        self.lbl_selected.setStyleSheet("font-weight: bold; padding: 2px;")
        ov.addWidget(self.lbl_selected)

        self.tabs = QTabWidget()
        ov.addWidget(self.tabs)

        # -- time series (one recording) ---------------------------------
        w1 = QWidget(); lv1 = QVBoxLayout(w1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Recording"))
        self.cb_rec = QComboBox()
        self.cb_rec.currentTextChanged.connect(self._draw_timeseries)
        row.addWidget(self.cb_rec, 1)
        self.ck_zscore = QCheckBox("z-score when plotting")
        self.ck_zscore.setChecked(True)
        self.ck_zscore.stateChanged.connect(self._draw_timeseries)
        row.addWidget(self.ck_zscore)
        lv1.addLayout(row)
        self.glw = pg.GraphicsLayoutWidget()
        self.p_ts = self.glw.addPlot(title="engineered feature over time")
        self.p_ts.showGrid(x=True, y=True, alpha=0.25)
        self.p_ts.setLabel("bottom", "time within recording (min)")
        self.p_ts.addLegend(offset=(-10, 10))
        lv1.addWidget(self.glw)
        lv1.addWidget(QLabel("<i>Shaded band = CPT segment of the 10-min "
                             "protocol cycle (phase == During)</i>"))
        self.tabs.addTab(w1, "Time series")

        # -- phase distribution -------------------------------------------
        w2 = QWidget(); lv2 = QVBoxLayout(w2)
        self.canvas_phase = MplCanvas(width=6, height=4.5)
        lv2.addWidget(NavigationToolbar2QT(self.canvas_phase, self))
        lv2.addWidget(self.canvas_phase)
        lv2.addWidget(QLabel("<i>Pooled across the cohort. A feature engineered "
                             "for the CPT response should separate During from "
                             "Before/After here.</i>"))
        self.tabs.addTab(w2, "By phase")

        # -- correlation heatmap of the final feature set -------------------
        w3 = QWidget(); lv3 = QVBoxLayout(w3)
        self.canvas_corr = MplCanvas(width=7, height=6)
        lv3.addWidget(NavigationToolbar2QT(self.canvas_corr, self))
        lv3.addWidget(self.canvas_corr)
        lv3.addWidget(QLabel("<i>Pairwise |r| of the PRUNED set - should look "
                             "sparse. Blocks of dark cells mean the |r|>thresh "
                             "prune left a redundant cluster; lower the "
                             "threshold and re-run if so.</i>"))
        self.tabs.addTab(w3, "Correlation")
        return outer

    # -- helpers ---------------------------------------------------------
    def _browse(self, line_edit, title):
        p, _ = QFileDialog.getOpenFileName(self, title,
                                           os.path.dirname(line_edit.text()) or ROOT,
                                           "CSV (*.csv)")
        if p:
            line_edit.setText(p)

    def _browse_dir(self, line_edit):
        p = QFileDialog.getExistingDirectory(self, "Output folder",
                                             line_edit.text() or ROOT)
        if p:
            line_edit.setText(p)

    # -- actions -----------------------------------------------------------
    def on_run(self):
        ecg_csv, skna_csv = self.le_ecg.text(), self.le_skna.text()
        missing = [p for p in (ecg_csv, skna_csv) if not os.path.isfile(p)]
        if missing:
            QMessageBox.warning(
                self, "Missing input",
                "Not found:\n" + "\n".join(missing) +
                "\n\nBuild these first with build_ecg_features.py / "
                "build_skna_features.py (plain --out, no --npz needed).")
            return
        self.btn_run.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            method = self.cb_method.currentData()
            # 'none' is a real identity branch in normalize_within_recording();
            # this tab used to fake it by running 'robust' with keep_raw and
            # then dropping the _n columns, which deleted every LEVEL feature
            # and kept the temporal ones computed on NORMALIZED data - the
            # opposite of un-normalized. Pass it straight through instead.
            df, cols, notes = fe.engineer(
                ecg_csv, skna_csv, method=method,
                k=self.sb_k.value(), causal=self.ck_causal.isChecked(),
                corr_thresh=self.sb_corr.value(),
                keep_raw=self.ck_keep_raw.isChecked(),
                do_prune=self.ck_prune.isChecked(),
                winsorize=self.ck_wins.isChecked(),
                wins_lower=self.sb_wins_lo.value() / 100.0,
                wins_upper=self.sb_wins_hi.value() / 100.0)
            self.df, self.notes = df, notes
            self.cols_full = cols
            self.cols = list(cols)
            self._compute_stats()
            self.txt.setPlainText(
                "\n".join(f"[{i + 1}] {n}" for i, n in enumerate(notes)))
            self._fill_table()
            self.cb_rec.blockSignals(True)
            self.cb_rec.clear()
            self.cb_rec.addItems(sorted(df["Recording"].unique()))
            self.cb_rec.blockSignals(False)
            self.btn_save.setEnabled(True)
            self._draw_all()
        except Exception:
            QMessageBox.critical(self, "Feature engineering failed",
                                 traceback.format_exc())
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_run.setEnabled(True)

    def _compute_stats(self):
        """Pearson |r| (pooled + within-person) and Mutual Information of
        every pruned feature against SBP and DBP, computed once per Run and
        cached in self.feature_stats - the table, ranking dropdown and top-K
        filter all read from here rather than recomputing per redraw."""
        df, cols = self.df, self.cols_full
        sbp = df["SBP"].to_numpy(float)
        dbp = df["DBP"].to_numpy(float)
        person = df["person"].to_numpy()

        mi_sbp = mi_dbp = None
        if HAVE_SKLEARN and cols:
            X = df[cols].to_numpy(float)
            col_med = np.nanmedian(X, axis=0)
            inds = np.where(np.isnan(X))
            if inds[0].size:
                X[inds] = np.take(col_med, inds[1])
            ok = np.isfinite(sbp) & np.isfinite(dbp)
            try:
                mi_sbp = mutual_info_regression(X[ok], sbp[ok], random_state=0)
                mi_dbp = mutual_info_regression(X[ok], dbp[ok], random_state=0)
            except Exception:
                mi_sbp = mi_dbp = None

        def pearson(v, target):
            fin = np.isfinite(v) & np.isfinite(target)
            r_pool = (np.corrcoef(v[fin], target[fin])[0, 1]
                     if fin.sum() > 3 and v[fin].std() > 0 else np.nan)
            within = []
            for p in np.unique(person):
                m = fin & (person == p)
                if m.sum() > 4 and v[m].std() > 0 and target[m].std() > 0:
                    within.append(np.corrcoef(v[m], target[m])[0, 1])
            r_within = np.nanmedian(within) if within else np.nan
            return r_pool, r_within

        stats = {}
        for i, nm in enumerate(cols):
            v = df[nm].to_numpy(float)
            fin = np.isfinite(v)
            pct = 100.0 * fin.mean()
            f = v[fin]
            med, p5, p95 = ((np.median(f), np.percentile(f, 5),
                            np.percentile(f, 95)) if f.size else (np.nan,) * 3)
            r_pool_sbp, r_within_sbp = pearson(v, sbp)
            r_pool_dbp, r_within_dbp = pearson(v, dbp)
            stats[nm] = dict(
                pct=pct, median=med, p5=p5, p95=p95,
                r_pool_sbp=r_pool_sbp, r_within_sbp=r_within_sbp,
                r_pool_dbp=r_pool_dbp, r_within_dbp=r_within_dbp,
                mi_sbp=(float(mi_sbp[i]) if mi_sbp is not None else np.nan),
                mi_dbp=(float(mi_dbp[i]) if mi_dbp is not None else np.nan))
        self.feature_stats = stats

    def on_filter_topk(self):
        if not self.cols_full or not self.feature_stats:
            return
        key = self.cb_rank.currentData()
        if key.startswith("mi") and not HAVE_SKLEARN:
            QMessageBox.warning(self, "scikit-learn not available",
                                "Mutual information ranking needs "
                                "scikit-learn installed. Falling back is "
                                "not automatic - pick a Pearson option, or "
                                "`pip install scikit-learn` and Run again.")
            return

        def score(nm):
            v = self.feature_stats[nm].get(key, np.nan)
            return abs(v) if np.isfinite(v) else -1.0

        ranked = sorted(self.cols_full, key=score, reverse=True)
        k = min(self.sb_topk.value(), len(ranked))
        self.cols = ranked[:k]
        label = self.cb_rank.currentText()
        self.notes.append(f"manually filtered to top {k}/{len(self.cols_full)} "
                          f"features by {label}")
        self.txt.setPlainText(
            "\n".join(f"[{i + 1}] {n}" for i, n in enumerate(self.notes)))
        self._fill_table()
        self._draw_all()

    def on_reset_filter(self):
        if not self.cols_full:
            return
        self.cols = list(self.cols_full)
        self._fill_table()
        self._draw_all()

    def _fill_table(self):
        df, cols = self.df, self.cols
        self.lbl_final.setText(
            f"<b>Kept: {len(cols)}</b> of {len(self.cols_full)} pruned "
            f"features (from {len(df)} windows, {df['person'].nunique()} people)")
        self.tbl.setSortingEnabled(False)
        self.tbl.setRowCount(len(cols))
        for i, nm in enumerate(cols):
            s = self.feature_stats[nm]
            r_txt = (f"{s['r_pool_sbp']:+.2f} / {s['r_within_sbp']:+.2f}"
                     if np.isfinite(s['r_pool_sbp']) and np.isfinite(s['r_within_sbp'])
                     else "-")
            mi_txt = (f"{s['mi_sbp']:.3f} / {s['mi_dbp']:.3f}"
                     if np.isfinite(s['mi_sbp']) and np.isfinite(s['mi_dbp'])
                     else ("needs scikit-learn" if not HAVE_SKLEARN else "-"))

            vals = [nm, f"{s['pct']:.0f}", f"{s['median']:.3g}",
                    f"{s['p5']:.3g}", f"{s['p95']:.3g}", r_txt]
            for c, v in enumerate(vals):
                it = QTableWidgetItem()
                data = v if c in (0, 5) else float(v)
                it.setData(Qt.DisplayRole, data)
                it.setData(Qt.EditRole, data)
                if c:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl.setItem(i, c, it)
            mi_item = QTableWidgetItem()
            mi_item.setData(Qt.DisplayRole, mi_txt)
            mi_item.setData(Qt.EditRole, mi_txt)
            mi_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.tbl.setItem(i, 6, mi_item)
            self.tbl.setItem(i, 7, QTableWidgetItem(describe_feature(nm)))
        self.tbl.setSortingEnabled(True)
        if self.tbl.rowCount() and self.tbl.currentRow() < 0:
            self.tbl.setCurrentCell(0, 0)

    def _current_feature(self):
        i = self.tbl.currentRow()
        if i < 0 or self.cols is None:
            return None
        name_item = self.tbl.item(i, 0)
        return name_item.text() if name_item else None

    def _draw_all(self):
        nm = self._current_feature()
        if nm:
            self.lbl_selected.setText(
                f"Selected: <b>{nm}</b> &nbsp;-&nbsp; {describe_feature(nm)}")
        else:
            self.lbl_selected.setText("<i>no feature selected</i>")
        self._draw_timeseries()
        self._draw_phase()
        self._draw_corr()

    def _draw_timeseries(self):
        if self.df is None:
            return
        nm = self._current_feature()
        rec = self.cb_rec.currentText()
        self.p_ts.clear()
        try:
            self.p_ts.legend.clear()
        except Exception:
            pass
        if not nm or not rec:
            return
        d = self.df[self.df["Recording"] == rec].sort_values("t_center_sec")
        if d.empty:
            return
        t = d["t_center_sec"].to_numpy(float) / 60.0   # seconds -> minutes
        v = d[nm].to_numpy(float)

        during = d["is_cpt"].to_numpy(bool)
        if during.any():
            edges = np.where(np.diff(during.astype(int)) != 0)[0]
            starts = [0] + list(edges + 1) if during[0] else list(edges + 1)
            for s in starts:
                if s < len(t) and during[s]:
                    e = s
                    while e + 1 < len(t) and during[e + 1]:
                        e += 1
                    reg = pg.LinearRegionItem(
                        values=(t[s], t[min(e + 1, len(t) - 1)]),
                        brush=CPT_BRUSH, movable=False)
                    reg.setZValue(-10)
                    self.p_ts.addItem(reg)

        z = self.ck_zscore.isChecked()

        def prep(y):
            f = y[np.isfinite(y)]
            if z and f.size and f.std() > 0:
                return (y - f.mean()) / f.std()
            return y

        self.p_ts.plot(t, prep(v), pen=PEN_FEAT, name=nm + (" (z)" if z else ""))
        for key, pen in (("SBP", PEN_SBP), ("DBP", PEN_DBP)):
            y = d[key].to_numpy(float)
            if np.isfinite(y).any():
                self.p_ts.plot(t, prep(y), pen=pen, name=key + (" (z)" if z else ""))
        self.p_ts.setTitle(f"{nm} - {rec}")
        self.p_ts.autoRange()

    def _draw_phase(self):
        fig = self.canvas_phase.fig
        fig.clear()
        if self.df is None:
            self.canvas_phase.draw()
            return
        nm = self._current_feature()
        if not nm:
            self.canvas_phase.draw()
            return
        ax = fig.add_subplot(111)
        v = self.df[nm].to_numpy(float)
        phase = self.df["phase"].to_numpy()
        data, labels, colors = [], [], []
        for ph in PHASE_ORDER:
            m = (phase == ph) & np.isfinite(v)
            if m.any():
                data.append(v[m]); labels.append(f"{ph}\n(n={m.sum()})")
                colors.append(PHASE_COLOR[ph])
        if not data:
            ax.set_title("no finite values"); self.canvas_phase.draw(); return
        bp = ax.boxplot(data, tick_labels=labels, showfliers=True,
                        flierprops=dict(marker=".", markersize=3, alpha=0.35),
                        medianprops=dict(color="black"), patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        ax.set_title(f"{nm} by protocol phase (pooled cohort)")
        ax.grid(axis="y", alpha=0.25)
        self.canvas_phase.draw()

    def _draw_corr(self):
        fig = self.canvas_corr.fig
        fig.clear()
        if self.df is None or not self.cols:
            self.canvas_corr.draw()
            return
        cols = self.cols
        C = self.df[cols].corr().to_numpy()
        ax = fig.add_subplot(111)
        im = ax.imshow(np.abs(C), vmin=0, vmax=1, cmap="magma_r")
        n = len(cols)
        step = max(1, n // 40)   # thin tick labels once the pruned set is large
        ax.set_xticks(range(0, n, step))
        ax.set_yticks(range(0, n, step))
        ax.set_xticklabels([cols[i] for i in range(0, n, step)],
                           rotation=90, fontsize=6)
        ax.set_yticklabels([cols[i] for i in range(0, n, step)], fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|r|")
        ax.set_title(f"pairwise |r|, final {n} features")
        self.canvas_corr.draw()

    def on_save(self):
        if self.df is None:
            return
        out_dir = self.le_out.text() or os.path.join(ROOT, "built")
        method = self.cb_method.currentData()
        method_label = "robust(unnormalized)" if method == "none" else method
        try:
            csv_path, npz_path, rep_path = fe.write_outputs(
                self.df, self.cols, self.notes, out_dir, method_label,
                self.sb_k.value(), self.ck_causal.isChecked())
        except Exception:
            QMessageBox.critical(self, "Save failed", traceback.format_exc())
            return
        QMessageBox.information(
            self, "Saved",
            f"{len(self.df)} rows x {len(self.cols)} features\n\n"
            f"{csv_path}\n{npz_path}\n{rep_path}\n\n"
            f"Load for training with:\n"
            f"  d = np.load({os.path.basename(npz_path)!r}, allow_pickle=True)\n"
            f"  X, y_sbp, y_dbp = d['X'], d['y_sbp'], d['y_dbp']\n"
            f"  groups = d['groups']   # person - group LOSO on this")

    def on_save_log(self):
        if not self.txt.toPlainText():
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save report",
                                           "feature_engineering_report.txt",
                                           "Text (*.txt)")
        if p:
            with open(p, "w") as f:
                f.write(self.txt.toPlainText())

    def on_first_open(self):
        """Called once, the first time this tab is switched to (see main.py) -
        the same convenience tab 4 gives for its npz files. If the cohort CSVs
        already sit at the default path (i.e. build_ecg_features.py /
        build_skna_features.py were already run), engineer and show them
        immediately instead of making you click Run just to check."""
        if os.path.isfile(self.le_ecg.text()) and os.path.isfile(self.le_skna.text()):
            self.on_run()