"""
SKNA Feature Explorer - Qt front-end for the 11-feature window extractor
========================================================================
Tab 3 of the Beat Pipeline app; same conventions as feature_window.py.

  * a table of all 11 with median / p5 / p95 and, when labels are available,
    the WITHIN-subject correlation against SBP - click a row to plot it
  * the selected feature over time with SBP/DBP overlaid
  * the signal chain for one window (bipolar SKNA, rectified, integrated) so
    the amplitude the features describe is visible, not assumed
  * the iSKNA periodogram with the dfSKNA search band shaded and the picked
    peak marked - a naked argmax lands on 60/120 Hz mains on some recordings,
    and this is where you SEE that rather than trusting a number

Read the within-subject r column, not the pooled one: a feature can correlate
with BP across the cohort purely by encoding subject identity, and the
absolute-threshold features (wampSKNA) are the likeliest to do it.

Run: python3 beat_pipeline/main.py
"""
import os
import traceback

import numpy as np
import pyqtgraph as pg
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSlider,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import beat_labels as bl
import beat_processing as bp
import skna_features as sf

PEN_FEAT = pg.mkPen((30, 90, 200), width=2)
PEN_SBP = pg.mkPen((200, 60, 60), width=2)
PEN_DBP = pg.mkPen((230, 140, 60), width=2, style=Qt.DashLine)
PEN_SKNA = pg.mkPen((90, 90, 90), width=1)
PEN_RECT = pg.mkPen((30, 150, 60), width=1)
PEN_INT = pg.mkPen((200, 40, 40), width=2)
PEN_PSD = pg.mkPen((30, 90, 200), width=1)
MAX_PLOT_POINTS = 40_000


def _decimate_minmax(t, y, max_points=MAX_PLOT_POINTS):
    """Envelope-preserving decimation. A 30 s SKNA window is 300 000 samples of
    a 500-1000 Hz carrier; stride decimation would draw an aliased fraction of
    the true amplitude, which is exactly what these features measure."""
    n = len(y)
    if n <= max_points:
        return t, y
    step = int(np.ceil(n / (max_points // 2)))
    m = (n // step) * step
    if m == 0:
        return t, y
    yy = y[:m].reshape(-1, step)
    tt = t[:m].reshape(-1, step)
    imin, imax = yy.argmin(axis=1), yy.argmax(axis=1)
    first, second = np.minimum(imin, imax), np.maximum(imin, imax)
    rows = np.arange(yy.shape[0])
    out_y = np.empty(yy.shape[0] * 2, dtype=y.dtype)
    out_t = np.empty(yy.shape[0] * 2, dtype=t.dtype)
    out_y[0::2], out_t[0::2] = yy[rows, first], tt[rows, first]
    out_y[1::2], out_t[1::2] = yy[rows, second], tt[rows, second]
    return out_t, out_y


class SknaFeatureWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SKNA Feature Explorer - 11 features on sliding windows")
        self.resize(1240, 780)
        self.res = None
        self.path = None

        ctrl = QScrollArea()
        ctrl.setWidget(self._build_controls())
        ctrl.setWidgetResizable(True)
        ctrl.setMinimumWidth(300)

        left = QSplitter(Qt.Vertical)
        left.addWidget(ctrl)
        left.addWidget(self._build_table())
        left.addWidget(self._build_log())
        left.setStretchFactor(0, 2)
        left.setStretchFactor(1, 3)
        left.setStretchFactor(2, 2)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(self._build_plots())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 860])

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.addWidget(split)

    # -- controls ----------------------------------------------------------
    def _build_controls(self):
        panel = QWidget()
        v = QVBoxLayout(panel)

        gb = QGroupBox("Recording")
        f = QFormLayout(gb)
        self.cb_subject = QComboBox()
        self.cb_subject.addItem("(pick a file manually)")
        self.cb_subject.addItems(list(bl.SUBJECT_RECORDING))
        self.cb_subject.currentTextChanged.connect(self.on_subject)
        self.le_path = QLineEdit(); self.le_path.setReadOnly(True)
        btn = QPushButton("Browse .txt...")
        btn.clicked.connect(self.on_browse)
        self.cb_channel = QComboBox(); self.cb_channel.addItems(bp.ECG_CHANNELS)
        self.cb_channel.setCurrentText("CH41")
        self.sb_dur = QDoubleSpinBox()
        self.sb_dur.setRange(0.0, 1e6); self.sb_dur.setValue(300.0)
        self.sb_dur.setSuffix(" s  (0 = whole file)")
        f.addRow("Subject", self.cb_subject)
        f.addRow(self.le_path); f.addRow(btn)
        f.addRow("SKNA channel", self.cb_channel)
        f.addRow("Duration", self.sb_dur)
        f.addRow(QLabel("<i>picking a subject also loads its BP<br>labels, "
                        "which the correlation column<br>needs</i>"))
        v.addWidget(gb)

        gb = QGroupBox("Windowing")
        f = QFormLayout(gb)
        self.sb_win = QDoubleSpinBox()
        self.sb_win.setRange(5.0, 600.0); self.sb_win.setValue(sf.DEFAULT_WINDOW_SEC)
        self.sb_win.setSuffix(" s")
        self.sb_stride = QDoubleSpinBox()
        self.sb_stride.setRange(1.0, 600.0); self.sb_stride.setValue(sf.DEFAULT_STRIDE_SEC)
        self.sb_stride.setSuffix(" s")
        f.addRow("Window", self.sb_win)
        f.addRow("Stride", self.sb_stride)
        f.addRow(QLabel(f"<i>no decimation: the {sf.SKNA_BAND[0]:.0f}-"
                        f"{sf.SKNA_BAND[1]:.0f} Hz band<br>needs the native "
                        f"rate. Keep window/stride<br>equal to the ECG build "
                        f"to join the two<br>tables.</i>"))
        v.addWidget(gb)

        gb = QGroupBox("Feature parameters")
        f = QFormLayout(gb)
        self.sb_theta = QDoubleSpinBox()
        self.sb_theta.setRange(0.0, 1000.0); self.sb_theta.setDecimals(3)
        self.sb_theta.setValue(sf.THETA_UV); self.sb_theta.setSuffix(" uV")
        self.sb_df_lo = QDoubleSpinBox()
        self.sb_df_lo.setRange(0.0, 5000.0); self.sb_df_lo.setDecimals(2)
        self.sb_df_lo.setValue(sf.DF_BAND[0]); self.sb_df_lo.setSuffix(" Hz")
        self.sb_df_hi = QDoubleSpinBox()
        self.sb_df_hi.setRange(0.1, 5000.0); self.sb_df_hi.setDecimals(2)
        self.sb_df_hi.setValue(sf.DF_BAND[1]); self.sb_df_hi.setSuffix(" Hz")
        self.ck_detrend = QCheckBox("cubic detrend before the bandpass")
        f.addRow("wamp threshold", self.sb_theta)
        f.addRow("df band low", self.sb_df_lo)
        f.addRow("df band high", self.sb_df_hi)
        f.addRow(self.ck_detrend)
        f.addRow(QLabel("<i>the df ceiling is the integrator's own<br>"
                        "16 Hz corner - widen it and the peak<br>"
                        "can land on 60/120 Hz mains</i>"))
        v.addWidget(gb)

        self.ck_zscore = QCheckBox("z-score the feature trace when plotting")
        self.ck_zscore.setChecked(True)
        self.ck_zscore.stateChanged.connect(self._draw_feature)
        v.addWidget(self.ck_zscore)

        self.btn_run = QPushButton("Compute features")
        self.btn_run.setMinimumHeight(34)
        self.btn_run.clicked.connect(self.on_run)
        v.addWidget(self.btn_run)

        self.btn_csv = QPushButton("Export CSV...")
        self.btn_csv.clicked.connect(self.on_export)
        self.btn_csv.setEnabled(False)
        v.addWidget(self.btn_csv)

        self.btn_npz = QPushButton("Export NPZ (for training)...")
        self.btn_npz.clicked.connect(self.on_export_npz)
        self.btn_npz.setEnabled(False)
        v.addWidget(self.btn_npz)
        v.addWidget(QLabel("<i>NPZ is this ONE recording. For the whole<br>"
                           "cohort in one training file use<br>"
                           "build_skna_features.py --npz</i>"))
        v.addStretch(1)
        return panel

    def _build_table(self):
        box = QWidget(); lv = QVBoxLayout(box)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel("<b>Features</b> (click a row to plot)"))
        self.tbl = QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(["feature", "median", "p5", "p95", "r|SBP"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.currentCellChanged.connect(lambda *_: self._draw_feature())
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
        self.txt.setMinimumHeight(110)
        lv.addWidget(self.txt)
        return box

    def _build_plots(self):
        box = QWidget(); lv = QVBoxLayout(box)
        lv.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()

        self.p_feat = self.glw.addPlot(row=0, col=0, colspan=2,
                                      title="feature over time")
        self.p_feat.showGrid(x=True, y=True, alpha=0.25)
        self.p_feat.setLabel("bottom", "window centre", units="s")
        self.p_feat.addLegend(offset=(-10, 10))

        self.p_scat = self.glw.addPlot(row=1, col=0, title="feature vs SBP")
        self.p_scat.showGrid(x=True, y=True, alpha=0.25)
        self.p_scat.setLabel("left", "SBP", units="mmHg")

        self.p_psd = self.glw.addPlot(row=1, col=1, title="iSKNA periodogram")
        self.p_psd.showGrid(x=True, y=True, alpha=0.25)
        self.p_psd.setLabel("bottom", "Hz")
        self.p_psd.setLogMode(x=True, y=True)

        self.p_sig = self.glw.addPlot(row=2, col=0, colspan=2,
                                      title="signal chain for this window")
        self.p_sig.showGrid(x=True, y=True, alpha=0.25)
        self.p_sig.setLabel("bottom", "s into window")
        self.p_sig.setLabel("left", "uV")
        self.p_sig.addLegend(offset=(-10, 10))
        lv.addWidget(self.glw)

        row = QHBoxLayout()
        row.addWidget(QLabel("window"))
        self.sl_win = QSlider(Qt.Horizontal)
        self.sl_win.setEnabled(False)
        self.sl_win.valueChanged.connect(self._draw_window)
        self.lbl_win = QLabel("-")
        row.addWidget(self.sl_win, 1); row.addWidget(self.lbl_win)
        lv.addLayout(row)
        return box

    # -- actions -----------------------------------------------------------
    def set_recording(self, path, channel=None):
        """Load a recording chosen elsewhere (tab 1 hands its file over here).

        The channel is deliberately NOT taken from tab 1: that one is the ECG
        channel, and this tab reads SKNA.
        """
        base = os.path.basename(path)
        subj = next((s for s, r in bl.SUBJECT_RECORDING.items() if r == base), None)
        self.cb_subject.blockSignals(True)
        self.cb_subject.setCurrentText(subj if subj else
                                       self.cb_subject.itemText(0))
        self.cb_subject.blockSignals(False)
        self.path = path
        self.le_path.setText(base + ("" if subj else "  (no subject -> no BP labels)"))

    def on_subject(self, name):
        if name in bl.SUBJECT_RECORDING:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            p = os.path.join(root, "dataset", "txt", bl.SUBJECT_RECORDING[name])
            if os.path.isfile(p):
                self.path = p
                self.le_path.setText(os.path.basename(p))

    def on_browse(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        start = os.path.join(root, "dataset", "txt")
        p, _ = QFileDialog.getOpenFileName(self, "Open recording",
                                           start if os.path.isdir(start) else "",
                                           "Text files (*.txt);;All files (*)")
        if p:
            self.path = p
            self.le_path.setText(os.path.basename(p))
            self.cb_subject.setCurrentIndex(0)

    def on_run(self):
        if not self.path:
            QMessageBox.warning(self, "No file", "Pick a subject or a .txt file first.")
            return
        lo, hi = self.sb_df_lo.value(), self.sb_df_hi.value()
        if lo >= hi:
            QMessageBox.warning(self, "Bad df band",
                                "The df band's low edge must be below its high edge.")
            return
        self.btn_run.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            subj = self.cb_subject.currentText()
            subj = subj if subj in bl.SUBJECT_RECORDING else None
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.res = sf.build_recording(
                self.path, subject=subj,
                window_sec=self.sb_win.value(), stride_sec=self.sb_stride.value(),
                skna_channel=self.cb_channel.currentText(),
                feature_root=os.path.join(root, "feature_result", "5sWindow"),
                theta_uv=self.sb_theta.value(), df_band=(lo, hi),
                detrend=self.ck_detrend.isChecked(),
                dur_sec=self.sb_dur.value() or None)
            self._fill_table()
            self.txt.setPlainText(sf.format_report(self.res))
            n = len(self.res["X"])
            self.sl_win.setEnabled(n > 0)
            self.sl_win.setRange(0, max(0, n - 1))
            self.sl_win.setValue(0)
            self.btn_csv.setEnabled(True)
            self.btn_npz.setEnabled(True)
            self._draw_feature()
            self._draw_window()
        except Exception:
            QMessageBox.critical(self, "Failed", traceback.format_exc())
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_run.setEnabled(True)

    def _fill_table(self):
        r = self.res
        X = r["X"]
        has_bp = "SBP" in r and np.isfinite(r["SBP"]).any()
        self.tbl.setRowCount(len(sf.FEATURE_NAMES))
        for i, nm in enumerate(sf.FEATURE_NAMES):
            v = X[:, i]
            f = v[np.isfinite(v)]
            vals = [nm]
            vals += [f"{np.median(f):.4g}", f"{np.percentile(f, 5):.4g}",
                     f"{np.percentile(f, 95):.4g}"] if f.size else ["-", "-", "-"]
            if has_bp:
                ok = np.isfinite(v) & np.isfinite(r["SBP"])
                # single recording -> this IS the within-subject correlation
                rr = (np.corrcoef(v[ok], r["SBP"][ok])[0, 1]
                      if ok.sum() > 3 and v[ok].std() > 0 else np.nan)
                vals.append(f"{rr:+.3f}" if np.isfinite(rr) else "-")
            else:
                vals.append("-")
            for c, s in enumerate(vals):
                it = QTableWidgetItem(s)
                if c:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl.setItem(i, c, it)
        if self.tbl.currentRow() < 0:
            self.tbl.setCurrentCell(0, 0)

    def _draw_feature(self):
        if self.res is None:
            return
        r = self.res
        i = max(0, self.tbl.currentRow())
        nm = sf.FEATURE_NAMES[i]
        t, v = r["t_center_sec"], r["X"][:, i]
        self.p_feat.clear(); self.p_scat.clear()
        try:
            self.p_feat.legend.clear()
        except Exception:
            pass

        vv = v.copy()
        lab = nm
        if self.ck_zscore.isChecked():
            f = vv[np.isfinite(vv)]
            if f.size and f.std() > 0:
                vv = (vv - f.mean()) / f.std()
                lab = f"{nm} (z)"
        self.p_feat.plot(t, vv, pen=PEN_FEAT, name=lab)
        self.p_feat.setTitle(f"{nm} over time")

        if "SBP" in r:
            for key, pen in (("SBP", PEN_SBP), ("DBP", PEN_DBP)):
                y = r[key].astype(float)
                f = y[np.isfinite(y)]
                if not f.size:
                    continue
                yy = (y - f.mean()) / f.std() if (self.ck_zscore.isChecked()
                                                  and f.std() > 0) else y
                self.p_feat.plot(t, yy, pen=pen, name=key + (" (z)" if
                                 self.ck_zscore.isChecked() else ""))
            ok = np.isfinite(v) & np.isfinite(r["SBP"])
            if ok.sum() > 2:
                self.p_scat.plot(v[ok], r["SBP"][ok], pen=None, symbol="o",
                                 symbolSize=5, symbolBrush=(30, 90, 200, 110),
                                 symbolPen=None)
                self.p_scat.setLabel("bottom", nm)
                rr = (np.corrcoef(v[ok], r["SBP"][ok])[0, 1]
                      if v[ok].std() > 0 else np.nan)
                self.p_scat.setTitle(f"{nm} vs SBP   (r = {rr:+.3f} within this "
                                     f"recording)")
        self.p_feat.autoRange(); self.p_scat.autoRange()

    def _draw_window(self):
        """The signal chain and the periodogram for the selected window.

        rSKNA and the full-rate iSKNA are recomputed from this window's slice
        rather than stored: keeping either at 10 kHz for a 32-minute recording
        costs ~150 MB, and a per-window recompute is milliseconds. The
        integrator's 10 ms transient at the window edge is 0.03% of a 30 s
        window, invisible here and not used for the features (those come from
        the whole-recording chain).
        """
        if self.res is None or "skna" not in self.res:
            return
        r = self.res
        k = self.sl_win.value()
        if k >= len(r["X"]):
            return
        fs = r["fs"]
        i0 = int(round(r["t_start_sec"][k] * fs))
        i1 = min(i0 + r["n_samples_win"], len(r["skna"]))
        w = np.asarray(r["skna"][i0:i1], dtype=np.float64)
        t = np.arange(len(w)) / fs
        rect = np.abs(w)
        isk = sf.integrate(rect, fs)

        self.p_sig.clear()
        try:
            self.p_sig.legend.clear()
        except Exception:
            pass
        td, wd = _decimate_minmax(t, w)
        self.p_sig.plot(td, wd, pen=PEN_SKNA, name="SKNA 500-999 Hz")
        td, rd = _decimate_minmax(t, rect)
        self.p_sig.plot(td, rd, pen=PEN_RECT, name="rSKNA |.|")
        td, idd = _decimate_minmax(t, isk)
        self.p_sig.plot(td, idd, pen=PEN_INT, name="iSKNA (tau 10 ms)")
        i = sf.FEATURE_NAMES.index
        self.p_sig.setTitle(
            f"window {k}/{len(r['X']) - 1} at t={r['t_start_sec'][k]:.0f}s   "
            f"aSKNA {r['X'][k, i('aSKNA')]:.2f} uV   "
            f"rms {r['X'][k, i('rmsSKNA')]:.2f} uV   "
            f"cf {r['X'][k, i('cfSKNA')]:.2f}")
        self.lbl_win.setText(f"{k}/{len(r['X']) - 1}")
        self.p_sig.autoRange()

        # periodogram of the DECIMATED iSKNA - the same samples dfSKNA used
        self.p_psd.clear()
        df_fs = r["df_fs"]
        j0 = int(round(i0 * df_fs / fs))
        j1 = int(round(i1 * df_fs / fs))
        seg = np.asarray(r["iskna_d"][j0:j1], dtype=np.float64)
        if len(seg) >= 8:
            from scipy import signal as _sig
            fr, P = _sig.periodogram(seg - seg.mean(), fs=df_fs)
            m = fr > 0
            self.p_psd.plot(fr[m], P[m], pen=PEN_PSD)
            lo, hi = r["df_band"]
            self.p_psd.addItem(pg.LinearRegionItem(
                values=(np.log10(max(lo, fr[m][0])), np.log10(hi)), movable=False,
                brush=(30, 90, 200, 40)))
            d = r["X"][k, i("dfSKNA")]
            if np.isfinite(d) and d > 0:
                self.p_psd.addLine(x=np.log10(d), pen=pg.mkPen((200, 40, 40),
                                   style=Qt.DashLine))
                self.p_psd.setTitle(f"iSKNA periodogram   dfSKNA = {d:.2f} Hz "
                                    f"(band {lo:g}-{hi:g} Hz shaded)")
            else:
                self.p_psd.setTitle("iSKNA periodogram   dfSKNA = n/a")
            self.p_psd.autoRange()

    def on_export(self):
        if self.res is None:
            return
        import pandas as pd
        p, _ = QFileDialog.getSaveFileName(self, "Export features",
                                           "skna_features.csv", "CSV (*.csv)")
        if not p:
            return
        r = self.res
        df = pd.DataFrame(r["X"], columns=sf.FEATURE_NAMES)
        df.insert(0, "Subject_ID", r.get("subject") or "")
        df.insert(1, "Recording", os.path.basename(r["path"]))
        df.insert(2, "window_idx", np.arange(len(df)))
        df.insert(3, "t_start_sec", r["t_start_sec"])
        df.insert(4, "t_center_sec", r["t_center_sec"])
        df.insert(5, "t_raw_center_sec", r["t_center_sec"] + r["clock_offset_sec"])
        for k in ("SBP", "DBP", "label_valid", "usable"):
            if k in r:
                df[k] = r[k]
        df.to_csv(p, index=False)
        QMessageBox.information(self, "Exported", f"{len(df)} windows -> {p}")

    def on_export_npz(self):
        if self.res is None:
            return
        subj = self.res.get("subject")
        default = f"{subj or 'skna'}_skna_features.npz"
        p, _ = QFileDialog.getSaveFileName(self, "Export features for training",
                                           default, "NumPy archive (*.npz)")
        if not p:
            return
        shape = sf.save_npz(p, self.res)
        note = ("" if subj else "\n\nNo subject was selected, so SBP/DBP are NaN "
                "and `usable` is all-False - this file has no labels to train on.")
        QMessageBox.information(
            self, "Exported",
            f"X{shape} -> {p}\n\nLoad it with:\n"
            f"  import skna_features as sf\n"
            f"  d = sf.load_npz({os.path.basename(p)!r}, usable_only=True)" + note)

    def on_save_log(self):
        if not self.txt.toPlainText():
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save report", "skna_report.txt",
                                           "Text (*.txt)")
        if p:
            with open(p, "w") as f:
                f.write(self.txt.toPlainText())
