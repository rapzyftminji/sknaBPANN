"""
ECG Feature Explorer - Qt front-end for the 19-feature window extractor
=======================================================================
Companion to beat_window.py, same conventions. Computes the 19 ECG features
on sliding windows for one recording and lets you inspect them:

  * a table of all 19 with median / p5 / p95 and, when labels are available,
    the WITHIN-subject correlation against SBP - click a row to plot it
  * the selected feature over time with SBP/DBP overlaid, so you can see
    whether it tracks BP or just drifts
  * a beat-delineation view (Q onset, J point, T peak, T end) with a slider,
    because features 13-19 are only as good as those landmarks

The within-subject correlation column is the one to read. A feature can
correlate strongly with BP across the pooled dataset purely because it
encodes subject identity - QTc scores -0.53 pooled and -0.09 within.

This is tab 2 of the Beat Pipeline app: python3 beat_pipeline/main.py
For the whole cohort in one CSV instead, use build_ecg_features.py.
"""
import os
import traceback

import numpy as np
import pyqtgraph as pg
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSlider,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import beat_processing as bp
import ecg_features as ef

PEN_FEAT = pg.mkPen((30, 90, 200), width=2)
PEN_SBP = pg.mkPen((200, 60, 60), width=2)
PEN_DBP = pg.mkPen((230, 140, 60), width=2, style=Qt.DashLine)
PEN_ECG = pg.mkPen((20, 20, 20), width=1)
LANDMARKS = [("q_onset", (30, 90, 200), "Q on"), ("s_offset", (30, 150, 60), "J"),
             ("t_peak", (200, 40, 40), "T pk"), ("t_end", (140, 60, 180), "T end")]


class FeatureWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Feature Explorer - 19 features on sliding windows")
        self.resize(1240, 780)
        self.res = None
        self.path = None
        self.subject = None
        self.channel = bp.ECG_CHANNELS[0]

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
        self.lbl_recording = QLabel(
            "<i>no recording loaded - pick a subject or file in tab 1</i>")
        self.lbl_recording.setWordWrap(True)
        f.addRow(self.lbl_recording)
        self.sb_dur = QDoubleSpinBox()
        self.sb_dur.setRange(0.0, 1e6); self.sb_dur.setValue(300.0)
        self.sb_dur.setSuffix(" s  (0 = whole file)")
        f.addRow("Duration", self.sb_dur)
        f.addRow(QLabel("<i>The file, subject and ECG channel are set once<br>"
                        "in tab 1 (Preprocessing && R-peak QC) and carried<br>"
                        "over here automatically when you switch tabs.</i>"))
        v.addWidget(gb)

        gb = QGroupBox("Windowing")
        f = QFormLayout(gb)
        self.sb_win = QDoubleSpinBox()
        self.sb_win.setRange(5.0, 600.0); self.sb_win.setValue(ef.DEFAULT_WINDOW_SEC)
        self.sb_win.setSuffix(" s")
        self.sb_stride = QDoubleSpinBox()
        self.sb_stride.setRange(1.0, 600.0); self.sb_stride.setValue(ef.DEFAULT_STRIDE_SEC)
        self.sb_stride.setSuffix(" s")
        self.sb_afs = QDoubleSpinBox()
        self.sb_afs.setRange(100.0, 2000.0); self.sb_afs.setValue(ef.ANALYSIS_FS)
        self.sb_afs.setSuffix(" Hz")
        f.addRow("Window", self.sb_win)
        f.addRow("Stride", self.sb_stride)
        f.addRow("Analysis rate", self.sb_afs)
        f.addRow(QLabel("<i>Hjorth/FD scale with sample rate -<br>"
                        "changing this changes features 1-3<br>"
                        "and makes them incomparable to<br>a table built at "
                        "another rate</i>"))
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
                           "build_ecg_features.py --npz</i>"))
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

        self.p_feat = self.glw.addPlot(row=0, col=0, title="feature over time")
        self.p_feat.showGrid(x=True, y=True, alpha=0.25)
        self.p_feat.setLabel("bottom", "window centre", units="s")
        self.p_feat.addLegend(offset=(-10, 10))

        self.p_scat = self.glw.addPlot(row=1, col=0, title="feature vs SBP")
        self.p_scat.showGrid(x=True, y=True, alpha=0.25)
        self.p_scat.setLabel("left", "SBP", units="mmHg")

        self.p_beat = self.glw.addPlot(row=2, col=0, title="beat delineation")
        self.p_beat.showGrid(x=True, y=True, alpha=0.25)
        self.p_beat.setLabel("bottom", "ms from R")
        lv.addWidget(self.glw)

        row = QHBoxLayout()
        row.addWidget(QLabel("beat"))
        self.sl_beat = QSlider(Qt.Horizontal)
        self.sl_beat.setEnabled(False)
        self.sl_beat.valueChanged.connect(self._draw_beat)
        self.lbl_beat = QLabel("-")
        row.addWidget(self.sl_beat, 1); row.addWidget(self.lbl_beat)
        lv.addLayout(row)
        return box

    # -- actions -----------------------------------------------------------
    def set_recording(self, path, channel=None, subject=None):
        """Recording chosen in tab 1 - the one place a file/subject is picked.

        `subject` comes straight from tab 1's own resolution (subject ->
        recording via beat_labels.SUBJECT_RECORDING), so this tab does not
        re-guess it from the filename and cannot get out of sync with tab 1.
        Without a subject there are no BP labels, so the r|SBP column and
        SBP/DBP overlay stay empty - that's expected for a manually-picked
        file with no known subject.
        """
        self.path = path
        self.subject = subject
        self.channel = channel if channel in bp.ECG_CHANNELS else bp.ECG_CHANNELS[0]
        base = os.path.basename(path)
        self.lbl_recording.setText(
            f"<b>{base}</b><br>subject: {subject or '(none - no BP labels)'}"
            f"<br>channel: {self.channel}")

    def on_run(self):
        if not self.path:
            QMessageBox.warning(self, "No recording",
                                "Pick a subject or a .txt file in tab 1 first.")
            return
        self.btn_run.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.res = ef.build_recording(
                self.path, subject=self.subject,
                window_sec=self.sb_win.value(), stride_sec=self.sb_stride.value(),
                analysis_fs=self.sb_afs.value(),
                ecg_channel=self.channel,
                feature_root=os.path.join(root, "feature_result", "5sWindow"),
                dur_sec=self.sb_dur.value() or None)
            self._fill_table()
            self.txt.setPlainText(ef.format_report(self.res))
            n = len(self.res["rp_a"])
            self.sl_beat.setEnabled(n > 0)
            self.sl_beat.setRange(0, max(0, n - 1))
            self.sl_beat.setValue(min(20, max(0, n - 1)))
            self.btn_csv.setEnabled(True)
            self.btn_npz.setEnabled(True)
            self._draw_feature()
            self._draw_beat()
        except Exception:
            QMessageBox.critical(self, "Failed", traceback.format_exc())
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_run.setEnabled(True)

    def _fill_table(self):
        r = self.res
        X = r["X"]
        has_bp = "SBP" in r and np.isfinite(r["SBP"]).any()
        self.tbl.setRowCount(len(ef.FEATURE_NAMES))
        for i, nm in enumerate(ef.FEATURE_NAMES):
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

    def _current_feature(self):
        i = self.tbl.currentRow()
        return max(0, i)

    def _draw_feature(self):
        if self.res is None:
            return
        r = self.res
        i = self._current_feature()
        nm = ef.FEATURE_NAMES[i]
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

    def _draw_beat(self):
        if self.res is None:
            return
        r = self.res
        k = self.sl_beat.value()
        ecg, rp, fs_a, beats = r["ecg_a"], r["rp_a"], r["analysis_fs"], r["beats"]
        if k >= len(rp):
            return
        R = int(rp[k])
        i0, i1 = max(0, R - int(0.20 * fs_a)), min(len(ecg), R + int(0.50 * fs_a))
        t = (np.arange(i0, i1) - R) / fs_a * 1000.0
        self.p_beat.clear()
        self.p_beat.plot(t, ecg[i0:i1], pen=PEN_ECG)
        base = beats["baseline"][k]
        if np.isfinite(base):
            self.p_beat.addLine(y=base, pen=pg.mkPen((150, 150, 150),
                                                     style=Qt.DotLine))
        for key, col, lbl in LANDMARKS:
            x = beats[key][k]
            if np.isfinite(x):
                ln = pg.InfiniteLine(pos=(x - R) / fs_a * 1000.0, angle=90,
                                     pen=pg.mkPen(col, style=Qt.DashLine),
                                     label=lbl,
                                     labelOpts={"position": 0.9, "color": col})
                self.p_beat.addItem(ln)
        self.p_beat.plot([0], [ecg[R]], pen=None, symbol="o", symbolSize=8,
                         symbolBrush=(220, 30, 30))
        q, tp, qt = beats["qrs_dur"][k], beats["tpte"][k], beats["qt"][k]
        self.p_beat.setTitle(f"beat {k}/{len(rp) - 1} at t={R / fs_a:.1f}s   "
                             f"QRS {q:.0f} ms   TpTe {tp:.0f} ms   QT {qt:.0f} ms")
        self.lbl_beat.setText(f"{k}/{len(rp) - 1}")
        self.p_beat.autoRange()

    def on_export(self):
        if self.res is None:
            return
        import pandas as pd
        p, _ = QFileDialog.getSaveFileName(self, "Export features",
                                           "ecg_features.csv", "CSV (*.csv)")
        if not p:
            return
        r = self.res
        df = pd.DataFrame(r["X"], columns=ef.FEATURE_NAMES)
        df.insert(0, "Subject_ID", r.get("subject") or "")
        df.insert(1, "Recording", os.path.basename(r["path"]))
        df.insert(2, "window_idx", np.arange(len(df)))
        df.insert(3, "t_center_sec", r["t_center_sec"])
        df.insert(4, "t_raw_center_sec", r["t_center_sec"] + r["clock_offset_sec"])
        df.insert(5, "n_beats", r["n_beats"])
        df.insert(6, "lf_reliable", r["lf_reliable"])
        for k in ("SBP", "DBP", "label_valid", "usable"):
            if k in r:
                df[k] = r[k]
        df.to_csv(p, index=False)
        QMessageBox.information(self, "Exported", f"{len(df)} windows -> {p}")

    def on_export_npz(self):
        if self.res is None:
            return
        subj = self.res.get("subject")
        default = f"{subj or 'ecg'}_features.npz"
        p, _ = QFileDialog.getSaveFileName(self, "Export features for training",
                                           default, "NumPy archive (*.npz)")
        if not p:
            return
        shape = ef.save_npz(p, self.res)
        note = ("" if subj else "\n\nNo subject was selected, so SBP/DBP are NaN "
                "and `usable` is all-False - this file has no labels to train on.")
        QMessageBox.information(
            self, "Exported",
            f"X{shape} -> {p}\n\nLoad it with:\n"
            f"  import ecg_features as ef\n"
            f"  d = ef.load_npz({os.path.basename(p)!r}, usable_only=True)" + note)

    def on_save_log(self):
        if not self.txt.toPlainText():
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save report", "ecg_report.txt",
                                           "Text (*.txt)")
        if p:
            with open(p, "w") as f:
                f.write(self.txt.toPlainText())
