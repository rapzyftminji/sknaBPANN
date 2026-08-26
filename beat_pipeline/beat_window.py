"""
Beat Pipeline - preprocessing & R-peak QC
=========================================
Standalone Qt tool for stage 1 of the beat-segment (ANN-LSTM) architecture:
load a BIOPAC .txt recording, high-pass it, run Pan-Tompkins, and INSPECT
the result before anything downstream trusts those peak positions.

Every Pan-Tompkins stage is plotted on a shared, linked x-axis so a missed
or doubled beat can be traced back to the stage that caused it (usually the
integration threshold during a drifty stretch).

The QC panel reports the 3-R-peak segment-duration distribution and the
resample length L implied by it - that is the number stage 2 needs, and it
should come from the real RR distribution rather than an assumed HR range.

Run: python3 beat_pipeline/main.py
"""
import os
import traceback

import numpy as np
import pyqtgraph as pg
# pyqtgraph defaults to a BLACK background, which hides the dark pens below.
# Matches src/main_window.py and recording_cutter so all tools look alike.
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)

import beat_labels as bl
import beat_processing as bp

MAX_PLOT_POINTS = 200_000     # DISPLAY decimation only; detection uses full data
PEN_RAW = pg.mkPen((170, 170, 170), width=1)
PEN_MAIN = pg.mkPen((20, 20, 20), width=1)
PEN_STAGE = pg.mkPen((30, 90, 200), width=1)
PEN_THR = pg.mkPen((200, 60, 60), width=1, style=Qt.DashLine)


def _decimate(t, y, max_points=MAX_PLOT_POINTS):
    """Stride-decimate a curve for display. Peak MARKERS are never decimated -
    they are drawn from true sample indices so their positions stay exact."""
    n = len(y)
    if n <= max_points:
        return t, y
    step = int(np.ceil(n / max_points))
    return t[::step], y[::step]


def _decimate_minmax(t, y, max_points=MAX_PLOT_POINTS):
    """Envelope-preserving decimation for the ECG traces.

    Plain stride decimation at 10 kHz throws away the R-peak SAMPLE itself,
    so the drawn trace sits below the true peak. Keeping each bin's min and
    max preserves peak amplitude.

    The two points must be emitted at DIFFERENT x, in the order they occur
    within the bin. Emitting both at the bin's start x (the obvious way)
    draws a vertical bar then a flat jump, which renders a clean ECG as a
    staircase and looks like bad data.
    """
    n = len(y)
    if n <= max_points:
        return t, y
    step = int(np.ceil(n / (max_points // 2)))
    m = (n // step) * step
    if m == 0:
        return t, y
    yy = y[:m].reshape(-1, step)
    tt = t[:m].reshape(-1, step)
    imin = yy.argmin(axis=1)
    imax = yy.argmax(axis=1)
    first = np.minimum(imin, imax)          # keep chronological order within
    second = np.maximum(imin, imax)         # the bin so the line still flows
    rows = np.arange(yy.shape[0])
    out_y = np.empty(yy.shape[0] * 2, dtype=y.dtype)
    out_t = np.empty(yy.shape[0] * 2, dtype=t.dtype)
    out_y[0::2] = yy[rows, first]; out_t[0::2] = tt[rows, first]
    out_y[1::2] = yy[rows, second]; out_t[1::2] = tt[rows, second]
    return out_t, out_y


class BeatPipelineWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beat Pipeline - preprocessing & R-peak QC")
        self.resize(1150, 720)

        self.path = None
        self.subject = None
        self.fs = bp.DEFAULT_FS
        self.raw = None
        self.filtered = None
        self.rpeaks = None
        self.stats = None

        # Controls go inside a QScrollArea and the QC log inside a vertical
        # splitter: on a laptop screen the fixed layout ran off the bottom
        # and the log could not be reached at all.
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidget(self._build_controls())
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMinimumWidth(300)

        left = QSplitter(Qt.Vertical)
        left.addWidget(ctrl_scroll)
        left.addWidget(self._build_log())
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 2)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._build_plots())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 760])

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.addWidget(splitter)

    # -- controls -----------------------------------------------------------
    def _build_controls(self):
        panel = QWidget()
        v = QVBoxLayout(panel)

        # file
        gb_file = QGroupBox("Recording (loaded once here, used by every tab)")
        fl = QFormLayout(gb_file)
        self.cb_subject = QComboBox()
        self.cb_subject.addItem("(pick a file manually)")
        self.cb_subject.addItems(list(bl.SUBJECT_RECORDING))
        self.cb_subject.currentTextChanged.connect(self.on_subject)
        self.le_path = QLineEdit(); self.le_path.setReadOnly(True)
        btn_browse = QPushButton("Browse .txt...")
        btn_browse.clicked.connect(self.on_browse)
        self.cb_channel = QComboBox(); self.cb_channel.addItems(bp.ECG_CHANNELS)
        self.sb_fs = QDoubleSpinBox()
        self.sb_fs.setRange(1.0, 100000.0); self.sb_fs.setDecimals(1)
        self.sb_fs.setValue(bp.DEFAULT_FS); self.sb_fs.setSuffix(" Hz")
        self.sb_start = QDoubleSpinBox()
        self.sb_start.setRange(0.0, 1e6); self.sb_start.setSuffix(" s")
        self.sb_dur = QDoubleSpinBox()
        self.sb_dur.setRange(0.0, 1e6); self.sb_dur.setValue(0.0)
        self.sb_dur.setSuffix(" s  (0 = whole file)")
        fl.addRow("Subject", self.cb_subject)
        fl.addRow(self.le_path); fl.addRow(btn_browse)
        fl.addRow("ECG channel", self.cb_channel)
        fl.addRow("Sample rate", self.sb_fs)
        fl.addRow("Start", self.sb_start)
        fl.addRow("Duration", self.sb_dur)
        fl.addRow(QLabel("<i>Picking a subject resolves both the .txt<br>"
                         "recording and its BP labels csv. Tabs 2 and 3<br>"
                         "pick up this same subject/file/channel automatically -<br>"
                         "no need to load it again there.</i>"))
        v.addWidget(gb_file)

        # FIR delay compensation
        gb_delay = QGroupBox("FIR delay vs CH1")
        fd = QFormLayout(gb_delay)
        self.ck_delay = QCheckBox("auto-measure && compensate")
        self.ck_delay.setChecked(True)
        self.lbl_delay = QLabel("<i>not measured</i>")
        self.lbl_delay.setWordWrap(True)
        fd.addRow(self.ck_delay)
        fd.addRow(self.lbl_delay)
        fd.addRow(QLabel("<i>CH40 is FIR-filtered and LAGS CH1 by<br>"
                         "0.7-7 s depending on the recording.<br>"
                         "CH1/CH41 have no delay. Leaving this<br>"
                         "off misaligns ECG against SKNA.</i>"))
        v.addWidget(gb_delay)

        # start-of-recording transient
        gb_lead = QGroupBox("Lead-in transient")
        gl = QFormLayout(gb_lead)
        self.ck_leadin = QCheckBox("auto-detect && skip")
        self.ck_leadin.setChecked(False)
        self.sb_leadin_factor = QDoubleSpinBox()
        self.sb_leadin_factor.setRange(1.5, 20.0); self.sb_leadin_factor.setValue(3.0)
        self.sb_leadin_factor.setSingleStep(0.5); self.sb_leadin_factor.setSuffix(" x steady")
        self.lbl_leadin = QLabel("<i>not measured</i>")
        gl.addRow(self.ck_leadin)
        gl.addRow("Threshold", self.sb_leadin_factor)
        gl.addRow("Skipped", self.lbl_leadin)
        gl.addRow(QLabel("<i>OFF by default: the detector already<br>"
                         "handles transients, and this trim is<br>"
                         "load-length dependent, which would shift<br>"
                         "BP labels. Inspection only.</i>"))
        v.addWidget(gb_lead)

        # high-pass
        gb_hpf = QGroupBox("High-pass (baseline wander)")
        fh = QFormLayout(gb_hpf)
        self.ck_hpf = QCheckBox("enable"); self.ck_hpf.setChecked(True)
        self.sb_fc = QDoubleSpinBox()
        self.sb_fc.setRange(0.001, 50.0); self.sb_fc.setDecimals(3)
        self.sb_fc.setValue(0.08); self.sb_fc.setSuffix(" Hz")
        self.sb_order = QSpinBox(); self.sb_order.setRange(1, 8); self.sb_order.setValue(2)
        fh.addRow(self.ck_hpf)
        fh.addRow("Corner fc", self.sb_fc)
        fh.addRow("Order", self.sb_order)
        fh.addRow(QLabel("<i>zero-phase (filtfilt) - R-peak<br>timing is preserved</i>"))
        v.addWidget(gb_hpf)

        # pan-tompkins
        gb_pt = QGroupBox("Pan-Tompkins")
        fp = QFormLayout(gb_pt)
        self.sb_bp_lo = QDoubleSpinBox(); self.sb_bp_lo.setRange(0.1, 100.0)
        self.sb_bp_lo.setValue(5.0); self.sb_bp_lo.setSuffix(" Hz")
        self.sb_bp_hi = QDoubleSpinBox(); self.sb_bp_hi.setRange(1.0, 200.0)
        self.sb_bp_hi.setValue(15.0); self.sb_bp_hi.setSuffix(" Hz")
        self.sb_integ = QDoubleSpinBox(); self.sb_integ.setRange(10.0, 500.0)
        self.sb_integ.setValue(150.0); self.sb_integ.setSuffix(" ms")
        self.sb_refr = QDoubleSpinBox(); self.sb_refr.setRange(50.0, 500.0)
        self.sb_refr.setValue(200.0); self.sb_refr.setSuffix(" ms")
        self.sb_twave = QDoubleSpinBox(); self.sb_twave.setRange(100.0, 800.0)
        self.sb_twave.setValue(360.0); self.sb_twave.setSuffix(" ms")
        self.sb_dfs = QDoubleSpinBox(); self.sb_dfs.setRange(50.0, 2000.0)
        self.sb_dfs.setValue(250.0); self.sb_dfs.setSuffix(" Hz")
        self.sb_refine = QDoubleSpinBox(); self.sb_refine.setRange(0.0, 200.0)
        self.sb_refine.setValue(50.0); self.sb_refine.setSuffix(" ms")
        fp.addRow("Bandpass low", self.sb_bp_lo)
        fp.addRow("Bandpass high", self.sb_bp_hi)
        fp.addRow("Integration window", self.sb_integ)
        fp.addRow("Refractory", self.sb_refr)
        fp.addRow("T-wave window", self.sb_twave)
        fp.addRow("Detection rate", self.sb_dfs)
        fp.addRow("Peak refine +/-", self.sb_refine)
        v.addWidget(gb_pt)

        # segmentation target
        gb_seg = QGroupBox("Stage-2 target (for L)")
        fs2 = QFormLayout(gb_seg)
        self.sb_target_fs = QDoubleSpinBox()
        self.sb_target_fs.setRange(100.0, 20000.0); self.sb_target_fs.setValue(2000.0)
        self.sb_target_fs.setSuffix(" Hz")
        fs2.addRow("Segment resample rate", self.sb_target_fs)
        v.addWidget(gb_seg)

        self.btn_run = QPushButton("Run preprocessing")
        self.btn_run.setMinimumHeight(34)
        self.btn_run.clicked.connect(self.on_run)
        v.addWidget(self.btn_run)
        v.addStretch(1)
        return panel

    def _build_log(self):
        box = QWidget()
        lv = QVBoxLayout(box)
        lv.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("<b>QC report</b>"))
        row.addStretch(1)
        btn_save = QPushButton("Save...")
        btn_save.clicked.connect(self.on_save_log)
        row.addWidget(btn_save)
        lv.addLayout(row)
        self.txt_qc = QPlainTextEdit(); self.txt_qc.setReadOnly(True)
        self.txt_qc.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.txt_qc.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.txt_qc.setMinimumHeight(150)
        lv.addWidget(self.txt_qc, 1)
        return box

    def on_save_log(self):
        if not self.txt_qc.toPlainText():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save QC report", "beat_qc.txt",
                                              "Text files (*.txt)")
        if path:
            with open(path, "w") as f:
                f.write(self.txt_qc.toPlainText())

    # -- plots --------------------------------------------------------------
    def _build_plots(self):
        self.plots = pg.GraphicsLayoutWidget()
        titles = [
            "1. ECG - raw (grey) vs high-passed (black)",
            "2. Bandpass 5-15 Hz (QRS isolation)",
            "3. Squared derivative",
            "4. Moving-window integration + adaptive threshold",
            "5. High-passed ECG with detected R-peaks",
        ]
        self.p = []
        for i, t in enumerate(titles):
            plot = self.plots.addPlot(row=i, col=0, title=t)
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel("bottom", "time", units="s")
            if i > 0:
                plot.setXLink(self.p[0])
            self.p.append(plot)
        self.scatter = pg.ScatterPlotItem(size=11, brush=pg.mkBrush(230, 30, 30),
                                          pen=pg.mkPen("k", width=1), symbol="o")
        self.scatter.setZValue(100)      # keep markers above the ECG curve
        self.p[4].addItem(self.scatter)
        return self.plots

    # -- actions ------------------------------------------------------------
    def on_browse(self):
        start_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "dataset", "txt")
        if not os.path.isdir(start_dir):
            start_dir = ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open BIOPAC .txt recording", start_dir, "Text files (*.txt);;All files (*)")
        if not path:
            return
        # A manually-picked file has no known subject, so it carries no BP
        # labels downstream - reset the combo rather than leave it pointing
        # at a subject this file no longer matches.
        self.cb_subject.blockSignals(True)
        self.cb_subject.setCurrentIndex(0)
        self.cb_subject.blockSignals(False)
        self.subject = None
        self.path = path
        self.le_path.setText(os.path.basename(path))
        self.sb_fs.setValue(bp.detect_fs(path))
        self.txt_qc.setPlainText(f"Loaded path: {path}\nHeader fs: {self.sb_fs.value():.1f} Hz\n"
                                 "No subject matched - BP labels will not be available "
                                 "downstream.\nPress 'Run preprocessing'.")

    def on_subject(self, name):
        if name not in bl.SUBJECT_RECORDING:
            self.subject = None
            return
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "dataset", "txt", bl.SUBJECT_RECORDING[name])
        self.subject = name
        if not os.path.isfile(path):
            self.txt_qc.setPlainText(f"Subject {name!r} maps to\n  {path}\n"
                                     "but that file was not found.")
            return
        self.path = path
        self.le_path.setText(os.path.basename(path))
        self.sb_fs.setValue(bp.detect_fs(path))
        self.txt_qc.setPlainText(f"Loaded path: {path}\nSubject: {name}\n"
                                 f"Header fs: {self.sb_fs.value():.1f} Hz\n"
                                 "Press 'Run preprocessing'.")

    def on_run(self):
        if not self.path:
            QMessageBox.warning(self, "No file", "Pick a .txt recording first.")
            return
        self.btn_run.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._run()
        except Exception:
            QMessageBox.critical(self, "Preprocessing failed", traceback.format_exc())
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_run.setEnabled(True)

    def _run(self):
        # All the actual work lives in bp.run_pipeline(), which the matplotlib
        # front-end (beat_plot.py) also calls - so the two tools cannot drift
        # apart in what they compute. This method only maps widgets to
        # arguments and the result back onto the display.
        res = bp.run_pipeline(
            self.path, channel=self.cb_channel.currentText(), fs=self.sb_fs.value(),
            start_sec=self.sb_start.value() or None, dur_sec=self.sb_dur.value() or None,
            compensate_delay=self.ck_delay.isChecked(),
            skip_leadin=self.ck_leadin.isChecked(),
            leadin_factor=self.sb_leadin_factor.value(),
            hpf=self.ck_hpf.isChecked(), fc=self.sb_fc.value(),
            order=self.sb_order.value(), target_fs=self.sb_target_fs.value(),
            bp_low=self.sb_bp_lo.value(), bp_high=self.sb_bp_hi.value(),
            integ_ms=self.sb_integ.value(), refractory_ms=self.sb_refr.value(),
            twave_ms=self.sb_twave.value(), detect_fs=self.sb_dfs.value(),
            refine_ms=self.sb_refine.value())
        self.res = res
        self.raw, self.filtered = res["raw"], res["filtered"]
        self.fs, self.rpeaks, self.stats = res["fs"], res["rpeaks"], res["stats"]

        r, delay = res["delay_r"], res["delay"]
        if not self.ck_delay.isChecked() or self.cb_channel.currentText() == "CH1":
            self.lbl_delay.setText("<i>not applied</i>")
        elif r is not None and r < 0.5:
            self.lbl_delay.setText(f"<b>measured but r={r:.2f} - NOT applied</b><br>"
                                   "<i>too weak to trust</i>")
        else:
            self.lbl_delay.setText(f"<b>{delay:+.2f} s</b> (r={r:.2f}) - compensated")

        if not self.ck_leadin.isChecked():
            self.lbl_leadin.setText("<i>not applied</i>")
        elif res["leadin"] > 0:
            self.lbl_leadin.setText(f"<b>{res['leadin']:.2f} s</b> skipped")
        else:
            self.lbl_leadin.setText("none detected")

        self._draw(res["stages"])
        self.txt_qc.setPlainText(bp.format_qc_report(res))

    def _draw(self, stages):
        # NOTE: ScatterPlotItem.implements('plotData') is True, so pyqtgraph
        # tracks it in dataItems and a blanket listDataItems()/removeItem()
        # sweep DELETES the R-peak markers - they then never reappear. Clear
        # the plots wholesale and re-add the scatter each redraw instead.
        for plot in self.p:
            plot.clear()
        self.scatter.clear()
        self.p[4].addItem(self.scatter)

        fs, raw, filt = self.fs, self.raw, self.filtered
        t = np.arange(len(raw)) / fs
        td, rd = _decimate_minmax(t, raw); self.p[0].plot(td, rd, pen=PEN_RAW)
        td, fd = _decimate_minmax(t, filt); self.p[0].plot(td, fd, pen=PEN_MAIN)

        fsd = stages["fs_detect"]
        t2 = np.arange(len(stages["bandpassed"])) / fsd
        for i, key in enumerate(("bandpassed", "squared", "integrated"), start=1):
            a, b = _decimate(t2, stages[key])
            self.p[i].plot(a, b, pen=PEN_STAGE)

        # threshold trace: THRESHOLD_I1 is adaptive, but its initial value is a
        # useful visual reference for "did this beat clear the bar?"
        integ = stages["integrated"]
        init = integ[: int(round(2.0 * fsd))] if len(integ) > 2 * fsd else integ
        if init.size:
            spki, npki = float(np.max(init)), float(np.mean(init))
            thr = npki + 0.25 * (spki - npki)
            self.p[3].addLine(y=thr, pen=PEN_THR)

        td, fd = _decimate_minmax(t, filt); self.p[4].plot(td, fd, pen=PEN_MAIN)
        if len(self.rpeaks):
            self.scatter.setData(self.rpeaks / fs, filt[self.rpeaks])
            self.p[4].setTitle(f"5. High-passed ECG with {len(self.rpeaks)} detected "
                               f"R-peaks (red)")
        else:
            self.p[4].setTitle("5. High-passed ECG - NO R-peaks detected")

        # Open on the first ~20 s: at full extent 300 markers collapse into a
        # solid band and you cannot tell a good detection from a doubled one.
        self.p[0].autoRange()
        span = min(20.0, t[-1] if len(t) else 20.0)
        self.p[0].setXRange(0, span, padding=0)
        for plot in self.p:
            plot.enableAutoRange(axis="y")
