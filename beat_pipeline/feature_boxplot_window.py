"""
Feature Distributions - box plots for every ECG and SKNA window feature
========================================================================
Tab 4 of the Beat Pipeline app. Loads the packed .npz cohort tables (built by
build_ecg_features.py --npz / build_skna_features.py --npz) and draws one box
plot per feature: 19 for ECG, 11 for SKNA, each its own subplot in a scrollable
grid so 19-30 small plots stay readable instead of being squeezed into one
window's worth of space.

GROUP BY MATTERS HERE MORE THAN USUAL. Pooled boxes mix all subjects into one
distribution and hide everything project memory has already found about this
cohort: several SKNA features separate the two recording batches ("*_10kHz.txt"
vs the rest) by 2-36x, which shows up as a pooled effect that vanishes within
subject. Group by Person (the default) to see each person's distribution
separately - Person, not Subject, because 16 subject IDs are only 14 people
(s13/s13_full are one person's two files, s5_session1/2 are one person's two
sessions; beat_labels.person_of). Pooled is kept for a quick single-glance
overview, not as the box to draw a conclusion from.
"""
import inspect
import os

import matplotlib
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

import ecg_features as ef
import skna_features as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ECG_NPZ = os.path.join(ROOT, "beat_pipeline", "built", "ecg_features_30s.npz")
DEFAULT_SKNA_NPZ = os.path.join(ROOT, "beat_pipeline", "built", "skna_features_30s.npz")
N_COLS = 4
SUBPLOT_W, SUBPLOT_H = 3.0, 2.6   # inches, per subplot - sets the canvas size
# boxplot's tick-label kwarg was renamed labels -> tick_labels in mpl 3.9,
# with the old name deprecated then removed (3.11 here); support whichever
# this environment has rather than pinning a version.
_BOXPLOT_LABEL_KW = ("tick_labels" if "tick_labels" in
                     inspect.signature(matplotlib.axes.Axes.boxplot).parameters
                     else "labels")


class MplCanvas(FigureCanvas):
    def __init__(self, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)


class _FamilyPanel(QWidget):
    """One family's (ECG or SKNA) scrollable grid of box plots."""

    def __init__(self, title):
        super().__init__()
        self.title = title
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.canvas = MplCanvas(width=N_COLS * SUBPLOT_W, height=SUBPLOT_H)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.canvas)
        v.addWidget(self.toolbar)
        v.addWidget(scroll)

    def draw(self, feature_names, X, groups, group_order, show_outliers):
        """X: (N, F). groups: (N,) label per row, only used when len(group_order) > 1.
        group_order=('pooled',) draws one box per feature from all rows."""
        n = len(feature_names)
        rows = int(np.ceil(n / N_COLS))
        self.canvas.fig.clear()
        fig_w_in, fig_h_in = N_COLS * SUBPLOT_W, rows * SUBPLOT_H
        self.canvas.fig.set_size_inches(fig_w_in, fig_h_in)

        # FigureCanvasQTAgg only syncs figure size <-> widget size in the
        # widget-drives-figure direction (its resizeEvent calls
        # set_size_inches). Growing the figure in code, as above, does NOT
        # grow the Qt widget - setMinimumSize alone doesn't either, because
        # Qt applies that through the layout system on the NEXT event loop
        # pass, after the canvas.draw() below already ran. Without an
        # explicit resize() here, paintEvent scales the full-resolution
        # raster buffer down into whatever (much smaller) widget rect Qt
        # still has, and nearest-neighbour image scaling turns the
        # anti-aliased tick labels and box lines into colored noise.
        dpi = self.canvas.fig.dpi
        px_w, px_h = int(round(fig_w_in * dpi)), int(round(fig_h_in * dpi))
        self.canvas.setMinimumSize(px_w, px_h)
        self.canvas.resize(px_w, px_h)

        pooled = len(group_order) == 1
        for i, nm in enumerate(feature_names):
            ax = self.canvas.fig.add_subplot(rows, N_COLS, i + 1)
            v = X[:, i]
            if pooled:
                data = [v[np.isfinite(v)]]
                labels = [""]
            else:
                data, labels = [], []
                for g in group_order:
                    gv = v[(groups == g) & np.isfinite(v)]
                    if gv.size:
                        data.append(gv)
                        labels.append(g)
            if not data or not any(d.size for d in data):
                ax.set_title(f"{nm}\n(no finite values)", fontsize=8)
                ax.set_xticks([])
                continue
            ax.boxplot(data, showfliers=show_outliers,
                      flierprops=dict(marker=".", markersize=2, alpha=0.4),
                      medianprops=dict(color="firebrick"),
                      **{_BOXPLOT_LABEL_KW: labels})
            ax.set_title(nm, fontsize=9)
            ax.tick_params(axis="both", labelsize=6)
            if not pooled:
                ax.tick_params(axis="x", rotation=90)
            else:
                ax.set_xticks([])
        self.canvas.fig.tight_layout()
        self.canvas.draw()


class FeatureBoxplotWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feature Distributions - box plots")
        self.resize(1200, 800)
        self.ecg = None
        self.skna = None

        root = QVBoxLayout(self)
        root.addLayout(self._build_controls())

        self.tabs = QTabWidget()
        self.panel_ecg = _FamilyPanel("ECG (19 features)")
        self.panel_skna = _FamilyPanel("SKNA (11 features)")
        self.tabs.addTab(self.panel_ecg, "ECG (19)")
        self.tabs.addTab(self.panel_skna, "SKNA (11)")
        root.addWidget(self.tabs, 1)

        self.le_ecg.setText(DEFAULT_ECG_NPZ)
        self.le_skna.setText(DEFAULT_SKNA_NPZ)

    # -- controls ------------------------------------------------------
    def _build_controls(self):
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("ECG npz"))
        self.le_ecg = QLabel(""); self.le_ecg.setStyleSheet("color: #555;")
        row1.addWidget(self.le_ecg, 1)
        b = QPushButton("Browse..."); b.clicked.connect(self.on_browse_ecg)
        row1.addWidget(b)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("SKNA npz"))
        self.le_skna = QLabel(""); self.le_skna.setStyleSheet("color: #555;")
        row2.addWidget(self.le_skna, 1)
        b = QPushButton("Browse..."); b.clicked.connect(self.on_browse_skna)
        row2.addWidget(b)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Group by"))
        self.cb_group = QComboBox()
        self.cb_group.addItems(["Person (recommended)", "Subject", "Pooled"])
        row3.addWidget(self.cb_group)
        self.ck_usable = QCheckBox("usable windows only")
        self.ck_usable.setChecked(True)
        row3.addWidget(self.ck_usable)
        self.ck_outliers = QCheckBox("show outlier points")
        self.ck_outliers.setChecked(False)
        row3.addWidget(self.ck_outliers)
        self.btn_reload = QPushButton("Reload && redraw")
        self.btn_reload.setMinimumHeight(28)
        self.btn_reload.clicked.connect(self.on_reload)
        row3.addWidget(self.btn_reload)
        row3.addStretch(1)
        row3.addWidget(QLabel(
            "<i>Pooled mixes every subject into one box and hides batch "
            "effects already found in this cohort - read Person, not Pooled, "
            "before drawing a conclusion.</i>"))

        v = QVBoxLayout()
        v.addLayout(row1); v.addLayout(row2); v.addLayout(row3)
        return v

    # -- actions ---------------------------------------------------------
    def on_browse_ecg(self):
        p, _ = QFileDialog.getOpenFileName(self, "ECG features npz",
                                           os.path.dirname(self.le_ecg.text()) or ROOT,
                                           "NumPy archive (*.npz)")
        if p:
            self.le_ecg.setText(p)

    def on_browse_skna(self):
        p, _ = QFileDialog.getOpenFileName(self, "SKNA features npz",
                                           os.path.dirname(self.le_skna.text()) or ROOT,
                                           "NumPy archive (*.npz)")
        if p:
            self.le_skna.setText(p)

    def on_reload(self):
        usable = self.ck_usable.isChecked()
        outliers = self.ck_outliers.isChecked()
        mode = self.cb_group.currentText()
        group_key = "pooled" if mode.startswith("Pooled") else (
            "person" if mode.startswith("Person") else "subject")

        errs = []
        for path, loader, panel, names, attr in (
            (self.le_ecg.text(), ef.load_npz, self.panel_ecg, ef.FEATURE_NAMES, "ecg"),
            (self.le_skna.text(), sf.load_npz, self.panel_skna, sf.FEATURE_NAMES, "skna"),
        ):
            if not path or not os.path.isfile(path):
                errs.append(f"{os.path.basename(path) or '(no path)'}: file not found "
                           f"- build it with build_{'ecg' if attr == 'ecg' else 'skna'}"
                           f"_features.py --npz first")
                setattr(self, attr, None)
                continue
            try:
                d = loader(path, usable_only=usable)
            except Exception as e:
                errs.append(f"{os.path.basename(path)}: {type(e).__name__}: {e}")
                setattr(self, attr, None)
                continue
            setattr(self, attr, d)
            if len(d["X"]) == 0:
                panel.canvas.fig.clear(); panel.canvas.draw()
                errs.append(f"{os.path.basename(path)}: no rows after filtering")
                continue
            if group_key == "pooled":
                groups, order = None, ("pooled",)
            else:
                groups = d[group_key]
                order = tuple(sorted(np.unique(groups)))
            panel.draw(names, d["X"], groups, order, outliers)

        if errs:
            QMessageBox.warning(self, "Some tables did not load", "\n".join(errs))
