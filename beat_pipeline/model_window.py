#!/usr/bin/env python3
"""
Tab 6: BP model - two-layer ANN, leave-one-subject-out.
=======================================================

The interactive front-end for ann_bp_loso.py. The target is SBP and DBP - the
project goal - reached with the Yao et al. two-layer feedforward network
(sigmoid hidden, two LINEAR output neurons) instead of the CNN-BiLSTM on raw
waveforms in src/model.py. CPT classification is kept as a secondary task
because it is the sanity check that the features carry autonomic information at
all; if CPT collapses, a BP number from the same features means nothing.

Feature source is a choice:

  * "Engineer here" runs feature_engineering.py with the controls below.
  * "From tab 5" takes the EXACT working set showing in the Feature
    Engineering tab, including any top-K filter applied there. Set the pipeline
    up in tab 5, look at it, then train on precisely that.

Calibration is the control that matters most for BP, and it changes which
baseline the model must beat:

  none    calibration-free, absolute mmHg for an unseen person; baseline is the
          constant train-mean predictor.
  offset  the held-out person's first N minutes of reference BP set an offset
          and the net predicts deviation from it; baseline is ZERO-DELTA.

Both baselines are always in the results table, and so is the ORACLE
subject-mean row - the one that reproduces Yao et al.'s headline accuracy using
no features at all. A BP MAE without its baseline beside it is not a result.
"""
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QRadioButton, QScrollArea, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_ablation as ma
import ann_bp_loso as abp
import ann_loso as al
import ann_personalize as apz
import ann_training_curve as atc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAMILIES = ["SKNA", "XSIG", "HRV", "ECG_MORPH", "NONLIN"]
FAMILY_HINT = {
    "SKNA": "burst content + SKNA-internal ratios",
    "XSIG": "cross-signal: SKNA burden per beat, per unit HRV",
    "HRV": "rate and rate variability (degenerate at 5 s windows)",
    "ECG_MORPH": "beat shape: QRS, QT, ST, T/R",
    "NONLIN": "complexity descriptors of the ECG trace",
}
METHODS = [
    ("expanding", "expanding - causal, past windows only (deployable)"),
    ("robust", "robust - whole-recording median/IQR (transductive)"),
    ("zscore", "zscore - whole-recording mean/std"),
    ("baseline", "baseline - pre-CPT rest only (LABEL-PEEKING)"),
    ("none", "none - raw levels, keeps the recording offset"),
]
BP_COLS = ["model", "MAE_SBP", "ME_SBP", "SDE_SBP", "MAE_DBP", "ME_DBP",
           "SDE_DBP", "features", "norm", "calib", "excluded", "seconds"]
CPT_COLS = ["model", "auc_pooled", "ap_pooled", "auc_fold_mean",
            "auc_fold_std", "sens_mean", "spec_mean", "folds_above_chance",
            "features", "norm", "excluded", "seconds"]


class MplCanvas(FigureCanvas):
    def __init__(self, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.fig)


class RunWorker(QThread):
    progress = pyqtSignal(str)      # short status line beside the busy bar
    log = pyqtSignal(str)           # streamed into the Log tab, fold by fold
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, cfg, supplied=None):
        super().__init__()
        self.cfg = cfg
        self.supplied = supplied      # (df, cols) handed over from tab 5

    def _features(self):
        """-> X, names, y_cpt, Y_bp, groups, t_center, cycle_min, recording."""
        c = self.cfg
        if self.supplied is not None:
            df, cols = self.supplied
            need = {"person", "SBP", "DBP", "is_cpt", "t_center_sec",
                    "cycle_min"}
            missing = need - set(df.columns)
            if missing:
                raise ValueError(f"tab 5 table is missing {sorted(missing)}")
            # Recording keys the personalization time-split, so a two-session
            # person is split inside each session. Falling back to `person`
            # would make session 1 "adapt" and session 2 "test".
            rec = (df["Recording"].to_numpy().astype(str)
                   if "Recording" in df.columns
                   else df["person"].to_numpy().astype(str))
            return (df[cols].to_numpy(float), list(cols),
                    df["is_cpt"].to_numpy(int),
                    df[["SBP", "DBP"]].to_numpy(float),
                    df["person"].to_numpy(),
                    df["t_center_sec"].to_numpy(float),
                    df["cycle_min"].to_numpy(float), rec)
        X, names, y_cpt, y_sbp, groups, meta = ma.build(
            c["method"], c["ecg"], c["skna"], c["out"], c["k"], True)
        return (np.asarray(X, float), names, np.asarray(y_cpt).astype(int),
                np.column_stack([np.asarray(y_sbp, float),
                                 np.asarray(meta["y_dbp"], float)]),
                np.asarray(groups), np.asarray(meta["t_center_sec"], float),
                np.asarray(meta["cycle_min"], float),
                np.asarray(meta["recording"]).astype(str))

    def run(self):
        try:
            c = self.cfg
            self.progress.emit("preparing features ...")
            X, names, y_cpt, Y, groups, t_center, cyc, rec = self._features()
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            if c.get("exclude"):
                drop = np.isin(rec, list(c["exclude"]))
                if not drop.any():
                    self.failed.emit(
                        "None of the excluded recordings are in this table:\n"
                        + "\n".join(c["exclude"])
                        + "\n\nHit 'Refresh list' - the exclusion list was "
                          "built from a different source.")
                    return
                gone = sorted(set(groups[drop]) - set(groups[~drop]))
                self.log.emit(
                    f"\nEXCLUDED {int(drop.sum())} windows from "
                    f"{len(c['exclude'])} recording(s): "
                    + ", ".join(sorted(c["exclude"]))
                    + (f"\n  people removed from LOSO: {', '.join(gone)}"
                       if gone else
                       "\n  no person fully removed"))
                keep = ~drop
                X, y_cpt, Y = X[keep], y_cpt[keep], Y[keep]
                groups, t_center = groups[keep], t_center[keep]
                cyc, rec = cyc[keep], rec[keep]
                if len(np.unique(groups)) < 3:
                    self.failed.emit(
                        f"Only {len(np.unique(groups))} people left after "
                        "exclusions - LOSO needs at least 3.")
                    return

            cols = [i for i, n in enumerate(names)
                    if ma.FAMILY[ma.split_column(n)[0]] in c["families"]
                    and (c["temporal"] or ma.is_level(n))]
            if not cols:
                self.failed.emit("No features match that family / block "
                                 "combination.")
                return
            Xf, feat = X[:, cols], [names[i] for i in cols]
            common = dict(n_windows=len(groups), n_features=len(cols),
                          n_people=len(np.unique(groups)), features=feat,
                          task=c["task"], groups=groups)

            if c["task"] == "BP":
                out = self._run_bp(c, Xf, Y, groups, t_center, common, rec)
            else:
                out = self._run_cpt(c, Xf, y_cpt, groups, cyc, common)
            self.done.emit(out)
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _run_bp(self, c, Xf, Y, groups, t_center, out, rec):
        out.update(Y=Y, runs=[], folds={}, pred={})
        specs = [dict(c["hp"], hidden=h) for h in c["hidden"]]
        todo = [None] if c["inner_select"] else specs
        for hp in todo:
            name = ("ANN(nested)" if hp is None
                    else abp.arch_name(hp["hidden"]))
            self.progress.emit(f"{name} BP - {out['n_people']} folds ...")
            self.log.emit(f"\n=== {name} | {out['n_features']} features | "
                          f"calibration={c['calibration']} | "
                          f"solver={c['hp']['solver']} "
                          f"activation={c['hp']['activation']} "
                          f"alpha={c['hp']['alpha']} "
                          f"max_iter={c['hp']['max_iter']} ===")
            t0 = time.time()
            oof, base, f = abp.run_loso(
                Xf, Y, groups, t_center,
                (None if hp is None else (lambda hp=hp: abp.make_ann(
                    hp, c["seed"]))),
                c["calibration"], c["calib_min"], progress=self.log.emit,
                specs=(specs if hp is None else None), seed=c["seed"])
            s = abp.score(Y, oof, name)
            s.update(seconds=time.time() - t0, features=out["n_features"],
                     norm=c["method"], calib=c["calibration"])
            out["runs"].append(s)
            out["folds"][name] = f
            out["pred"][name] = oof
            self.log.emit(f"  {name}: SBP MAE {s['MAE_SBP']:.2f}  "
                          f"DBP MAE {s['MAE_DBP']:.2f}  "
                          f"[{s['seconds']:.0f}s]")
        if c["compare"]:
            self.progress.emit("ridge reference ...")
            self.log.emit("\n=== ridge (linear reference) ===")
            t0 = time.time()
            oof, base, f = abp.run_loso(
                Xf, Y, groups, t_center, lambda: abp.make_ref(c["seed"]),
                c["calibration"], c["calib_min"], progress=self.log.emit)
            s = abp.score(Y, oof, "ridge")
            s.update(seconds=time.time() - t0, features=out["n_features"],
                     norm=c["method"], calib=c["calibration"])
            out["runs"].append(s)
            out["folds"][s["model"]] = f
            out["pred"][s["model"]] = oof

        bname = ("BASELINE zero-delta" if c["calibration"] == "offset"
                 else "BASELINE train mean")
        s = abp.score(Y, base, bname)
        s.update(features=0, norm=c["method"], calib=c["calibration"])
        out["runs"].append(s)
        out["baseline_mae"] = s["MAE_SBP"]
        s = abp.score(Y, abp.oracle_subject_mean(Y, groups),
                      "ORACLE subject mean")
        s.update(features=0, norm=c["method"], calib=c["calibration"])
        out["runs"].append(s)

        if c.get("curve"):
            self.progress.emit("training curve - warm-started refit ...")
            self.log.emit(
                f"\n=== training curve: to {c['hp']['max_iter']} iterations in "
                f"steps of {c['curve_step']} ===\n"
                "  lbfgs exposes no per-iteration loss, so the curve is a\n"
                "  SEPARATE warm-started refit, evaluated at every checkpoint.\n"
                "  Its curvature history restarts each step, so read the shape,\n"
                "  not the exact iteration counts.")
            hp0 = dict(c["hp"], hidden=c["hidden"][0])
            t0 = time.time()
            out["curve"] = atc.curves(
                Xf, Y, groups, t_center, hp0, c["calibration"], c["calib_min"],
                c["hp"]["max_iter"], c["curve_step"], c["seed"],
                progress=self.log.emit)
            out["curve_arch"] = abp.arch_name(c["hidden"][0])
            self.log.emit(f"  curve done [{time.time() - t0:.0f}s]")

        if c.get("personalize"):
            self.progress.emit("per-subject fine-tune ...")
            self.log.emit(
                "\n=== personalization: head-only fine-tune ===\n"
                "  Base ANN per fold trained WITHOUT the held-out person, then\n"
                "  its input->hidden layer frozen and only the head refit on\n"
                "  that person's adapt slice, early-stopped on the val slice.\n"
                "  Scored on the test slice only - NOT comparable to the MAEs\n"
                "  above, which score every window.")
            hp0 = dict(c["hp"], hidden=c["hidden"][0])
            ft, curves, preds = apz.run(
                Xf, Y, groups, t_center, rec, hp0, c["seed"], c["calib_min"],
                c["adapt_end"], c["val_end"], c["ft_lr"], c["ft_epochs"],
                c["ft_patience"], progress=self.log.emit)
            if len(ft):
                self.log.emit(apz.summarize(ft))
                out["ft"] = ft
                out["ft_curves"] = curves
                out["ft_preds"] = preds
            else:
                self.log.emit("  no subject produced a usable three-way "
                              "split; nothing to fine-tune.")
        return out

    def _run_cpt(self, c, Xf, y, groups, cyc, out):
        out.update(y=y, runs=[], folds={}, oof={}, thr={},
                   prevalence=float(y.mean()))
        # the CPT classifier keeps the paper's single hidden layer, so a depth
        # spec like 10x2 uses only its width here
        specs = [(abp.arch_name(h),
                  lambda h=h: al.make_ann(h[0], c["seed"], c["hp"]["alpha"]))
                 for h in c["hidden"]]
        if c["compare"]:
            specs.append(("logistic", lambda: al.make_ref(c["seed"])))
        for name, mk in specs:
            self.progress.emit(f"{name} CPT - {out['n_people']} folds ...")
            self.log.emit(f"\n=== {name} CPT | {out['n_features']} features | "
                          f"threshold FPR {100 * c['fpr']:.1f}% ===")
            t0 = time.time()
            oof, thr, f = al.run_loso(Xf, y, groups, cyc, mk, c["fpr"],
                                      c["seed"])
            self.log.emit(f"  per-subject AUC: " + ", ".join(
                f"{r.person} {r.auc:.3f}" for r in f.itertuples()))
            s = al.summarize(name, y, oof, f, time.time() - t0)
            s.update(features=out["n_features"], norm=c["method"])
            out["runs"].append(s)
            out["folds"][name] = f
            out["oof"][name] = oof
            out["thr"][name] = thr
        return out


class ModelWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BP model - ANN, leave-one-subject-out")
        self.history = []
        self.last = None
        self.worker = None
        self.supplied = None          # (df, cols) from tab 5
        self.ft_curves = None         # per-epoch fine-tune history
        self.curve = None             # per-iteration base-ANN training curve
        self.excluded = []            # recording names dropped before the split

        ctrl = QScrollArea()
        ctrl.setWidget(self._build_controls())
        ctrl.setWidgetResizable(True)
        ctrl.setMinimumWidth(350)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(ctrl)
        split.addWidget(self._build_right())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([410, 880])

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.addWidget(split)
        self._on_task()
        # the list depends on which table the run would use, so keep it in step
        self.cb_method.currentIndexChanged.connect(self._refresh_excl)
        self.rb_tab5.toggled.connect(self._refresh_excl)
        self.le_ecg.editingFinished.connect(self._refresh_excl)
        self._refresh_excl()

    # -- handoff from tab 5 --------------------------------------------
    def set_feature_set(self, df, cols, notes=None):
        """Called by main.py when tab 5 has a table and this tab opens."""
        self.supplied = (df, list(cols))
        self.lbl_tab5.setText(
            f"<b>{len(cols)}</b> features x {len(df)} windows available "
            f"from tab 5")
        self.rb_tab5.setEnabled(True)
        if self.rb_tab5.isChecked():
            self._refresh_excl()

    # -- controls ------------------------------------------------------
    def _build_controls(self):
        panel = QWidget()
        v = QVBoxLayout(panel)

        gb = QGroupBox("Target")
        f = QFormLayout(gb)
        self.cb_task = QComboBox()
        self.cb_task.addItem("BP  (SBP + DBP regression)", "BP")
        self.cb_task.addItem("CPT (rest vs cold-pressor)", "CPT")
        self.cb_task.currentIndexChanged.connect(self._on_task)
        f.addRow("Predict", self.cb_task)
        f.addRow(QLabel("<i>BP is the goal. CPT is the sanity check that<br>"
                        "these features carry autonomic information<br>"
                        "at all - if it collapses, a BP number from the<br>"
                        "same features means nothing.</i>"))
        v.addWidget(gb)

        gb = QGroupBox("Feature source")
        fv = QVBoxLayout(gb)
        self.rb_here = QRadioButton("Engineer here (controls below)")
        self.rb_tab5 = QRadioButton("From tab 5 (current working set)")
        self.rb_here.setChecked(True)
        self.rb_tab5.setEnabled(False)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_here); grp.addButton(self.rb_tab5)
        self.lbl_tab5 = QLabel("<i>Run tab 5 first to enable.</i>")
        self.lbl_tab5.setWordWrap(True)
        fv.addWidget(self.rb_here); fv.addWidget(self.rb_tab5)
        fv.addWidget(self.lbl_tab5)
        v.addWidget(gb)

        self.gb_eng = QGroupBox("Engineering (used by 'Engineer here')")
        f = QFormLayout(self.gb_eng)
        self.le_ecg = QLineEdit(os.path.join(ROOT, "ecg_features_30s.csv"))
        self.le_skna = QLineEdit(os.path.join(ROOT, "skna_features_30s.csv"))
        for le, what in ((self.le_ecg, "ECG features CSV"),
                         (self.le_skna, "SKNA features CSV")):
            b = QPushButton("Browse...")
            b.clicked.connect(lambda _, l=le, w=what: self._browse(l, w))
            f.addRow(what.split()[0] + " csv", le)
            f.addRow(b)
        self.cb_method = QComboBox()
        for key, label in METHODS:
            self.cb_method.addItem(label, key)
        self.sb_k = QSpinBox(); self.sb_k.setRange(1, 10); self.sb_k.setValue(2)
        f.addRow("Normalization", self.cb_method)
        f.addRow("Temporal span (k)", self.sb_k)
        v.addWidget(self.gb_eng)

        gb = QGroupBox("Exclude recordings")
        fv = QVBoxLayout(gb)
        self.lst_excl = QListWidget()
        self.lst_excl.setMaximumHeight(150)
        self.lst_excl.setToolTip(
            "Ticked recordings are dropped before the split, so the person "
            "disappears from LOSO entirely.\nSame effect as --drop-recording "
            "on ann_bp_loso.py / ann_personalize.py.")
        self.lst_excl.itemChanged.connect(self._on_excl_changed)
        fv.addWidget(self.lst_excl)
        row = QHBoxLayout()
        b = QPushButton("Refresh list")
        b.clicked.connect(self._refresh_excl)
        row.addWidget(b)
        b = QPushButton("Clear")
        b.clicked.connect(lambda: self._set_excl([]))
        row.addWidget(b)
        row.addStretch(1)
        fv.addLayout(row)
        self.lbl_excl = QLabel("<i>Nothing excluded.</i>")
        self.lbl_excl.setWordWrap(True)
        fv.addWidget(self.lbl_excl)
        fv.addWidget(QLabel(
            "<i>Every exclusion is stamped into the run row, the<br>"
            "results header and the saved history, so a number<br>"
            "can never be quoted without it. Dropping a subject<br>"
            "AFTER seeing it score badly is selection on the test<br>"
            "set - decide from data quality, and say so.</i>"))
        v.addWidget(gb)

        gb = QGroupBox("Feature families")
        fv = QVBoxLayout(gb)
        self.ck_fam = {}
        for name in FAMILIES:
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.setToolTip(FAMILY_HINT[name])
            self.ck_fam[name] = cb
            fv.addWidget(cb)
        self.ck_temporal = QCheckBox("include dynamics (_d1/_rm/_rs/_slope)")
        self.ck_temporal.setChecked(True)
        fv.addWidget(self.ck_temporal)
        v.addWidget(gb)

        self.gb_calib = QGroupBox("Calibration (BP only)")
        f = QFormLayout(self.gb_calib)
        self.cb_calib = QComboBox()
        self.cb_calib.addItem("none - calibration-free, absolute mmHg", "none")
        self.cb_calib.addItem("offset - per-subject, predict deviation",
                              "offset")
        self.sb_calib_min = QDoubleSpinBox()
        self.sb_calib_min.setRange(0.5, 30.0); self.sb_calib_min.setValue(2.0)
        self.sb_calib_min.setSuffix(" min")
        f.addRow("Mode", self.cb_calib)
        f.addRow("Calibration window", self.sb_calib_min)
        f.addRow(QLabel("<i>'none' must beat the constant train mean.<br>"
                        "'offset' must beat ZERO-DELTA - predicting the<br>"
                        "calibration value and never moving. Both rows<br>"
                        "are always in the results table.</i>"))
        v.addWidget(self.gb_calib)

        gb = QGroupBox("ANN hyperparameters")
        f = QFormLayout(gb)
        self.le_hidden = QLineEdit("10")
        self.le_hidden.setToolTip(
            "Architecture. '10' = one hidden layer of 10 (the paper).\n"
            "'10x2' = two hidden layers of 10.\n"
            "'5,10,20' = sweep three architectures.")
        self.cb_activation = QComboBox()
        for act in ("logistic", "tanh", "relu", "identity"):
            self.cb_activation.addItem(
                act + (" (paper)" if act == "logistic" else ""), act)
        self.cb_solver = QComboBox()
        self.cb_solver.addItem("lbfgs (stands in for the paper's SCG)",
                               "lbfgs")
        self.cb_solver.addItem("adam", "adam")
        self.cb_solver.addItem("sgd", "sgd")
        self.sb_alpha = QDoubleSpinBox()
        self.sb_alpha.setRange(1e-6, 10.0); self.sb_alpha.setDecimals(5)
        self.sb_alpha.setSingleStep(0.005); self.sb_alpha.setValue(0.01)
        self.sb_maxiter = QSpinBox()
        self.sb_maxiter.setRange(50, 100000); self.sb_maxiter.setValue(1000)
        self.sb_maxiter.setSingleStep(500)
        self.sb_lr = QDoubleSpinBox()
        self.sb_lr.setRange(1e-6, 1.0); self.sb_lr.setDecimals(6)
        self.sb_lr.setValue(0.001)
        self.sb_tol = QDoubleSpinBox()
        self.sb_tol.setRange(1e-8, 1.0); self.sb_tol.setDecimals(8)
        self.sb_tol.setValue(0.0001)
        self.ck_early = QCheckBox("early stopping (adam/sgd only)")
        self.ck_early.setToolTip(
            "sklearn holds out a RANDOM slice of the training rows. With "
            "overlapping windows that leaks across the split, so prefer "
            "nested selection below.")
        self.ck_inner = QCheckBox("nested selection (inner LOSO)")
        self.ck_inner.setToolTip(
            "Choose the architecture inside each outer fold, using only that "
            "fold's training people. Use this whenever you sweep more than "
            "one architecture and intend to quote the winner.")
        self.sb_seed = QSpinBox(); self.sb_seed.setRange(0, 9999)
        self.ck_compare = QCheckBox("also run the linear reference")
        self.ck_compare.setChecked(True)
        f.addRow("Architecture", self.le_hidden)
        f.addRow("Activation", self.cb_activation)
        f.addRow("Solver", self.cb_solver)
        f.addRow("L2 alpha", self.sb_alpha)
        f.addRow("Max iterations", self.sb_maxiter)
        f.addRow("Learning rate", self.sb_lr)
        f.addRow("Tolerance", self.sb_tol)
        f.addRow(self.ck_early)
        f.addRow(self.ck_inner)
        f.addRow("Seed", self.sb_seed)
        f.addRow(self.ck_compare)
        f.addRow(QLabel("<i>Two LINEAR output neurons for SBP and DBP in<br>"
                        "one network, as in the paper. Watch the Log tab<br>"
                        "for 'NOT CONVERGED' - if you see it, raise max<br>"
                        "iterations before believing the MAE.</i>"))
        v.addWidget(gb)

        self.gb_curve = QGroupBox("Training curve")
        f = QFormLayout(self.gb_curve)
        self.ck_curve = QCheckBox("train vs held-out loss per iteration")
        self.ck_curve.setToolTip(
            "lbfgs reports no per-epoch loss, so the curve cannot be read out "
            "of a fitted network.\nIt is generated instead: refit warm-started "
            "in steps, scoring the training block and the\nheld-out person at "
            "every checkpoint. Roughly DOUBLES the run time.")
        self.sb_curvestep = QSpinBox()
        self.sb_curvestep.setRange(5, 5000); self.sb_curvestep.setValue(50)
        self.sb_curvestep.setSingleStep(25)
        self.sb_curvestep.setToolTip(
            "Checkpoint spacing, up to 'Max iterations' above. Smaller = finer "
            "curve but more\nrestarts of lbfgs's curvature history, so further "
            "from one uninterrupted fit.")
        f.addRow(self.ck_curve)
        f.addRow("Checkpoint every", self.sb_curvestep)
        f.addRow(QLabel(
            "<i>Held-out is the LOSO test person, so this is a<br>"
            "generalization curve, not a validation split: the<br>"
            "gap between the two lines is the inter-subject<br>"
            "transfer gap. The dashed line is the baseline the<br>"
            "model must cross to have learned anything.<br>"
            "Uses the FIRST architecture when sweeping.</i>"))
        v.addWidget(self.gb_curve)

        self.gb_ft = QGroupBox("Personalization (per-subject fine-tune)")
        f = QFormLayout(self.gb_ft)
        self.ck_ft = QCheckBox("head-only fine-tune per subject")
        self.ck_ft.setToolTip(
            "After the LOSO fit, freeze the input->hidden layer and refit only "
            "the output head on the held-out person's own early windows.\n"
            "Same protocol as the CNN-BiLSTM personalization in "
            "src/personalize_finetune.py.")
        self.sb_adapt = QDoubleSpinBox()
        self.sb_adapt.setRange(0.1, 0.9); self.sb_adapt.setSingleStep(0.05)
        self.sb_adapt.setValue(0.50)
        self.sb_adapt.setToolTip("Adapt slice = the first this-much of the "
                                 "recording (the calibration data).")
        self.sb_valend = QDoubleSpinBox()
        self.sb_valend.setRange(0.15, 0.95); self.sb_valend.setSingleStep(0.05)
        self.sb_valend.setValue(0.60)
        self.sb_valend.setToolTip("Val slice ends here; everything after is "
                                  "the test slice, never seen while adapting.")
        self.sb_ftlr = QDoubleSpinBox()
        self.sb_ftlr.setRange(1e-5, 1.0); self.sb_ftlr.setDecimals(5)
        self.sb_ftlr.setSingleStep(1e-3); self.sb_ftlr.setValue(0.01)
        self.sb_ftep = QSpinBox()
        self.sb_ftep.setRange(1, 5000); self.sb_ftep.setValue(400)
        self.sb_ftpat = QSpinBox()
        self.sb_ftpat.setRange(1, 500); self.sb_ftpat.setValue(40)
        self.sb_ftpat.setToolTip("Early-stop patience, in fine-tune epochs, "
                                 "on the val slice.")
        f.addRow(self.ck_ft)
        f.addRow("Adapt ends at", self.sb_adapt)
        f.addRow("Val ends at", self.sb_valend)
        f.addRow("Fine-tune LR", self.sb_ftlr)
        f.addRow("Max FT epochs", self.sb_ftep)
        f.addRow("FT patience", self.sb_ftpat)
        f.addRow(QLabel(
            "<i>Adds a 'Fine-tune curve' tab with train/validation loss<br>"
            "per epoch, for SBP and DBP. This is a real held-out<br>"
            "VALIDATION slice of the same person; the base ANN's curve<br>"
            "above is train vs held-out SUBJECT instead.<br>"
            "Scored on the test slice only, so its MAEs do NOT compare<br>"
            "with the Runs table. The bar is still ZERO-DELTA.</i>"))
        v.addWidget(self.gb_ft)

        self.gb_cpt = QGroupBox("CPT only")
        f = QFormLayout(self.gb_cpt)
        self.sb_fpr = QDoubleSpinBox()
        self.sb_fpr.setRange(0.1, 50.0); self.sb_fpr.setValue(5.0)
        self.sb_fpr.setSuffix(" %")
        f.addRow("Threshold FPR", self.sb_fpr)
        f.addRow(QLabel("<i>Classifier only, and nothing to do with BP.<br>"
                        "The net outputs a probability; to quote a<br>"
                        "sensitivity and specificity you need a cut-off.<br>"
                        "This sets it at the value that gives this<br>"
                        "false-positive rate on the TRAINING folds'<br>"
                        "pre-CPT rest windows - so the operating point<br>"
                        "is chosen without seeing the test subject.<br>"
                        "Lower = fewer false alarms, lower sensitivity.<br>"
                        "It does not affect AUC or AP, which are<br>"
                        "threshold-free.</i>"))
        v.addWidget(self.gb_cpt)

        self.btn_run = QPushButton("Run LOSO")
        self.btn_run.setMinimumHeight(34)
        self.btn_run.clicked.connect(self.on_run)
        v.addWidget(self.btn_run)

        self.bar = QProgressBar(); self.bar.setRange(0, 0)
        self.bar.setVisible(False)
        v.addWidget(self.bar)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        v.addWidget(self.lbl_status)

        row = QHBoxLayout()
        self.btn_clear = QPushButton("Clear history")
        self.btn_clear.clicked.connect(self.on_clear)
        row.addWidget(self.btn_clear)
        v.addLayout(row)
        v.addStretch(1)
        return panel

    # -- exclusions -----------------------------------------------------
    def _excl_source(self):
        """(person, recording) arrays for whatever the run would train on,
        without paying for a full engineer. Prefers the tab-5 table, then the
        ablation cache that ma.build() would hit; returns None if neither is
        available yet."""
        if self.rb_tab5.isChecked() and self.supplied is not None:
            df = self.supplied[0]
            rec = (df["Recording"] if "Recording" in df.columns
                   else df["person"])
            return df["person"].to_numpy().astype(str), rec.to_numpy().astype(str)
        # mirror ma.build()'s cache naming - keep_s13 stays at its 'full'
        # default here, exactly as _features() calls it
        tag = os.path.splitext(os.path.basename(self.le_ecg.text()))[0].replace(
            "ecg_features_", "")
        cache = os.path.join(ROOT, "beat_pipeline", "built",
                             f"ablation_cache_{tag}_"
                             f"{self.cb_method.currentData()}.npz")
        if not os.path.exists(cache):
            return None
        with np.load(cache, allow_pickle=True) as d:
            return (d["groups"].astype(str), d["recording"].astype(str))

    def _refresh_excl(self):
        keep = set(self.excluded)
        src = self._excl_source()
        self.lst_excl.blockSignals(True)
        self.lst_excl.clear()
        if src is None:
            self.lst_excl.blockSignals(False)
            self.lbl_excl.setText(
                "<i>No engineered table cached for this source yet - run once, "
                "or switch to tab 5, then Refresh.</i>")
            return
        person, rec = src
        order = sorted(set(zip(person, rec)))
        for p, r in order:
            n = int((rec == r).sum())
            it = QListWidgetItem(f"{p}  ·  {r}  ·  {n} windows")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if r in keep else Qt.Unchecked)
            it.setData(Qt.UserRole, r)
            self.lst_excl.addItem(it)
        self.lst_excl.blockSignals(False)
        self._on_excl_changed()

    def _set_excl(self, names):
        names = set(names)
        self.lst_excl.blockSignals(True)
        for i in range(self.lst_excl.count()):
            it = self.lst_excl.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in names
                             else Qt.Unchecked)
        self.lst_excl.blockSignals(False)
        self._on_excl_changed()

    def _on_excl_changed(self):
        names, people = [], set()
        src = self._excl_source()
        for i in range(self.lst_excl.count()):
            it = self.lst_excl.item(i)
            if it.checkState() == Qt.Checked:
                names.append(it.data(Qt.UserRole))
                people.add(it.text().split("·")[0].strip())
        self.excluded = names
        if not names:
            self.lbl_excl.setText("<i>Nothing excluded.</i>")
            return
        n_win = 0
        gone = []
        if src is not None:
            person, rec = src
            drop = np.isin(rec, names)
            n_win = int(drop.sum())
            # a person only leaves LOSO if ALL of their recordings are dropped
            gone = sorted({p for p in people
                           if not (person[~drop] == p).any()})
        self.lbl_excl.setText(
            f"<b style='color:#b3261e'>Excluding {len(names)} recording"
            f"{'s' if len(names) > 1 else ''}</b> ({n_win} windows). "
            + (f"Leaves LOSO entirely: <b>{', '.join(gone)}</b>."
               if gone else
               "No person is fully removed - their other recordings remain."))

    def _on_task(self):
        bp = self.cb_task.currentData() == "BP"
        self.gb_calib.setVisible(bp)
        self.gb_curve.setVisible(bp)
        self.gb_ft.setVisible(bp)
        self.gb_cpt.setVisible(not bp)

    # -- results -------------------------------------------------------
    def _build_right(self):
        w = QWidget(); v = QVBoxLayout(w)
        self.lbl_head = QLabel("Run a model to see results.")
        self.lbl_head.setWordWrap(True)
        v.addWidget(self.lbl_head)
        self.lbl_verdict = QLabel("")
        self.lbl_verdict.setWordWrap(True)
        v.addWidget(self.lbl_verdict)

        self.tabs = QTabWidget()
        self.tbl_runs = self._table()
        self.tabs.addTab(self._wrap(
            self.tbl_runs, "Every model this session, with the baselines it "
            "has to beat. Click a row to plot it."), "Runs")
        self.tbl_folds = self._table()
        self.tabs.addTab(self._wrap(
            self.tbl_folds, "Per-subject metrics. base_mae_* is that person's "
            "own baseline - a model worth anything beats it more often than "
            "not."), "Per subject")
        self.canvas = MplCanvas()
        self.tabs.addTab(self.canvas, "Plots")

        tcw = QWidget(); tcv = QVBoxLayout(tcw)
        self.lbl_tc = QLabel("Tick 'train vs held-out loss per iteration' and "
                             "run a BP model to see the training curve.")
        self.lbl_tc.setWordWrap(True)
        tcv.addWidget(self.lbl_tc)
        self.canvas_tc = MplCanvas(width=9, height=7)
        tcv.addWidget(self.canvas_tc)
        row = QHBoxLayout()
        self.btn_tc_save = QPushButton("Save curve CSV + PNG ...")
        self.btn_tc_save.setEnabled(False)
        self.btn_tc_save.clicked.connect(self._save_curve)
        row.addWidget(self.btn_tc_save); row.addStretch(1)
        tcv.addLayout(row)
        self.tabs.addTab(tcw, "Training curve")

        ftw = QWidget(); ftv = QVBoxLayout(ftw)
        row = QHBoxLayout()
        row.addWidget(QLabel("Subject:"))
        self.cb_ftsubj = QComboBox()
        self.cb_ftsubj.currentIndexChanged.connect(self._plot_ft)
        row.addWidget(self.cb_ftsubj); row.addStretch(1)
        ftv.addLayout(row)
        self.lbl_ft = QLabel("Tick 'head-only fine-tune per subject' and run "
                             "a BP model to see per-epoch curves.")
        self.lbl_ft.setWordWrap(True)
        ftv.addWidget(self.lbl_ft)
        self.canvas_ft = MplCanvas(width=7, height=5)
        ftv.addWidget(self.canvas_ft)
        self.tabs.addTab(ftw, "Fine-tune curve")

        self.tabs.addTab(self._build_ft_report(), "Fine-tune")

        self.txt = QPlainTextEdit(); self.txt.setReadOnly(True)
        self.tabs.addTab(self.txt, "Log")
        self.tbl_runs.itemSelectionChanged.connect(self._on_pick)
        v.addWidget(self.tabs)
        return w

    def _build_ft_report(self):
        """The personalization report: verdict, the bars it has to clear, the
        per-subject margin over ADAPT-MEAN, and the raw table.

        Laid out verdict-first on purpose. `apz.summarize()` already computes
        all of this, but it only ever reached the Log tab, where the one number
        that decides the result (ADAPT-MEAN) sits below a screen of per-subject
        rows and gets read as a footnote.
        """
        w = QWidget(); v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)

        self.lbl_ft_verdict = QLabel("No personalization in this run.")
        self.lbl_ft_verdict.setWordWrap(True)
        self.lbl_ft_verdict.setTextFormat(Qt.RichText)
        v.addWidget(self.lbl_ft_verdict)

        self.lbl_ft_warn = QLabel("")
        self.lbl_ft_warn.setWordWrap(True)
        self.lbl_ft_warn.setVisible(False)
        v.addWidget(self.lbl_ft_warn)

        self.tbl_ft_bars = self._table()
        self.tbl_ft_bars.setMaximumHeight(170)
        v.addWidget(self.tbl_ft_bars)

        split = QSplitter(Qt.Vertical)
        self.canvas_ftsum = MplCanvas(width=9, height=4)
        split.addWidget(self.canvas_ftsum)
        self.tbl_ft = self._table()
        split.addWidget(self._wrap(
            self.tbl_ft, "Per subject, TEST slice only. margin_SBP = "
            "amean_SBP - post_SBP, so POSITIVE means the features earned "
            "something beyond re-anchoring. flat_SBP = prediction SD / true "
            "SD; near 0 means the head collapsed to a constant and is not "
            "tracking anything."))
        split.setSizes([340, 300])
        v.addWidget(split, 1)
        return w

    def _table(self):
        t = QTableWidget()
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setSelectionMode(QAbstractItemView.SingleSelection)
        t.setAlternatingRowColors(True)
        return t

    def _wrap(self, table, hint):
        w = QWidget(); v = QVBoxLayout(w)
        lab = QLabel(f"<i>{hint}</i>"); lab.setWordWrap(True)
        v.addWidget(lab); v.addWidget(table)
        return w

    # -- actions -------------------------------------------------------
    def on_run(self):
        use_tab5 = self.rb_tab5.isChecked() and self.supplied is not None
        if not use_tab5:
            missing = [p for p in (self.le_ecg.text(), self.le_skna.text())
                       if not os.path.isfile(p)]
            if missing:
                QMessageBox.warning(self, "Missing input",
                                    "Not found:\n" + "\n".join(missing))
                return
        fams = {n for n, cb in self.ck_fam.items() if cb.isChecked()}
        if not fams:
            QMessageBox.warning(self, "No features",
                                "Tick at least one feature family.")
            return
        try:
            hidden = abp.parse_hidden(self.le_hidden.text())
        except ValueError:
            QMessageBox.warning(
                self, "Architecture",
                "Use '10', '10x2' for depth, or '5,10,20' to sweep.")
            return
        if len(hidden) > 1 and not self.ck_inner.isChecked():
            if QMessageBox.question(
                    self, "Sweeping without nested selection",
                    f"You are sweeping {len(hidden)} architectures with "
                    "nested selection off.\n\nPicking the best of them by the "
                    "LOSO score is selection on the test folds - the winner "
                    "is partly chosen by luck on the very subjects it is then "
                    "reported against.\n\nRun anyway?") != QMessageBox.Yes:
                return

        hp = dict(activation=self.cb_activation.currentData(),
                  solver=self.cb_solver.currentData(),
                  alpha=self.sb_alpha.value(),
                  max_iter=self.sb_maxiter.value(),
                  learning_rate_init=self.sb_lr.value(),
                  early_stopping=self.ck_early.isChecked(),
                  validation_fraction=0.1, tol=self.sb_tol.value())
        cfg = dict(task=self.cb_task.currentData(),
                   ecg=self.le_ecg.text(), skna=self.le_skna.text(),
                   out=os.path.join(ROOT, "beat_pipeline", "built"),
                   method=("tab5" if use_tab5
                           else self.cb_method.currentData()),
                   k=self.sb_k.value(), families=fams,
                   temporal=self.ck_temporal.isChecked(), hidden=hidden,
                   hp=hp, inner_select=self.ck_inner.isChecked(),
                   fpr=self.sb_fpr.value() / 100.0, seed=self.sb_seed.value(),
                   compare=self.ck_compare.isChecked(),
                   calibration=self.cb_calib.currentData(),
                   calib_min=self.sb_calib_min.value(),
                   exclude=list(self.excluded),
                   curve=self.ck_curve.isChecked(),
                   curve_step=self.sb_curvestep.value(),
                   personalize=self.ck_ft.isChecked(),
                   adapt_end=self.sb_adapt.value(),
                   val_end=self.sb_valend.value(),
                   ft_lr=self.sb_ftlr.value(),
                   ft_epochs=self.sb_ftep.value(),
                   ft_patience=self.sb_ftpat.value())
        self.cfg = cfg
        self.stamp = time.strftime("%Y%m%d-%H%M%S")
        self.btn_run.setEnabled(False)
        self.bar.setVisible(True)
        # by index this breaks every time a tab is inserted above it
        self.tabs.setCurrentIndex(self.tabs.indexOf(self.txt))
        self.txt.appendPlainText(
            f"\n{'=' * 70}\n[{self.stamp}] {cfg['task']} | "
            f"{'+'.join(sorted(fams))} | norm={cfg['method']} | "
            f"source={'tab 5' if use_tab5 else 'engineered here'}"
            + (f" | EXCLUDED {', '.join(sorted(cfg['exclude']))}"
               if cfg["exclude"] else ""))
        self.worker = RunWorker(cfg, self.supplied if use_tab5 else None)
        self.worker.progress.connect(self.lbl_status.setText)
        self.worker.log.connect(self._append_log)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _excl_note(self):
        """Rendered next to every headline number, so the exclusion travels
        with the result instead of living only in the controls."""
        ex = (self.cfg or {}).get("exclude") if hasattr(self, "cfg") else None
        if not ex:
            return ""
        return (f" &nbsp;|&nbsp; <b style='color:#b3261e'>EXCLUDED: "
                f"{', '.join(sorted(ex))}</b>")

    def _append_log(self, msg):
        self.txt.appendPlainText(msg)
        self.txt.verticalScrollBar().setValue(
            self.txt.verticalScrollBar().maximum())

    def _on_finished(self):
        self.btn_run.setEnabled(True)
        self.bar.setVisible(False)

    def _on_failed(self, msg):
        self.lbl_status.setText("failed")
        # append, never replace: a crash at the END of a run used to wipe the
        # fold-by-fold log that had just been streamed in, losing hours of it
        self._append_log("\nRUN FAILED\n" + msg)
        self.tabs.setCurrentIndex(self.tabs.indexOf(self.txt))
        try:
            d = os.path.join(ROOT, "beat_pipeline", "built")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(
                d, f"ui_run_{getattr(self, 'stamp', 'nostamp')}_FAILED.log")
            with open(path, "w") as fh:
                fh.write(self.txt.toPlainText())
            self._append_log(f"\nsaved  {os.path.basename(path)}")
        except Exception:
            pass
        QMessageBox.critical(self, "Run failed", msg.strip().split("\n")[-1])

    def _on_done(self, out):
        self.last = out
        # stamped per run, not per session: the history table accumulates runs
        # made under different exclusions and they must stay distinguishable
        ex = "+".join(sorted(self.cfg.get("exclude") or [])) or "-"
        for s in out["runs"]:
            s["_folds"] = out["folds"].get(s["model"])
            s["_task"] = out["task"]
            s["excluded"] = ex
        self.history.extend(out["runs"])
        self._fill_curve(out)
        self._fill_ft(out.get("ft"), out.get("ft_curves"))

        if out["task"] == "BP":
            self.lbl_head.setText(
                f"<b>{out['n_features']} features</b> | {out['n_windows']} "
                f"windows, {out['n_people']} people | target SBP + DBP"
                + self._excl_note())
            models = [s for s in out["runs"]
                      if not s["model"].startswith(("BASELINE", "ORACLE"))]
            best = min(models, key=lambda s: s["MAE_SBP"]) if models else None
            bl = out["baseline_mae"]
            if best:
                ok = best["MAE_SBP"] < bl
                self.lbl_verdict.setText(
                    f"<b style='color:{'#0a7d38' if ok else '#b3261e'}'>"
                    f"{'BEATS' if ok else 'DOES NOT BEAT'} the baseline</b> - "
                    f"best {best['model']} SBP MAE {best['MAE_SBP']:.2f} vs "
                    f"baseline {bl:.2f} mmHg. "
                    + ("" if ok else "Do not report this MAE on its own."))
        else:
            self.lbl_head.setText(
                f"<b>{out['n_features']} features</b> | {out['n_windows']} "
                f"windows, {out['n_people']} people | prevalence "
                f"{100 * out['prevalence']:.1f}% (AP baseline "
                f"{out['prevalence']:.3f}, AUC 0.500)" + self._excl_note())
            self.lbl_verdict.setText("")
        self._fill_runs()
        self._persist(out)
        self.lbl_status.setText("done")
        self.tabs.setCurrentIndex(0)

    def _persist(self, out):
        """Write this run to disk: log, per-fold CSV, and an appended history
        row so the session survives closing the app."""
        d = os.path.join(ROOT, "beat_pipeline", "built")
        os.makedirs(d, exist_ok=True)
        stamp, c = self.stamp, self.cfg
        try:
            with open(os.path.join(d, f"ui_run_{stamp}.log"), "w") as fh:
                fh.write(self.txt.toPlainText())
            folds = [f.assign(model=m) for m, f in out["folds"].items()
                     if f is not None]
            if folds:
                pd.concat(folds).to_csv(
                    os.path.join(d, f"ui_run_{stamp}_folds.csv"), index=False)
            if out.get("curve") is not None and len(out["curve"]):
                out["curve"].to_csv(
                    os.path.join(d, f"ui_run_{stamp}_curve.csv"), index=False)
            hist = os.path.join(d, "ui_run_history.csv")
            rows = pd.DataFrame([{k: v for k, v in s.items()
                                  if not k.startswith("_")}
                                 for s in out["runs"]]).assign(
                stamp=stamp, task=c["task"], families="+".join(sorted(
                    c["families"])), source=c["method"],
                excluded="+".join(sorted(c["exclude"])) or "none",
                temporal=c["temporal"], calibration=c["calibration"],
                calib_min=c["calib_min"], hidden=self.le_hidden.text(),
                inner_select=c["inner_select"], seed=c["seed"],
                **{f"hp_{k}": v for k, v in c["hp"].items()})
            rows.to_csv(hist, mode="a", header=not os.path.exists(hist),
                        index=False)
            self._append_log(
                f"\nsaved  ui_run_{stamp}.log\n"
                f"       ui_run_{stamp}_folds.csv\n"
                + (f"       ui_run_{stamp}_curve.csv\n"
                   if out.get("curve") is not None else "")
                + f"       ui_run_history.csv  (appended)")
        except Exception:
            self._append_log("could not persist run:\n"
                             + traceback.format_exc())

    def _fill_runs(self):
        task = self.history[-1]["_task"] if self.history else "BP"
        cols = BP_COLS if task == "BP" else CPT_COLS
        t = self.tbl_runs
        t.clear(); t.setColumnCount(len(cols)); t.setRowCount(len(self.history))
        t.setHorizontalHeaderLabels(cols)
        for r, s in enumerate(self.history):
            for c, k in enumerate(cols):
                v = s.get(k, "")
                txt = f"{v:.2f}" if isinstance(v, float) else str(v)
                it = QTableWidgetItem(txt)
                if s["model"].startswith(("BASELINE", "ORACLE")):
                    it.setForeground(Qt.darkGray)
                t.setItem(r, c, it)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        # Land on the newest REAL model, not on the baseline rows appended
        # after it - those have no per-subject breakdown to show.
        real = [i for i, s in enumerate(self.history)
                if not s["model"].startswith(("BASELINE", "ORACLE"))]
        if real:
            t.selectRow(real[-1])
        elif self.history:
            t.selectRow(len(self.history) - 1)

    def _on_pick(self):
        rows = self.tbl_runs.selectionModel().selectedRows()
        if not rows:
            return
        s = self.history[rows[0].row()]
        if s.get("_folds") is None:
            self.tbl_folds.setRowCount(0)
            return
        self._fill_folds(s["_folds"])
        try:
            self._plot(s)
        except Exception:
            self.txt.appendPlainText(traceback.format_exc())

    def _fill_folds(self, folds):
        t = self.tbl_folds
        cols = list(folds.columns)
        t.clear(); t.setColumnCount(len(cols)); t.setRowCount(len(folds))
        t.setHorizontalHeaderLabels(cols)
        for r in range(len(folds)):
            for c, k in enumerate(cols):
                v = folds.iloc[r][k]
                txt = (f"{v:.2f}" if isinstance(v, (float, np.floating))
                       else str(v))
                it = QTableWidgetItem(txt)
                if k == "mae_SBP" and "base_mae_SBP" in folds.columns:
                    if v > folds.iloc[r]["base_mae_SBP"]:
                        it.setForeground(Qt.red)
                t.setItem(r, c, it)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def _fill_table(self, t, df, red_if=()):
        """Generic DataFrame -> QTableWidget. red_if is (value_col, bar_col)
        pairs painted red when the value loses to its bar."""
        cols = list(df.columns)
        t.clear(); t.setColumnCount(len(cols)); t.setRowCount(len(df))
        t.setHorizontalHeaderLabels(cols)
        for r in range(len(df)):
            for c, k in enumerate(cols):
                v = df.iloc[r][k]
                txt = (f"{v:.2f}" if isinstance(v, (float, np.floating))
                       else str(v))
                it = QTableWidgetItem(txt)
                for val_col, bar_col in red_if:
                    if k == val_col and bar_col in df.columns:
                        it.setForeground(Qt.red if v > df.iloc[r][bar_col]
                                         else Qt.darkGreen)
                t.setItem(r, c, it)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    # -- training curve -------------------------------------------------
    def _fill_curve(self, out):
        """Draw the warm-started train/held-out curve, or clear the tab."""
        df = out.get("curve")
        self.curve = df
        self.btn_tc_save.setEnabled(df is not None and len(df) > 0)
        if df is None or not len(df):
            self.canvas_tc.fig.clear(); self.canvas_tc.draw()
            self.lbl_tc.setText(
                "No training curve in this run - tick 'train vs held-out loss "
                "per iteration' (BP task only) and re-run.")
            return
        mode = self.cfg["calibration"]
        h = df[df.split == "held-out"]
        end = h[h["iter"] == h["iter"].max()]
        won = sum(1 for _, g in end.groupby("fold")
                  if bool((g.mae.values < g.base.values).all()))
        self.lbl_tc.setText(
            f"{out.get('curve_arch', 'ANN')} - {h.fold.nunique()} folds, "
            f"checkpoints every {self.cfg['curve_step']} iterations. "
            f"At the last checkpoint <b>{won}/{h.fold.nunique()}</b> subjects "
            f"beat their own baseline on both channels. "
            "Held-out here is the LOSO TEST person, so a widening gap is the "
            "inter-subject transfer gap, not classical overfitting to noise.")
        atc.draw(self.canvas_tc.fig, df,
                 f"{h.fold.nunique()} people, calibration={mode}, "
                 f"alpha={self.cfg['hp']['alpha']}, "
                 f"{out['n_features']} features",
                 label_size=8, base_label=atc.baseline_label(mode),
                 solver=self.cfg["hp"]["solver"])
        self.canvas_tc.draw()

    def _save_curve(self):
        if self.curve is None or not len(self.curve):
            return
        png, _ = QFileDialog.getSaveFileName(
            self, "Save training curve", os.path.join(
                ROOT, "figures", f"ann_training_curves_{self.stamp}.png"),
            "PNG (*.png)")
        if not png:
            return
        try:
            os.makedirs(os.path.dirname(png) or ".", exist_ok=True)
            csv = os.path.splitext(png)[0] + ".csv"
            self.curve.to_csv(csv, index=False)
            mode = self.cfg["calibration"]
            h = self.curve[self.curve.split == "held-out"]
            atc.plot(self.curve, png,
                     f"{h.fold.nunique()} people, calibration={mode}, "
                     f"alpha={self.cfg['hp']['alpha']}",
                     atc.baseline_label(mode))
            self._append_log(f"\nsaved  {png}\n       {csv}")
        except Exception:
            QMessageBox.critical(self, "Save failed", traceback.format_exc())

    # -- personalization ------------------------------------------------
    def _fill_ft(self, ft, curves):
        """Populate the fine-tune table and the per-subject curve picker."""
        self.ft_curves = curves
        self.cb_ftsubj.blockSignals(True)
        self.cb_ftsubj.clear()
        if ft is None or not len(ft):
            for t in (self.tbl_ft, self.tbl_ft_bars):
                t.clear(); t.setRowCount(0); t.setColumnCount(0)
            self.cb_ftsubj.blockSignals(False)
            self.lbl_ft.setText("No personalization in this run - tick "
                                "'head-only fine-tune per subject' and re-run.")
            self.lbl_ft_verdict.setText(
                "No personalization in this run - tick 'head-only fine-tune "
                "per subject' and re-run.")
            self.lbl_ft_warn.setVisible(False)
            for c in (self.canvas_ft, self.canvas_ftsum):
                c.fig.clear(); c.draw()
            return

        ft = self._ft_derived(ft)
        # colour post_* against the bar that matters, not against pre_*
        self._fill_table(self.tbl_ft, ft,
                         red_if=(("post_SBP", "amean_SBP"),
                                 ("post_DBP", "amean_DBP")))
        for p in curves["person"].drop_duplicates():
            self.cb_ftsubj.addItem(str(p))
        self.cb_ftsubj.blockSignals(False)
        n = len(ft)
        am = (int(ft.post_beats_amean.sum()) if "post_beats_amean" in ft
              else None)
        self.lbl_ft.setText(
            f"Beats the un-personalized model on "
            f"<b>{int(ft.post_beats_pre.sum())}/{n}</b> &nbsp;|&nbsp; "
            f"beats zero-delta on <b>{int(ft.post_beats_zero.sum())}/{n}</b>"
            + ("" if am is None else
               f" &nbsp;|&nbsp; beats <b>adapt-mean</b> on <b>{am}/{n}</b> "
               f"&mdash; that is the real test: predicting this person's own "
               f"average BP using no features at all, which is exactly the "
               f"recalibration the fine-tune gets free from seeing the adapt "
               f"slice. Only what it wins ABOVE that came from the features.")
            + " Epoch -1 is the base model before any fine-tuning.")
        self._fill_ft_bars(ft)
        self._ft_verdict(ft)
        self._plot_ft_summary(ft)
        self._plot_ft()

    @staticmethod
    def _ft_derived(ft):
        """Add the two columns the verdict actually turns on."""
        ft = ft.copy()
        for ch in ("SBP", "DBP"):
            if f"amean_{ch}" in ft:
                ft[f"margin_{ch}"] = ft[f"amean_{ch}"] - ft[f"post_{ch}"]
            if f"predsd_{ch}" in ft and f"truesd_{ch}" in ft:
                ft[f"flat_{ch}"] = (ft[f"predsd_{ch}"]
                                    / ft[f"truesd_{ch}"].replace(0, np.nan))
        return ft

    def _fill_ft_bars(self, ft):
        """The five bars side by side - the comparison the whole result is."""
        n = len(ft)
        def wins(col):
            return f"{int(ft[col].sum())}/{n}" if col in ft else "-"
        rows = [
            ("PRE - population ANN, no adaptation", "pre", "",
             "what the LOSO model gives an unseen person"),
            ("POST - head-only fine-tune", "post", "post_beats_pre",
             "the personalized model"),
            ("ZERO-DELTA", "zero", "post_beats_zero",
             "never move from the calibration reading"),
            ("ADAPT-MEAN  <-- THE BAR", "amean", "post_beats_amean",
             "this person's own mean over the adapt slice, NO features"),
            ("BIAS-ONLY (do not quote)", "bias", "post_beats_bias",
             "flatters POST - see the Log tab"),
        ]
        cols = ["bar", "SBP", "DBP", "POST wins", "what it is"]
        t = self.tbl_ft_bars
        t.clear(); t.setColumnCount(len(cols)); t.setRowCount(len(rows))
        t.setHorizontalHeaderLabels(cols)
        post_sbp = ft["post_SBP"].mean()
        for r, (label, key, wcol, note) in enumerate(rows):
            if f"{key}_SBP" not in ft:
                continue
            sbp, dbp = ft[f"{key}_SBP"].mean(), ft[f"{key}_DBP"].mean()
            vals = [label, f"{sbp:.2f}", f"{dbp:.2f}",
                    "-" if key == "pre" else wins(wcol), note]
            for c, txt in enumerate(vals):
                it = QTableWidgetItem(txt)
                if key == "amean":
                    it.setForeground(Qt.darkRed if sbp < post_sbp
                                     else Qt.darkGreen)
                    f = it.font(); f.setBold(True); it.setFont(f)
                elif key == "bias":
                    it.setForeground(Qt.darkGray)
                t.setItem(r, c, it)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def _ft_verdict(self, ft):
        """Same rule as apz.summarize()'s VERDICT block, stated up front."""
        n = len(ft)
        pre, post = ft["pre_SBP"].mean(), ft["post_SBP"].mean()
        am = ft["amean_SBP"].mean() if "amean_SBP" in ft else None
        nam = int(ft.post_beats_amean.sum()) if "post_beats_amean" in ft else 0
        flat = ft["flat_SBP"].median() if "flat_SBP" in ft else float("nan")
        rmed = ft["r_SBP"].median() if "r_SBP" in ft else float("nan")

        if am is None:
            head, col = "No ADAPT-MEAN control in this run - cannot judge.", "#b3261e"
            body = ("Re-run with a build of ann_personalize.py that emits "
                    "amean_*; without it POST vs PRE means nothing.")
        elif nam <= n / 2:
            head, col = "LEVEL ADAPTATION, NOT TRACKING", "#b3261e"
            body = (
                f"POST ({post:.2f}) improves on PRE ({pre:.2f}) but loses to "
                f"ADAPT-MEAN ({am:.2f}), which uses <b>no features at all</b> "
                f"- beaten on only <b>{nam}/{n}</b> subjects. The fine-tune is "
                f"re-anchoring the BP level, which it gets free from seeing "
                f"the adapt slice. Report this as calibration, not estimation.")
        else:
            head, col = "BEATS THE NO-FEATURE BAR", "#0a7d38"
            body = (f"POST ({post:.2f}) beats ADAPT-MEAN ({am:.2f}) on "
                    f"<b>{nam}/{n}</b> subjects. Check flat_SBP and r_SBP "
                    f"below before quoting it.")
        extra = ""
        if flat == flat:
            extra = (f" &nbsp;|&nbsp; median prediction SD / true SD "
                     f"<b>{flat:.3f}</b>"
                     + (" - the head collapsed to a near-constant"
                        if flat < 0.2 else "")
                     + f", median deviation r <b>{rmed:.3f}</b>.")
        self.lbl_ft_verdict.setText(
            f"<span style='color:{col};font-size:13pt'><b>{head}</b></span><br>"
            f"{body}{extra}")

        cap = self.cfg.get("ft_epochs")
        pinned = int((ft.best_epoch >= cap - 1).sum()) if cap else 0
        self.lbl_ft_warn.setVisible(bool(pinned))
        if pinned:
            self.lbl_ft_warn.setText(
                f"<b style='color:#b3261e'>TRUNCATED:</b> best_epoch is pinned "
                f"at the {cap}-epoch cap on <b>{pinned}/{n}</b> subjects, so "
                f"the fine-tune was still improving when it stopped. Raise "
                f"'Max FT epochs' (and patience) and re-run before believing "
                f"any number on this tab.")

    def _plot_ft_summary(self, ft):
        """Left: who actually gained over ADAPT-MEAN. Right: the inversion -
        the more the head still uses the features, the worse it does."""
        fig = self.canvas_ftsum.fig
        fig.clear()
        fig.set_layout_engine("none")
        if "margin_SBP" not in ft:
            self.canvas_ftsum.draw()
            return
        d = ft.sort_values("margin_SBP")
        ax1, ax2 = fig.subplots(1, 2)
        GREEN, RED, INK, MUTED = "#008300", "#b3261e", "#0b0b0b", "#52514e"

        y = np.arange(len(d))
        ax1.barh(y, d.margin_SBP.values,
                 color=[GREEN if m > 0 else RED for m in d.margin_SBP],
                 height=.68)
        ax1.set_yticks(y); ax1.set_yticklabels(d.person.astype(str),
                                               fontsize=8)
        ax1.axvline(0, color=INK, lw=1.4)
        ax1.set_xlabel("SBP MAE: adapt-mean − fine-tuned (mmHg)", fontsize=9,
                       color=MUTED)
        ax1.set_title("right of 0 = the FEATURES earned it", fontsize=10,
                      fontweight="bold", loc="left", color=INK)

        if "flat_SBP" in d and d.flat_SBP.notna().sum() > 2:
            ax2.scatter(d.flat_SBP, d.margin_SBP, s=34, color="#2a78d6",
                        zorder=3)
            # only the points that carry the argument - the collapsed cluster
            # sits on top of itself at (0, 0) and labelling it is a smear
            for _, r in d.iterrows():
                if abs(r.margin_SBP) > .1 or r.flat_SBP > .4:
                    ax2.annotate(str(r.person), (r.flat_SBP, r.margin_SBP),
                                 xytext=(4, 3), textcoords="offset points",
                                 fontsize=7.5, color=MUTED)
            ax2.axhline(0, color=INK, lw=1.4)
            ax2.axvline(0.2, color=MUTED, ls=(0, (4, 3)), lw=1)
            ok = d.flat_SBP.notna() & d.margin_SBP.notna()
            if ok.sum() > 2:
                rr = float(np.corrcoef(d.flat_SBP[ok], d.margin_SBP[ok])[0, 1])
                ax2.set_title(f"feature use vs gain — r = {rr:+.2f}",
                              fontsize=10, fontweight="bold", loc="left",
                              color=INK)
            ax2.set_xlabel("prediction SD / true SD  (<0.2 = flat line)",
                           fontsize=9, color=MUTED)
            ax2.set_ylabel("gain over adapt-mean (mmHg)", fontsize=9,
                           color=MUTED)
        for ax in (ax1, ax2):
            ax.grid(True, color="#d8d7d2", lw=.7)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=8, colors=MUTED)
        fig.tight_layout()
        self.canvas_ftsum.draw()

    def _plot_ft(self):
        c = getattr(self, "ft_curves", None)
        fig = self.canvas_ft.fig
        fig.clear()
        if c is None or not len(c) or not self.cb_ftsubj.count():
            self.canvas_ft.draw()
            return
        who = self.cb_ftsubj.currentText()
        d = c[c["person"].astype(str) == who].sort_values("epoch")
        if not len(d):
            self.canvas_ft.draw()
            return
        for i, ch in enumerate(("SBP", "DBP")):
            ax = fig.add_subplot(2, 1, i + 1)
            ax.plot(d.epoch, d[f"train_MAE_{ch}"], color="#2a78d6", lw=2,
                    label="training loss")
            ax.plot(d.epoch, d[f"val_MAE_{ch}"], color="#eb6834", lw=2,
                    label="validation loss")
            best = d.loc[d[f"val_MAE_SBP"] + d["val_MAE_DBP"] ==
                         (d["val_MAE_SBP"] + d["val_MAE_DBP"]).min(), "epoch"]
            if len(best):
                ax.axvline(float(best.iloc[0]), color="#52514e",
                           ls=(0, (4, 3)), lw=1.2,
                           label=f"best epoch {int(best.iloc[0])}")
            ax.set_ylabel(f"{ch} MAE (mmHg)", fontsize=9)
            ax.grid(True, color="#d8d7d2", lw=.7)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_title(f"{who} - head fine-tune, train vs validation",
                             fontsize=11, fontweight="bold", loc="left")
                ax.legend(fontsize=8, frameon=False)
        fig.axes[-1].set_xlabel("fine-tune epoch (-1 = base model)",
                                fontsize=9)
        self.canvas_ft.draw()

    def _plot(self, s):
        fig = self.canvas.fig
        fig.clear()
        if s["_task"] == "BP":
            self._plot_bp(fig, s)
        else:
            self._plot_cpt(fig, s)
        self.canvas.draw()

    def _plot_bp(self, fig, s):
        pred = self.last["pred"].get(s["model"])
        if pred is None:
            return
        Y = self.last["Y"]
        # Both output neurons, not just SBP: DBP flatters every absolute-error
        # statistic because it simply moves less, so showing SBP alone makes
        # the model look better than it is.
        axes = fig.subplots(2, 2)
        for j, ch in enumerate(("SBP", "DBP")):
            ax1, ax2 = axes[j]
            ax1.scatter(Y[:, j], pred[:, j], s=4, alpha=0.25, color="#2a78d6",
                        edgecolors="none")
            lo = float(min(Y[:, j].min(), pred[:, j].min()))
            hi = float(max(Y[:, j].max(), pred[:, j].max()))
            ax1.plot([lo, hi], [lo, hi], lw=1, ls=(0, (4, 3)), color="#52514e")
            ax1.set_xlabel(f"reference {ch} (mmHg)")
            ax1.set_ylabel(f"estimated {ch} (mmHg)")
            mae = s.get(f"MAE_{ch}")
            if mae is None:
                mae = float(np.abs(pred[:, j] - Y[:, j]).mean())
            ax1.set_title(f"{s['model']}  {ch}  MAE {mae:.2f} mmHg",
                          fontsize=10, loc="left")

            mean = (Y[:, j] + pred[:, j]) / 2.0
            diff = pred[:, j] - Y[:, j]
            md, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
            ax2.scatter(mean, diff, s=4, alpha=0.25, color="#2a78d6",
                        edgecolors="none")
            for yv, st in ((md, "-"), (md + 1.96 * sd, (0, (4, 3))),
                           (md - 1.96 * sd, (0, (4, 3)))):
                ax2.axhline(yv, lw=1.2, ls=st, color="#eb6834")
            ax2.set_xlabel("mean of reference and estimate (mmHg)")
            ax2.set_ylabel("estimate - reference (mmHg)")
            ax2.set_title(f"{ch}  bias {md:+.1f}, LoA {1.96 * sd:.0f}",
                          fontsize=10, loc="left")
            for a in (ax1, ax2):
                a.grid(True, color="#dcdcd8", lw=0.6); a.set_axisbelow(True)
                for sp in ("top", "right"):
                    a.spines[sp].set_visible(False)
        fig.subplots_adjust(hspace=0.42, wspace=0.26)

    def _plot_cpt(self, fig, s):
        from sklearn.metrics import roc_curve
        oof = self.last["oof"].get(s["model"])
        if oof is None:
            return
        y = self.last["y"]
        ax1, ax2 = fig.subplots(1, 2)
        ok = ~np.isnan(oof)
        fpr, tpr, _ = roc_curve(y[ok], oof[ok])
        ax1.plot(fpr, tpr, lw=2, color="#2a78d6")
        ax1.plot([0, 1], [0, 1], lw=1, ls=(0, (4, 3)), color="#52514e")
        ax1.set_xlabel("false-positive rate"); ax1.set_ylabel("true-positive rate")
        ax1.set_title(f"{s['model']}  AUC {s['auc_pooled']:.3f}", fontsize=10,
                      loc="left")
        f = s["_folds"].sort_values("auc")
        ax2.barh(f.person, f.auc, color="#2a78d6", height=0.65)
        ax2.axvline(0.5, lw=1, ls=(0, (4, 3)), color="#52514e")
        ax2.set_xlim(0, 1); ax2.set_xlabel("AUC (held-out subject)")
        ax2.set_title("per-subject transfer", fontsize=10, loc="left")
        ax2.tick_params(labelsize=8)
        for a in (ax1, ax2):
            a.grid(True, color="#dcdcd8", lw=0.6); a.set_axisbelow(True)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)

    def on_clear(self):
        self.history.clear()
        self.tbl_runs.setRowCount(0)
        self.tbl_folds.setRowCount(0)
        self.lbl_verdict.setText("")
        self.canvas.fig.clear(); self.canvas.draw()
        self._fill_curve({})
        self._fill_ft(None, None)

    def _browse(self, line_edit, what):
        path, _ = QFileDialog.getOpenFileName(self, what, ROOT, "CSV (*.csv)")
        if path:
            line_edit.setText(path)
