#!/usr/bin/env python3
"""Entry point for the standalone Beat Pipeline tool. Run: python3 beat_pipeline/main.py

Six tabs, in the order you use them:

  1. Preprocessing & R-peak QC  (beat_window.py) - THE ONLY PLACE the
     recording is loaded: pick a subject (resolves both the .txt and its BP
     labels) or browse a file manually, filter, run Pan-Tompkins, inspect
     every stage before anything downstream trusts the peaks.
  2. ECG Features               (feature_window.py) - the 19 sliding-window
     features on that same recording, with the within-subject r against SBP.
  3. SKNA Features              (skna_feature_window.py) - the 11 SKNA
     features on the SAME window grid, so the two tables join row-for-row.
  4. Feature Distributions      (feature_boxplot_window.py) - box plots of
     every ECG/SKNA feature over the whole COHORT (loads the packed .npz
     files, not the current recording).
  5. Feature Engineering        (feature_engineering_window.py) - runs
     feature_engineering.py's merge/normalize/temporal/prune pipeline on the
     cohort CSVs and lets you inspect the RESULT: per-feature pooled vs.
     within-person |r| against SBP, one recording's engineered trace over
     time (CPT segment shaded), the pooled by-phase distribution, and a
     correlation heatmap of the final pruned set - the checks you'd otherwise
     only get by re-reading the report .txt after a command-line run.
  6. BP Model (ANN)             (model_window.py) - the project target: SBP and
     DBP from the engineered features, using the two-layer feedforward ANN
     (ann_bp_loso.py) instead of the CNN-BiLSTM on raw waveforms in src/. Set
     the pipeline up in tab 5, switch here, and "From tab 5" trains on exactly
     the working set you were looking at, top-K filter included. Evaluated
     leave-one-PERSON-out, with the calibration mode, feature families and
     hidden width as controls. CPT classification is selectable as the sanity
     check that the features carry autonomic information at all. Every results
     table carries the baseline the model has to beat - constant train-mean or
     zero-delta - plus the oracle subject-mean row.

Picking a subject or file in tab 1 carries the path, ECG channel AND subject
over to tabs 2 and 3 automatically on tab switch - neither tab has its own
file picker any more, so there is exactly one place to load a recording.
Each feature tab still re-runs the analysis pipeline itself on that shared
file (they need different analysis rates - 500 Hz for ECG, native for
SKNA's 500-1000 Hz band), so 2 and 3 stay independent of tab 1 having been
RUN, just not of it having been LOADED. Tabs 4 and 5 are independent of all
three - tab 4 reads whatever cohort .npz files build_ecg_features.py/
build_skna_features.py last wrote, and tab 5 reads their cohort CSVs directly.
"""
import os
import sys

from PyQt5.QtWidgets import QApplication, QTabWidget

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from beat_window import BeatPipelineWindow
from feature_window import FeatureWindow
from skna_feature_window import SknaFeatureWindow
from feature_boxplot_window import FeatureBoxplotWindow
from feature_engineering_window import FeatureEngineeringWindow
from model_window import ModelWindow


class BeatPipelineApp(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beat Pipeline - R-peak QC & ECG features")
        self.resize(1280, 820)

        self.tab_beat = BeatPipelineWindow()
        self.tab_feat = FeatureWindow()
        self.tab_skna = SknaFeatureWindow()
        self.tab_box = FeatureBoxplotWindow()
        self.tab_eng = FeatureEngineeringWindow()
        self.tab_model = ModelWindow()
        self.addTab(self.tab_beat, "1. Preprocessing && R-peak QC")
        self.addTab(self.tab_feat, "2. ECG Features")
        self.addTab(self.tab_skna, "3. SKNA Features")
        self.addTab(self.tab_box, "4. Feature Distributions")
        self.addTab(self.tab_eng, "5. Feature Engineering")
        self.addTab(self.tab_model, "6. BP Model (ANN)")

        # Hand the file over on tab switch rather than on browse: the user may
        # browse several times while iterating on the QC, and only the file
        # that is loaded when they move on is the one they meant.
        self._handed_over = {}
        self.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index):
        # Each file from tab 1 is pushed to each tab exactly once. Without that
        # guard, a file picked directly in a feature tab would be silently
        # replaced every time the user looked back at the QC tab.
        tab = self.widget(index)
        if (tab in (self.tab_feat, self.tab_skna) and self.tab_beat.path
                and self.tab_beat.path != self._handed_over.get(index)):
            tab.set_recording(self.tab_beat.path,
                              self.tab_beat.cb_channel.currentText(),
                              self.tab_beat.subject)
            self._handed_over[index] = self.tab_beat.path
        # Tab 4 reads the cohort .npz files, not the current recording - load
        # them the first time the tab is opened rather than at startup, so a
        # missing built/ file does not pop a warning before the user asked.
        elif tab is self.tab_box and index not in self._handed_over:
            self.tab_box.on_reload()
            self._handed_over[index] = True
        # Same idea for tab 5: if the cohort CSVs already exist at the
        # default path, engineer and show them on first visit instead of
        # requiring a click just to check what's already there.
        elif tab is self.tab_eng and index not in self._handed_over:
            self.tab_eng.on_first_open()
            self._handed_over[index] = True
        # Tab 6 trains on whatever tab 5 is currently showing. Unlike the tab-1
        # handoff this is NOT once-only: the working set is exactly what the
        # user is looking at, top-K filter included, so it is re-read every
        # time tab 6 is opened.
        elif tab is self.tab_model and self.tab_eng.df is not None:
            self.tab_model.set_feature_set(self.tab_eng.df, self.tab_eng.cols,
                                           self.tab_eng.notes)


def main():
    app = QApplication(sys.argv)
    window = BeatPipelineApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
