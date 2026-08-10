# Jadi untuk sekarang, kita boleh memakai frame_pitch untuk mengujikan adaptive baseline, sama seperti yang kita lakukan pada loudness.

# Tetapi nanti pipeline final harus menjadi:

# Patient utterance
# ->
# compute_pitch()
# ->
# median_pitch
# ->
# PitchBaseline
# ->
# delta_P
# ->
# Fuzzy

# Bukan:

# Patient utterance
# ->
# 50 frame F0
# ->
# 50 kali update baseline

# Itu penting supaya konsisten dengan pseudocode awalmu.

import numpy as np
import pandas as pd

from baseline.initialization import initialize_baseline
from baseline.adaptive import update_baseline
from fuzzy.inference import ChangeFuzzyInference

class NonverbalChangeDetector:

    def __init__(self, initial_duration_frames):
        self.initial_duration_frames = initial_duration_frames
        self.fuzzy_engine = ChangeFuzzyInference()

    def detect(
        self,
        loudness_values,
        pitch_values
    ):
        """
        Detect loudness and pitch changes
        using adaptive baseline and fuzzy inference.
        """

        loudness_values = np.asarray(loudness_values)
        pitch_values = np.asarray(pitch_values)

        # ==============================
        # Initialization
        # ==============================

        loudness_baseline = initialize_baseline(
            loudness_values[
                :self.initial_duration_frames
            ]
        )

        pitch_baseline = initialize_baseline(
            pitch_values[
                :self.initial_duration_frames
            ]
        )

        loudness_results = []
        pitch_results = []

        # ==============================
        # Detection
        # ==============================

        for i in range(
            self.initial_duration_frames,
            len(loudness_values)
        ):

            # ==================================
            # Loudness
            # ==================================

            current_loudness = loudness_values[i]

            delta_loudness = (
                current_loudness
                - loudness_baseline.baseline
            )

            current_loudness_deviation = (
                loudness_baseline.deviation
            )

            if (
                abs(delta_loudness)
                <= current_loudness_deviation
            ):

                loudness_category = (
                    "No Significant Change"
                )

            else:

                loudness_category = (
                    self.fuzzy_engine.infer(
                        delta_loudness,
                        current_loudness_deviation
                    )
                )

            loudness_results.append([
                i,
                current_loudness,
                loudness_baseline.baseline,
                delta_loudness,
                current_loudness_deviation,
                loudness_category
            ])

            # ==================================
            # Pitch
            # ==================================

            current_pitch = pitch_values[i]

            if np.isnan(current_pitch):

                # Unvoiced frame
                pitch_results.append([
                    i,
                    np.nan,
                    pitch_baseline.baseline,
                    np.nan,
                    pitch_baseline.deviation,
                    "Unvoiced"
                ])

            else:

                delta_pitch = (
                    current_pitch
                    - pitch_baseline.baseline
                )

                current_pitch_deviation = (
                    pitch_baseline.deviation
                )

                if (
                    abs(delta_pitch)
                    <= current_pitch_deviation
                ):

                    pitch_category = (
                        "No Significant Change"
                    )

                else:

                    pitch_category = (
                        self.fuzzy_engine.infer(
                            delta_pitch,
                            current_pitch_deviation
                        )
                    )

                pitch_results.append([
                    i,
                    current_pitch,
                    pitch_baseline.baseline,
                    delta_pitch,
                    current_pitch_deviation,
                    pitch_category
                ])

            # ==================================
            # Adaptive Baseline
            # ==================================

            loudness_baseline = update_baseline(
                loudness_baseline,
                current_loudness
            )

            if not np.isnan(current_pitch):

                pitch_baseline = update_baseline(
                    pitch_baseline,
                    current_pitch
                )

        # ==============================
        # Convert to DataFrame
        # ==============================

        loudness_df = pd.DataFrame(
            loudness_results,
            columns=[
                "Frame",
                "Current_Loudness",
                "Baseline_Loudness",
                "Delta_Loudness",
                "Baseline_Loudness_Deviation",
                "Loudness_Category"
            ]
        )

        pitch_df = pd.DataFrame(
            pitch_results,
            columns=[
                "Frame",
                "Current_Pitch",
                "Baseline_Pitch",
                "Delta_Pitch",
                "Baseline_Pitch_Deviation",
                "Pitch_Category"
            ]
        )

        return loudness_df, pitch_df