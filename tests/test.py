import librosa
import matplotlib.pyplot as plt
from features.loudness import compute_loudness
from features.pitch import compute_pitch
from baseline.initialization import initialize_baseline
from baseline.adaptive import update_baseline
import pandas as pd
from fuzzy.inference import ChangeFuzzyInference
from config import FRAME_DURATION, FRAME_OVERLAP, INITIAL_DURATION

audio, sr = librosa.load(
    "/workspaces/Verbamind_SpeechToNonverbalInformation/Audio_Input/sweep_100-300Hz_log_30s_44k_16bit.wav",
    sr=None
)

median_loudness, frame_loudness = compute_loudness(audio, sr)
print(f"median_loudness = {median_loudness:.2f} dB")

median_pitch, frame_pitch = compute_pitch(audio, sr)
print(f"Median Fundamental Frequency = {median_pitch:.2f} Hz")

# # Loudness Plot
# plt.plot(frame_loudness)
# plt.xlabel("Frame")
# plt.ylabel("Loudness (dB)")
# plt.savefig("tests/loudness_plot.png")
# print("Plot berhasil disimpan di tests/loudness_plot.png")

# # Pitch Plot
# plt.figure(figsize=(12, 4))

# plt.plot(frame_pitch)

# plt.xlabel("Frame")
# plt.ylabel("F0 (Hz)")
# plt.title("Fundamental Frequency (NACF)")

# plt.ylim(50, 350)
# plt.grid(True)

# plt.savefig("tests/pitch_plot.png")
# plt.close()

# print("Plot berhasil disimpan di tests/pitch_plot.png")

# 1. Hitung frame_length dan hop_length dalam jumlah SAMPEL
frame_length = int(FRAME_DURATION * sr)             # Contoh (sr=16k): 0.025 * 16000 = 400 sampel
hop_length = int(frame_length * (1 - FRAME_OVERLAP)) # Contoh: 400 * 0.5 = 200 sampel

# 2. Ubah INITIAL_DURATION ke sampel juga (dikalikan sr)
initial_samples = int(INITIAL_DURATION * sr)         # Contoh: 1 * 16000 = 16000 sampel

# 3. Hitung jumlah frame (dibagi dalam satuan yang sama: sampel / sampel)
num_frames_initial = int((initial_samples - frame_length) // hop_length) + 1

# Safety check agar tidak 0 atau melebihi total frame yang ada
num_frames_initial = max(1, min(num_frames_initial, len(frame_loudness)))

LoudnessBaseline = initialize_baseline(frame_loudness[:num_frames_initial])
PitchBaseline = initialize_baseline(frame_pitch[:num_frames_initial])

# print("Baseline:", LoudnessBaseline.baseline)
# print("BaselineDeviation:", LoudnessBaseline.deviation)

# baseline_loudness_update = []
# baseline_deviation_update = []

fuzzy_engine = ChangeFuzzyInference()
loudness_category_frame_array = []
pitch_category_frame_array = []

for i in range(num_frames_initial, len(frame_loudness)):
    # ==============================
    # Deviation Baseline Initialization
    # ==============================
    delta_loudness = frame_loudness[i] - LoudnessBaseline.baseline
    delta_pitch = frame_pitch[i] - PitchBaseline.baseline

    # baseline_loudness_update.append(LoudnessBaseline.baseline)
    # baseline_deviation_update.append(LoudnessBaseline.deviation)

    # Print new baseline
    # print("New baseline:", LoudnessBaseline.baseline)
    # print("New baseline deviation:", LoudnessBaseline.deviation)
    # print("Delta_loudness:", delta_loudness)

    # ==============================
    # Loudness Fuzzy Inference
    # ==============================
    if abs(delta_loudness) <= LoudnessBaseline.deviation:
        loudness_category = "No Significant Change"
    else:
        loudness_category = fuzzy_engine.infer(
            delta_loudness,
            LoudnessBaseline.deviation
        )
    loudness_category_frame_array.append([i, frame_loudness[i], LoudnessBaseline.baseline, delta_loudness, LoudnessBaseline.deviation, loudness_category])

    # ==============================
    # Pitch Fuzzy Inference
    # ==============================
    if abs(delta_pitch) <= PitchBaseline.deviation:
        pitch_category = "No Significant Change"
    else:
        pitch_category = fuzzy_engine.infer(
            delta_pitch,
            PitchBaseline.deviation
        )
    pitch_category_frame_array.append([i, frame_pitch[i], PitchBaseline.baseline, delta_pitch, PitchBaseline.deviation, pitch_category])

    # ==============================
    # Adaptive Baseline
    # ==============================
    LoudnessBaseline = update_baseline(LoudnessBaseline, frame_loudness[i])
    PitchBaseline = update_baseline(PitchBaseline, frame_pitch[i])

# loudness_df = pd.DataFrame(loudness_category_frame_array, columns=["Frame", "Current_Loudness", "Baseline_Loudness", "Delta_Loudness", "Baseline_Loudness_Deviation", "Loudness_Category"])
# loudness_df.to_csv(
#     "/workspaces/Verbamind_SpeechToNonverbalInformation/Model_Output/Detection_csv/LoudnessCategoryPerFrame.csv",
#     index=False,
# )

# pitch_df = pd.DataFrame(pitch_category_frame_array, columns=["Frame", "Current_Pitch", "Baseline_Pitch", "Delta_Pitch", "Baseline_Pitch_Deviation", "Pitch_Category"])
# pitch_df.to_csv(
#     "/workspaces/Verbamind_SpeechToNonverbalInformation/Model_Output/Detection_csv/PitchCategoryPerFrame.csv",
#     index=False,
# )

# plt.figure(figsize=(12,4))

# plt.plot(frame_loudness, label="Loudness")
# plt.plot(
#     range(num_frames_initial, len(frame_loudness)),
#     baseline_loudness_update,
#     label="Adaptive Baseline"
# )
# plt.legend()
# plt.savefig("tests/baseline_loudness_update_plot.png")

# plt.figure(figsize=(12, 4))

# plt.plot(
#     frame_pitch,
#     label="F0"
# )

# plt.plot(
#     range(
#         num_frames_initial,
#         num_frames_initial + len(pitch_baseline_history)
#     ),
#     pitch_baseline_history,
#     label="Adaptive Baseline"
# )

# plt.xlabel("Frame")
# plt.ylabel("F0 (Hz)")
# plt.title("Pitch and Adaptive Baseline")

# plt.legend()
# plt.grid(True)

# plt.savefig(
#     "Model_Output/Plot/pitch_baseline_plot.png"
# )

# plt.close()
