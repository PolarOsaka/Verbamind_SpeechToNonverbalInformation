import librosa
import matplotlib.pyplot as plt
from features.loudness import compute_loudness
from baseline.initialization import initialize_baseline
from baseline.adaptive import update_baseline
import pandas as pd
from fuzzy.inference import ChangeFuzzyInference
from config import FRAME_DURATION, FRAME_OVERLAP, INITIAL_DURATION

audio, sr = librosa.load(
    "/workspaces/Verbamind_SpeechToNonverbalInformation/tests/test_audio.wav",
    sr=None
)

median_loudness, frame_loudness = compute_loudness(audio, sr)

print(f"median_loudness = {median_loudness:.2f} dB")

plt.plot(frame_loudness)
plt.xlabel("Frame")
plt.ylabel("Loudness (dB)")
plt.savefig("tests/loudness_plot.png")
print("Plot berhasil disimpan di tests/loudness_plot.png")

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

# print("Baseline:", LoudnessBaseline.baseline)
# print("BaselineDeviation:", LoudnessBaseline.deviation)

# baseline_loudness_update = []
# baseline_deviation_update = []

fuzzy_engine = ChangeFuzzyInference()
category_frame_array = []

for i in range(num_frames_initial, len(frame_loudness)):
    delta_loudness = frame_loudness[i] - LoudnessBaseline.baseline

    # baseline_loudness_update.append(LoudnessBaseline.baseline)
    # baseline_deviation_update.append(LoudnessBaseline.deviation)

    # Print new baseline
    # print("New baseline:", LoudnessBaseline.baseline)
    # print("New baseline deviation:", LoudnessBaseline.deviation)
    # print("Delta_loudness:", delta_loudness)

    # Fuzzy Inference
    if abs(delta_loudness) <= LoudnessBaseline.deviation:
        category = "No Significant Change"

    else:
        category = fuzzy_engine.infer(
            delta_loudness,
            LoudnessBaseline.deviation
        )
    
    category_frame_array.append([i, frame_loudness[i], LoudnessBaseline.baseline, delta_loudness, LoudnessBaseline.deviation, category])

    # Adaptive Baseline
    LoudnessBaseline = update_baseline(LoudnessBaseline, frame_loudness[i])

df = pd.DataFrame(category_frame_array, columns=["Frame", "Current_Loudness", "Baseline_Loudness", "Delta_Loudness", "Baseline_Deviation", "Category"])
df.to_csv(
    "/workspaces/Verbamind_SpeechToNonverbalInformation/tests/categoryperframe.csv",
    index=False,
)

# plt.figure(figsize=(12,4))

# plt.plot(frame_loudness, label="Loudness")
# plt.plot(
#     range(num_frames_initial, len(frame_loudness)),
#     baseline_loudness_update,
#     label="Adaptive Baseline"
# )
# plt.legend()
# plt.savefig("tests/baseline_loudness_update_plot.png")


