import librosa
from features.loudness import compute_loudness
from features.pitch import compute_pitch
from pipeline.detector import NonverbalChangeDetector
from config import (
    FRAME_DURATION,
    FRAME_OVERLAP,
    INITIAL_DURATION
)

AUDIO_PATH = (
    "Audio_Input/sine_300Hz_5s_44k_16bit.wav"
)

def main():

    # ==============================
    # Load audio
    # ==============================

    audio, sr = librosa.load(
        AUDIO_PATH,
        sr=None
    )

    # ==============================
    # Feature extraction
    # ==============================

    median_loudness, frame_loudness = (
        compute_loudness(audio, sr)
    )

    median_pitch, frame_pitch = (
        compute_pitch(audio, sr)
    )

    print(
        f"Median Loudness: "
        f"{median_loudness:.2f} dB"
    )

    print(
        f"Median F0: "
        f"{median_pitch:.2f} Hz"
    )

    # ==============================
    # Initial duration -> frame count
    # ==============================

    frame_length = int(
        FRAME_DURATION * sr
    )

    hop_length = int(
        frame_length
        * (1 - FRAME_OVERLAP)
    )

    initial_samples = int(
        INITIAL_DURATION * sr
    )

    num_frames_initial = (
        (initial_samples - frame_length)
        // hop_length
    ) + 1

    num_frames_initial = max(
        1,
        min(
            num_frames_initial,
            len(frame_loudness)
        )
    )

    print(
        f"Initial frames: "
        f"{num_frames_initial}"
    )

    # ==============================
    # Detector
    # ==============================

    detector = NonverbalChangeDetector(
        initial_duration_frames=
        num_frames_initial
    )

    loudness_df, pitch_df = detector.detect(
        frame_loudness,
        frame_pitch
    )

    # ==============================
    # Save results
    # ==============================

    loudness_df.to_csv(
        "Model_Output/Detection_csv/LoudnessCategoryPerFrame.csv",
        index=False
    )

    pitch_df.to_csv(
        "Model_Output/Detection_csv/PitchCategoryPerFrame.csv",
        index=False
    )

    print(
        "Detection selesai."
    )


if __name__ == "__main__":
    main()