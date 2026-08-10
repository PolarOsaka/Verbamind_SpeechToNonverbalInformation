import numpy as np
import librosa
from scipy.signal import find_peaks
from config import FRAME_DURATION, FRAME_OVERLAP, PITCH_MAX, PITCH_MIN, PITCH_PROMINENCE, PITCH_CORRELATION_THRESHOLD

def frame_audio(audio, sr):
    """
    Split audio into overlapping frames.
    """

    frame_length = int(FRAME_DURATION * sr)

    hop_length = int(frame_length * (1 - FRAME_OVERLAP))

    frames = librosa.util.frame(
        audio,
        frame_length=frame_length,
        hop_length=hop_length
    )

    return frames.T


def compute_autocorrelation(frame):
    """
    Calculate autocorrelation of the frame
    """

    frame = frame - np.mean(frame) # Menghilangkan dc offset supaya tidak memengaruhi korelasi

    corr = np.correlate(
        frame,
        frame,
        mode="full"
    ) # Mengalikan sinyal dengan dirinya sendiri yang digeser-geser

    corr = corr[len(corr)//2:] # Karena hasil autokorelasi simetris, ambil hanya jeda waktu positif

    # Normalisasi
    # corr /= np.max(np.abs(corr)) 
    max_corr = np.max(np.abs(corr))
    if max_corr == 0:
        return np.zeros_like(corr)
    corr /= max_corr

    return corr

def compute_frame_pitch(frame, sr):
    """
    Estimate fundamental frequency (F0)
    using Normalized Autocorrelation Function (NACF).

    Returns
    -------
    float
        Fundamental frequency (Hz)
        or np.nan for unvoiced frames.
    """
    min_lag = int(sr / PITCH_MAX)
    max_lag = int(sr / PITCH_MIN)

    corr = compute_autocorrelation(frame)

    # Ambil potongan korelasi di area pencarian
    search_area = corr[min_lag:max_lag]

    # Cari puncak-puncak yang menonjol
    peaks, properties = find_peaks(search_area, prominence=PITCH_PROMINENCE)

    # Jika tidak ada puncak yang ditemukan (misal frame hening/noise/unvoiced)
    if len(peaks) == 0:
        return np.nan

    # Ambil puncak dengan nilai korelasi tertinggi di antara puncak yang lolos prominence
    # best_peak = peaks[np.argmax(search_area[peaks])]
    best_peak = peaks[
        np.argmax(properties["prominences"])
    ]
    peak_value = search_area[best_peak]

    # THRESHOLD UNVOICED: Jika korelasinya di bawah 0.3, anggap Unvoiced/Noise
    if peak_value < PITCH_CORRELATION_THRESHOLD:
        return np.nan

    # Hitung lag sebenarnya
    lag = best_peak + min_lag

    # Hitung pitch
    f0 = sr / lag

    return f0

def compute_pitch(audio, sr):

    frames = frame_audio(audio, sr)

    pitch = []

    for frame in frames:

        pitch.append(
            compute_frame_pitch(frame, sr)
        )

    pitch = np.asarray(pitch)

    if np.all(np.isnan(pitch)):
        return np.nan, pitch.tolist()

    return float(np.nanmedian(pitch)), pitch.tolist()