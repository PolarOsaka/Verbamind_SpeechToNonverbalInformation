# Verbamind: Speech-to-Nonverbal-Information

**Verbamind** is an audio processing tool designed to extract nonverbal speech cues—such as changes in loudness, pitch, speech rate, and pause detection. It is specifically tailored to assist in **psychological documentation and verbatim transcription analysis**. The system extracts acoustic features from speech, establishes an adaptive baseline, and classifies significant changes using fuzzy inference.

## 📋 Features & Roadmap

- [x] **Loudness Change Detection** *(Active & Improved)*
- [X] **Pitch Change Detection** *(Active & Improved)*
- [ ] **Speech Rate Change Detection** *(In Development)*
- [ ] **Pause Detection** *(In Development)*

## 🚀 Current Functionality

Currently, the **Loudness and Pitch Change Detector** is fully updated and operational. 

- **Input:** WAV audio file (`audio.wav`)
- **Output:** Frame-by-frame loudness classification:
  - `Very Low`
  - `Low`
  - `No Significant Change`
  - `High`
  - `Very High`
  
The current pipeline that operates at the **audio-frame level**:

```text
Audio Recording
      │
      ├───────────────┐
      ↓               ↓
Loudness            Pitch
(RMS)               (NACF)
      │               │
      ↓               ↓
Frame-level         Frame-level
Feature             Feature
      │               │
      └───────┬───────┘
              ↓
      Initial Baseline
              ↓
       Adaptive Baseline
              ↓
        Δ Feature Value
              ↓
        Fuzzy Inference
              ↓
      Change Category
```

## 📋 Project Structure

```text
Verbamind_SpeechToNonverbalInformation/
│
├── config.py
│
├── input
│   ├── audio.wav
│
├── output
│   ├── LoudnessCategoryPerFrame.csv
│   └── PitchCategoryPerFrame.csv
│
├── features/
│   ├── loudness.py
│   └── pitch.py
│
├── baseline/
│   ├── initialization.py
│   └── adaptive.py
│
├── fuzzy/
│   └── inference.py
│
├── pipeline/
│   └── detector.py
│
├── tests/
│   ├── test.py
│
└── main.py
```
---

## 🚀 Module Responsibilities

### `config.py`

Contains global configuration parameters used throughout the pipeline.

Examples:

```text
FRAME_DURATION
FRAME_OVERLAP
PITCH_MIN
PITCH_MAX
INITIAL_DURATION
EMA coefficients
Fuzzy thresholds
```
---

### Feature Extraction Module
The feature extraction module extracts loudness and pitch (accoustic features) from the audio signal.

---

#### `features/loudness.py`

Extracts frame-level loudness from an audio recording.

##### Processing

```text
Audio
 ↓
Overlapping Frames
 ↓
RMS
 ↓
Frame Loudness
 ↓
Median Loudness
```
Example:

```python
median_loudness, frame_loudness = compute_loudness(
    audio,
    sr
)
```

`frame_loudness[i]` represents the loudness of frame `i`.

---

#### `features/pitch.py`

Estimates the fundamental frequency (F0) of each audio frame using the Normalized Autocorrelation Function (NACF).

##### Processing

```text
Audio
 ↓
Overlapping Frames
 ↓
DC Removal
 ↓
Autocorrelation
 ↓
Normalization
 ↓
Peak Detection
 ↓
F0 Estimation
```

The pitch search range is currently defined by:

```text
PITCH_MIN = 75 Hz
PITCH_MAX = 300 Hz
```

Unvoiced or unreliable frames are represented by:

```python
np.nan
```
Example:

```python
median_pitch, frame_pitch = compute_pitch(
    audio,
    sr
)
```

`frame_pitch[i]` represents the estimated F0 of frame `i`.

---

### Baseline Module

The baseline module provides a reference point against which changes in acoustic features are measured.

The same baseline implementation is reused for different features.

```text
Feature Value
     ↓
Initial Baseline
     ↓
Adaptive Baseline
```

---

#### `baseline/initialization.py`
Initializes the baseline and baseline deviation from the first portion of the recording.

---

#### `baseline/adaptive.py`

Updates the baseline using an Exponential Moving Average (EMA).

##### Processing

```text
Current Feature
      ↓
Δ Feature
      ↓
EMA
      ↓
Updated Baseline
Updated Deviation
```
The module is feature-independent and can therefore be reused for:

* Loudness
* Pitch
* Speech Rate

---

### Fuzzy Inference

#### `fuzzy/inference.py`

Provides a generic fuzzy classifier for significant feature changes.

The same fuzzy inference engine is intended to be used for:

```text
Loudness
Pitch
Speech Rate
```
`No Significant Change` is handled by the detector before fuzzy inference.

Conceptually:

```text
              Δ / deviation
                    │
                    ↓
             Fuzzy Inference
                    │
       ┌────────────┼────────────┐
       ↓            ↓            ↓
   Very Low        Low       High / Very High
```

---

### Detection Pipeline

#### `pipeline/detector.py`

Combines feature extraction results, adaptive baseline tracking, and fuzzy inference.

The current detector operates at the **frame level**.

##### Processing

For each frame after the initial baseline period:

```text
Current Feature
      ↓
Calculate Δ
      ↓
Compare against Baseline Deviation
      │
      ├── Not significant
      │       ↓
      │   No Significant Change
      │
      └── Significant
              ↓
        Fuzzy Inference
              ↓
        Change Category
              ↓
       Update EMA Baseline
```
---

##### Loudness Output

Current columns:

```text
Frame
Current_Loudness
Baseline_Loudness
Delta_Loudness
Baseline_Loudness_Deviation
Loudness_Category
```

Example:

```text
Frame | Current_Loudness | Baseline_Loudness | Delta_Loudness | Baseline_Loudness_Deviation | Loudness_Category
```

---

##### Pitch Output

Current columns:

```text
Frame
Current_Pitch
Baseline_Pitch
Delta_Pitch
Baseline_Pitch_Deviation
Pitch_Category
```

Example:

```text
Frame | Current_Pitch | Baseline_Pitch | Delta_Pitch | Baseline_Pitch_Deviation | Pitch_Category
```

For unvoiced frames:

```text
Pitch_Category = Unvoiced
```

and the pitch baseline is not updated.

---

### `main.py`

`main.py` serves as the entry point for running the current pipeline.

Its responsibilities are:

```text
Load Audio
   ↓
Extract Loudness
   ↓
Extract Pitch
   ↓
Determine Initial Baseline Period
   ↓
Run Nonverbal Change Detector
   ↓
Save Detection Results
```

#### Input

Filepath to an audio recording:

```text
.wav
```

The current prototype is tested using speech/audio recordings loaded with `librosa`.

Example:

```python
audio, sr = librosa.load(
    AUDIO_PATH,
    sr=None
)
```

#### Output

CSV files containing frame-level detection results:

```text
tests/
├── LoudnessCategoryPerFrame.csv
└── PitchCategoryPerFrame.csv
```

---

## 📋 Current Input / Output Summary

```text
┌───────────────────────┐
│ Audio Recording       │
│ .wav                  │
└───────────┬───────────┘
            │
            ↓
┌───────────────────────┐
│ Feature Extraction    │
│                       │
│ Loudness → dB         │
│ Pitch    → Hz         │
└───────────┬───────────┘
            │
            ↓
┌───────────────────────┐
│ Adaptive Baseline     │
│                       │
│ Baseline              │
│ Deviation             │
└───────────┬───────────┘
            │
            ↓
┌───────────────────────┐
│ Fuzzy Change          │
│ Classification        │
└───────────┬───────────┘
            │
            ↓
┌───────────────────────┐
│ Frame-level CSV       │
│                       │
│ Loudness Category     │
│ Pitch Category        │
└───────────────────────┘
```
---
