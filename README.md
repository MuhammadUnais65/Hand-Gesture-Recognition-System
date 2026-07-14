# Hand Gesture Recognition System

A real-time hand gesture recognition system built with **MediaPipe**, **OpenCV**, and a **KNN classifier**. The system lets you record your own hand gestures, trains a model on them, and then recognizes those gestures live through your webcam — displaying the predicted label and confidence score on screen.

---

## How It Works

The pipeline has two stages:

**1. Data Collection (`record_gesture.py`)**
MediaPipe extracts 21 hand landmarks (x, y, z) from each video frame — 63 values per sample. You record 100 samples per gesture, with a small delay between captures to keep the data diverse. Samples are saved as `.npy` files.

**2. Training + Live Prediction (`main.py`)**
On first run, the script loads all `.npy` files, trains a KNN classifier, and saves the model (`gesture_model.pkl`). Every run after that loads the saved model directly — no retraining needed. Predictions include a confidence score; anything below the threshold shows as "Unknown."

---

## Features

- Record any gesture you want — just add it to `config.py`
- 100 samples per gesture with frame-skipping for data diversity
- Visual progress bar during recording
- Model saved after first training — fast startup on subsequent runs
- Live confidence score displayed on screen
- "Unknown" label for low-confidence predictions
- FPS counter
- Clean shared config — no duplicate settings across files

---

## Project Structure

```
Hand-Gesture-Recognition-System/
│
├── Gesture Recognition   # sample video
├── README.md
├── config.py            # All settings in one place (gestures, thresholds, paths)
├── main.py              # Step 2: train model and run live prediction
├── record_gesture.py    # Step 1: record samples for a gesture

│
├── open.npy             # Recorded samples (will generate after recording)
├── fist.npy
├── point.npy
│
└── gesture_model.pkl    # Trained KNN model (will generate after first run of main.py)
```

---

## Setup

```bash
pip install opencv-python mediapipe scikit-learn joblib numpy
```

---

## Usage

### Step 1 — Record gestures

Run this once for each gesture:

```bash
python record_gesture.py
```

You'll be prompted to enter a gesture name (`open`, `fist`, or `point`). Hold your hand in front of the camera and stay still — the recorder captures 100 samples and saves them automatically.

Repeat for each gesture.

### Step 2 — Run live recognition

```bash
python main.py
```

The model trains on your recorded data (first run only), then opens your webcam. Show any recorded gesture to see the prediction and confidence live. Press `q` to quit.

---

## Configuration

All key settings are in `config.py`:

```python
GESTURES             = ["open", "fist", "point"]  # add your own here
SAMPLES_PER_GESTURE  = 100
SAMPLE_DELAY_FRAMES  = 3      # frames skipped between saves
KNN_NEIGHBORS        = 3
CONFIDENCE_THRESHOLD = 0.6    # below this → "Unknown"
MODEL_PATH           = "gesture_model.pkl"
```

To add a new gesture: add its name to `GESTURES`, record it with `record_gesture.py`, then delete `gesture_model.pkl` so the model retrains with the new data.

---

## Tech Stack

| Component | Library |
|---|---|
| Hand landmark detection | MediaPipe |
| Video capture & display | OpenCV |
| Classification | scikit-learn (KNN) |
| Model persistence | joblib |
| Data storage | NumPy (.npy) |
| Language | Python 3 |

---

## Possible Next Steps

- Add more gestures (thumbs up, peace sign, etc.)
- Replace KNN with a small neural network for better accuracy
- Trigger keyboard/mouse actions based on detected gesture
- Two-hand gesture support
- Build a simple GUI for the recorder

---

## Author

Part of an ongoing portfolio of Data Science, Machine Learning, and Computer Vision projects focused on practical, real-world applications.
