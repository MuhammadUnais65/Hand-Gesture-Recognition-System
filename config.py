# config.py — central settings for the gesture recognition project

GESTURES = ["open", "fist", "point"]

SAMPLES_PER_GESTURE = 100
SAMPLE_DELAY_FRAMES = 3   # skip frames between saves so samples stay diverse

DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE  = 0.7

KNN_NEIGHBORS        = 3
CONFIDENCE_THRESHOLD = 0.6   # below this, prediction shows as "Unknown"

MODEL_PATH = "gesture_model.pkl"
