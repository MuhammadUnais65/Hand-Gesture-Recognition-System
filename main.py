import os
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from config import (
    GESTURES,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE,
    KNN_NEIGHBORS,
    CONFIDENCE_THRESHOLD,
    MODEL_PATH,
)

def load_data():
    X, y = [], []
    for gesture in GESTURES:
        path = f"{gesture}.npy"
        if not os.path.exists(path):
            print(f"Missing: {path} — run record_gesture.py for '{gesture}' first.")
            exit()
        data = np.load(path)
        X.append(data)
        y.extend([gesture] * len(data))
    return np.vstack(X), np.array(y)

def get_model():
    # reuse saved model if it exists, otherwise train and save
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH} ...")
        return joblib.load(MODEL_PATH)
    print("Training model ...")
    X, y = load_data()
    model = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

def predict_gesture(model, keypoints):
    if len(keypoints) != 63:
        return "Unknown", 0.0

    proba  = model.predict_proba([keypoints])[0]
    confidence = proba.max()

    if confidence < CONFIDENCE_THRESHOLD:
        return "Unknown", confidence

    label = model.classes_[proba.argmax()]
    return label, confidence

def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        exit()
    return cap

def run():
    model = get_model()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,)
    draw = mp.solutions.drawing_utils
    cap  = open_camera()

    print("Camera running — press 'q' to quit.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed.")
            break

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture_label = "No hand detected"
        confidence    = 0.0

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                keypoints = []
                for lm in hand_lm.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

                gesture_label, confidence = predict_gesture(model, keypoints)

        # FPS
        now      = time.time()
        fps      = 1.0 / (now - prev_time + 1e-6)
        prev_time = now

        # overlay
        color = (0, 255, 0) if gesture_label not in ("Unknown", "No hand detected") else (0, 140, 255)

        cv2.putText(frame, f"Gesture: {gesture_label}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if confidence > 0:
            cv2.putText(frame, f"Confidence: {confidence:.0%}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
