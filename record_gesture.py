import cv2
import mediapipe as mp
import numpy as np

from config import (
    GESTURES,
    SAMPLES_PER_GESTURE,
    SAMPLE_DELAY_FRAMES,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE,
)

def get_gesture_name():
    print(f"Available gestures: {', '.join(GESTURES)}")
    name = input("Enter gesture name to record: ").strip().lower()
    if name not in GESTURES:
        print(f"'{name}' is not in the gesture list. Add it to config.py first.")
        exit()
    return name

def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera. Check if it's connected.")
        exit()
    return cap

def record(gesture_name):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )
    draw = mp.solutions.drawing_utils

    cap = open_camera()
    data = []
    frame_counter = 0

    print(f"\nRecording '{gesture_name}' — target: {SAMPLES_PER_GESTURE} samples")
    print("Hold your hand clearly in front of the camera.")
    print("Press 'q' to quit early.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        count = len(data)

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                # only save every Nth frame to keep samples diverse
                if frame_counter % SAMPLE_DELAY_FRAMES == 0 and count < SAMPLES_PER_GESTURE:
                    row = []
                    for lm in hand_lm.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                    data.append(row)
                    count += 1

        frame_counter += 1

        # progress bar drawn on frame
        bar_width = int((count / SAMPLES_PER_GESTURE) * (frame.shape[1] - 40))
        cv2.rectangle(frame, (20, frame.shape[0] - 40),
                      (frame.shape[1] - 20, frame.shape[0] - 20), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, frame.shape[0] - 40),
                      (20 + bar_width, frame.shape[0] - 20), (0, 200, 0), -1)

        cv2.putText(frame, f"Recording: {gesture_name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {count}/{SAMPLES_PER_GESTURE}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Gesture Recorder", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped early.")
            break

        if count >= SAMPLES_PER_GESTURE:
            # show a brief "done" screen before closing
            done_frame = frame.copy()
            cv2.putText(done_frame, "Done! Saving...",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Gesture Recorder", done_frame)
            cv2.waitKey(1000)
            break

    cap.release()
    cv2.destroyAllWindows()

    if data:
        np.save(f"{gesture_name}.npy", np.array(data))
        print(f"\nSaved {len(data)} samples → {gesture_name}.npy")
    else:
        print("No samples recorded. Make sure your hand is visible.")

if __name__ == "__main__":
    gesture_name = get_gesture_name()
    record(gesture_name)
