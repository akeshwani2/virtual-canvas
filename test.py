# 1. We are going to capture the camera feed using openCV
# 2. Use mediapipe to detect the hand and track the position of the pointer finger
# 3. Draw on a canvas where the index finger is moving

import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Set up a blank canvas for drawing
canvas = None

# Initialize the previous position
prev_x_tip, prev_y_tip = None, None

def is_hand_open(hand_landmarks):
    # Check if the thumb finger tip is more than 4 cm away from the wrist
    # hand_landmarks.landmark is a list of landmark points, each with an x, y, z coordinate
    # mp_hands.HandLandmark.THUMB_TIP is a constant that represents the thumb tip landmark
    # .y is the y coordinate of the thumb tip
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y

    # Check if all finger tips are less than 4 cm away from the wrist
    # tip.y < wrist means the finger tip is less than 4 cm away from the wrist
    # all() is a function that returns True if all the items in an iterable are true
    # tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip] is an iterable
    # tip.y < wrist for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip] is a generator expression
    # that returns True if the finger tip is less than 4 cm away from the wrist
    return all(tip < wrist for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip])

# Initializing the hand detection model
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if canvas is None:
            canvas = np.zeros_like(frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x_tip = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                y_tip = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

                # Draw a line from the previous position to the current position
                if prev_x_tip is not None and prev_y_tip is not None:
                    cv2.line(canvas, (prev_x_tip, prev_y_tip), (x_tip, y_tip), (0, 255, 0), 2)

                # Update the previous position
                prev_x_tip, prev_y_tip = x_tip, y_tip

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            # Reset the previous position when no hand is detected
            prev_x_tip, prev_y_tip = None, None

        # Move the display update outside of the else block
        combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
        cv2.imshow("Drawing", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()