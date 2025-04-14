import numpy as np
import cv2
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from visualization.visualizer import HandPoseVisualizer
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Random pose for demo
pose = np.random.rand(21, 3) * 0.2

viz = HandPoseVisualizer()
COLORS = {
    "landmarks": [1, 0, 0],
    "proximals": [0.5,0,1],
    "intermediates": [0,1,0.5],
    "distals": [0,0.5,1],
    "palm": [1,1,0],
}

# Initialize camera
cam = cv2.VideoCapture(0)

# Update the landmark points with our random pose
viz.set_landmark_points(pose)
viz.set_colors(COLORS)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cam.isOpened():
        success, frame = cam.read()
        # apply hand tracking
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        frame.flags.writeable = True

        # Detections
        results = hands.process(frame)
        # RGB 2 BGR
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            viz.read_multi_landmarks(results.multi_hand_landmarks)
            viz.show_pose(finger_tips_shown=True, ligaments_shown=True, palm_shown=True)
            print(viz.return_landmark_points())
            
        # cv2.imshow("K", frame)
        # if cv2.waitKey(5) & 0xFF == ord("q"):
        #     break

cam.release()
cv2.destroyAllWindows()
viz.close()