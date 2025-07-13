import cv2
import mediapipe as mp
import numpy as np
from data.data_reader import DataReader
from visualization.visualized_pose import VisualizedPose
from visualization.visualizer import HandPoseVisualizer

reader = DataReader()
viz = HandPoseVisualizer()
viz.initialize_window()
mp_hands = mp.solutions.hands
cam = cv2.VideoCapture(0)

hand_pose = None

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            continue

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for landmark, hand_label in zip(results.multi_hand_landmarks, results.multi_handedness):
                side = hand_label.classification[0].label.lower()  # "left" or "right"
                new_pose = DataReader.convert_mediapipe_to_HandPose(landmark, handedness=side)

                print(side)

                hand_pose = VisualizedPose(new_pose.get_all_coordinates(), new_pose.get_handedness())
                viz.set_hand_poses([hand_pose])

            viz.update_pose(finger_tips_shown=True, ligaments_shown=True, palm_shown=True)
            print("HandPose: ", hand_pose)
            print("Hand Side: ", hand_pose.get_handedness())

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

cam.release()
viz.close()
