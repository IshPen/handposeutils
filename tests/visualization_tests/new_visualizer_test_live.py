import cv2
import mediapipe as mp
from handposeutils.data.data_reader import DataReader
from handposeutils.visualization.visualized_pose import VisualizedPose
from handposeutils.visualization.visualizer import HandPoseVisualizer

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
            hand_poses = []
            for landmark, hand_label in zip(results.multi_hand_landmarks, results.multi_handedness):
                side = hand_label.classification[0].label.lower()
                new_pose = DataReader.convert_mediapipe_to_HandPose(landmark, handedness=side)
                vis_pose = VisualizedPose(new_pose.get_all_coordinates(), new_pose.get_handedness())
                vis_pose.straighten_finger("middle")
                vis_pose.annotate(side.upper())
                vis_pose.highlight("pinky")
                hand_poses.append(vis_pose)

                print("HandPose: ", vis_pose)
                print("Hand Side: ", vis_pose.get_handedness())

            viz.set_hand_poses(hand_poses)
            viz.update_pose()


        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

cam.release()
viz.close()
