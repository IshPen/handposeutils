import cv2
import mediapipe as mp
import sys, os
from datetime import datetime
import json
from handposeutils.data.data_reader import DataReader
from handposeutils.visualization.visualizer import HandPoseVisualizer
import threading

reader = DataReader()
viz = HandPoseVisualizer()

mp_hands = mp.solutions.hands
cam = cv2.VideoCapture(0)

hand_pose = None


# Set output directory
SAVE_DIR = "saved_poses"
os.makedirs(SAVE_DIR, exist_ok=True)

should_exit = False

def pose_saver_thread(get_current_pose_fn):
    global should_exit
    print("[🖐️] Type 's' to save pose, or 'q' to quit:")
    while True:
        user_input = sys.stdin.readline().strip().lower()
        if user_input == "s":
            pose = get_current_pose_fn()
            if pose:
                filename = f"pose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                path = os.path.join(SAVE_DIR, filename)
                json_data = DataReader.export_HandPose_to_json(pose)
                with open(path, "w") as f:
                    json.dump(json_data, f, indent=2)
                print(f"[✓] Saved hand pose to {path}")
            else:
                print("[!] No hand pose detected.")
        elif user_input == "q":
            print("[✓] Quitting...")
            should_exit = True
            break

# Global reference to latest hand pose
current_pose = None

def get_latest_pose():
    return current_pose

threading.Thread(target=pose_saver_thread, args=(get_latest_pose,), daemon=True).start()


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
                hand_pose = DataReader.convert_mediapipe_to_HandPose(landmark, handedness=side)

                viz.set_hand_poses([hand_pose])

                current_pose = hand_pose # For the 'save function'

            viz.update_pose(finger_tips_shown=True, ligaments_shown=True, palm_shown=True)

        if should_exit:
            break

cam.release()
viz.close()
