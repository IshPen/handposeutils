import cv2
import mediapipe as mp
import sys, os
from datetime import datetime
import json
from data.data_reader import DataReader
from data.handpose_sequence import HandPoseSequence
from visualization.visualizer import HandPoseVisualizer
import threading

reader = DataReader()
viz = HandPoseVisualizer()

sequence = HandPoseSequence()
recording_mode = False

mp_hands = mp.solutions.hands
cam = cv2.VideoCapture(0)

hand_pose = None


# Set output directory
SAVE_DIR = "saved_poses"
os.makedirs(SAVE_DIR, exist_ok=True)

should_exit = False

def pose_control_thread(get_current_pose_fn):
    global should_exit, recording_mode
    print("[🖐️] Type 'r' to start recording, 'e' to end and export, or 'q' to quit:")
    while True:
        user_input = sys.stdin.readline().strip().lower()
        if user_input == "r":
            if not recording_mode:
                sequence.start_recording(get_current_pose_fn, fps=30)
                recording_mode = True
            else:
                print("[!] Already recording.")
        elif user_input == "e":
            if recording_mode:
                sequence.stop_recording()
                filename = f"sequence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                path = os.path.join(SAVE_DIR, filename)
                json_data = DataReader.convert_HandPoseSequence_to_json(sequence, fps=30)

                with open(path, "w") as f:
                    json.dump(json_data, f, indent=2)
                print(f"[✓] Saved HandPoseSequence to {path}")
                recording_mode = False

                DataReader.save_frames_to_folder(
                    sequence,
                    folder_name="saved_poses/split_rock_frames",
                    file_prefix="frame",
                    handpose_prefix_name="rock"
                )

            else:
                print("[!] Not currently recording.")
        elif user_input == "q":
            print("[✓] Quitting...")
            should_exit = True
            break

# Global reference to latest hand pose
current_pose = None

def get_latest_pose():
    return current_pose

threading.Thread(target=pose_control_thread, args=(get_latest_pose,), daemon=True).start()


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
