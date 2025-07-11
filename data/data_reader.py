import json
import pandas as pd
from typing import List, Dict, Any
from .handpose import HandPose
from .handpose_sequence import HandPoseSequence, TimedHandPose
from .coordinate import Coordinate
from .constants import POINTS_NAMES_LIST, FINGER_MAPPING
from mediapipe.framework.formats import landmark_pb2

## The DataReader class for
# I highkey don't think you'll ever need to convert from OpenPose to HandPoses,
# but I somehow found myself in a situation where I did.
# Thus, I implemented the OpenPose conversions.
# Functions are self explanatory.
# TODO: Create unified json structure.

class DataReader:
    # --- MediaPipe Conversion ---
    @staticmethod
    def convert_mediapipe_to_HandPose(mp_landmarks, side: str = "right_hand") -> HandPose:
        coords = [Coordinate(lm.x, lm.y, lm.z) for lm in mp_landmarks.landmark]
        return HandPose(coords, side)

    @staticmethod
    def convert_HandPose_to_mediapipe(pose: HandPose):
        from mediapipe.framework.formats import landmark_pb2
        landmarks = [
            landmark_pb2.NormalizedLandmark(x=c.x, y=c.y, z=c.z)
            for c in pose.get_all_coordinates()
        ]
        return landmark_pb2.NormalizedLandmarkList(landmark=landmarks)

    # --- OpenPose Conversion ---

    @staticmethod
    def convert_openpose_to_HandPose(openpose_data: List[float], side="right_hand") -> HandPose:
        # OpenPose flat format: [x0, y0, c0, x1, y1, c1, ...]
        coords = []
        for i in range(21):
            x = openpose_data[i * 3]
            y = openpose_data[i * 3 + 1]
            z = 0.0  # OpenPose does not provide depth
            coords.append(Coordinate(x, y, z))
        return HandPose(coords, side)

    @staticmethod
    def convert_HandPose_to_openpose(pose: HandPose) -> List[float]:
        openpose_format = []
        for coord in pose.get_all_coordinates():
            openpose_format.extend([coord.x, coord.y, 1.0])  # default confidence -> 1.0
        return openpose_format

    # --- CSV Conversion ---

    @staticmethod
    def convert_csv_to_HandPose(df: pd.DataFrame, side="right_hand") -> HandPose:
        coords = [Coordinate(row['x'], row['y'], row['z']) for _, row in df.iterrows()]
        return HandPose(coords, side)

    @staticmethod
    def export_HandPose_to_csv(pose: HandPose) -> pd.DataFrame:
        data = []
        for i in range(21):
            coord = pose[i]
            data.append({
                "name": POINTS_NAMES_LIST[i],
                "x": coord.x,
                "y": coord.y,
                "z": coord.z
            })
        return pd.DataFrame(data)