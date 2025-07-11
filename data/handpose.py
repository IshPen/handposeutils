from typing import List, Literal, Dict
from coordinate import Coordinate


# List of Mediapipe point names in order (index == Mediapipe index)
POINTS_NAMES_LIST = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

FINGER_MAPPING = {
    "THUMB": range(1, 5),
    "INDEX": range(5, 9),
    "MIDDLE": range(9, 13),
    "RING": range(13, 17),
    "PINKY": range(17, 21)
}

class HandPose:
    def __init__(self, coordinates: List[Coordinate], side: Literal["left_hand", "right_hand"]):
        if len(coordinates) != 21:
            raise ValueError("Expected 21 coordinates for hand landmarks (MediaPipe format).")

        self.side = side
        self.points: Dict[int, Dict] = {}

        for i, coord in enumerate(coordinates):
            finger = next((name for name, idxs in FINGER_MAPPING.items() if i in idxs), "PALM")
            self.points[i] = {
                "coordinate": coord,
                "side": side,
                "common_name": POINTS_NAMES_LIST[i],
                "finger": finger
            }

    def get_coordinate_by_index(self, index: int) -> Coordinate:
        return self.points[index]["coordinate"]

    def get_index_by_common_name(self, name: str) -> int:
        for i, data in self.points.items():
            if data["common_name"] == name:
                return i
        raise ValueError(f"No point found with common name {name}")

    def get_all_coordinates(self) -> List[Coordinate]:
        return [self.points[i]["coordinate"] for i in range(21)]

    def __str__(self):
        return f"<HandPose {self.side}, {len(self.points)} landmarks>"

    def __getitem__(self, index_or_name: int | str) -> Coordinate:
        if isinstance(index_or_name, int):
            return self.get_coordinate_by_index(index_or_name)
        elif isinstance(index_or_name, str):
            idx = self.get_index_by_common_name(index_or_name)
            return self.get_coordinate_by_index(idx)
        raise TypeError("Index must be int (0–20) or common_name string.")
