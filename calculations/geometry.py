import numpy as np
from data.handpose import HandPose

def vector_between(c1, c2):
    """
    Returns a NumPy vector from Coordinate c1 to c2.

    :param c1: Coordinate object.
    :param c2: Coordinate object.
    :return: A NumPy array representing the vector from c1 to c2.
    """
    return np.array([c2.x - c1.x, c2.y - c1.y, c2.z - c1.z])


def get_finger_length(finger_name: str, pose) -> float:
    """
    Computes the total length of a finger by summing Euclidean distances between joints.

    :param finger_name: Name of the finger ('thumb', 'index', etc.).
    :return: float — total 3D length of the finger in the given pose.
    """
    from data.constants import FINGER_MAPPING
    indices = FINGER_MAPPING[finger_name]
    coords = [pose[i] for i in indices]

    # Sum distances between adjacent joints along the finger
    length = 0.0
    for i in range(len(coords) - 1):
        v = vector_between(coords[i], coords[i+1])
        length += np.linalg.norm(v)
    return length

def get_finger_segment_lengths(finger_name: str, pose) -> list[float]:
    """
    Computes the individual segment lengths of a finger (proximal, intermediate, distal).

    :param finger_name: Name of the finger.
    :return: List of three floats representing segment lengths.
    """
    from data.constants import FINGER_MAPPING
    indices = FINGER_MAPPING[finger_name]
    coords = [pose[i] for i in indices]

    # Return lengths between successive joints (3 segments per finger)
    return [np.linalg.norm(vector_between(coords[i], coords[i+1])) for i in range(3)]

def get_finger_curvature(finger_name: str, pose) -> float:
    """
    Estimates the average angular curvature of a finger.

    :param finger_name: Name of the finger.
    :return: Float — average angle (in radians) between finger segments. Lower is straighter.
    """
    from data.constants import FINGER_MAPPING
    indices = FINGER_MAPPING[finger_name]
    a, b, c, d = [pose[i] for i in indices]

    # Get vectors between adjacent joints
    v1 = vector_between(a, b)
    v2 = vector_between(b, c)
    v3 = vector_between(c, d)

    def angle_between(v1, v2):
        # Classic cosine angle formula
        dot = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_theta = np.clip(dot / (norms + 1e-6), -1.0, 1.0)
        return np.arccos(cos_theta)

    # Average the two segment angles
    return (angle_between(v1, v2) + angle_between(v2, v3)) / 2.0

def get_total_hand_span(pose) -> float:
    """
    Measures total hand span between thumb tip and pinky tip.

    :return: Float distance between landmarks 4 and 20.
    """
    thumb_tip = pose[4]
    pinky_tip = pose[20]

    # Simple Euclidean distance
    return np.linalg.norm(vector_between(thumb_tip, pinky_tip))

def get_finger_spread(pose) -> dict[str, float]:
    """
    Measures the angular spread between adjacent fingers at their MCP joints.

    :return: Dict mapping each finger pair (e.g., "INDEX-MIDDLE") to angle in radians.
    """
    base_indices = [5, 9, 13, 17]  # MCPs for index → pinky
    names = ["INDEX", "MIDDLE", "RING", "PINKY"]
    spread = {}

    for i in range(len(base_indices) - 1):
        a = pose[base_indices[i]]
        b = pose[0]  # Wrist
        c = pose[base_indices[i + 1]]

        # Vectors from wrist to adjacent MCPs
        v1 = vector_between(b, a)
        v2 = vector_between(b, c)

        # Angle between MCP direction vectors
        dot = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        angle = np.arccos(np.clip(dot / (norms + 1e-6), -1.0, 1.0))

        spread[f"{names[i]}-{names[i+1]}"] = angle
    return spread
