# transforms.py

import math
from typing import Literal
from ..data.handpose import HandPose

def normalize_handpose_positioning(pose: HandPose) -> HandPose:
    coords = pose.get_all_coordinates()
    center_x = sum(c.x for c in coords) / len(coords)
    center_y = sum(c.y for c in coords) / len(coords)
    center_z = sum(c.z for c in coords) / len(coords)

    for coord in coords:
        coord.x -= center_x
        coord.y -= center_y
        coord.z -= center_z
    return pose

def normalize_handpose_scaling(pose: HandPose) -> HandPose:
    coords = pose.get_all_coordinates()
    min_x = min(c.x for c in coords)
    max_x = max(c.x for c in coords)
    min_y = min(c.y for c in coords)
    max_y = max(c.y for c in coords)
    min_z = min(c.z for c in coords)
    max_z = max(c.z for c in coords)

    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z
    max_range = max(range_x, range_y, range_z)

    if max_range == 0:
        return pose

    for coord in coords:
        coord.x = (coord.x - min_x) / max_range * 2 - 1
        coord.y = (coord.y - min_y) / max_range * 2 - 1
        coord.z = (coord.z - min_z) / max_range * 2 - 1
    return pose

def normalize_handpose(pose: HandPose) -> HandPose:
    pose = normalize_handpose_positioning(pose)
    pose = normalize_handpose_scaling(pose)
    return pose

def mirror_pose(pose: HandPose, axis: Literal['x', 'y', 'z'] = 'x') -> HandPose:
    coords = pose.get_all_coordinates()
    for coord in coords:
        if axis == 'x':
            coord.x = -coord.x
        elif axis == 'y':
            coord.y = -coord.y
        elif axis == 'z':
            coord.z = -coord.z
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
    return pose

def rotate_pose_by_axis(pose: HandPose, degrees: float, axis: Literal['x', 'y', 'z']) -> HandPose:
    radians = math.radians(degrees)
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)
    coords = pose.get_all_coordinates()

    for c in coords:
        x, y, z = c.x, c.y, c.z
        if axis == 'x':
            c.y = y * cos_a - z * sin_a
            c.z = y * sin_a + z * cos_a
        elif axis == 'y':
            c.x = x * cos_a + z * sin_a
            c.z = -x * sin_a + z * cos_a
        elif axis == 'z':
            c.x = x * cos_a - y * sin_a
            c.y = x * sin_a + y * cos_a
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
    return pose

def straighten_finger(pose: HandPose, finger: str) -> HandPose:
    from ..data.constants import FINGER_MAPPING
    indices = FINGER_MAPPING.get(finger)
    if not indices:
        raise ValueError(f"Invalid finger: {finger}")
    base = pose.get_coordinate_by_index(indices[0])
    next_point = pose.get_coordinate_by_index(indices[1])
    direction = next_point - base
    direction.normalize()

    for i in range(1, len(indices)):
        pose.points[indices[i]]["coordinate"] = base + direction.scale(i * 0.05)
    return pose


# Angles.py returns angles between consecutive fingers in 3d space.
# Functions include cosine angle between fingers with an intermediary joint
# And theta/phi angle return functions for angles relative to axes
import numpy as np

def get_angle(end_point1, end_point2, common_joint):
    """
    :param end_point1: the coordinates of the first finger
    :param end_point2: the coordinates of the second finger
    :param common_joint: the coordinates of the common joint
    :return: the angle between the two fingers
    """
    vector1 = end_point1 - common_joint
    vector2 = end_point2 - common_joint
    angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return angle

def get_theta_angle(end_point1, common_joint): # theta angle between finger and x-axis
    """
    :param end_point1: the coordinates of the first finger
    :return: the theta angle between the finger and the x-axis
    """
    vector1 = end_point1 - common_joint
    vector2 = np.array([1, 0, 0])
    angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return angle

def get_phi_angle(end_point1, common_joint): # phi angle between finger and y-axis
    """
    :param end_point1: the coordinates of the first finger
    :param end_point2: the coordinates of the second finger
    :param common_joint: the coordinates of the common joint
    :return: the phi angle between the finger and the y-axis
    """
    vector1 = end_point1 - common_joint
    vector2 = np.array([0, 1, 0])
    angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return angle

def get_theta_subangle_between_fingers(end_point1, end_point2, common_joint): # theta angle between two fingers
    """
    :param end_point1: the coordinates of the first finger
    :param end_point2: the coordinates of the second finger
    :param common_joint: the coordinates of the common joint
    :return: the absolute theta angle between the two fingers (only the axis-based subangle)
    """
    theta_angle1 = get_theta_angle(end_point1, common_joint)
    theta_angle2 = get_theta_angle(end_point2, common_joint)
    return abs(theta_angle1 - theta_angle2)

## Monkey patching

HandPose.normalize_positioning = lambda self: normalize_handpose_positioning(self)
HandPose.normalize_scaling = lambda self: normalize_handpose_scaling(self)
HandPose.normalize = lambda self: normalize_handpose(self)
HandPose.mirror = lambda self, axis='x': mirror_pose(self, axis)
HandPose.rotate = lambda self, degrees=0, axis='z': rotate_pose_by_axis(self, degrees, axis)
HandPose.straighten_finger = lambda self, finger='index': straighten_finger(self, finger)


def get_phi_subangle_between_fingers(end_point1, end_point2, common_joint): # phi angle between two fingers
    """
    :param end_point1: the coordinates of the first finger
    :param end_point2: the coordinates of the second finger
    :param common_joint: the coordinates of the common joint
    :return: the absolute phi angle between the two fingers (only the axis-based subangle)
    """
    phi_angle1 = get_phi_angle(end_point1, common_joint)
    phi_angle2 = get_phi_angle(end_point2, common_joint)
    return abs(phi_angle1 - phi_angle2)

