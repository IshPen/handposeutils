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


def get_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def get_planar_distance(point1, point2, plane):
    '''
    :param point1: the coordinates of the first point
    :param point2: the coordinates of the second point
    :param plane (str): the plane to get distance in
        options: "xy", "yz", "xz"
    :return: the distance between the two points projected onto the plane
    '''
    if plane == "xy":
        return np.linalg.norm(point1[:2] - point2[:2]) # distance in the xy plane
    elif plane == "yz":
        return np.linalg.norm(point1[1:] - point2[1:]) # distance in the yz plane
    elif plane == "xz":
        return np.linalg.norm(point1[2:] - point2[2:]) # distance in the xz plane
    else:
        raise ValueError("Invalid plane")

def get_linear_distance(point1, point2, axis):
    '''
    :param point1: the coordinates of the first point
    :param point2: the coordinates of the second point
    :param axis (str): the axis to get distance in
        options: "x", "y", "z"
    :return: the distance between the two points projected onto the axis
    '''
    if axis == "x":
        return np.linalg.norm(point1[0] - point2[0]) # distance in the x axis
    elif axis == "y":
        return np.linalg.norm(point1[1] - point2[1]) # distance in the y axis
    elif axis == "z":
        return np.linalg.norm(point1[2] - point2[2]) # distance in the z axis
    else:
        raise ValueError("Invalid axis")
