# Angles.py returns angles between consecutive fingers in 3d space.
# Functions include cosine angle between fingers with an intermediary joint
# And theta/phi angle return functions for angles relative to axes
import numpy as np

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

