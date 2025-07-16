# embeddings.py
# encoder for geometric, latent, and graph-based embeddings

import numpy as np
from math import acos
from typing import List
from data.handpose import HandPose
from data.coordinate import Coordinate


def get_joint_angle_vector(pose: HandPose) -> np.ndarray:
    """
    Generate a 15D joint-angle embedding vector for a HandPose.
    Each finger contributes 3 angles: two intra-finger and one base-to-knuckle angle.

    :param pose: HandPose
    :return: np.ndarray of shape (15,) containing angles in radians.
    """

    def compute_angle(a: Coordinate, b: Coordinate, c: Coordinate) -> float:
        """Compute angle at point b formed by (a - b - c) using cosine rule."""
        v1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        cos_angle = np.clip(dot / norm, -1.0, 1.0)
        return acos(cos_angle)

    # Define angle triplets (a, b, c)
    triplets: List[tuple[int, int, int]] = [
        # Thumb
        (1, 2, 3), (2, 3, 4), (0, 1, 2),
        # Index
        (5, 6, 7), (6, 7, 8), (0, 5, 6),
        # Middle
        (9, 10, 11), (10, 11, 12), (0, 9, 10),
        # Ring
        (13, 14, 15), (14, 15, 16), (0, 13, 14),
        # Pinky
        (17, 18, 19), (18, 19, 20), (0, 17, 18),
    ]

    angles = []
    for a_idx, b_idx, c_idx in triplets:
        a, b, c = pose[a_idx], pose[b_idx], pose[c_idx]
        angle = compute_angle(a, b, c)
        angles.append(angle)

    return np.array(angles)


def get_bone_length_vector(pose: HandPose) -> np.ndarray:
    """
    Compute a 20D bone length vector from a HandPose, representing each bone segment.

    :param pose: HandPose
    :return: np.ndarray of shape (20,) containing bone lengths.
    """
    bone_pairs: List[tuple[int, int]] = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    lengths = []
    for i, j in bone_pairs:
        coord_i = pose[i]
        coord_j = pose[j]
        dist = np.linalg.norm(np.array(coord_j.as_tuple()) - np.array(coord_i.as_tuple()))
        lengths.append(dist)

    return np.array(lengths)


def get_relative_vector_embedding(pose: HandPose) -> np.ndarray:
    """
    Compute a 63D vector of relative positions of each landmark from the wrist.

    :param pose: HandPose
    :return: np.ndarray of shape (63,) representing relative landmark positions.
    """
    wrist = pose[0]
    relative_coords = []

    for i in range(21):
        pt = pose[i]
        vec = np.array([pt.x - wrist.x, pt.y - wrist.y, pt.z - wrist.z])
        relative_coords.extend(vec)

    return np.array(relative_coords)
