import numpy as np
from typing import Tuple
from data.handpose import HandPose

def procrustes_alignment(pose1: HandPose, pose2: HandPose) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform Procrustes alignment between two HandPose objects.

    This aligns pose1 to pose2 by removing translation, scale, and rotation,
    and returns the aligned poses and their similarity score.

    Parameters:
        pose1: First HandPose
        pose2: Second HandPose

    Returns:
        (aligned_pose1, aligned_pose2, distance) — aligned numpy arrays and Procrustes distance
    """

    # Step 1: Convert HandPoses to N x 3 numpy arrays
    p1 = np.array([coord.as_tuple() for coord in pose1.get_all_coordinates()])
    p2 = np.array([coord.as_tuple() for coord in pose2.get_all_coordinates()])

    if p1.shape != p2.shape:
        raise ValueError(f"Shape mismatch: pose1 has shape {p1.shape}, pose2 has shape {p2.shape}")

    # Step 2: Center both poses at the origin
    p1_centered = p1 - p1.mean(axis=0)
    p2_centered = p2 - p2.mean(axis=0)

    # Step 3: Normalize scale (Frobenius norm)
    p1_scaled = p1_centered / np.linalg.norm(p1_centered)
    p2_scaled = p2_centered / np.linalg.norm(p2_centered)

    # Step 4: Compute optimal rotation matrix using Kabsch algorithm
    H = p1_scaled.T @ p2_scaled
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix reflection issues
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Step 5: Apply rotation to pose1
    p1_aligned = p1_scaled @ R
    p2_aligned = p2_scaled

    # Step 6: Compute Procrustes distance (residual sum of squares)
    distance = np.sum((p1_aligned - p2_aligned) ** 2)

    return p1_aligned, p2_aligned, distance


def pose_similarity(pose1: HandPose, pose2: HandPose, method: str = 'procrustes') -> float:
    """
    Compute similarity between two HandPose objects.

    Supported methods:
        - 'procrustes': Procrustes distance (lower = more similar)

    Returns:
        float similarity score (lower is more similar for Procrustes)
    """
    if method == 'procrustes':
        _, _, distance = procrustes_alignment(pose1, pose2)
        return distance
    else:
        raise NotImplementedError(f"Similarity method '{method}' is not implemented.")
