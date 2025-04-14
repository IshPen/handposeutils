import numpy as np

def get_cosine_similarity(pose1, pose2):
    '''
    :param pose1: the first pose
    :param pose2: the second pose
    :return: the cosine similarity between the two poses
    '''
    return np.dot(pose1, pose2) / (np.linalg.norm(pose1) * np.linalg.norm(pose2))

def get_euclidean_similarity(pose1, pose2):
    '''
    :param pose1: the first pose
    :param pose2: the second pose
    :return: the euclidean similarity between the two poses
    '''
    return np.linalg.norm(pose1 - pose2)

def get_manhattan_similarity(pose1, pose2):
    '''
    :param pose1: the first pose
    :param pose2: the second pose
    :return: the manhattan similarity between the two poses
    '''
    return np.sum(np.abs(pose1 - pose2))

def get_chebyshev_similarity(pose1, pose2):
    '''
    :param pose1: the first pose
    :param pose2: the second pose
    :return: the chebyshev similarity between the two poses
    '''
    return np.max(np.abs(pose1 - pose2))

def get_minkowski_similarity(pose1, pose2, p=2):
    '''
    :param pose1: the first pose
    :param pose2: the second pose
    :param p: the power to use for the minkowski similarity
    :return: the minkowski similarity between the two poses
    '''
    return np.sum(np.abs(pose1 - pose2) ** p) ** (1/p)


