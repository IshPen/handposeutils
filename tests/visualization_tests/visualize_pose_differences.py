from handposeutils.data.data_reader import DataReader
from handposeutils.visualization import HandPoseVisualizer
import json

visualizer = HandPoseVisualizer()

with open('poses/rock_on.json') as f:
    pose1 = DataReader.convert_json_to_HandPose(json_data=json.load(f))
    pose1.normalize()

with open('poses/rock_on.json') as f:
    pose2 = DataReader.convert_json_to_HandPose(json_data=json.load(f))
    pose2.normalize()
    pose2.straighten_finger("middle")
    pose2.straighten_finger("ring")

#== CHANGE method to 'cosine' or 'joint_angle' to see differences==#
visualizer.visualize_pose_similarity(pose1, pose2, method='euclidean', offset=False)
