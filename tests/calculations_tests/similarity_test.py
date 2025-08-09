from handposeutils.data.data_reader import DataReader
from handposeutils.calculations.similarity import pose_similarity
import json

with open('poses/rock_on.json') as f:
    pose1 = DataReader.convert_json_to_HandPose(json_data=json.load(f))
    pose1.normalize()

with open('poses/rock_on.json') as f:
    pose2 = DataReader.convert_json_to_HandPose(json_data=json.load(f))
    pose2.normalize()
    #pose2.straighten_finger("pinky")
    #pose2.straighten_finger("index")
    #pose2.straighten_finger("thumb")
    #pose2.straighten_finger("middle")
    #pose2.straighten_finger("ring")

print(pose_similarity(pose1, pose2, "procrustes"))
