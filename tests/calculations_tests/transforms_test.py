import json

from data.data_reader import DataReader
from visualization import HandPoseVisualizer

with open('poses/rock_on.json') as f:
    pose = DataReader.convert_json_to_HandPose(json_data=json.load(f))

# Visualize
vis1 = HandPoseVisualizer()
vis2 = HandPoseVisualizer()

vis1.set_hand_poses([pose])
vis1.show_pose()

pose.normalize()
pose.straighten_finger("pinky")
pose.straighten_finger("thumb")
pose.straighten_finger("ring")
pose.straighten_finger("middle")
pose.straighten_finger("index")
pose.mirror("z")
# pose.rotate(degrees=-90, axis="z")

vis2.set_hand_poses([pose])
vis2.show_pose()

vis1.close()
vis2.close()