import json

from data.data_reader import DataReader
from visualization import HandPoseVisualizer

with open('poses/rock_on.json') as f:
    pose = DataReader.convert_json_to_HandPose(json_data=json.load(f))

# Visualize
visualizer = HandPoseVisualizer()
visualizer.set_hand_poses([pose])
visualizer.show_pose()

print("Press 'q' to normalize pose...")

pose.normalize()
visualizer.set_hand_poses([pose])
visualizer.show_pose()

input("Press 'q' to close window...")

visualizer.close()