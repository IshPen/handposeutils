import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.data_reader import DataReader
from visualization.visualizer import HandPoseVisualizer

reader = DataReader()
landmarks = reader.read('mock_hand_21_joints.c3d')  # or 'path_to_file.bvh'
print(landmarks.shape)  # Should output: (23, 3)
print(landmarks)        # Outputs the 2D array of landmarks

viz = HandPoseVisualizer()

viz.set_landmark_points(landmarks)
viz.show_pose(finger_tips_shown=True, ligaments_shown=True, palm_shown=True) # don't put this in the while loop, it'll block camera movement in the o3d window

while True:
    viz.vis.poll_events()
    viz.vis.update_renderer()
