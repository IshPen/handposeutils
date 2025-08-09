import json
from handposeutils.data.data_reader import DataReader
from handposeutils.visualization.visualized_pose import VisualizedPose
from handposeutils.visualization.visualizer import HandPoseVisualizer

def main(json_path):
    # Load saved HandPose JSON
    with open(json_path, 'r') as f:
        pose_data = json.load(f)

    # Convert to HandPose and then to VisualizedPose
    base_pose = DataReader.convert_json_to_HandPose(pose_data)
    vis_pose = VisualizedPose(base_pose.get_all_coordinates(), base_pose.side)
    vis_pose.annotate("Loaded from JSON")
    vis_pose.setColorScheme(
        fingers=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        palm=(1.0, 1.0, 0.0),
        landmarks=(0.2, 0.8, 0.8)
    )
    vis_pose.setOpacity(1.0)
    vis_pose.highlight("ring", (1.0, 1.0, 0.0))  # Optional highlight

    # Visualize
    visualizer = HandPoseVisualizer()
    visualizer.set_hand_poses([vis_pose])
    visualizer.show_pose()

    input("Press Enter to close window...")
    visualizer.close()

if __name__ == "__main__":
    main("../calculations_tests/poses/rock_on.json")
