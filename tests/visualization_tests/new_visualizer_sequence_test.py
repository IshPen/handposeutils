import json
from data.data_reader import DataReader
from data.handpose_sequence import HandPoseSequence, TimedHandPose
from visualization.visualized_pose import VisualizedPose
from visualization.visualizer import HandPoseVisualizer

def main(json_path):
    with open(json_path, 'r') as f:
        seq_data = json.load(f)

    fps = seq_data.get("fps", 30)
    sequence = []

    for entry in seq_data["sequence"]:
        base_pose = DataReader.convert_json_to_HandPose(entry["pose"])
        base_pose.normalize_position()
        vis_pose = VisualizedPose(base_pose.get_all_coordinates(), base_pose.side)
        vis_pose.annotate(f"{entry['start_time']:.2f}s")
        sequence.append(TimedHandPose(vis_pose, entry["start_time"], entry["end_time"]))

    hand_sequence = HandPoseSequence(sequence)

    visualizer = HandPoseVisualizer()
    loop = False
    visualizer.play_sequence(hand_sequence, fps=fps, loop=loop)

    input("Press Enter to close window...")
    visualizer.close()

if __name__ == "__main__":
    main("../data_tests/saved_poses/rock_on_recording.json")
    # main("../embeddings_tests/poses/rock_sequence.json")
