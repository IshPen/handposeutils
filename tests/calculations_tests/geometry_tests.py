from handposeutils.data import DataReader
import json

from handposeutils.visualization import HandPoseVisualizer
from handposeutils.calculations.geometry import *

def test_finger_length(pose):
    print("== Finger Lengths ==")
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        length = get_finger_length(finger, pose)
        print(f"{finger.title()} Length: {length:.3f}")

def test_finger_segment_lengths(pose):
    print("\n== Finger Segment Lengths ==")
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        segments = get_finger_segment_lengths(finger, pose)
        print(f"{finger.title()} Segments: {['%.3f' % s for s in segments]}")

def test_finger_curvature(pose):
    print("\n== Finger Curvatures ==")
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        angle = get_finger_curvature(finger, pose)
        print(f"{finger.title()} Curvature (radians): {angle:.3f}")

def test_total_hand_span(pose):
    span = get_total_hand_span(pose)
    print(f"\n== Total Hand Span ==\nThumb to Pinky Distance: {span:.3f}")

def test_finger_spread(pose):
    print("\n== Finger Spread Angles ==")
    spreads = get_finger_spread(pose)
    for pair, angle in spreads.items():
        print(f"{pair.title()} MCP Angle: {np.degrees(angle):.2f}°")

def test_hand_aspect_ratio(pose):
    aspect = get_hand_aspect_ratio(pose)
    print(f"\n== Hand Aspect Ratio ==\nWidth / Height: {aspect:.3f}")

def test_pose_flatness(pose):
    print("\n== Pose Flatness ==")
    for axis in ['x', 'y', 'z']:
        flatness = get_pose_flatness(pose, axis)
        print(f"Flatness along {axis.upper()}-axis: std = {flatness:.4f}")

def test_joint_angle(pose):
    print("\n== Joint Angles ==")
    joint_triplets = [(1, 2, 3), (2, 3, 4), (5, 6, 7), (6, 7, 8)]  # Thumb and index finger joints
    for t in joint_triplets:
        angle = get_joint_angle(t, pose)
        print(f"Angle at {t[1]} (via {t[0]}-{t[2]}): {np.degrees(angle):.2f}°")

def test_palm_normal_vector(pose):
    normal = get_palm_normal_vector(pose)
    print("\n== Palm Normal Vector ==")
    print(f"Vector: {normal.round(3)}")

def test_cross_finger_angles(pose):
    print("\n== Cross-Finger Angles ==")
    angles = get_cross_finger_angles(pose)
    for pair, angle in angles.items():
        print(f"{pair.title()}: {np.degrees(angle):.2f}°")

def main(pose):
    test_finger_length(pose)
    test_finger_segment_lengths(pose)
    test_finger_curvature(pose)
    test_total_hand_span(pose)
    test_finger_spread(pose)
    test_hand_aspect_ratio(pose)
    test_pose_flatness(pose)
    test_joint_angle(pose)
    test_palm_normal_vector(pose)
    test_cross_finger_angles(pose)

if __name__ == "__main__":
    with open('poses/rock_on.json') as f:
        pose1 = DataReader.convert_json_to_HandPose(json_data=json.load(f))
        pose1.normalize()
    main(pose1)
    vis = HandPoseVisualizer()
    vis.set_hand_poses([pose1])
    vis.show_pose()