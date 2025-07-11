import numpy as np
import ezc3d

# Create a mock 23-joint hand pose
c3d = ezc3d.c3d()
c3d['header']['points']['frame_rate'] = 30

# Shape: (4, nbPoints, nbFrames) => (4, 23, 1)
points = np.zeros((4, 23, 1))
for i in range(23):
    points[0, i, 0] = (i % 5) * 10        # x - spread fingers
    points[1, i, 0] = (i // 5) * 20       # y - stacked joints
    points[2, i, 0] = 0                   # z - flat for now
    points[3, i, 0] = 0                   # residual

c3d['data']['points'] = points

# Add 23 joint labels
c3d['parameters']['POINT']['LABELS']['value'] = [f"J{i}" for i in range(23)]

# Save the file
c3d.write("actual_hand_23_joints.c3d")
print("C3D file created: mock_hand_21_joints.c3d")
