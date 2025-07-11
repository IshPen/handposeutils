import numpy as np
import ezc3d
from bvh import Bvh

class DataReader:
    def __init__(self):
        pass

    def read(self, filepath):
        if filepath.endswith('.c3d'):
            try:
                return self._read_c3d_binary(filepath)
            except Exception as e:
                print(f"[Info] Binary C3D read failed: {e}. Trying text fallback...")
                return self._read_c3d_text(filepath)

        elif filepath.endswith('.bvh'):
            return self._read_bvh(filepath)

        else:
            raise ValueError(f"Unsupported file type: {filepath}")

    def _is_binary_c3d(self, filepath):
        with open(filepath, 'rb') as f:
            first_byte = f.read(1)
            return first_byte == b'\x50'  # C3D magic = 0x50
        # highkey broken though :sad: need to find a better determinant for true c3d vs text c3d

    def _read_c3d_binary(self, filepath):
        """
        Reads a binary .c3d file via ezc3d and returns a (21, 3) array of first‐frame points.
        """
        c3d = ezc3d.c3d(filepath)
        points = c3d['data']['points']  # shape: (4, n_markers, n_frames)
        n_markers = points.shape[1]
        if points.shape[2] == 0:
            raise ValueError("No frames found in binary C3D.")
        if n_markers < self.expected_points:
            raise ValueError(f"Expected ≥ {self.expected_points} markers, found {n_markers}.")

        # Take first frame, first 21 markers, X/Y/Z only
        frame0 = points[:3, :self.expected_points, 0].T  # shape (21, 3)
        return frame0

    def _read_c3d_text(self, filepath):
        """
        Reads a text‐based C3D (still .c3d extension) with lines:
          POINTS 21
          FRAME_RATE 30
          LABELS J0 J1 ... J20
          J0 x y z conf
          ...
        Returns a (21, 3) array.
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Skip header lines: those starting with POINTS, FRAME_RATE, LABELS, or '#'
        data_lines = [
            line for line in lines
            if line.strip() and
               not line.startswith(('POINTS', 'FRAME_RATE', 'LABELS', '#'))
        ]

        if len(data_lines) < self.expected_points:
            raise ValueError(f"Expected ≥ {self.expected_points} data lines, found {len(data_lines)}.")

        landmarks = []
        for line in data_lines[:self.expected_points]:
            parts = line.strip().split()
            if len(parts) < 5:
                raise ValueError(f"Malformed line in text C3D: {line}")
            x, y, z = map(float, parts[1:4])
            landmarks.append([x, y, z])

        return np.array(landmarks)

    def _read_bvh(self, filepath):
        """
        Reads the first frame of a BVH file, returning a (21, 3) array of joint positions.
        """
        with open(filepath) as f:
            mocap = Bvh(f.read())

        # Collect joint names in hierarchy order
        joints = mocap.get_joints()
        if len(joints) < self.expected_points:
            raise ValueError(f"Expected ≥ {self.expected_points} joints, found {len(joints)}.")

        # Grab first frame's flat list of motion values
        frames = mocap.frames
        if not frames:
            raise ValueError("No motion frames in BVH.")
        first_frame_vals = list(map(float, frames[0]))

        # Iterate through joints, extracting X/Y/Z if present
        joint_positions = []
        idx = 0
        for joint in joints:
            channels = mocap.joint_channels(joint.name)
            # Check for position channels on root; for children, usually only rotations
            if 'Xposition' in channels and 'Yposition' in channels and 'Zposition' in channels:
                x = first_frame_vals[idx + channels.index('Xposition')]
                y = first_frame_vals[idx + channels.index('Yposition')]
                z = first_frame_vals[idx + channels.index('Zposition')]
                joint_positions.append([x, y, z])
            else:
                # No explicit position: you may compute via forward kinematics, but for now, use zeros
                joint_positions.append([0.0, 0.0, 0.0])
            idx += len(channels)
            if len(joint_positions) >= self.expected_points:
                break

        if len(joint_positions) < self.expected_points:
            raise ValueError(f"Could not extract {self.expected_points} joint positions from BVH.")

        return np.array(joint_positions)
