import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull

class HandPoseVisualizer:
    def __init__(self, window_name="Hand Pose Visualizer", color_profile: dict = None):
        self.window_name = window_name
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=self.window_name)
        self.geometry_added = False

        self.landmark_points = []
        self.geometry = []

        # Landmark connections for each finger
        self.FINGERS = {
            "thumb": [1, 2, 3, 4],
            "index": [5, 6, 7, 8],
            "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16],
            "pinky": [17, 18, 19, 20]
        }

        self.COLORS_DEFAULT = {
            "landmarks": [0.1, 0.6, 0.9],
            "proximals": [0.5,0,1],
            "intermediates": [0,1,0.5],
            "distals": [0,0.5,1],
            "palm": [1,1,0],
        }
        self.colors = None
        if color_profile is None:
            self.set_colors(self.COLORS_DEFAULT)
        else:
            self.set_colors(color_profile)

    def __create_sphere(self, center, radius=5.0, color=None):
        if color is None:
            color = self.colors.get("landmarks")
        else:
            color = color
        """
        :param center: center of sphere
        :param radius: radius of sphere
        :param color: color of sphere (rgb scaled to [[0.0-1.0],_,_])
        :return: o3d sphere object
        """
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
        sphere.translate(center)
        sphere.paint_uniform_color(color)
        return sphere


    def __create_cylinder_between(self, p1, p2, radius=0.8, resolution=20, color=[1, 0, 0]):
        """
        :param p1: point 1 (start point of cylinder)
        :param p2: point 2 (end point of cylinder)
        :param radius: radius of cylinder
        :param resolution: resolution of cylinder
        :param color: color of cylinder (rgb scaled [[0.0-1.0],_,_])
        :return: o3d cylinder object
        """
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        axis = p2 - p1
        length = np.linalg.norm(axis)
        if length == 0:
            return None
        axis /= length

        # Step 1: Create the default cylinder along z-axis centered at origin
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=resolution)
        cylinder.paint_uniform_color(color)

        # Step 2: Align Z-axis with the target axis (rotate cylinder)
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, axis)
        c = np.dot(z_axis, axis)
        if np.linalg.norm(v) < 1e-6:
            R = np.eye(3) if c > 0 else -np.eye(3)  # 180° flip if facing backward
        else:
            skew = np.array([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]])
            R = np.eye(3) + skew + skew @ skew * ((1 - c) / (np.linalg.norm(v) ** 2))

        cylinder.rotate(R, center=(0, 0, 0))

        # Step 3: Translate to the midpoint between p1 and p2
        midpoint = (p1 + p2) / 2
        cylinder.translate(midpoint)

        return cylinder

    def add_geometry(self, geometry):
        """
        :param geometry: single open3d shape to add to visulizer window
        """
        self.geometry.append(geometry)

    def return_geometry(self):
        """
        :return: visualizer geometry
        """
        return self.geometry

    def set_landmark_points(self, points_array):
        """
        :param points_array: array containing landmark points to update the screen with
        """
        self.landmark_points = points_array

    def return_landmark_points(self):
        return np.array(self.landmark_points)

    def get_landmark_point(self, index):
        """
        :param index: index of the landmark point to get
        NOTE: will probably not work as expected if more than one hand is stored in the landmark_points list
        :return: landmark point at index
        """
        try:
            return self.landmark_points[index]
        except Exception as e:
            raise e

    def update_visualizer(self):
        # Typically always called after show_pose()
        # Add all geometries to the visualizer window
        self.vis.clear_geometries()
        self.vis.update_renderer()

        for geo in self.geometry:
            self.vis.add_geometry(geo)

        self.vis.poll_events()
        self.vis.update_renderer()


    def show_pose(self, finger_tips_shown = True, ligaments_shown = True, palm_shown = True):
        """
        Shows the pose in the visualizer window
        :param finger_tips_shown: boolean to show/hide finger tips
        :param ligaments_shown: boolean to show/hide ligaments
        :param palm_shown: boolean to show/hide palm
        :return: None
        """

        # Clear previous frame geometries
        self.vis.clear_geometries()
        self.geometry = []

        # Update the geometries list
        if finger_tips_shown:
            for joint in self.landmark_points:
                s = self.__create_sphere(np.array(joint), 1.0)
                self.add_geometry(s)

        if ligaments_shown:
            # Check if we have multiple hands (more than 21 landmarks)
            num_hands = len(self.landmark_points) // 21

            # NOTE: this probably isn't the best implementation of this, but it works for now lol
            temp_finger_colors_array = []
            temp_finger_colors_array.extend([self.colors.get("proximals")])
            temp_finger_colors_array.extend([self.colors.get("intermediates")])
            temp_finger_colors_array.extend([self.colors.get("distals")])

            for hand_idx in range(num_hands):
                # Calculate offset for this hand
                offset = hand_idx * 21

                # Create adjusted finger indices for this hand
                adjusted_fingers = {}
                for finger_name, indices in self.FINGERS.items():
                    adjusted_fingers[finger_name] = [idx + offset for idx in indices]

                # Draw ligaments for this hand
                for finger_indices in adjusted_fingers.values():
                    for i in range(len(finger_indices) - 1):
                        p1 = self.landmark_points[finger_indices[i]]
                        p2 = self.landmark_points[finger_indices[i+1]]
                        c = self.__create_cylinder_between(p1, p2, radius=0.8, color=temp_finger_colors_array[i])
                        if c: self.add_geometry(c)

        if palm_shown:
            # Handle multiple hands for palm visualization
            num_hands = len(self.landmark_points) // 21

            for hand_idx in range(num_hands):
                # Calculate offset for this hand
                offset = hand_idx * 21

                # Indices of points forming the palm boundary (adjusted for this hand)
                palm_indices = [0, 1, 5, 9, 13, 17]
                adjusted_palm_indices = [idx + offset for idx in palm_indices]

                # Extract palm points
                palm_points = np.array([self.landmark_points[i] for i in adjusted_palm_indices])
                palm_2d = palm_points[:, :2]

                # Create convex hull in 2D to find palm outline
                hull = ConvexHull(palm_2d)
                hull_indices = hull.vertices
                hull_triangles = []

                # Triangulate using fan method (good for convex shapes)
                for i in range(1, len(hull_indices) - 1):
                    hull_triangles.append([hull_indices[0], hull_indices[i], hull_indices[i + 1]])
                # Front face (blue side)
                plane_front = o3d.geometry.TriangleMesh()
                plane_front.vertices = o3d.utility.Vector3dVector(palm_points)
                plane_front.triangles = o3d.utility.Vector3iVector(hull_triangles)
                plane_front.paint_uniform_color([0, 0, 1])
                plane_front.compute_vertex_normals()

                self.add_geometry(plane_front)

        self.update_visualizer()

    def read_multi_landmarks(self, multi_hand_landmarks):
        """
        :param multi_hand_landmarks: landmarks from Mediapipe input
        :return: None
        """
        self.landmark_points = []

        for _, hand in enumerate(multi_hand_landmarks):
            self.read_hand_landmarks(hand)



    def read_hand_landmarks(self, hand, POSE_CENTER = np.array([0,0,0])):
        """
        :param landmarks: landmarks from Mediapipe input
        :param POSE_CENTER: center of hand - EITHER np.array([x,y,z]) OR list index from landmarks of desired center [0-20]
        :return: None
        """
        # Add points to list
        try:
            if type(POSE_CENTER) == np.ndarray:
                POSE_CENTER = POSE_CENTER # center at pose
            elif type(POSE_CENTER) == int:
                POSE_CENTER = np.array([hand.landmark[POSE_CENTER].x * 100, (1-hand.landmark[POSE_CENTER].y) * 100, (1-hand.landmark[POSE_CENTER].z) * 100])
                # Hand is centered at landmark --> landmark[int]
            else:
                POSE_CENTER = np.array([0,0,0]) # Hand is not centered

            for _, landmark in enumerate(hand.landmark):
                # Center points @ POSE_CENTER
                point = np.array([round(landmark.x * 100, 3), round((1 - landmark.y) * 100, 3),
                                            round((1 - landmark.z) * 100, 3)]) - POSE_CENTER
                self.landmark_points.append(point)

        except Exception as e:
            raise e

    def set_colors(self, colors: dict):
        '''
        :param colors: dictionary of colors for each part of the hand
        default: {"landmarks": [0.1, 0.6, 0.9], "thumb": [0.5,0,1], "index": [0,0.5,1], "middle": [0,1,0.5], "ring": [1,1,0], "pinky": [1,0,0], "palm": [0,1,1]}
        :return: None
        '''
        self.colors = colors


    def close(self):
        try:
            self.vis.destroy_window()
        except Exception as e:
            raise e