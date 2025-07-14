import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
import time

# TODO: fix hand pose duplication issue

class HandPoseVisualizer:
    def __init__(self, window_name="Hand Pose Visualizer", color_profile: dict = None):
        self.window_name = window_name
        self.vis = o3d.visualization.Visualizer()
        self.window_created = False

        self.hand_poses = []
        self.geometry = []

        self.landmark_spheres = []  # List of list[Mesh] — one sublist per hand
        self.ligament_cylinders = []  # List of list[Mesh]
        self.palm_meshes = []  # List[Mesh] — one palm mesh per hand
        self.cache_initialized = False

        self.FINGERS = {
            "thumb": [1, 2, 3, 4],
            "index": [5, 6, 7, 8],
            "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16],
            "pinky": [17, 18, 19, 20]
        }

        self.COLORS_DEFAULT = {
            "landmarks": [0.1, 0.6, 0.9],
            "proximals": [0.5, 0, 0],
            "intermediates": [0, 1, 0.5],
            "distals": [0, 0.5, 1],
            "palm": [0, 0, 1],
        }

        self.colors = self.COLORS_DEFAULT if color_profile is None else color_profile

    def initialize_window(self):
        if not self.window_created:
            self.vis.create_window(window_name=self.window_name)
            self.window_created = True

    def set_hand_poses(self, hand_pose_list):
        if len(self.hand_poses) != len(hand_pose_list):
            self.cache_initialized = False
        self.hand_poses = hand_pose_list


    def set_colors(self, colors: dict):
        self.colors = colors

    def __create_sphere(self, center, radius=1.0, resolution=5, color=None):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=resolution)
        sphere.translate(center)
        sphere.paint_uniform_color(color or self.colors["landmarks"])
        return sphere

    def __create_cylinder_between(self, p1, p2, radius=0.8, resolution=5, color=[1, 0, 0]):
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        axis = p2 - p1
        length = np.linalg.norm(axis)
        if length == 0:
            return None
        axis /= length

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=resolution)
        cylinder.paint_uniform_color(color)

        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, axis)
        c = np.dot(z_axis, axis)
        if np.linalg.norm(v) < 1e-6:
            R = np.eye(3) if c > 0 else -np.eye(3)
        else:
            skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + skew + skew @ skew * ((1 - c) / (np.linalg.norm(v) ** 2))
        cylinder.rotate(R, center=(0, 0, 0))
        midpoint = (p1 + p2) / 2
        cylinder.translate(midpoint)
        return cylinder

    def _build_geometry(self, finger_tips_shown=True, ligaments_shown=True, palm_shown=True):
        self.geometry.clear()

        for hand_pose in self.hand_poses:
            coords = hand_pose.get_all_coordinates()
            landmark_points = np.array([[pt.x, pt.y, pt.z] for pt in coords])

            highlighted_finger = getattr(hand_pose, "getHighlightedFinger", lambda: None)()
            highlight_color = np.array(getattr(hand_pose, "getHighlightColor", lambda: (1.0, 1.0, 0.0))()) * 0.3

            # Landmarks
            if finger_tips_shown:
                for pt in landmark_points:
                    self.geometry.append(self.__create_sphere(pt, radius=1.0))

            # Ligaments
            if ligaments_shown:
                for finger_name, indices in self.FINGERS.items():
                    for i in range(len(indices) - 1):
                        p1 = landmark_points[indices[i]]
                        p2 = landmark_points[indices[i + 1]]
                        color = self.colors["proximals"] if i == 0 else \
                                self.colors["intermediates"] if i == 1 else \
                                self.colors["distals"]
                        cyl = self.__create_cylinder_between(p1, p2, radius=0.8, color=color)
                        if cyl:
                            self.geometry.append(cyl)

            # Palm
            if palm_shown:
                palm_indices = [0, 1, 5, 9, 13, 17]
                palm_points = landmark_points[palm_indices]
                hull = ConvexHull(palm_points[:, :2], qhull_options='QJ')
                hull_indices = hull.vertices
                triangles = []
                for i in range(1, len(hull_indices) - 1):
                    triangles.append([hull_indices[0], hull_indices[i], hull_indices[i + 1]])
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(palm_points)
                mesh.triangles = o3d.utility.Vector3iVector(triangles)
                mesh.paint_uniform_color(self.colors["palm"])
                mesh.compute_vertex_normals()
                self.geometry.append(mesh)

            # Highlight Glow
            if highlighted_finger in self.FINGERS:
                indices = self.FINGERS[highlighted_finger]
                for i in range(len(indices) - 1):
                    p1 = landmark_points[indices[i]]
                    p2 = landmark_points[indices[i + 1]]
                    glow_cyl = self.__create_cylinder_between(p1, p2, radius=1.5, color=highlight_color)
                    if glow_cyl:
                        self.geometry.append(glow_cyl)

            # Annotation
            annotation = getattr(hand_pose, "getAnnotation", lambda: None)()
            if annotation:
                wrist_coord = landmark_points[0]
                self.geometry.append(self.__create_sphere(wrist_coord + np.array([0, 0.02, 0]),
                                                          radius=1.0, color=(1.0, 1.0, 1.0)))
                print(f"[Annotation] {annotation} @ wrist: {wrist_coord}")

    def show_pose(self, finger_tips_shown=True, ligaments_shown=True, palm_shown=True):
        self.initialize_window()

        if not self.hand_poses:
            print("[!] No pose to show.")
            return

        self.build_cached_geometry(self.hand_poses[0])

        print("[🖱️] Use left mouse to rotate, right mouse to pan, scroll to zoom. Press 'q' to quit.")
        self.vis.run()
        self.vis.destroy_window()

    def update_pose(self, finger_tips_shown=True, ligaments_shown=True, palm_shown=True):
        if not self.hand_poses:
            return

        if not self.cache_initialized:
            self.build_cached_geometry(self.hand_poses)

        self.update_cached_geometry(self.hand_poses)

    def close(self):
        try:
            self.vis.destroy_window()
        except Exception as e:
            raise e

    def play_sequence(self, hand_pose_sequence, fps=30, loop=False):
        self.initialize_window()
        frame_duration = 1.0 / fps
        index = 0

        print(f"[▶️] Playing sequence at {fps} FPS...")

        # === Build geometry from the first pose ===
        if len(hand_pose_sequence) == 0:
            print("[!] Empty sequence.")
            return

        first_pose = hand_pose_sequence[0].pose
        self.build_cached_geometry(first_pose)

        try:
            while True:
                if index >= len(hand_pose_sequence):
                    if loop:
                        index = 0
                    else:
                        break

                timed_pose = hand_pose_sequence[index]
                pose = timed_pose.pose

                self.update_cached_geometry(pose)

                time.sleep(frame_duration)
                index += 1
        except KeyboardInterrupt:
            print("[⏹️] Playback interrupted.")

    def build_cached_geometry(self, hand_poses):
        self.hand_poses = hand_poses
        self.landmark_spheres.clear()
        self.ligament_cylinders.clear()
        self.palm_meshes.clear()

        for pose in hand_poses:
            coords = pose.get_all_coordinates()
            landmark_points = np.array([[pt.x, pt.y, pt.z] for pt in coords])

            # === Landmarks ===
            hand_spheres = []
            for pt in landmark_points:
                sphere = self.__create_sphere(pt, radius=1.0)
                self.vis.add_geometry(sphere)
                hand_spheres.append(sphere)
            self.landmark_spheres.append(hand_spheres)

            # === Ligaments ===
            hand_cyls = []
            for finger_name, indices in self.FINGERS.items():
                for i in range(len(indices) - 1):
                    p1 = landmark_points[indices[i]]
                    p2 = landmark_points[indices[i + 1]]
                    color = self.colors["proximals"] if i == 0 else \
                        self.colors["intermediates"] if i == 1 else \
                            self.colors["distals"]
                    cyl = self.__create_cylinder_between(p1, p2, radius=0.8, color=color)
                    if cyl:
                        self.vis.add_geometry(cyl)
                        hand_cyls.append(cyl)
            self.ligament_cylinders.append(hand_cyls)

            # === Palm ===
            palm_indices = [0, 1, 5, 9, 13, 17]
            palm_points = landmark_points[palm_indices]
            hull = ConvexHull(palm_points[:, :2], qhull_options='QJ')
            triangles = [[hull.vertices[0], hull.vertices[i], hull.vertices[i + 1]]
                         for i in range(1, len(hull.vertices) - 1)]

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(palm_points)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.paint_uniform_color(self.colors["palm"])
            mesh.compute_vertex_normals()
            self.vis.add_geometry(mesh)
            self.palm_meshes.append(mesh)

        self.cache_initialized = True

    def update_cached_geometry(self, hand_poses):
        #if len(hand_poses) != len(self.landmark_spheres):
        #    self.build_cached_geometry(hand_poses)
        #    return

        for h_index, pose in enumerate(hand_poses):
            coords = pose.get_all_coordinates()
            landmark_points = np.array([[pt.x, pt.y, pt.z] for pt in coords])

            # === Update Landmarks ===
            for i, pt in enumerate(landmark_points):
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=5)
                mesh.translate(pt)
                self.landmark_spheres[h_index][i].vertices = mesh.vertices
                self.landmark_spheres[h_index][i].compute_vertex_normals()
                self.vis.update_geometry(self.landmark_spheres[h_index][i])

            # === Update Ligaments ===
            lig_index = 0
            for finger_name, indices in self.FINGERS.items():
                for i in range(len(indices) - 1):
                    p1 = landmark_points[indices[i]]
                    p2 = landmark_points[indices[i + 1]]
                    new_cyl = self.__create_cylinder_between(p1, p2, radius=0.8, resolution=5)
                    cyl = self.ligament_cylinders[h_index][lig_index]
                    cyl.vertices = new_cyl.vertices
                    cyl.triangles = new_cyl.triangles
                    cyl.compute_vertex_normals()
                    self.vis.update_geometry(cyl)
                    lig_index += 1

            # === Update Palm ===
            palm_points = landmark_points[[0, 1, 5, 9, 13, 17]]
            palm = self.palm_meshes[h_index]
            palm.vertices = o3d.utility.Vector3dVector(palm_points)
            palm.compute_vertex_normals()
            self.vis.update_geometry(palm)

        self.vis.poll_events()
        self.vis.update_renderer()


class DeprecatedHandPoseVisualizer:
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


    def __create_cylinder_between(self, p1, p2, radius=0.8, resolution=5, color=[1, 0, 0]):
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
                hull = ConvexHull(palm_2d, qhull_options='QJ')
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