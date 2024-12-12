import numpy as np


class JointAnglesCalculator:
    """
    A class for calculating joint angles using 3D pose landmarks.
    """
    def __init__(self):
        # MediaPipe landmark indices
        self.landmark_indices = {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
        }
        # Visibility threshold
        self.visibility_threshold = 0.2

    def calculate_angle(self, a, b, c):
        """
        Calculate the angle at point b formed by points a, b, and c in 3D space.
        :param a: First point as (x, y, z).
        :param b: Second point (middle) as (x, y, z).
        :param c: Third point as (x, y, z).
        :return: Angle in degrees.
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0:
            return None  # Cannot calculate angle due to zero-length vector
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))

    def is_landmark_visible(self, landmark):
        """
        Check if a landmark is visible based on its visibility attribute.
        """
        return landmark.visibility >= self.visibility_threshold

    def calculate_joint_angle(self, landmarks, point_a_name, point_b_name, point_c_name):
        """
        Calculate the joint angle for specified points in 3D space.
        :param landmarks: List of pose landmarks.
        :param point_a_name: Name of the first point.
        :param point_b_name: Name of the middle point.
        :param point_c_name: Name of the third point.
        :return: Angle in degrees or None if not visible.
        """
        lm = self.landmark_indices
        try:
            point_a = landmarks[lm[point_a_name]]
            point_b = landmarks[lm[point_b_name]]
            point_c = landmarks[lm[point_c_name]]

            # Check visibility
            if not (self.is_landmark_visible(point_a) and
                    self.is_landmark_visible(point_b) and
                    self.is_landmark_visible(point_c)):
                return None

            # Use 3D coordinates
            return self.calculate_angle(
                (point_a.x, point_a.y, point_a.z),
                (point_b.x, point_b.y, point_b.z),
                (point_c.x, point_c.y, point_c.z)
            )
        except (IndexError, AttributeError):
            return None

    def get_joint_angles(self, landmarks, exercise_type='all'):
        """
        Calculate joint angles based on landmarks and exercise type.
        :param landmarks: List of pose landmarks.
        :param exercise_type: Type of exercise ('pushup', 'squat', 'lunge', or 'all').
        :return: Dictionary of joint angles.
        """
        joints = {}

        if exercise_type in ['pushup', 'all']:
            joints['left_elbow'] = self.calculate_joint_angle(landmarks, 'left_shoulder', 'left_elbow', 'left_wrist')
            joints['right_elbow'] = self.calculate_joint_angle(landmarks, 'right_shoulder', 'right_elbow', 'right_wrist')

        if exercise_type in ['squat', 'lunge', 'all']:
            joints['left_knee'] = self.calculate_joint_angle(landmarks, 'left_hip', 'left_knee', 'left_ankle')
            joints['right_knee'] = self.calculate_joint_angle(landmarks, 'right_hip', 'right_knee', 'right_ankle')

        return joints