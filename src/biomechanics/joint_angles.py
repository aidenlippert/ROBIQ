import numpy as np

class JointAnglesCalculator:
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

    def calculate_angle(self, a, b, c):
        """Calculate the angle at point b formed by points a, b, and c."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Numerical stability
        angle = np.degrees(np.arccos(cosine_angle))
        return angle

    def get_joint_angles(self, landmarks):
        """Compute joint angles based on landmarks."""
        lm = self.landmark_indices
        joints = {}

        # Example: Calculating left elbow angle
        left_shoulder = landmarks[lm['left_shoulder']]
        left_elbow = landmarks[lm['left_elbow']]
        left_wrist = landmarks[lm['left_wrist']]
        left_elbow_angle = self.calculate_angle(
            (left_shoulder.x, left_shoulder.y, left_shoulder.z),
            (left_elbow.x, left_elbow.y, left_elbow.z),
            (left_wrist.x, left_wrist.y, left_wrist.z)
        )
        joints['left_elbow'] = left_elbow_angle

        # Right Elbow Angle
        right_shoulder = landmarks[lm['right_shoulder']]
        right_elbow = landmarks[lm['right_elbow']]
        right_wrist = landmarks[lm['right_wrist']]
        right_elbow_angle = self.calculate_angle(
            (right_shoulder.x, right_shoulder.y, right_shoulder.z),
            (right_elbow.x, right_elbow.y, right_elbow.z),
            (right_wrist.x, right_wrist.y, right_wrist.z)
        )
        joints['right_elbow'] = right_elbow_angle

        # Add more joints as needed for knees, shoulders, etc.

        return joints