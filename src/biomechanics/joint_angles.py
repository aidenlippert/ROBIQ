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
        # Visibility threshold
        self.visibility_threshold = 0.2

    def calculate_angle(self, a, b, c):
        """Calculate the angle at point b formed by points a, b, and c."""
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
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Numerical stability
        angle = np.degrees(np.arccos(cosine_angle))
        return angle

    def is_landmark_visible(self, landmark):
        """Check if a landmark is visible based on the visibility attribute."""
        return landmark.visibility >= self.visibility_threshold

    def get_joint_angles(self, landmarks, exercise_type):
        """Compute joint angles based on landmarks and exercise type."""
        lm = self.landmark_indices
        joints = {}

        if exercise_type in ['pushup', 'all']:
            # Calculate elbow angles
            joints['left_elbow'] = self.calculate_joint_angle(landmarks, 'left_shoulder', 'left_elbow', 'left_wrist')
            joints['right_elbow'] = self.calculate_joint_angle(landmarks, 'right_shoulder', 'right_elbow', 'right_wrist')

        if exercise_type in ['squat', 'lunge', 'all']:
            # Calculate knee angles
            joints['left_knee'] = self.calculate_joint_angle(landmarks, 'left_hip', 'left_knee', 'left_ankle')
            joints['right_knee'] = self.calculate_joint_angle(landmarks, 'right_hip', 'right_knee', 'right_ankle')

        return joints

def calculate_joint_angle(self, landmarks, point_a_name, point_b_name, point_c_name):
    lm = self.landmark_indices
    try:
        point_a = landmarks[lm[point_a_name]]
        point_b = landmarks[lm[point_b_name]]
        point_c = landmarks[lm[point_c_name]]

        # Debug prints for visibility
        print(f'Calculating angle for {point_b_name}')
        print(f'{point_a_name} visibility: {point_a.visibility}')
        print(f'{point_b_name} visibility: {point_b.visibility}')
        print(f'{point_c_name} visibility: {point_c.visibility}')

        # Check landmarks visibility
        if not (self.is_landmark_visible(point_a) and
                self.is_landmark_visible(point_b) and
                self.is_landmark_visible(point_c)):
            print(f'One or more landmarks not visible for {point_b_name}')
            return None

        angle = self.calculate_angle(
            (point_a.x, point_a.y, point_a.z),
            (point_b.x, point_b.y, point_b.z),
            (point_c.x, point_c.y, point_c.z)
        )
        print(f'Angle for {point_b_name}: {angle}')
        return angle
    except (IndexError, AttributeError) as e:
        print(f'Error calculating angle for {point_b_name}: {e}')
        return None