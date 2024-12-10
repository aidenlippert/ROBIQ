import numpy as np

class CenterOfMassEstimator:
    def __init__(self):
        # Approximate mass percentages of body segments
        self.segment_masses = {
            'head': 0.08,
            'upper_torso': 0.20,
            'lower_torso': 0.32,
            'upper_arm': 0.03,  # Each
            'forearm': 0.02,    # Each
            'hand': 0.01,       # Each
            'thigh': 0.10,      # Each
            'shank': 0.04,      # Each
            'foot': 0.015       # Each
        }
        # Segment to landmarks mapping
        self.segments = {
            'head': [0],  # Nose
            'upper_torso': [11, 12],  # Shoulders
            'lower_torso': [23, 24],  # Hips
            # Limbs
            'left_upper_arm': [11, 13],
            'right_upper_arm': [12, 14],
            'left_forearm': [13, 15],
            'right_forearm': [14, 16],
            'left_hand': [15, 17, 19, 21],  # Including hand landmarks
            'right_hand': [16, 18, 20, 22],
            'left_thigh': [23, 25],
            'right_thigh': [24, 26],
            'left_shank': [25, 27],
            'right_shank': [26, 28],
            'left_foot': [27, 31],
            'right_foot': [28, 32]
        }

    def calculate_segment_com(self, landmarks, indices):
        points = [np.array([landmarks[i].x, landmarks[i].y, landmarks[i].z]) for i in indices]
        segment_com = np.mean(points, axis=0)
        return segment_com

    def estimate_com(self, landmarks):
        total_mass = sum(self.segment_masses.values())
        com_numerator = np.zeros(3)

        for segment, mass in self.segment_masses.items():
            segment_name = segment
            # Handle left and right segments
            if segment in ['upper_arm', 'forearm', 'hand', 'thigh', 'shank', 'foot']:
                for side in ['left', 'right']:
                    seg_key = f"{side}_{segment}"
                    indices = self.segments.get(seg_key, [])
                    if indices:
                        seg_com = self.calculate_segment_com(landmarks, indices)
                        com_numerator += seg_com * (mass / 2)
            else:
                indices = self.segments.get(segment_name, [])
                if indices:
                    seg_com = self.calculate_segment_com(landmarks, indices)
                    com_numerator += seg_com * mass

        body_com = com_numerator / total_mass
        return body_com