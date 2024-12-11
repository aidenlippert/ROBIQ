import numpy as np
from collections import deque

class TemporalSmoothing:
    def __init__(self, window_size=10):  # Increased window size for stricter smoothing
        self.window_size = window_size
        self.landmark_queues = [deque(maxlen=self.window_size) for _ in range(33)]  # 33 landmarks

    def smooth_landmarks(self, landmarks):
        smoothed_landmarks = []
        for idx, landmark in enumerate(landmarks):
            self.landmark_queues[idx].append(landmark)
            avg_x = np.mean([lm.x for lm in self.landmark_queues[idx]])
            avg_y = np.mean([lm.y for lm in self.landmark_queues[idx]])
            avg_z = np.mean([lm.z for lm in self.landmark_queues[idx]])
            smoothed_landmark = type('Landmark', (object,), {'x': avg_x, 'y': avg_y, 'z': avg_z})()
            smoothed_landmarks.append(smoothed_landmark)
        return smoothed_landmarks