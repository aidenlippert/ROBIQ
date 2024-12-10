import numpy as np
from collections import deque

class TemporalSmoothing:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.landmark_queues = [deque(maxlen=self.window_size) for _ in range(33)]  # 33 landmarks

    def smooth_landmarks(self, landmarks):
        smoothed_landmarks = []
        for idx, landmark in enumerate(landmarks):
            self.landmark_queues[idx].append(landmark)
            avg_x = np.mean([lm.x for lm in self.landmark_queues[idx]])
            avg_y = np.mean([lm.y for lm in self.landmark_queues[idx]])
            avg_z = np.mean([lm.z for lm in self.landmark_queues[idx]])
            smoothed_landmarks.append((avg_x, avg_y, avg_z))
        return smoothed_landmarks