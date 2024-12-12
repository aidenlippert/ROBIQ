import torch
import torch.nn as nn
import numpy as np
import time

class MotionAnalyzer:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.landmark_histories = {}
        self.time_stamps = deque(maxlen=self.window_size)

    def update_landmarks(self, landmarks):
        current_time = time.time()
        self.time_stamps.append(current_time)
        for idx, landmark in enumerate(landmarks):
            if idx not in self.landmark_histories:
                self.landmark_histories[idx] = deque(maxlen=self.window_size)
            self.landmark_histories[idx].append((landmark.x, landmark.y, landmark.z))

    def calculate_velocity(self, p1, p2, dt):
        """Calculate velocity between two points over time dt."""
        displacement = np.array(p2) - np.array(p1)
        velocity = displacement / dt
        return velocity

    def calculate_acceleration(self, v1, v2, dt):
        """Calculate acceleration between two velocities over time dt."""
        acceleration = (v2 - v1) / dt
        return acceleration

    def get_motion_parameters(self):
        velocities = {}
        accelerations = {}
        if len(self.time_stamps) < 2:
            return velocities, accelerations  # Not enough data

        dt = self.time_stamps[-1] - self.time_stamps[-2]
        for idx, history in self.landmark_histories.items():
            if len(history) >= 2:
                v = self.calculate_velocity(history[-2], history[-1], dt)
                velocities[idx] = v

            if len(history) >= 3:
                dt_prev = self.time_stamps[-2] - self.time_stamps[-3]
                v_prev = self.calculate_velocity(history[-3], history[-2], dt_prev)
                a = self.calculate_acceleration(v_prev, velocities[idx], dt)
                accelerations[idx] = a

        return velocities, accelerations