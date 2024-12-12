import numpy as np
from collections import deque
import time

class MotionAnalyzer:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.landmark_histories = {}
        self.time_stamps = deque(maxlen=self.window_size)

    def update_landmarks(self, landmarks):
        """
        Update the history of landmark positions with the current time.
        
        Args:
            landmarks (list): List of Landmark objects (with x, y, z attributes).
        """
        current_time = time.perf_counter()  # Use a more precise time source
        self.time_stamps.append(current_time)
        
        for idx, landmark in enumerate(landmarks):
            if idx not in self.landmark_histories:
                self.landmark_histories[idx] = deque(maxlen=self.window_size)
            self.landmark_histories[idx].append((landmark.x, landmark.y, landmark.z))

    def calculate_velocity(self, p1, p2, dt):
        """
        Calculate velocity between two points over time.
        
        Args:
            p1 (tuple): Coordinates (x, y, z) of the first point.
            p2 (tuple): Coordinates (x, y, z) of the second point.
            dt (float): Time difference between the two points.
            
        Returns:
            np.ndarray: Velocity vector.
        """
        if dt <= 0:
            return np.array([0.0, 0.0, 0.0])  # Prevent division by zero
        displacement = np.array(p2) - np.array(p1)
        velocity = displacement / dt
        return velocity

    def calculate_acceleration(self, v1, v2, dt):
        """
        Calculate acceleration between two velocities over time.
        
        Args:
            v1 (np.ndarray): Velocity vector at time t-1.
            v2 (np.ndarray): Velocity vector at time t.
            dt (float): Time difference between the two velocities.
            
        Returns:
            np.ndarray: Acceleration vector.
        """
        if dt <= 0:
            return np.array([0.0, 0.0, 0.0])  # Prevent division by zero
        acceleration = (v2 - v1) / dt
        return acceleration

    def get_motion_parameters(self):
        """
        Get velocity and acceleration for each landmark based on their historical positions.
        
        Returns:
            dict: Velocities for each landmark.
            dict: Accelerations for each landmark.
        """
        velocities = {}
        accelerations = {}
        
        if len(self.time_stamps) < 2:
            return velocities, accelerations  # Not enough data for motion calculations
        
        # Compute the time difference between the latest two time stamps
        dt = self.time_stamps[-1] - self.time_stamps[-2]

        for idx, history in self.landmark_histories.items():
            if len(history) >= 2:
                # Calculate velocity
                v = self.calculate_velocity(history[-2], history[-1], dt)
                velocities[idx] = v

            if len(history) >= 3:
                # Calculate acceleration
                dt_prev = self.time_stamps[-2] - self.time_stamps[-3]
                v_prev = self.calculate_velocity(history[-3], history[-2], dt_prev)
                a = self.calculate_acceleration(v_prev, velocities[idx], dt)
                accelerations[idx] = a

        return velocities, accelerations