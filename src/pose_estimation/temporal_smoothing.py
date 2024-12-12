import numpy as np
from collections import deque, namedtuple
from pykalman import KalmanFilter  # Install via pip install pykalman

# Define a simple Landmark class
Landmark = namedtuple('Landmark', ['x', 'y', 'z', 'visibility'])


class TemporalSmoothing:
    """
    Temporal smoothing of landmarks using Kalman Filter, Weighted Average, or Exponential Moving Average.
    """
    def __init__(self, window_size=10, confidence_threshold=0.5, method='kalman', alpha=0.3):
        """
        Initialize TemporalSmoothing class.

        Args:
            window_size (int): Number of previous frames to consider for smoothing.
            confidence_threshold (float): Visibility threshold for valid landmarks.
            method (str): Smoothing method. Options: ['kalman', 'weighted_avg', 'ema'].
            alpha (float): Smoothing factor for EMA. Closer to 1 makes it more reactive.
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.method = method.lower()
        self.alpha = alpha

        # Storage for smoothing
        self.landmark_queues = [deque(maxlen=window_size) for _ in range(33)]  # 33 landmarks assumed

        # Kalman Filters for each landmark index
        if self.method == 'kalman':
            self.kalman_filters = [self._initialize_kalman_filter() for _ in range(33)]

    def _initialize_kalman_filter(self):
        """
        Initialize a Kalman Filter for landmark smoothing.
        """
        kf = KalmanFilter(initial_state_mean=[0, 0, 0],
                          n_dim_obs=3,
                          transition_matrices=np.eye(3),  # Simple motion model
                          observation_matrices=np.eye(3),
                          observation_covariance=np.eye(3) * 0.1,  # Measurement noise
                          transition_covariance=np.eye(3) * 0.01)  # Process noise
        return kf

    def smooth_landmarks(self, landmarks):
        """
        Apply smoothing to the given landmarks.

        Args:
            landmarks (list of Landmark): List of raw landmarks to be smoothed.

        Returns:
            list of Landmark: Smoothed landmarks.
        """
        smoothed_landmarks = []

        for idx, landmark in enumerate(landmarks):
            # Handle low-confidence landmarks
            if landmark.visibility < self.confidence_threshold:
                smoothed_landmarks.append(self._handle_missing_landmark(idx))
                continue

            # Add current landmark to the queue
            self.landmark_queues[idx].append(landmark)

            # Perform smoothing based on method
            if self.method == 'kalman':
                smoothed_landmark = self._kalman_smooth(idx, landmark)
            elif self.method == 'weighted_avg':
                smoothed_landmark = self._weighted_average_smooth(idx, landmark)
            elif self.method == 'ema':
                smoothed_landmark = self._ema_smooth(idx, landmark)
            else:
                raise ValueError(f"Unsupported smoothing method: {self.method}")

            smoothed_landmarks.append(smoothed_landmark)

        return smoothed_landmarks

    def _handle_missing_landmark(self, idx):
        """
        Handle missing landmarks by returning the last valid position or default values.
        """
        if len(self.landmark_queues[idx]) > 0:
            return self.landmark_queues[idx][-1]
        return Landmark(0.0, 0.0, 0.0, 0.0)  # Default to zero position if no history

    def _kalman_smooth(self, idx, landmark):
        """
        Smooth landmarks using the Kalman Filter.

        Args:
            idx (int): Index of the landmark.
            landmark (Landmark): Current landmark.

        Returns:
            Landmark: Smoothed landmark.
        """
        kalman_filter = self.kalman_filters[idx]
        observations = np.array([landmark.x, landmark.y, landmark.z])
        kalman_filter = kalman_filter.em(observations, n_iter=1)
        filtered_state_means, _ = kalman_filter.filter(observations)

        smoothed_x, smoothed_y, smoothed_z = filtered_state_means[-1]
        return Landmark(smoothed_x, smoothed_y, smoothed_z, landmark.visibility)

    def _weighted_average_smooth(self, idx, landmark):
        """
        Smooth landmarks using weighted average, considering visibility as confidence.

        Args:
            idx (int): Index of the landmark.
            landmark (Landmark): Current landmark.

        Returns:
            Landmark: Smoothed landmark.
        """
        landmarks = self.landmark_queues[idx]
        weights = [lm.visibility for lm in landmarks]  # Confidence as weight

        if not weights:  # Fallback to the current position
            return landmark

        avg_x = np.average([lm.x for lm in landmarks], weights=weights)
        avg_y = np.average([lm.y for lm in landmarks], weights=weights)
        avg_z = np.average([lm.z for lm in landmarks], weights=weights)
        return Landmark(avg_x, avg_y, avg_z, landmark.visibility)

    def _ema_smooth(self, idx, landmark):
        """
        Smooth landmarks using an Exponential Moving Average (EMA).

        Args:
            idx (int): Index of the landmark.
            landmark (Landmark): Current landmark.

        Returns:
            Landmark: Smoothed landmark.
        """
        landmarks = self.landmark_queues[idx]
        if len(landmarks) < 2:
            return landmark  # Not enough history for EMA

        prev_landmark = landmarks[-2]  # Previous smoothed landmark
        smoothed_x = self.alpha * landmark.x + (1 - self.alpha) * prev_landmark.x
        smoothed_y = self.alpha * landmark.y + (1 - self.alpha) * prev_landmark.y
        smoothed_z = self.alpha * landmark.z + (1 - self.alpha) * prev_landmark.z

        return Landmark(smoothed_x, smoothed_y, smoothed_z, landmark.visibility)