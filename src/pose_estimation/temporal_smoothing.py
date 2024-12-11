import numpy as np
from collections import deque
from collections import namedtuple

# You can define a simple Landmark class or use namedtuple or dataclass for efficiency
Landmark = namedtuple('Landmark', ['x', 'y', 'z', 'visibility'])

class TemporalSmoothing:
    def __init__(self, window_size=10, confidence_threshold=0.5, use_weighted_avg=False):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.use_weighted_avg = use_weighted_avg
        
        # Deques for storing landmark positions per index
        self.landmark_queues = [deque(maxlen=self.window_size) for _ in range(33)]  # Assume 33 landmarks
        
    def smooth_landmarks(self, landmarks):
        smoothed_landmarks = []
        
        for idx, landmark in enumerate(landmarks):
            # Handle missing landmarks (if they have low confidence)
            if landmark.visibility < self.confidence_threshold:
                if len(self.landmark_queues[idx]) > 0:
                    prev_landmark = self.landmark_queues[idx][-1]  # Use last known position
                    smoothed_landmarks.append(prev_landmark)
                    continue
            
            # Add new landmarks to the queue
            self.landmark_queues[idx].append(landmark)
            
            # Weighted averaging (optional)
            if self.use_weighted_avg:
                weights = np.linspace(1, 0, num=len(self.landmark_queues[idx]))  # Linearly decrease weights
                avg_x = np.average([lm.x for lm in self.landmark_queues[idx]], weights=weights)
                avg_y = np.average([lm.y for lm in self.landmark_queues[idx]], weights=weights)
                avg_z = np.average([lm.z for lm in self.landmark_queues[idx]], weights=weights)
            else:
                # Simple average
                avg_x = np.mean([lm.x for lm in self.landmark_queues[idx]])
                avg_y = np.mean([lm.y for lm in self.landmark_queues[idx]])
                avg_z = np.mean([lm.z for lm in self.landmark_queues[idx]])

            # Create a new Landmark with smoothed values
            smoothed_landmark = Landmark(avg_x, avg_y, avg_z, landmark.visibility)
            smoothed_landmarks.append(smoothed_landmark)
        
        return smoothed_landmarks