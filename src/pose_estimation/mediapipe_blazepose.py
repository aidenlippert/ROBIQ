import mediapipe as mp
import cv2


class BlazePoseEstimator:
    """
    A class for detecting human pose landmarks using MediaPipe Pose.
    Provides functionality for drawing landmarks and returning pose landmarks.
    """
    def __init__(self, static_image_mode=False, model_complexity=2, enable_segmentation=False,
                 min_detection_confidence=0.5, min_tracking_confidence=0.7):
        """
        Initialize MediaPipe Pose with the specified configuration.
        """
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.enable_segmentation = enable_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=self.static_image_mode,
                                      model_complexity=self.model_complexity,
                                      enable_segmentation=self.enable_segmentation,
                                      min_detection_confidence=self.min_detection_confidence,
                                      min_tracking_confidence=self.min_tracking_confidence)

        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame):
        """
        Process a video frame to detect pose landmarks.
        :param frame: Input video frame (BGR format).
        :return: Processed pose landmarks or None if not detected.
        """
        # Convert the frame to RGB as MediaPipe uses RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None

    def draw_landmarks(self, frame, landmarks):
        """
        Draw pose landmarks on the frame.
        :param frame: Input video frame.
        :param landmarks: MediaPipe landmarks to be drawn.
        """
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

    def close(self):
        """
        Release MediaPipe resources.
        """
        self.pose.close()