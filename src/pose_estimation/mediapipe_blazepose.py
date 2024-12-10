import mediapipe as mp
import cv2

class BlazePoseEstimator:
    def __init__(self, static_image_mode=False, model_complexity=2, enable_segmentation=False,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity  # 0, 1, or 2
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

    def process_frame(self, frame):
        # Convert the image to RGB as MediaPipe uses RGB images
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the image and detect the pose
        results = self.pose.process(image_rgb)
        return results

    def draw_landmarks(self, frame, results):
        # Draw pose landmarks on the frame
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        return frame

    def close(self):
        # Release the MediaPipe resources
        self.pose.close()