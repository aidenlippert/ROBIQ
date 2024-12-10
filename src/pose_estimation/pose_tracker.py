from biomechanics.joint_angles import JointAnglesCalculator
from biomechanics.motion_analysis import MotionAnalyzer
from biomechanics.center_of_mass import CenterOfMassEstimator
from biomechanics.symmetry_analysis import SymmetryAnalyzer
import cv2
from .mediapipe_blazepose import BlazePoseEstimator
from .temporal_smoothing import TemporalSmoothingss PoseTracker:
 

class PoseTracker:
    def __init__(self):
        self.pose_estimator = BlazePoseEstimator()
        self.temporal_smoother = TemporalSmoothing(window_size=5)
        # Instantiate biomechanics classes
        self.joint_angles_calculator = JointAnglesCalculator()
        self.motion_analyzer = MotionAnalyzer(window_size=5)
        self.com_estimator = CenterOfMassEstimator()
        self.symmetry_analyzer = SymmetryAnalyzer()

    def process_stream(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.pose_estimator.process_frame(frame)
            if results.pose_landmarks:
                smoothed_landmarks = self.temporal_smoother.smooth_landmarks(results.pose_landmarks.landmark)
                
                # Update motion analyzer
                self.motion_analyzer.update_landmarks(smoothed_landmarks)
                
                # Calculate biomechanics metrics
                joint_angles = self.joint_angles_calculator.get_joint_angles(smoothed_landmarks)
                velocities, accelerations = self.motion_analyzer.get_motion_parameters()
                com = self.com_estimator.estimate_com(smoothed_landmarks)
                symmetry = self.symmetry_analyzer.analyze_symmetry(joint_angles)
                
                # Draw landmarks and visualize metrics
                frame = self.pose_estimator.draw_landmarks(frame, results)
                # Optionally overlay metrics on the frame
                
            cv2.imshow('FitWizard - Pose Tracker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.pose_estimator.close()