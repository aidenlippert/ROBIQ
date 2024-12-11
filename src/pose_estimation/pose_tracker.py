from biomechanics.joint_angles import JointAnglesCalculator
from biomechanics.motion_analysis import MotionAnalyzer
from biomechanics.center_of_mass import CenterOfMassEstimator
from biomechanics.symmetry_analysis import SymmetryAnalyzer
from pose_estimation.mediapipe_blazepose import BlazePoseEstimator
from pose_estimation.temporal_smoothing import TemporalSmoothing
from pose_estimation.pose_refinement import PoseRefiner
from pose_estimation.activity_recognition import ActivityRecognizer
from pose_estimation.pose_similarity import PoseSimilarity
from pose_estimation.depth_estimation import DepthEstimator
from utils.visualization_utils import draw_auto_corrections, draw_pose_accuracy_overlay
from biomechanics.pose_similarity import PoseSimilarityModel
from feedback.audio_feedback import AudioFeedback
from feedback.adaptive_coach import AdaptiveCoach
from biomechanics.injury_risk import InjuryRiskAnalyzer
import cv2
import numpy as np

class PoseTracker:
    def __init__(self, activity_model_path=None, pose_refinement_model_path=None):
        # Core pose estimation components
        self.pose_estimator = BlazePoseEstimator()
        self.temporal_smoother = TemporalSmoothing(window_size=10)  # Stricter smoothing
        self.pose_refiner = PoseRefiner(pose_refinement_model_path)

        # Activity recognition and biomechanics components
        self.activity_recognizer = ActivityRecognizer(activity_model_path)
        self.pose_similarity_model = PoseSimilarityModel()
        self.depth_estimator = DepthEstimator()
        self.joint_angles_calculator = JointAnglesCalculator()
        self.motion_analyzer = MotionAnalyzer(window_size=5)
        self.com_estimator = CenterOfMassEstimator()
        self.symmetry_analyzer = SymmetryAnalyzer()

        # Feedback and coaching components
        self.audio_feedback = AudioFeedback()
        self.coach = AdaptiveCoach()
        self.injury_analyzer = InjuryRiskAnalyzer()

        # State variables
        self.exercise_type = 'all'  # Default: calculate all joints
        self.skill_level = None
        self.rep_count = 0
        self.rom_status = 'white'
        self.rep_started = False
        self.previous_activity = None  # Track recognized activities
        self.similarity_threshold = 80  # Threshold for triggering feedback (percentage)

    def update_exercise(self, exercise_type, skill_level):
        self.exercise_type = exercise_type
        self.skill_level = skill_level
        self.rep_count = 0
        self.rom_status = 'white'
        self.rep_started = False
        print(f"Exercise updated: {exercise_type}, Skill level: {skill_level}")

    def process_frame(self, frame):
        # Step 1: Pose estimation
        results = self.pose_estimator.process_frame(frame)

        if results.pose_landmarks:
            # Step 2: Pose refinement
            landmarks = results.pose_landmarks.landmark
            smoothed_landmarks = self.temporal_smoother.smooth_landmarks(landmarks)
            refined_landmarks = self.pose_refiner.refine_pose(self._generate_heatmap(frame, smoothed_landmarks))

            # Step 3: Depth estimation
            depth_map = self.depth_estimator.estimate_depth(frame)

            # Step 4: Draw landmarks
            annotated_image = self._draw_landmarks(frame, refined_landmarks)

            # Step 5: Joint angle calculations
            joint_angles = self.joint_angles_calculator.get_joint_angles(refined_landmarks, self.exercise_type)
            self.update_rom_status_and_rep_count(joint_angles)

            # Step 6: Activity recognition
            keypoints_sequence = np.array([[lm.x, lm.y] for lm in refined_landmarks]).flatten()
            activity = self.activity_recognizer.predict_activity([keypoints_sequence])
            if activity != self.previous_activity:
                print(f"Detected Activity: {activity}")
                self.previous_activity = activity

            # Step 7: Pose similarity scoring (auto-correction)
            similarity_score, correction_data = self.pose_similarity_model.compare(smoothed_landmarks)
            
            # Only proceed with corrections if similarity score is above the threshold
            if similarity_score >= self.similarity_threshold:
                annotated_image = draw_auto_corrections(annotated_image, correction_data)
                annotated_image = draw_pose_accuracy_overlay(annotated_image, similarity_score)

                # Step 8: Injury risk analysis
                overuse_joints = self.injury_analyzer.analyze_joint_stress(joint_angles)
                if overuse_joints:
                    print(f"Warning: Overuse detected in {overuse_joints}")

                # Step 9: Adaptive Coaching
                feedback_message = self.coach.adjust_workout(similarity_score, self.rep_count)
                print(feedback_message)

                # Step 10: Audio feedback for corrections
                if similarity_score < 70:
                    self.audio_feedback.provide_correction("Please adjust your posture!")

            else:
                # If accuracy is too low, suppress corrections and feedback
                feedback_message = "Pose accuracy too low for feedback."

            return annotated_image, joint_angles, self.rom_status, self.rep_count, activity, feedback_message
        else:
            # No landmarks detected
            self.rom_status = 'white'
            return frame, {}, self.rom_status, self.rep_count, None, None

    def update_rom_status_and_rep_count(self, joint_angles):
        """
        Update range of motion (ROM) status and repetition count.
        Integrates 3D information from depth estimation.
        """
        if self.exercise_type == 'squat':
            knee_angle = joint_angles.get('left_knee')
            if knee_angle is not None:
                if knee_angle > 160:
                    self.rom_status = 'white'
                    if self.rep_started:
                        self.rep_count += 1
                        self.rep_started = False
                elif 120 < knee_angle <= 160:
                    self.rom_status = 'yellow'
                elif 90 < knee_angle <= 120:
                    self.rom_status = 'light_green'
                elif knee_angle <= 90:
                    self.rom_status = 'dark_green'
                    self.rep_started = True
            else:
                self.rom_status = 'white'

    def _draw_landmarks(self, frame, landmarks):
        """
        Draw pose landmarks on the frame.
        """
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, f'{idx}', (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        return frame

    def _generate_heatmap(self, frame, landmarks):
        """
        Generates a simple grayscale heatmap from keypoints.
        """
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(heatmap, (x, y), 10, 255, -1)
        return heatmap