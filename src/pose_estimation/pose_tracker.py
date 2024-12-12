import cv2
import numpy as np
from biomechanics.joint_angles import JointAnglesCalculator
from biomechanics.motion_analysis import MotionAnalyzer
from biomechanics.center_of_mass import CenterOfMassEstimator
from biomechanics.symmetry_analysis import SymmetryAnalyzer
from biomechanics.injury_risk import InjuryRiskAnalyzer
from pose_estimation.mediapipe_blazepose import BlazePoseEstimator
from pose_estimation.temporal_smoothing import TemporalSmoothing
from pose_estimation.pose_refinement import PoseRefiner
from pose_estimation.activity_recognition import ActivityRecognizer
from pose_estimation.pose_similarity import PoseSimilarityModel
from pose_estimation.depth_estimation import DepthEstimator
from utils.visualization_utils import draw_auto_corrections, draw_pose_accuracy_overlay
from feedback.audio_feedback import AudioFeedback
from feedback.adaptive_coach import AdaptiveCoach


class PoseTracker:
    def __init__(self, activity_model_path=None, pose_refinement_model_path=None):
        """
        Initialize the PoseTracker and all required components.
        """
        # Core components
        self.pose_estimator = BlazePoseEstimator()
        self.temporal_smoother = TemporalSmoothing(window_size=10)
        self.pose_refiner = PoseRefiner(pose_refinement_model_path)
        self.depth_estimator = DepthEstimator()

        # Analysis and recognition components
        self.joint_angles_calculator = JointAnglesCalculator()
        self.motion_analyzer = MotionAnalyzer(window_size=5)
        self.activity_recognizer = ActivityRecognizer(activity_model_path)
        self.pose_similarity_model = PoseSimilarityModel()
        self.injury_analyzer = InjuryRiskAnalyzer()

        # Feedback components
        self.audio_feedback = AudioFeedback()
        self.adaptive_coach = AdaptiveCoach()

        # State variables
        self.exercise_type = 'all'
        self.skill_level = None
        self.rep_count = 0
        self.rom_status = 'white'
        self.rep_started = False
        self.previous_activity = None
        self.similarity_threshold = 80

    def update_exercise(self, exercise_type, skill_level):
        """
        Update the selected exercise and skill level.
        """
        self.exercise_type = exercise_type
        self.skill_level = skill_level
        self.rep_count = 0
        self.rom_status = 'white'
        self.rep_started = False
        print(f"Exercise updated: {exercise_type}, Skill level: {skill_level}")

    def process_frame(self, frame):
        """
        Process a video frame for pose estimation, biomechanics, and feedback.

        Args:
            frame: Input video frame from OpenCV.

        Returns:
            Tuple containing processed frame, joint angles, ROM status, rep count, activity, and feedback message.
        """
        if frame is None or not frame.size:
            return frame, {}, 'white', 0, None, "Invalid frame"

        # Step 1: Lighting adjustment
        frame = self._preprocess_lighting(frame)

        # Step 2: Pose estimation
        results = self.pose_estimator.process_frame(frame)
        if not results or not results.pose_landmarks:
            return frame, {}, 'white', self.rep_count, None, None

        landmarks = results.pose_landmarks.landmark

        # Step 3: Temporal smoothing and pose refinement
        smoothed_landmarks = self.temporal_smoother.smooth_landmarks(landmarks)
        refined_landmarks = self.pose_refiner.refine_pose(self._generate_heatmap(frame, smoothed_landmarks))

        # Step 4: Depth estimation (optional visualizations)
        depth_map = self.depth_estimator.estimate_depth(frame)

        # Step 5: Joint angle calculations
        joint_angles = self.joint_angles_calculator.get_joint_angles(refined_landmarks, self.exercise_type)
        self._update_rom_and_reps(joint_angles)

        # Step 6: Activity recognition
        activity = self._recognize_activity(refined_landmarks)

        # Step 7: Pose similarity scoring
        similarity_score, corrections = self.pose_similarity_model.compare(smoothed_landmarks)
        feedback_message = self._handle_pose_feedback(similarity_score, corrections, joint_angles)

        # Draw feedback overlays
        annotated_frame = self._draw_overlays(frame, refined_landmarks, similarity_score, corrections)

        return annotated_frame, joint_angles, self.rom_status, self.rep_count, activity, feedback_message

    def _preprocess_lighting(self, frame):
        """Equalizes histogram of the input frame for better visibility."""
        frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        frame_yuv[:, :, 0] = cv2.equalizeHist(frame_yuv[:, :, 0])
        return cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)

    def _generate_heatmap(self, frame, landmarks):
        """Generate a simple heatmap from landmarks."""
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        for lm in landmarks:
            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            cv2.circle(heatmap, (x, y), 10, 255, -1)
        return heatmap

    def _recognize_activity(self, landmarks):
        """Perform activity recognition using keypoints."""
        keypoints_sequence = np.array([[lm.x, lm.y] for lm in landmarks]).flatten()
        activity = self.activity_recognizer.predict_activity([keypoints_sequence])
        if activity != self.previous_activity:
            print(f"Activity: {activity}")
            self.previous_activity = activity
        return activity

    def _handle_pose_feedback(self, similarity_score, corrections, joint_angles):
        """
        Provide feedback based on pose similarity and joint angles.
        """
        if similarity_score >= self.similarity_threshold:
            overuse_joints = self.injury_analyzer.analyze_joint_stress(joint_angles)
            if overuse_joints:
                print(f"Warning: Overuse in {overuse_joints}")
            return self.adaptive_coach.adjust_workout(similarity_score, self.rep_count)
        elif similarity_score < 70:
            self.audio_feedback.provide_correction("Adjust your posture!")
            return "Posture needs improvement."
        return "Pose accuracy too low for feedback."

    def _update_rom_and_reps(self, joint_angles):
        """
        Update the range of motion (ROM) status and repetition count based on joint angles.
        """
        exercise_logic = {
            'squat': 'left_knee',
            'push_up': 'left_elbow'
        }
        joint = exercise_logic.get(self.exercise_type)
        if joint and joint in joint_angles:
            angle = joint_angles[joint]
            if angle > 160:
                self._reset_rep()
            elif angle <= 90:
                self._start_rep()

    def _reset_rep(self):
        if self.rep_started:
            self.rep_count += 1
            self.rep_started = False
            print(f"Rep Count: {self.rep_count}")
        self.rom_status = 'white'

    def _start_rep(self):
        self.rom_status = 'dark_green'
        self.rep_started = True

    def _draw_overlays(self, frame, landmarks, similarity_score, corrections):
        """
        Draw pose landmarks and auto-corrections on the frame.
        """
        for idx, lm in enumerate(landmarks):
            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        if corrections:
            frame = draw_auto_corrections(frame, corrections)
            frame = draw_pose_accuracy_overlay(frame, similarity_score)
        return frame