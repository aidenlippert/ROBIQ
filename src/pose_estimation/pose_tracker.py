from biomechanics.joint_angles import JointAnglesCalculator
from biomechanics.motion_analysis import MotionAnalyzer
from biomechanics.center_of_mass import CenterOfMassEstimator
from biomechanics.symmetry_analysis import SymmetryAnalyzer
from pose_estimation.mediapipe_blazepose import BlazePoseEstimator
from pose_estimation.temporal_smoothing import TemporalSmoothing
import cv2

class PoseTracker:
    def __init__(self):
        self.pose_estimator = BlazePoseEstimator()
        self.temporal_smoother = TemporalSmoothing(window_size=10)  # Stricter smoothing
        # Instantiate biomechanics classes
        self.joint_angles_calculator = JointAnglesCalculator()
        self.motion_analyzer = MotionAnalyzer(window_size=5)
        self.com_estimator = CenterOfMassEstimator()
        self.symmetry_analyzer = SymmetryAnalyzer()
        self.exercise_type = 'all'  # Default to 'all' to calculate all joints
        self.skill_level = None
        self.rep_count = 0
        self.rom_status = 'white'
        self.rep_started = False

    def update_exercise(self, exercise_type, skill_level):
        self.exercise_type = exercise_type
        self.skill_level = skill_level
        self.rep_count = 0
        self.rom_status = 'white'
        self.rep_started = False
        print(f"Exercise updated: {exercise_type}, Skill level: {skill_level}")

    def process_frame(self, frame):
        results = self.pose_estimator.process_frame(frame)
        joint_angles = {}
        if results.pose_landmarks:
            smoothed_landmarks = self.temporal_smoother.smooth_landmarks(results.pose_landmarks.landmark)
            
            # Draw landmarks with visibility
            annotated_image = frame.copy()
            for idx, landmark in enumerate(smoothed_landmarks):
                if landmark.visibility > 0:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    # Draw a circle at each landmark position
                    cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
                    # Put the index and visibility score near the landmark
                    cv2.putText(annotated_image, f'{idx}:{landmark.visibility:.2f}', (x + 5, y - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            frame = annotated_image

            # Proceed with calculations
            joint_angles = self.joint_angles_calculator.get_joint_angles(smoothed_landmarks, self.exercise_type)
            self.update_rom_status_and_rep_count(joint_angles)
        else:
            # No landmarks detected
            joint_angles = {}
            self.rom_status = 'white'
        return frame, joint_angles, self.rom_status, self.rep_count

    def update_rom_status_and_rep_count(self, joint_angles):
        if self.exercise_type == 'squat':
            knee_angle = joint_angles.get('left_knee')
            if knee_angle is not None:
                # Update ROM status and rep count logic as before
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
                # Knee angle not available
                self.rom_status = 'white'
        # Add similar logic for other exercises (lunge, pushup) as needed