import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from .exercise_selection import ExerciseSelectionWidget
from .settings import SettingsDialog
from .metrics_display import MetricsDisplayWidget
from pose_estimation.pose_tracker import PoseTracker
from biomechanics.symmetry_analysis import SymmetryAnalyzer
from biomechanics.joint_angles import JointAnglesCalculator
from biomechanics.motion_analysis import MotionAnalyzer
from biomechanics.center_of_mass import CenterOfMassEstimator
from pose_estimation.mediapipe_blazepose import BlazePoseEstimator
from pose_estimation.temporal_smoothing import TemporalSmoothing

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ROBIQ - AI Personal Trainer')
        self.setGeometry(100, 100, 1200, 200)  # Make the window wider and less tall

        # Initialize pose tracker
        self.pose_tracker = PoseTracker()

        # Initialize UI components
        self.init_ui()

        # Initialize video thread
        self.video_thread = VideoThread(self.pose_tracker)
        self.video_thread.frame_updated.connect(self.update_video_frame)
        self.video_thread.joint_angles_updated.connect(self.update_joint_angles)
        self.video_thread.rom_status_updated.connect(self.update_rom_status)
        self.video_thread.rep_count_updated.connect(self.update_rep_count)

    def init_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layouts
        main_layout = QHBoxLayout()
        video_layout = QVBoxLayout()
        control_layout = QHBoxLayout()

        # Video display area
        self.video_label = QLabel()
        self.video_label.setFixedSize(350, 350)  # Set to a square size
        self.video_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)

        # Joint angles display area
        self.joint_angles_label = QLabel()
        self.joint_angles_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.joint_angles_label)

        # ROM status display area
        self.rom_status_label = QLabel()
        self.rom_status_label.setFixedSize(100, 100)
        self.rom_status_label.setAlignment(Qt.AlignCenter)
        self.rom_status_label.setStyleSheet("background-color: white;")
        video_layout.addWidget(self.rom_status_label)

        # Rep count display area
        self.rep_count_label = QLabel("Reps: 0")
        self.rep_count_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.rep_count_label)

        # Metrics display widget
        self.metrics_display = MetricsDisplayWidget()
        video_layout.addWidget(self.metrics_display)

        # Control buttons
        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.start_session)
        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_session)
        self.settings_button = QPushButton('Settings')
        self.settings_button.clicked.connect(self.open_settings)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.settings_button)
        
        video_layout.addLayout(control_layout)

        # Exercise selection widget
        self.exercise_selection = ExerciseSelectionWidget()
        self.exercise_selection.exercise_selected.connect(self.on_exercise_selected)

        # Add layouts to main layout
        main_layout.addLayout(video_layout, 3)
        main_layout.addWidget(self.exercise_selection, 1)
        central_widget.setLayout(main_layout)

    def start_session(self):
        # Start the video thread (pose tracking)
        self.video_thread.start()

    def stop_session(self):
        # Stop the video thread and clean up
        self.video_thread.stop()

    def update_video_frame(self, qt_image):
        # Scale the image to fit the QLabel
        scaled_image = qt_image.scaled(self.video_label.size(), Qt.KeepAspectRatioByExpanding)
        # Center the image in the QLabel
        self.video_label.setPixmap(QPixmap.fromImage(scaled_image))

    def update_joint_angles(self, joint_angles):
        # Update the joint angles display
        angles_text_lines = []
        joints_to_display = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']
        for joint in joints_to_display:
            angle = joint_angles.get(joint)
            if angle is not None:
                angles_text_lines.append(f"{joint}: {angle:.2f}")
            else:
                angles_text_lines.append(f"{joint}: -")
        angles_text = "\n".join(angles_text_lines)
        self.joint_angles_label.setText(angles_text)

    def update_rom_status(self, rom_status):
        # Update the ROM status display
        color_map = {
            'white': 'white',
            'yellow': 'yellow',
            'light_green': 'lightgreen',
            'dark_green': 'darkgreen'
        }
        self.rom_status_label.setStyleSheet(f"background-color: {color_map[rom_status]};")

    def update_rep_count(self, rep_count):
        # Update the rep count display
        self.rep_count_label.setText(f"Reps: {rep_count}")

    def on_exercise_selected(self, exercise_type, skill_level):
        # Update the pose tracker with the selected exercise type and skill level
        self.pose_tracker.update_exercise(exercise_type, skill_level)

    def open_settings(self):
        # Open the settings dialog
        settings_dialog = SettingsDialog(self)
        settings_dialog.exec_()

    def closeEvent(self, event):
        # Ensure the session is stopped when the window is closed
        self.stop_session()
        event.accept()

class VideoThread(QThread):
    frame_updated = pyqtSignal(QImage)
    joint_angles_updated = pyqtSignal(dict)
    rom_status_updated = pyqtSignal(str)
    rep_count_updated = pyqtSignal(int)

    def __init__(self, pose_tracker):
        super().__init__()
        self.pose_tracker = pose_tracker
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame, joint_angles, rom_status, rep_count = self.pose_tracker.process_frame(frame)

            # Convert frame to QImage for display in PyQt5
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
            self.frame_updated.emit(qt_image)
            self.joint_angles_updated.emit(joint_angles)
            self.rom_status_updated.emit(rom_status)
            self.rep_count_updated.emit(rep_count)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()