import cv2
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from .exercise_selection import ExerciseSelectionWidget
from .settings import SettingsDialog
from .metrics_display import MetricsDisplayWidget
from pose_estimation.pose_tracker import PoseTracker
from pose_estimation.pose_refiner import PoseRefiner
from motion_analysis.motion_analyzer import MotionAnalyzer
from symmetry_analysis.symmetry_analyzer import SymmetryAnalyzer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ROBIQ - AI Personal Trainer')
        self.setGeometry(100, 100, 1200, 600)  # Wider window to display all info

        # Initialize Pose Tracker and auxiliary components
        self.pose_tracker = PoseTracker()
        self.pose_refiner = PoseRefiner()
        self.motion_analyzer = MotionAnalyzer(window_size=5)
        self.symmetry_analyzer = SymmetryAnalyzer()

        # Initialize UI components
        self.init_ui()

        # Initialize video thread
        self.video_thread = VideoThread(self.pose_tracker, self.pose_refiner, self.motion_analyzer, self.symmetry_analyzer)
        self.video_thread.frame_updated.connect(self.update_video_frame)
        self.video_thread.joint_angles_updated.connect(self.update_joint_angles)
        self.video_thread.rom_status_updated.connect(self.update_rom_status)
        self.video_thread.rep_count_updated.connect(self.update_rep_count)
        self.video_thread.activity_updated.connect(self.update_activity)
        self.video_thread.pose_similarity_updated.connect(self.update_pose_similarity)
        self.video_thread.motion_metrics_updated.connect(self.update_motion_metrics)
        self.video_thread.symmetry_scores_updated.connect(self.update_symmetry_scores)

    def init_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout()
        video_layout = QVBoxLayout()
        control_layout = QHBoxLayout()

        # Video display area
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)

        # Joint angles display
        self.joint_angles_label = QLabel("Joint Angles")
        self.joint_angles_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.joint_angles_label)

        # ROM status display
        self.rom_status_label = QLabel()
        self.rom_status_label.setFixedSize(100, 100)
        self.rom_status_label.setAlignment(Qt.AlignCenter)
        self.rom_status_label.setStyleSheet("background-color: white;")
        video_layout.addWidget(self.rom_status_label)

        # Rep count display
        self.rep_count_label = QLabel("Reps: 0")
        self.rep_count_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.rep_count_label)

        # Pose similarity display
        self.pose_similarity_label = QLabel("Pose Similarity: N/A")
        self.pose_similarity_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.pose_similarity_label)

        # Motion metrics display
        self.motion_metrics_label = QLabel("Motion Metrics: N/A")
        self.motion_metrics_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.motion_metrics_label)

        # Symmetry scores display
        self.symmetry_scores_label = QLabel("Symmetry Scores: N/A")
        self.symmetry_scores_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.symmetry_scores_label)

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
        self.video_thread.start()

    def stop_session(self):
        self.video_thread.stop()

    def update_video_frame(self, qt_image):
        scaled_image = qt_image.scaled(self.video_label.size(), Qt.KeepAspectRatioByExpanding)
        self.video_label.setPixmap(QPixmap.fromImage(scaled_image))

    def update_joint_angles(self, joint_angles):
        angles_text = "\n".join([f"{joint}: {angle:.2f}" for joint, angle in joint_angles.items()])
        self.joint_angles_label.setText(angles_text)

    def update_rom_status(self, rom_status):
        color_map = {
            'white': 'white',
            'yellow': 'yellow',
            'light_green': 'lightgreen',
            'dark_green': 'darkgreen'
        }
        self.rom_status_label.setStyleSheet(f"background-color: {color_map.get(rom_status, 'white')};")

    def update_rep_count(self, rep_count):
        self.rep_count_label.setText(f"Reps: {rep_count}")

    def update_activity(self, activity):
        self.exercise_selection.update_activity_label(f"Detected: {activity}")

    def update_pose_similarity(self, similarity_score):
        self.pose_similarity_label.setText(f"Pose Similarity: {similarity_score:.2f}%")

    def update_motion_metrics(self, motion_metrics):
        self.motion_metrics_label.setText(f"Motion Metrics: {motion_metrics}")

    def update_symmetry_scores(self, symmetry_scores):
        symmetry_text = "\n".join([f"{joint}: {score:.2f}" for joint, score in symmetry_scores.items()])
        self.symmetry_scores_label.setText(f"Symmetry Scores:\n{symmetry_text}")

    def on_exercise_selected(self, exercise_type, skill_level):
        self.pose_tracker.update_exercise(exercise_type, skill_level)

    def open_settings(self):
        settings_dialog = SettingsDialog(self)
        settings_dialog.exec_()

    def closeEvent(self, event):
        self.stop_session()
        event.accept()