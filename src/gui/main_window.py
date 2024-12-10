import sys
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from .exercise_selection import ExerciseSelectionWidget
from .settings import SettingsDialog
from .metrics_display import MetricsDisplayWidget
from pose_estimation.pose_tracker import PoseTracker

class VideoThread(QThread):
    frame_updated = pyqtSignal(QImage)

    def __init__(self, pose_tracker):
        super().__init__()
        self.pose_tracker = pose_tracker
        self.running = True

    def run(self):
        # Check if the camera is available before starting video capture
        cap = self.pose_tracker.cap
        if not cap.isOpened():
            print("Error: Camera not found!")
            return

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame!")
                break
            frame = self.pose_tracker.process_frame(frame)

            # Convert frame to QImage for display in Qt
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_updated.emit(qt_image)

    def stop(self):
        self.running = False
        self.pose_tracker.release_resources()  # Ensure pose tracker resources are released
        self.quit()
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('FitWizard - AI Personal Trainer')
        self.setGeometry(100, 100, 1200, 800)

        # Initialize pose tracker (with a default video source)
        self.pose_tracker = PoseTracker(video_source=0)

        # Initialize UI components
        self.init_ui()

        # Initialize video thread
        self.video_thread = VideoThread(self.pose_tracker)
        self.video_thread.frame_updated.connect(self.update_video_frame)

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
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)

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
        main_layout.addLayout(video_layout, 2)
        main_layout.addWidget(self.exercise_selection, 1)
        central_widget.setLayout(main_layout)

    def start_session(self):
        # Start the video thread (pose tracking)
        self.video_thread.start()

    def stop_session(self):
        # Stop the video thread and clean up
        self.video_thread.stop()

    def update_video_frame(self, qt_image):
        # Update the displayed video frame
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

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

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()