# src/gui/settings.py
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton, QCheckBox, QSpinBox

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Settings')
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Camera selection
        camera_label = QLabel('Select Camera:')
        self.camera_combo = QComboBox()
        self.camera_combo.addItem('Default Camera')  # Assume a default camera
        layout.addWidget(camera_label)
        layout.addWidget(self.camera_combo)

        # Feedback options
        feedback_label = QLabel('Feedback Options:')
        self.audio_feedback_checkbox = QCheckBox('Enable Audio Feedback')
        self.visual_feedback_checkbox = QCheckBox('Enable Visual Feedback')
        self.audio_feedback_checkbox.setChecked(True)
        self.visual_feedback_checkbox.setChecked(True)
        layout.addWidget(feedback_label)
        layout.addWidget(self.audio_feedback_checkbox)
        layout.addWidget(self.visual_feedback_checkbox)

        # Feedback interval
        interval_label = QLabel('Feedback Interval (seconds):')
        self.interval_spinbox = QSpinBox()
        self.interval_spinbox.setRange(1, 10)
        self.interval_spinbox.setValue(2)
        layout.addWidget(interval_label)
        layout.addWidget(self.interval_spinbox)

        # Save button
        save_button = QPushButton('Save')
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)

        self.setLayout(layout)

    def save_settings(self):
        self.accept()