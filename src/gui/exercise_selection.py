# src/gui/exercise_selection.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox
from PyQt5.QtCore import pyqtSignal

class ExerciseSelectionWidget(QWidget):
    exercise_selected = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Exercise selection
        exercise_label = QLabel('Select Exercise:')
        self.exercise_combo = QComboBox()
        self.exercise_combo.addItems(['Squat', 'Push-up', 'Lunge'])  # Add more exercises as needed
        layout.addWidget(exercise_label)
        layout.addWidget(self.exercise_combo)

        # Skill level selection
        skill_label = QLabel('Select Skill Level:')
        self.skill_combo = QComboBox()
        self.skill_combo.addItems(['Beginner', 'Intermediate', 'Advanced'])
        layout.addWidget(skill_label)
        layout.addWidget(self.skill_combo)

        # Connect signals
        self.exercise_combo.currentTextChanged.connect(self.emit_selection)
        self.skill_combo.currentTextChanged.connect(self.emit_selection)

        self.setLayout(layout)

    def emit_selection(self):
        exercise = self.exercise_combo.currentText().lower()
        skill_level = self.skill_combo.currentText().lower()
        self.exercise_selected.emit(exercise, skill_level)