# src/gui/metrics_display.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout
from PyQt5.QtCore import Qt

class MetricsDisplayWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        title_label = QLabel('Metrics')
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Metrics grid
        self.metrics_grid = QGridLayout()
        self.metrics_labels = {}
        metrics = ['Left Knee Angle', 'Right Knee Angle', 'Symmetry Score', 'Average Speed']
        for i, metric in enumerate(metrics):
            label = QLabel(f'{metric}:')
            value = QLabel('--')
            self.metrics_labels[metric] = value
            self.metrics_grid.addWidget(label, i, 0)
            self.metrics_grid.addWidget(value, i, 1)

        layout.addLayout(self.metrics_grid)
        self.setLayout(layout)

    def update_metrics(self, joint_angles, symmetry_scores, avg_speed):
        left_knee_angle = joint_angles.get('left_knee', '--')
        right_knee_angle = joint_angles.get('right_knee', '--')
        symmetry = symmetry_scores.get('knee', '--')
        self.metrics_labels['Left Knee Angle'].setText(f'{left_knee_angle:.1f}')
        self.metrics_labels['Right Knee Angle'].setText(f'{right_knee_angle:.1f}')
        self.metrics_labels['Symmetry Score'].setText(f'{symmetry:.2f}')
        self.metrics_labels['Average Speed'].setText(f'{avg_speed:.2f} m/s')