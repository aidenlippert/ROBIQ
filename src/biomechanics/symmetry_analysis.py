class SymmetryAnalyzer:
    def __init__(self):
        # Pairs of left and right landmarks
        self.left_right_pairs = {
            'shoulder': (11, 12),
            'elbow': (13, 14),
            'wrist': (15, 16),
            'hip': (23, 24),
            'knee': (25, 26),
            'ankle': (27, 28)
        }

    def calculate_symmetry(self, left_metric, right_metric):
        """Calculate symmetry ratio between left and right metrics."""
        if left_metric + right_metric == 0:
            return 1.0  # Perfect symmetry
        symmetry_ratio = min(left_metric, right_metric) / max(left_metric, right_metric)
        return symmetry_ratio

    def analyze_symmetry(self, metrics):
        """Analyze symmetry across various metrics."""
        symmetry_scores = {}
        for joint_name, (left_idx, right_idx) in self.left_right_pairs.items():
            left_metric = metrics.get(f"left_{joint_name}", None)
            right_metric = metrics.get(f"right_{joint_name}", None)
            if left_metric is not None and right_metric is not None:
                symmetry = self.calculate_symmetry(left_metric, right_metric)
                symmetry_scores[joint_name] = symmetry
        return symmetry_scores