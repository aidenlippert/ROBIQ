class SymmetryAnalyzer:
    def __init__(self):
        # Pairs of left and right landmarks, defined by their indices
        self.left_right_pairs = {
            'shoulder': (11, 12),
            'elbow': (13, 14),
            'wrist': (15, 16),
            'hip': (23, 24),
            'knee': (25, 26),
            'ankle': (27, 28)
        }

    def calculate_symmetry(self, left_metric, right_metric):
        """
        Calculate symmetry ratio between left and right metrics.
        
        Args:
            left_metric (float): Metric value for the left side (e.g., joint angle or distance).
            right_metric (float): Metric value for the right side (e.g., joint angle or distance).
        
        Returns:
            float: Symmetry ratio (between 0.0 and 1.0), where 1.0 means perfect symmetry.
        """
        # Prevent division by zero and handle cases where both metrics are zero
        if left_metric == 0 and right_metric == 0:
            return 1.0  # Perfect symmetry if both sides are zero
        
        if left_metric + right_metric == 0:
            return 0.0  # This might represent a complete asymmetry (both sides are zero but unequal)

        # Symmetry is calculated as the ratio of the smaller to the larger metric
        symmetry_ratio = min(left_metric, right_metric) / max(left_metric, right_metric)
        return symmetry_ratio

    def analyze_symmetry(self, metrics):
        """
        Analyze symmetry across various joint metrics and return symmetry scores.
        
        Args:
            metrics (dict): A dictionary containing the metric values for both left and right sides.
                            Example format:
                            {
                                "left_shoulder": 45.0,
                                "right_shoulder": 46.0,
                                "left_knee": 80.0,
                                "right_knee": 80.0,
                                ...
                            }
        
        Returns:
            dict: A dictionary with the symmetry scores for each joint.
            Example:
            {
                'shoulder': 0.978,
                'elbow': 1.0,
                'wrist': 0.9,
                ...
            }
        """
        symmetry_scores = {}
        
        for joint_name, (left_idx, right_idx) in self.left_right_pairs.items():
            # Fetch the metrics for the left and right joint
            left_metric = metrics.get(f"left_{joint_name}")
            right_metric = metrics.get(f"right_{joint_name}")
            
            if left_metric is not None and right_metric is not None:
                # Calculate symmetry for this pair
                symmetry = self.calculate_symmetry(left_metric, right_metric)
                symmetry_scores[joint_name] = symmetry
        
        return symmetry_scores