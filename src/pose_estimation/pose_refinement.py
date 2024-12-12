import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

class PoseRefinementModel(nn.Module):
    def __init__(self):
        super(PoseRefinementModel, self).__init__()
        
        # Updated architecture with more layers for feature extraction
        self.refine_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Increased channels
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling to reduce dimensions
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling to reduce dimensions
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 34 * 2)  # Adjusted to output 34 keypoints (x, y)
        )

    def forward(self, x):
        return self.refine_cnn(x)

class PoseRefiner:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PoseRefinementModel().to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        
        # Improved transformations: normalization and resizing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),  # Resize to a common dimension
            transforms.Normalize(mean=[0.485], std=[0.229])  # Example normalization (adjust as needed)
        ])

    def refine_pose(self, heatmap):
        """
        Input:
            - heatmap: 1-channel (grayscale) heatmap from pose estimation
        Returns:
            - refined keypoints [(x1, y1), (x2, y2), ...]
        """
        # Ensure input is in the correct format (e.g., grayscale 2D numpy array)
        if heatmap.ndim != 2:
            raise ValueError("Heatmap should be a 2D array (grayscale).")

        with torch.no_grad():
            # Apply the necessary transformations
            input_tensor = self.transform(heatmap).unsqueeze(0).to(self.device)
            
            # Forward pass through the model
            outputs = self.model(input_tensor)
            
            # Reshape and scale keypoints back to the original size
            refined_points = outputs.cpu().numpy().reshape(-1, 2)
            return refined_points