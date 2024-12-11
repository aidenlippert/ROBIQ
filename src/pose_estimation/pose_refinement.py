import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

class PoseRefinementModel(nn.Module):
    def __init__(self):
        super(PoseRefinementModel, self).__init__()
        # Lightweight CNN for pose keypoint refinement
        self.refine_cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 32 * 32, 34 * 2)  # 34 keypoints x 2 (x, y)
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
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32))
        ])

    def refine_pose(self, heatmap):
        """
        Input:
            - heatmap: 1-channel (grayscale) heatmap from pose estimation
        Returns:
            - refined keypoints [(x1, y1), (x2, y2), ...]
        """
        with torch.no_grad():
            input_tensor = self.transform(heatmap).unsqueeze(0).to(self.device)
            outputs = self.model(input_tensor)
            refined_points = outputs.cpu().numpy().reshape(-1, 2)
        return refined_points