import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class PoseRefinementModel(nn.Module):
    def __init__(self):
        super(PoseRefinementModel, self).__init__()
        # Using pretrained ResNet18 as the backbone for feature extraction
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 34 * 2)  # 34 keypoints (x, y)

    def forward(self, x):
        return self.backbone(x)

class PoseRefiner:
    def __init__(self, model_path=None, alpha=0.9):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PoseRefinementModel().to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.RandomRotation(30),  # Random rotation up to 30 degrees
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandomAffine(10),  # Random affine transformation
            transforms.Normalize(mean=[0.485], std=[0.229])  # Normalization
        ])
        self.alpha = alpha  # EMA smoothing factor
        self.prev_keypoints = None

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

            # Apply EMA smoothing to the refined keypoints
            if self.prev_keypoints is None:
                self.prev_keypoints = refined_points
            else:
                self.prev_keypoints = self.alpha * refined_points + (1 - self.alpha) * self.prev_keypoints
        return self.prev_keypoints