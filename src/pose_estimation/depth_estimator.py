import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize

class DepthEstimator:
    def __init__(self, model_type='DPT_Large'):
        # Use MiDaS v3.1 (best version for monocular depth estimation)
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_v3.1")
        self.model.eval()
        self.transforms = Compose([
            Resize((384, 384)),
            ToTensor()
        ])

    def estimate_depth(self, frame):
        """
        Input:
            - frame: BGR image from OpenCV
        Returns:
            - depth_map: Numpy array (grayscale depth map)
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transforms(img_rgb).unsqueeze(0)
        with torch.no_grad():
            depth_map = self.model(input_tensor).squeeze().numpy()
        return depth_map