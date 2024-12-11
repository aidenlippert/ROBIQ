import torch
import torch.nn as nn
import numpy as np

class ActivityRecognitionModel(nn.Module):
    def __init__(self, input_size=68, hidden_size=128, num_layers=2, num_classes=5):
        super(ActivityRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])
        return out

class ActivityRecognizer:
    def __init__(self, model_path=None, num_classes=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActivityRecognitionModel(num_classes=num_classes).to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        self.num_classes = num_classes

    def predict_activity(self, keypoints_sequence):
        """
        Input:
            - keypoints_sequence: Numpy array (sequence_length x num_keypoints*2)
        Returns:
            - Predicted activity label (class index)
        """
        with torch.no_grad():
            input_tensor = torch.tensor(keypoints_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()