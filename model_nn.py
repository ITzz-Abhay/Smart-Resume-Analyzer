import torch.nn as nn
import torch.nn.functional as F

class MyNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 25)  # 25 classes
        )
    def forward(self, x):
        return self.model(x)
