
import torch.nn as nn


class MetaNet(nn.Module):
    def __init__(self, feature_dim=64, hidden_dim=200, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.net(features)
