
import torch.nn as nn
from .resnet_cifar import ResNetCIFAR


class MainNet(nn.Module):
    def __init__(self, num_classes=10, feature_dim=64):
        super().__init__()
        self.backbone = ResNetCIFAR(feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x, return_features=False):
        features = self.extract_features(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits
