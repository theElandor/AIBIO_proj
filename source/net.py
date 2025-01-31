import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torchvision import models


class SimCLR(nn.Module):
    def __init__(self):
        super(SimCLR, self).__init__()
        self.backbone = models.resnet18(weights='DEFAULT')
        self.backbone.fc = nn.Identity()  # fully-connected removed
        self.projection_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return projections
