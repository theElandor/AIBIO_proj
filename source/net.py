import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torchvision import models
from source.utils import load_weights


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


class FcHead(nn.Module):
    def __init__(self, num_classes: int):
        super(FcHead, self).__init__()
        self.num_classes = num_classes
        # need to remove hardcoded 512,
        # need to take the output of the last layer of the backbone
        # head used for cell type classification, so 4 is hardcoded.
        self.fc = nn.Linear(512, self.num_classes)

    def forward(self, x):
        return self.fc(x)


class CellClassifier(nn.Module):
    """!CellClassifier is a class that combines the backbone and the head of the network."""

    def __init__(self, backbone: torch.nn.Module, head: torch.nn.Module):
        super(CellClassifier, self).__init__()
        self.backbone = backbone
        self.head = head

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def load_backbone_weights(self, config: dict, device: torch.cuda.device):
        assert 'backbone_weights' in config.keys(), "Please provide a valid checkpoint to load the backbone weights from."
        load_weights(config['backbone_weights'], self.backbone, device)
        print(f"Loaded the following backbone weights: {config['backbone_weights']}")

    def load_head_weights(self, config: dict, device: torch.cuda.device):
        assert 'head_weights' in config.keys(), "Please provide a valid checkpoint to load the head weights from."
        load_weights(config['head_weights'], self.head, device)
        print(f"Loaded the following head weights: {config['head_weights']}")

    def forward(self, x):
        # in simCLR they don't use the last layer of the backbone
        x = self.backbone.backbone(x)
        x = self.head(x)
        return x
