import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torchvision import models
from source.utils import load_weights, initialize_weights


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
class SimCLR34_norm(nn.Module):
    def __init__(self):
        super(SimCLR34_norm, self).__init__()
        self.backbone = models.resnet34(weights='DEFAULT')
        self.backbone.fc = nn.Identity()  # fully-connected removed
        self.projection_head = nn.Sequential(
            nn.Linear(512,256, bias=False),  
            nn.BatchNorm1d(256),  
            nn.ReLU(),
            nn.Linear(256, 128, bias=False),  
            nn.BatchNorm1d(128)  
        )

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return projections
    
class SimCLR50_norm(nn.Module):
    def __init__(self):
        super(SimCLR50_norm, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()  # fully-connected removed
        self.projection_head = nn.Sequential(
            nn.Linear(2048,1024, bias=False),  
            nn.BatchNorm1d(1024),  
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256, bias=False),  
            nn.BatchNorm1d(256)  
        )

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return projections
    
class SimCLR50_v2(nn.Module):
    '''
    This net discards the droppable projection head and incorporates a droppable classification head
    '''
    def __init__(self,drop_head:bool,num_classes:int,embedding_size = 512):
        super(SimCLR50_v2, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Linear(in_features = 2048,out_features = embedding_size)
        
        self.backbone = resnet
        self.classifier = nn.Linear(in_features = embedding_size,out_features = num_classes)
        self.drop_head = drop_head
        
        initialize_weights(self.classifier)
    def forward(self, x):
        embedding = self.backbone(x)
        if self.drop_head:
            return embedding
        else:
            output = self.classifier(embedding)
            return output
    
class FcHead(nn.Module):
    def __init__(self, num_classes: int):
        super(FcHead, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(512, self.num_classes)

    def forward(self, x):
        return self.fc(x)
    
class FcHead50(nn.Module):
    def __init__(self, num_classes: int):
        super(FcHead50, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        return self.fc(x)


class ResNet(nn.Module):
    """!Simple resnet to try end-to-end classification of siRNA or cell type."""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes        
        super(ResNet, self).__init__()
        self.resnet = models.resnet101(weights='DEFAULT')
        self.resnet.fc = nn.Identity()  # fully-connected removed
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes)
        )
    def forward(self, x):
        x = self.resnet(x)
        logits = self.fc(x)
        return logits        


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
