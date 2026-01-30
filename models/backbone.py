"""ResNet backbone for SSL"""

import torch.nn as nn
from torchvision.models import resnet50

class ResNetBackbone(nn.Module):
    """ResNet-50 backbone for feature extraction"""
    
    def __init__(self, output_dim=2048):
        super().__init__()
        self.output_dim = output_dim
        
        # Load ResNet50
        resnet = resnet50(pretrained=False)
        
        # Remove classification head
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
    
    def forward(self, x):
        """Extract features"""
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        return features
