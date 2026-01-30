"""Barlow Twins self-supervised learning implementation"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

class BarlowTwins(nn.Module):
    """Barlow Twins SSL model"""
    
    def __init__(self, backbone, projection_dim=2048, lambda_param=5e-3):
        super().__init__()
        self.backbone = backbone
        self.lambda_param = lambda_param
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(backbone.output_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, x1, x2):
        """Compute Barlow Twins loss"""
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))
        
        # Normalize
        z1 = (z1 - z1.mean(0)) / z1.std(0)
        z2 = (z2 - z2.mean(0)) / z2.std(0)
        
        # Cross-correlation matrix
        c = (z1.T @ z2) / z1.shape[0]
        
        # Loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambda_param * off_diag
        
        return loss
    
    @staticmethod
    def off_diagonal(x):
        """Return off-diagonal elements of matrix"""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class TileDataset(Dataset):
    """Dataset for tile-based SSL"""
    
    def __init__(self, tiles, transform=None):
        self.tiles = tiles
        self.transform = transform
    
    def __getitem__(self, idx):
        tile = self.tiles[idx]
        x1 = self.transform(tile)
        x2 = self.transform(tile)
        return x1, x2
    
    def __len__(self):
        return len(self.tiles)

def get_barlow_transform():
    """Get augmentation transform for Barlow Twins"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
