"""Self-supervised learning trainer"""

import torch
from torch.utils.data import DataLoader
from models.barlow_twins import BarlowTwins, TileDataset, get_barlow_transform
from models.backbone import ResNetBackbone
import numpy as np
from torchvision import transforms

def train_barlow_twins(tiles_np, batch_size=64, epochs=10, device='cuda'):
    """Train Barlow Twins SSL model"""
    
    # Create dataset
    transform = get_barlow_transform()
    dataset = TileDataset(tiles_np, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    backbone = ResNetBackbone().to(device)
    model = BarlowTwins(backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x1, x2 in dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            
            loss = model(x1, x2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    return backbone

def extract_representations(backbone, tiles_np, batch_size=64, device='cuda'):
    """Extract feature representations using trained backbone"""
    
    backbone.eval()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(tiles_np), batch_size):
            batch = [transform(img) for img in tiles_np[i:i+batch_size]]
            batch = torch.stack(batch).to(device)
            emb = backbone(batch).cpu().numpy()
            embeddings.append(emb)
    
    return np.vstack(embeddings)
