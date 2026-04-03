import torch
import torch.nn as nn
from typing import List
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List
from pathlib import Path

try:
    from .utils import get_loaders
except ImportError:  # pragma: no cover - supports direct script execution
    from utils import get_loaders

class AutoEncoder(nn.Module):
    def __init__(self, layers: List[int]):
        super().__init__()

        self.encoder = nn.Sequential(*[
            nn.Linear(input_dim, output_dim)
            for input_dim, output_dim in zip(layers[:-1], layers[1:])
        ])

        self.decoder = nn.Sequential(*[
            nn.Linear(input_dim, output_dim)
            for input_dim, output_dim in zip(layers[::-1][:-1], layers[::-1][1:])
        ])

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, y):
        return torch.norm(pred - y)

def train_AE(model: AutoEncoder,
             optimizer: torch.optim.Optimizer,
             loss: CustomLoss,
             train_loader,
             test_loader,
             plot: bool):
    
    epochs = 100
    train_losses = []
    test_losses = []
    
    for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
        model.train()
        running_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch", leave=False):
            x = batch[0]
            
            optimizer.zero_grad()
            pred = model(x)
            l = loss(pred, x)
            l.backward()
            optimizer.step()
            
            running_train_loss += l.item()
            
        train_losses.append(running_train_loss / len(train_loader))

        model.eval()
        running_test_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0]
                pred = model(x)
                l = loss(pred, x)
                running_test_loss += l.item()
                
        test_losses.append(running_test_loss / len(test_loader))

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(test_losses, label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        
    return model, train_losses, test_losses


