#!/usr/bin/env python3
"""
Barnacle Segmentation Training Script
Runs the complete U-Net training pipeline for barnacle detection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from tqdm import tqdm
import sys

# Add the unet module to path
sys.path.append('unet')
from unet_model import UNet

def main():
    print("Starting Barnacle Segmentation Training")
    print("=" * 50)
    
    # Check device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check data organization
    print("\nData Overview:")
    train_tiles = len(glob('data/tiles/train/imgs/*.png'))
    val_tiles = len(glob('data/tiles/val/imgs/*.png'))
    positive_patches = len(glob('data/patches/positives/*.png'))
    
    print(f"Training tiles: {train_tiles}")
    print(f"Validation tiles: {val_tiles}")
    print(f"Individual barnacle patches: {positive_patches}")
    
    # Dataset class
    class BarnacleDataset(Dataset):
        def __init__(self, data_dir, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            
            # Get all image files
            self.img_files = sorted(glob(os.path.join(data_dir, 'imgs', '*.png')))
            
        def __len__(self):
            return len(self.img_files)
        
        def __getitem__(self, idx):
            img_path = self.img_files[idx]
            mask_path = img_path.replace('imgs', 'masks')
            
            # Load image and mask
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Binarize mask
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = mask / 255.0  # Normalize to [0, 1]
            
            # Convert to tensors
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            
            if self.transform:
                img = self.transform(img)
                mask = self.transform(mask)
            
            return img, mask
    
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = BarnacleDataset('data/tiles/train', transform=train_transform)
    val_dataset = BarnacleDataset('data/tiles/val')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    print("\nInitializing U-Net model...")
    model = UNet(n_channels=3, n_classes=1)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training functions
    def dice_score(pred, target):
        """Calculate Dice score"""
        pred = torch.sigmoid(pred) > 0.5
        target = target > 0.5
        
        intersection = (pred & target).sum()
        union = pred.sum() + target.sum()
        
        return (2 * intersection) / (union + 1e-6)
    
    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        total_loss = 0
        total_dice = 0
        
        for imgs, masks in tqdm(loader, desc="Training"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_dice += dice_score(outputs, masks).item()
        
        return total_loss / len(loader), total_dice / len(loader)
    
    def validate_epoch(model, loader, criterion, device):
        model.eval()
        total_loss = 0
        total_dice = 0
        
        with torch.no_grad():
            for imgs, masks in tqdm(loader, desc="Validation"):
                imgs = imgs.to(device)
                masks = masks.to(device)
                
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                
                total_loss += loss.item()
                total_dice += dice_score(outputs, masks).item()
        
        return total_loss / len(loader), total_dice / len(loader)
    
    # Training loop
    print("\nStarting training...")
    num_epochs = 20  # Reduced for faster training
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    
    best_val_dice = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_dice = validate_epoch(model, val_loader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), 'models/best_barnacle_unet.pth')
            print(f"New best model saved! Dice: {val_dice:.4f}")
        
        # Early stopping
        if epoch > 5 and val_dice < 0.1:
            print("Early stopping due to poor performance")
            break
    
    # Training summary
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best validation Dice score: {best_val_dice:.4f}")
    print(f"Model saved to: models/best_barnacle_unet.pth")
    
    # Plot training curves
    print("\n📈 Generating training plots...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_dices, label='Train Dice')
    plt.plot(val_dices, label='Val Dice')
    plt.title('Training and Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved to: training_curves.png")
    
    # Test on validation data
    print("\n🔍 Testing on validation data...")
    model.load_state_dict(torch.load('models/best_barnacle_unet.pth'))
    model.eval()
    
    test_batch = next(iter(val_loader))
    test_imgs, test_masks = test_batch
    
    with torch.no_grad():
        test_imgs = test_imgs.to(device)
        predictions = model(test_imgs)
        predictions = torch.sigmoid(predictions)
    
    # Calculate final metrics
    final_dice = dice_score(predictions, test_masks.to(device)).item()
    print(f"Final Dice Score: {final_dice:.4f}")
    
    # Save sample predictions
    print("\n💾 Saving sample predictions...")
    plt.figure(figsize=(15, 12))
    for i in range(4):
        plt.subplot(4, 3, i*3+1)
        plt.imshow(test_imgs[i].cpu().permute(1, 2, 0))
        plt.title(f'Input Image {i+1}')
        plt.axis('off')
        
        plt.subplot(4, 3, i*3+2)
        plt.imshow(test_masks[i].squeeze().cpu(), cmap='gray')
        plt.title(f'Ground Truth {i+1}')
        plt.axis('off')
        
        plt.subplot(4, 3, i*3+3)
        plt.imshow(predictions[i].squeeze().cpu(), cmap='gray')
        plt.title(f'Prediction {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
    print("Sample predictions saved to: sample_predictions.png")
    
    print("\n🎉 Training pipeline complete!")
    print("Files generated:")
    print("- models/best_barnacle_unet.pth (trained model)")
    print("- training_curves.png (training progress)")
    print("- sample_predictions.png (sample results)")

if __name__ == "__main__":
    main() 