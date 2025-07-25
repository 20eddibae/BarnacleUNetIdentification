import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import matplotlib.pyplot as plt
import numpy as np
from Pytorch_UNet.train_barnacle_unet import BarnacleDataset
from torch.utils.data import DataLoader

# Load the dataset
train_dataset = BarnacleDataset('../data/tiles/train')
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

# Get a batch
imgs, masks = next(iter(train_loader))

# Print unique values and coverage for each mask in the batch
for i, mask in enumerate(masks):
    unique = torch.unique(mask)
    coverage = (mask > 0.5).float().mean().item() * 100
    print(f"Mask {i}: unique values: {unique.tolist()}, barnacle coverage: {coverage:.2f}%")

# Visualize the first 4 image-mask pairs
fig, axs = plt.subplots(4, 2, figsize=(8, 12))
for i in range(4):
    axs[i, 0].imshow(imgs[i].permute(1, 2, 0).cpu().numpy())
    axs[i, 0].set_title(f'Image {i}')
    axs[i, 0].axis('off')
    axs[i, 1].imshow(masks[i].squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axs[i, 1].set_title(f'Mask {i}')
    axs[i, 1].axis('off')
plt.tight_layout()
plt.savefig('notebook_mask_batch_debug.png')
plt.show() 