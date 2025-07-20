#!/usr/bin/env python3
import cv2
import numpy as np

# Load mask
mask_path = 'data/masks/train/mask1.png'
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

print(f"Mask shape: {mask.shape}")
print(f"Mask unique values: {np.unique(mask)}")
print(f"Mask min/max: {mask.min()}/{mask.max()}")

# Binarize mask
_, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
print(f"Binary mask unique values: {np.unique(bin_mask)}")

# Check background regions
background_mask = (bin_mask == 0)
print(f"Background pixels: {background_mask.sum()}")
print(f"Total pixels: {background_mask.size}")
print(f"Background ratio: {background_mask.sum() / background_mask.size:.3f}")

# Check if we can find background regions
h, w = mask.shape
patch_size = 64
found_background = False

for attempt in range(100):
    x0 = np.random.randint(0, w - patch_size)
    y0 = np.random.randint(0, h - patch_size)
    
    region = background_mask[y0:y0+patch_size, x0:x0+patch_size]
    background_ratio = region.sum() / region.size
    
    if background_ratio > 0.9:
        print(f"Found background region at ({x0}, {y0}) with ratio {background_ratio:.3f}")
        found_background = True
        break

if not found_background:
    print("No suitable background regions found with 90% threshold")
    
    # Try with lower threshold
    for attempt in range(100):
        x0 = np.random.randint(0, w - patch_size)
        y0 = np.random.randint(0, h - patch_size)
        
        region = background_mask[y0:y0+patch_size, x0:x0+patch_size]
        background_ratio = region.sum() / region.size
        
        if background_ratio > 0.5:
            print(f"Found background region at ({x0}, {y0}) with ratio {background_ratio:.3f}")
            break 