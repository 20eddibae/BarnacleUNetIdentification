#!/usr/bin/env python3

import os
import cv2
import numpy as np
from glob import glob

def verify_setup():
    
    print("Verifying Barnacle Segmentation Setup\n")
    
    # Check directory structure
    print("Directory Structure:")
    required_dirs = [
        'data/imgs/train',
        'data/imgs/val', 
        'data/masks/train',
        'data/masks/val',
        'data/patches/positives',
        'data/patches/negatives',
        'data/tiles/train/imgs',
        'data/tiles/train/masks',
        'data/tiles/val/imgs',
        'data/tiles/val/masks',
        'notebooks',
        'models',
        'scripts'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - MISSING")
    
    # Check data files
    print("\nData Files:")
    
    # Original images and masks
    train_imgs = len(glob('data/imgs/train/*.png'))
    train_masks = len(glob('data/masks/train/*.png'))
    val_imgs = len(glob('data/imgs/val/*.png'))
    val_masks = len(glob('data/masks/val/*.png'))
    
    print(f"  Original training images: {train_imgs}")
    print(f"  Original training masks: {train_masks}")
    print(f"  Original validation images: {val_imgs}")
    print(f"  Original validation masks: {val_masks}")
    
    # Patches
    positive_patches = len(glob('data/patches/positives/*.png'))
    negative_patches = len(glob('data/patches/negatives/*.png'))
    
    print(f"  Positive patches: {positive_patches}")
    print(f"  Negative patches: {negative_patches}")
    
    # Tiles
    train_tiles = len(glob('data/tiles/train/imgs/*.png'))
    train_tile_masks = len(glob('data/tiles/train/masks/*.png'))
    val_tiles = len(glob('data/tiles/val/imgs/*.png'))
    val_tile_masks = len(glob('data/tiles/val/masks/*.png'))
    
    print(f"  Training tiles: {train_tiles}")
    print(f"  Training tile masks: {train_tile_masks}")
    print(f"  Validation tiles: {val_tiles}")
    print(f"  Validation tile masks: {val_tile_masks}")
    
    # Check mask quality
    print("\nMask Quality Check:")
    try:
        mask_path = 'data/masks/train/mask1.png'
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        unique_values = np.unique(bin_mask)
        barnacle_pixels = (bin_mask == 255).sum()
        total_pixels = bin_mask.size
        barnacle_ratio = barnacle_pixels / total_pixels
        
        print(f"  Mask shape: {mask.shape}")
        print(f"  Unique values: {unique_values}")
        print(f"  Barnacle pixels: {barnacle_pixels:,}")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Barnacle ratio: {barnacle_ratio:.3f}")
        
        if len(unique_values) == 2 and 0 in unique_values and 255 in unique_values:
            print("  Mask is properly binarized")
        else:
            print("  Mask may need binarization")
            
    except Exception as e:
        print(f"  ❌ Error checking mask: {e}")
    
    # Check tile quality
    print("\nTile Quality Check:")
    try:
        tile_path = 'data/tiles/train/imgs/img1_tile_0000.png'
        tile_mask_path = 'data/tiles/train/masks/img1_tile_0000.png'
        
        if os.path.exists(tile_path) and os.path.exists(tile_mask_path):
            tile = cv2.imread(tile_path)
            tile_mask = cv2.imread(tile_mask_path, cv2.IMREAD_GRAYSCALE)
            
            print(f"  Tile shape: {tile.shape}")
            print(f"  Tile mask shape: {tile_mask.shape}")
            print(f"  Tile mask unique values: {np.unique(tile_mask)}")
            
            if tile.shape[:2] == tile_mask.shape:
                print("  Tile and mask dimensions match")
            else:
                print("  Tile and mask dimensions mismatch")
        else:
            print("  Sample tiles not found")
            
    except Exception as e:
        print(f"  ❌ Error checking tiles: {e}")
    
    # Check scripts
    print("\nScripts:")
    scripts = [
        'scripts/extract_patches.py',
        'scripts/create_tiles.py',
        'scripts/debug_mask.py',
        'scripts/verify_setup.py'
    ]
    
    for script in scripts:
        if os.path.exists(script):
            print(f"  {script}")
        else:
            print(f"  {script} - MISSING")
    
    # Check notebook
    print("\nNotebook:")
    notebook_path = 'notebooks/barnacle_unet.ipynb'
    if os.path.exists(notebook_path):
        print(f"  {notebook_path}")
    else:
        print(f"  {notebook_path} - MISSING")
    
    # Summary
    print("\nSummary:")
    total_training_data = positive_patches + train_tiles
    print(f"  Total training samples: {total_training_data}")
    print(f"  Individual barnacle patches: {positive_patches}")
    print(f"  Tiled training samples: {train_tiles}")
    print(f"  Validation samples: {val_tiles}")
    
    if total_training_data > 1000:
        print("  Sufficient training data available")
    else:
        print("  Limited training data - consider adding more annotations")
    
    print("\nReady for Training!")
    print("  Run: jupyter notebook notebooks/barnacle_unet.ipynb")

if __name__ == "__main__":
    verify_setup() 