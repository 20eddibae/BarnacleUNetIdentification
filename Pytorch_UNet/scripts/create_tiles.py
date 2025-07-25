#!/usr/bin/env python3

import os
import cv2
import numpy as np
from glob import glob
import random
import shutil

def create_tiles(data_dir='data', tile_size=128, overlap=32, val_split=0.2):
    
    # Delete previous tiles
    for split in ['train', 'val']:
        for sub in ['imgs', 'masks']:
            dir_path = f'{data_dir}/tiles/{split}/{sub}'
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)
    
    # Process training data
    print("Processing training data...")
    train_tiles = process_split(data_dir, 'train', tile_size, overlap)
    
    # Process validation data
    print("Processing validation data...")
    val_tiles = process_split(data_dir, 'val', tile_size, overlap)
    
    print(f"\nTotal tiles created:")
    print(f"  Training tiles: {train_tiles}")
    print(f"  Validation tiles: {val_tiles}")
    print(f"  Total tiles: {train_tiles + val_tiles}")

def process_split(data_dir, split, tile_size, overlap):
    """Process a specific split (train/val) of the data."""
    
    # Get all image files for this split
    img_paths = glob(f'{data_dir}/imgs/{split}/*.png')
    
    total_tiles = 0
    
    for img_path in img_paths:
        # Construct corresponding mask path for mask1.png, mask2.png, etc.
        base = os.path.basename(img_path)
        mask_base = base.replace('img', 'mask')
        mask_path = os.path.join(os.path.dirname(img_path).replace('imgs', 'masks'), mask_base)
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask {mask_path} not found, skipping {img_path}")
            continue
            
        print(f"  Processing {img_path}...")
        
        # Load image and mask
        img = cv2.imread(img_path)
        mask_color = cv2.imread(mask_path)
        # Resize mask to match image size if needed
        if mask_color.shape[:2] != img.shape[:2]:
            print(f"Resizing mask from {mask_color.shape[:2]} to {img.shape[:2]} for {mask_path}")
            mask_color = cv2.resize(mask_color, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if img is None or mask_color is None:
            print(f"Warning: Could not load {img_path} or {mask_path}")
            continue

        # Convert all blue (on black) in the mask to white (255) on black (0)
        hsv = cv2.cvtColor(mask_color, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 50, 50])  # Lower bound for blue in HSV
        upper_blue = np.array([140, 255, 255])  # Upper bound for blue in HSV
        bin_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        print(f"Unique values in bin_mask for {mask_path}: {np.unique(bin_mask)}")

        h, w = img.shape[:2]
        stride = tile_size - overlap

        tiles_count = 0

        # Find bounding box of the annotated region in the mask
        ys, xs = np.where(bin_mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
        else:
            print(f"No annotation found in mask for {img_path}, skipping.")
            continue

        # Create overlapping tiles only within the bounding box
        for y in range(y_min, y_max - tile_size + 1, stride):
            for x in range(x_min, x_max - tile_size + 1, stride):
                img_tile = img[y:y+tile_size, x:x+tile_size]
                mask_tile = bin_mask[y:y+tile_size, x:x+tile_size]
                # Only save tiles that have some barnacles (positive examples)
                if mask_tile.sum() > 0:
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    tile_name = f"{base_name}_tile_{tiles_count:04d}"
                    cv2.imwrite(f'{data_dir}/tiles/{split}/imgs/{tile_name}.png', img_tile)
                    cv2.imwrite(f'{data_dir}/tiles/{split}/masks/{tile_name}.png', mask_tile)
                    tiles_count += 1
        
        total_tiles += tiles_count
        print(f"    Created {tiles_count} tiles")
    
    return total_tiles

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create tiled patches from full images')
    parser.add_argument('--data_dir', default='data', help='Data directory')
    parser.add_argument('--tile_size', type=int, default=256, help='Size of tiles')
    parser.add_argument('--overlap', type=int, default=64, help='Overlap between tiles')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    
    args = parser.parse_args()
    
    create_tiles(args.data_dir, args.tile_size, args.overlap, args.val_split) 