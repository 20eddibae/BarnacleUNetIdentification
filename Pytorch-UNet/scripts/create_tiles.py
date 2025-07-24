#!/usr/bin/env python3

import os
import cv2
import numpy as np
from glob import glob
import random

def create_tiles(data_dir='data', tile_size=256, overlap=64, val_split=0.2):
    
    # Create output directories
    os.makedirs(f'{data_dir}/tiles/train/imgs', exist_ok=True)
    os.makedirs(f'{data_dir}/tiles/train/masks', exist_ok=True)
    os.makedirs(f'{data_dir}/tiles/val/imgs', exist_ok=True)
    os.makedirs(f'{data_dir}/tiles/val/masks', exist_ok=True)
    
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
        # Construct corresponding mask path
        mask_path = img_path.replace('imgs', 'masks').replace('img', 'mask')
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask {mask_path} not found, skipping {img_path}")
            continue
            
        print(f"  Processing {img_path}...")
        
        # Load image and mask
        img = cv2.imread(img_path)
        # Load mask in color
        mask_color = cv2.imread(mask_path)
        if img is None or mask_color is None:
            print(f"Warning: Could not load {img_path} or {mask_path}")
            continue

        # Detect exact blue as foreground
        bin_mask = cv2.inRange(mask_color, np.array([191, 57, 39]), np.array([191, 57, 39]))

        h, w = img.shape[:2]
        stride = tile_size - overlap

        tiles_count = 0
        debug_tiles_saved = 0

        # Create overlapping tiles
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                # Extract tile
                img_tile = img[y:y+tile_size, x:x+tile_size]
                mask_tile = bin_mask[y:y+tile_size, x:x+tile_size]

                # Debug output for the first 5 tiles
                if debug_tiles_saved < 5:
                    print(f"DEBUG: {img_path} tile {tiles_count} unique mask values: {np.unique(mask_tile)}, sum: {mask_tile.sum()}")
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    tile_name = f"{base_name}_tile_{tiles_count:04d}_debug"
                    cv2.imwrite(f'{data_dir}/tiles/{split}/imgs/{tile_name}.png', img_tile)
                    cv2.imwrite(f'{data_dir}/tiles/{split}/masks/{tile_name}.png', mask_tile)
                    debug_tiles_saved += 1

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