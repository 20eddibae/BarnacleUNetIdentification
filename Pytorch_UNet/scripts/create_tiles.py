#!/usr/bin/env python3

import os
import cv2
import numpy as np
from glob import glob

def create_tiles(data_dir='data', tile_size=128, overlap=32):
    # Create output directories for all tiles
    all_img_dir = f'{data_dir}/tiles/all/imgs'
    all_mask_dir = f'{data_dir}/tiles/all/masks'
    os.makedirs(all_img_dir, exist_ok=True)
    os.makedirs(all_mask_dir, exist_ok=True)

    # Remove any existing tiles in all/
    for d in [all_img_dir, all_mask_dir]:
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))

    # Get all image files (from both train and val source images)
    img_paths = glob(f'{data_dir}/imgs/train/*.png') + glob(f'{data_dir}/imgs/val/*.png')
    total_tiles = 0
    for img_path in img_paths:
        base = os.path.basename(img_path)
        # Construct corresponding mask path: replace 'img' with 'mask' in filename
        mask_base = base.replace('img', 'mask')
        if 'train' in img_path:
            mask_path = os.path.join(data_dir, 'masks', 'train', mask_base)
        elif 'val' in img_path:
            mask_path = os.path.join(data_dir, 'masks', 'val', mask_base)
        else:
            print(f"Warning: Could not determine split for {img_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"Warning: Mask {mask_path} not found, skipping {img_path}")
            continue
        print(f"  Processing {img_path}...")
        img = cv2.imread(img_path)
        mask_color = cv2.imread(mask_path)
        if mask_color.shape[:2] != img.shape[:2]:
            print(f"Resizing mask from {mask_color.shape[:2]} to {img.shape[:2]} for {mask_path}")
            mask_color = cv2.resize(mask_color, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if img is None or mask_color is None:
            print(f"Warning: Could not load {img_path} or {mask_path}")
            continue
        hsv = cv2.cvtColor(mask_color, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        bin_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        h, w = img.shape[:2]
        stride = tile_size - overlap
        tiles_count = 0
        ys, xs = np.where(bin_mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
        else:
            print(f"No annotation found in mask for {img_path}, skipping.")
            continue
        for y in range(y_min, y_max - tile_size + 1, stride):
            for x in range(x_min, x_max - tile_size + 1, stride):
                img_tile = img[y:y+tile_size, x:x+tile_size]
                mask_tile = bin_mask[y:y+tile_size, x:x+tile_size]
                if mask_tile.sum() > 0:
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    tile_name = f"{base_name}_tile_{tiles_count:04d}.png"
                    cv2.imwrite(os.path.join(all_img_dir, tile_name), img_tile)
                    cv2.imwrite(os.path.join(all_mask_dir, tile_name), mask_tile)
                    tiles_count += 1
        total_tiles += tiles_count
        print(f"    Created {tiles_count} tiles")
    print(f"\nTotal tiles created: {total_tiles}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create all tiled patches from full images')
    parser.add_argument('--data_dir', default='data', help='Data directory')
    parser.add_argument('--tile_size', type=int, default=128, help='Size of tiles')
    parser.add_argument('--overlap', type=int, default=32, help='Overlap between tiles')
    args = parser.parse_args()
    create_tiles(args.data_dir, args.tile_size, args.overlap) 