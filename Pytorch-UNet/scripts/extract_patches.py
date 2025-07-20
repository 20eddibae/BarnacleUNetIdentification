#!/usr/bin/env python3
"""
Automated patch extraction for barnacle segmentation.
Extracts positive patches (barnacles) and negative patches (background) from training data.
"""

import os
import cv2
import numpy as np
from glob import glob
import argparse

def extract_patches(data_dir='data', min_area=20, patch_size=64):
    """
    Extract positive and negative patches from training data.
    
    Args:
        data_dir: Directory containing imgs/ and masks/ subdirectories
        min_area: Minimum area for a connected component to be considered a barnacle
        patch_size: Size of negative patches to extract
    """
    
    # Create output directories
    os.makedirs(f'{data_dir}/patches/positives', exist_ok=True)
    os.makedirs(f'{data_dir}/patches/negatives', exist_ok=True)
    
    # Get all mask files
    mask_paths = glob(f'{data_dir}/masks/train/*.png')
    
    total_positives = 0
    total_negatives = 0
    
    for mask_path in mask_paths:
        # Construct corresponding image path
        img_path = mask_path.replace('masks', 'imgs').replace('mask', 'img')
        
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping {mask_path}")
            continue
            
        print(f"Processing {mask_path}...")
        
        # Load image and mask
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            print(f"Warning: Could not load {img_path} or {mask_path}")
            continue
            
        # Binarize mask
        _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Find connected components (barnacles)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask)
        
        # Extract positive patches (barnacles)
        positives_count = 0
        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, w, h, area = stats[i]
            
            if area < min_area:
                continue  # Skip tiny components
                
            # Extract patch with some padding
            pad = 5
            y1 = max(0, y - pad)
            y2 = min(img.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(img.shape[1], x + w + pad)
            
            patch = img[y1:y2, x1:x2]
            
            if patch.size > 0:
                patch_name = f"{os.path.splitext(os.path.basename(mask_path))[0]}_pos_{i}.png"
                cv2.imwrite(f'{data_dir}/patches/positives/{patch_name}', patch)
                positives_count += 1
        
        total_positives += positives_count
        
        # Extract negative patches (background regions)
        negatives_count = 0
        h, w = mask.shape
        
        # Create a mask for background regions
        background_mask = (bin_mask == 0)
        
        attempts = 0
        max_attempts = positives_count * 10  # Increase attempts for better coverage
        
        while negatives_count < positives_count and attempts < max_attempts:
            # Random crop
            x0 = np.random.randint(0, w - patch_size)
            y0 = np.random.randint(0, h - patch_size)
            
            # Check if this region is mostly background (at least 50% background pixels)
            region = background_mask[y0:y0+patch_size, x0:x0+patch_size]
            if region.sum() / region.size > 0.5:
                patch = img[y0:y0+patch_size, x0:x0+patch_size]
                patch_name = f"{os.path.splitext(os.path.basename(mask_path))[0]}_neg_{negatives_count}.png"
                cv2.imwrite(f'{data_dir}/patches/negatives/{patch_name}', patch)
                negatives_count += 1
            
            attempts += 1
        
        total_negatives += negatives_count
        print(f"  Extracted {positives_count} positive and {negatives_count} negative patches")
    
    print(f"\nTotal extracted:")
    print(f"  Positive patches: {total_positives}")
    print(f"  Negative patches: {total_negatives}")
    print(f"  Total patches: {total_positives + total_negatives}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract patches from training data')
    parser.add_argument('--data_dir', default='data', help='Data directory')
    parser.add_argument('--min_area', type=int, default=20, help='Minimum area for barnacle detection')
    parser.add_argument('--patch_size', type=int, default=64, help='Size of negative patches')
    
    args = parser.parse_args()
    
    extract_patches(args.data_dir, args.min_area, args.patch_size) 