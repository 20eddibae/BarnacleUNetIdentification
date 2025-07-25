import os
import random
import shutil

# Directories
all_img_dir = 'Pytorch_UNet/data/tiles/all/imgs'
all_mask_dir = 'Pytorch_UNet/data/tiles/all/masks'
train_img_dir = 'Pytorch_UNet/data/tiles/train/imgs'
train_mask_dir = 'Pytorch_UNet/data/tiles/train/masks'
val_img_dir = 'Pytorch_UNet/data/tiles/val/imgs'
val_mask_dir = 'Pytorch_UNet/data/tiles/val/masks'

# List all tile filenames (assume all masks have matching names)
all_tiles = sorted([f for f in os.listdir(all_img_dir) if f.endswith('.png')])
random.shuffle(all_tiles)

split_idx = int(0.8 * len(all_tiles))
train_tiles = all_tiles[:split_idx]
val_tiles = all_tiles[split_idx:]

# Helper to clear a directory
def clear_dir(d):
    for f in os.listdir(d):
        path = os.path.join(d, f)
        if os.path.isfile(path):
            os.remove(path)

# Helper to copy tiles
def copy_tiles(tile_list, img_src, mask_src, img_dst, mask_dst):
    for tile in tile_list:
        shutil.copy(os.path.join(img_src, tile), os.path.join(img_dst, tile))
        shutil.copy(os.path.join(mask_src, tile), os.path.join(mask_dst, tile))

# Clear out old train/val folders
for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
    clear_dir(d)

# Copy new splits
copy_tiles(train_tiles, all_img_dir, all_mask_dir, train_img_dir, train_mask_dir)
copy_tiles(val_tiles, all_img_dir, all_mask_dir, val_img_dir, val_mask_dir)

print(f"Total tiles: {len(all_tiles)}")
print(f"Training tiles: {len(train_tiles)}")
print(f"Validation tiles: {len(val_tiles)}") 