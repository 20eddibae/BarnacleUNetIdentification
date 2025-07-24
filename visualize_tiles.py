import cv2
import matplotlib.pyplot as plt
import numpy as np

tiles = [
    'Pytorch-UNet/data/tiles/val/masks/img2_tile_0000_debug.png',
    'Pytorch-UNet/data/tiles/val/masks/img2_tile_0001_debug.png',
    'Pytorch-UNet/data/tiles/val/masks/img2_tile_0002_debug.png',
    'Pytorch-UNet/data/tiles/val/masks/img2_tile_0003_debug.png',
    'Pytorch-UNet/data/tiles/val/masks/img2_tile_0004_debug.png',
]

fig, axs = plt.subplots(1, len(tiles), figsize=(15, 3))
for i, tile in enumerate(tiles):
    mask = cv2.imread(tile, cv2.IMREAD_GRAYSCALE)
    axs[i].imshow(mask, cmap='gray', vmin=0, vmax=255)
    axs[i].set_title(f'Tile {i}')
    axs[i].axis('off')
plt.tight_layout()
plt.savefig('tile_visualization_blue_debug_grid.png')
plt.show() 