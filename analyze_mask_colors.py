import cv2
import numpy as np

mask_path = 'Pytorch-UNet/data/masks/val/mask2.png'
mask = cv2.imread(mask_path)
if mask is None:
    print(f'Could not load {mask_path}')
else:
    print('Mask shape:', mask.shape)
    unique_colors = np.unique(mask.reshape(-1, 3), axis=0)
    print('Unique BGR values in mask:')
    for color in unique_colors:
        print(color)
    print(f'Total unique colors: {len(unique_colors)}') 