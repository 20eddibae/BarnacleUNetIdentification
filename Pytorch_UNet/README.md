# Barnacle Segmentation Challenge

This project tackles the problem of automating barnacle counting in coastal tide pool images, as described in the DALI Data Challenge. The goal is to help scientists process images faster by segmenting barnacles using both deep learning and classical computer vision approaches.

## Approach

- **Deep Learning:**
  - Used a U-Net model (PyTorch) for pixel-wise segmentation.
  - Only two annotated images were available, so we split them into many smaller tiles, focusing on regions with barnacles.
  - Randomized the train/validation split to avoid overfitting to a single image.
  - Evaluated the model on both held-out tiles and full-size unseen images.

## Project Structure

- `notebooks/barnacle_unet.ipynb` — Main notebook: data loading, training, evaluation, and visualizations.
- `scripts/create_tiles.py` — Slices large images/masks into smaller, aligned tiles for training.
- `scripts/random_tile_split.py` — Randomly splits tiles into train/val sets.
- `unet/` — U-Net model (code taken from original repo here: https://github.com/milesial/Pytorch-UNet)
- `test_unseen_images.py` — Runs the trained model on full-size unseen images.
- `opencv-barnacle/pipeline.py` — Classical OpenCV pipeline for comparison.
- `data/` — Contains all images, masks, and generated tiles.

## Visualizations/Evaluation of Results 

- `training_curves.png` — Shows loss and Dice score during training.
- `sample_predictions.png` — Example predictions on validation tiles.
- `unseen_predictions.png` — Model output on full-size unseen images.

`Note: these are in the notebooks subdirectory`

## Hurdles & Solutions

- **Limited Data:** Only two labeled images. Solved by tiling and careful data splitting.
- **Mask Alignment:** Masks were a different scale and only covered the center. Fixed by resizing and tiling only annotated regions.
- **Overfitting:** Prevented by mixing tiles from both images in train/val sets.
- **Classical CV Limitations:** Color thresholding alone was not robust enough for barnacle detection.

## How to Run

1. Install dependencies (see requirements.txt).
2. Run `scripts/create_tiles.py` and `scripts/random_tile_split.py` to prepare data.
3. Train the model in the notebook (`notebooks/barnacle_unet.ipynb`).
4. Run `test_unseen_images.py` to test on new images.

## Reflections

This project demonstrates a practical approach to a hard segmentation problem with minimal data. The U-Net model, while not perfect, shows promise and outperforms basic OpenCV methods. Key learnings included robust data preprocessing, careful validation, and the importance of visual inspection at every step.


### Citations
Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox:

U-Net: Convolutional Networks for Biomedical Image Segmentation
