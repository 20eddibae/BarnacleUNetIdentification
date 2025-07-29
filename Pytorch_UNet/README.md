# Barnacle Segmentation & Counting Challenge

This project tackles the problem of automating barnacle counting in coastal tide pool images, as described in the DALI Data Challenge. The goal is to help scientists process images faster by both segmenting and counting barnacles using both deep learning and classical computer vision approaches.

## Approach

- Deep Learning (U-Net):
  - Used a U-Net model (PyTorch) for pixel-wise segmentation.
  - Only two annotated images were available, so I split them into many smaller tiles, focusing on regions with barnacles.
  - Randomized the train/validation split to avoid overfitting to a single image.
  - Post-processing: Connected component analysis to count individual barnacles from segmentation masks.
  - Evaluated the model on both held-out tiles and full-size unseen images.

## Evaluation Metrics

- **Segmentation Quality**: Dice score (intersection over union) for pixel-level accuracy
- **Counting Accuracy**: Number of detected barnacles vs. ground truth count
- **Processing Speed**: Time to process a single image

## Project Structure (besides unet all which is code I created)

- `notebooks/barnacle_unet.ipynb` — Main notebook: data loading, training, evaluation, and visualizations (had some LLM help with some of the cells)
- `scripts/create_tiles.py` — Slices large images/masks into smaller, aligned tiles for training.
- `scripts/random_tile_split.py` — Randomly splits tiles into train/val sets.
- `unet/` — U-Net model (code taken from original repo here: https://github.com/milesial/Pytorch-UNet)
- `test_unseen_images.py` — Runs the trained model on full-size unseen images (NOTE: used gpt to generate to test)
- `data/` — Contains all images, masks, and generated tiles.

## Training Results

The U-Net model achieved excellent performance on the training data:

Final Training Values:
- **Train Loss**: 0.1023
- **Train Dice**: 0.8904
- **Val Loss**: 0.2173
- **Val Dice**: 0.6921

The training curves show consistent improvement in both loss and Dice score, with the model learning effectively from the limited training data. Visualizations of the training progress are available in the Jupyter notebook.

## Barnacle Counting Results

The segmentation model was successfully applied to count barnacles in unseen images:

Counting Results:
- **unseen_prediction_1.png**: 2,101 barnacles detected
- **unseen_prediction_2.png**: 139 barnacles detected

The counting visualization shows:
- **Original Image**: Raw input image with barnacles
- **Segmentation Mask**: Binary mask output from U-Net
- **Detected Barnacles**: Individual barnacles labeled with different colors and counted
`Note: this visualization is available in the notebooks folder with "unseen_prediction.png" showing the total segmentation results `

Complete visualizations of the counting results are available in the Jupyter notebook.

## Visualizations/Evaluation of Results 

Under notebooks directory is the segmentation/counting results
In particular: 
-  **unseen_predictions.png** - shows the total segmentation images on the unseen samples
-  **unseen_counting.png** - shows the total counting on the segmentation images of the unseen samples 

All visualizations are available in the Jupyter notebook (`notebooks/barnacle_unet.ipynb`):
- Training curves showing loss and Dice score progression
- Barnacle counting results on unseen images
- Sample predictions on validation tiles
- Model output on full-size unseen images

## Hurdles & Solutions

- **Limited Data:** Only two labeled images. Solved by tiling and careful data splitting.
- **Mask Alignment:** Masks were a different scale and only covered the center. Fixed by resizing and tiling only annotated regions.
- **Overfitting:** The training images were vastly different so I fixed this by mixing tiles from both images in train/val sets.

## How to Run

1. Install dependencies (see requirements.txt).
2. Run `scripts/create_tiles.py` and `scripts/random_tile_split.py` to prepare data.
3. Train the model in the notebook (`notebooks/barnacle_unet.ipynb`).
4. Run `test_unseen_images.py` to test on new images.

## Results & Conclusions

- **U-Net Approach**: Achieved excellent training performance with Train Dice: 0.8904 and Val Dice: 0.6921. The model successfully counted 2,101 barnacles in the first unseen image and 139 in the second, which demonstrated the effectiveness of the segmentation-to-counting pipeline.
- **Counting Accuracy**: The U-Net approach provides automated counting that can significantly speed up scientific analysis, though manual verification is still recommended for critical applications.
- **Practical Application**: This system could dramatically improve the scientists' workflow efficiency by providing accurate initial estimates that can be quickly verified and corrected.

## Reflections

This project demonstrates a practical approach to a hard segmentation and counting problem with minimal data. The U-Net model achieved excellent performance, successfully counting thousands of barnacles across different image types. This was a really interesting project to apply my theoretical understanding of UNet segmentation that I know from medical imaging into this real life context. Through this approach I learned a lot about data preprocessing, careful validation, and the importance of visual inspection at every step. In particular, I learned that the most important quality in deep learning models is the data and how clean it is and that is what I spent the most time on in this project.

The key insight is that while full automation might not be achievable with limited training data, a semi-automated system that provides good initial estimates can still dramatically improve the scientists' workflow efficiency. The successful counting of 2,101 barnacles in one image and 139 in another demonstrates the practical value of this approach.

### Citations
Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox:

U-Net: Convolutional Networks for Biomedical Image Segmentation
