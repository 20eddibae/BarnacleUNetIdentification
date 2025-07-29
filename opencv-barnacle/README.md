# Barnacle Detection with OpenCV

## What this does

This project tries to automatically count barnacles in photos taken by National Park Service scientists. Right now they have to manually count hundreds of barnacles in each image, which takes forever. This OpenCV script gives it a shot at automating that process.

## How it works

1. Find brownish colors that look like barnacles
2. Clean up the image to remove noise
3. Separate connected barnacles from each other
4. Count individual barnacle shapes
5. Draw red circles around what it finds

## Running it

First install dependencies:
```bash
pip install opencv-python numpy matplotlib
```

To test on a different image:
1. Put your image in the `imgs/` folder
2. Change this line in `pipeline.py`:
   ```python
   img = cv2.imread('imgs/your_image.png')
   ```
3. Run it:
   ```bash
   python3 pipeline.py
   ```

## Results

The script saves:
- `opencv_result.png` - shows numbered barnacles with red outlines
- `opencv_mask.png` - the black/white mask it creates
- `opencv_results.png` - side by side comparison

I tested it on:
- img1.png - opencv_img1results.png (1533 barnacles)
- img2.png - opencv_img2results.png (3807 barnacles)
- unseen_img1.png - opencv_unseenimg1results.png (2089 barnacles)
- unseen_img2.png →-opencv_unseenimg2results.png (122 barnacles)

## What I tried and what happened

### First attempt: Too much noise
Started with basic color detection but it picked up way too much stuff that wasn't barnacles. Had to add filtering.

### Second attempt: Connected barnacles
The barnacles in the middle were all stuck together and being counted as one big thing. Added erosion to separate them.

### Third attempt: Finding the right balance
Too strict filtering missed real barnacles, too loose included noise. Tried different circularity and area thresholds.

### Fourth attempt: Making it look better
The red outlines were too thin and there was too much green clutter. Made the red thicker and removed the green.

## What works and what doesn't

**Works:**
- Finds most barnacles pretty quickly
- Works on different images
- Shows numbered results
- Faster than deep learning methods (like the UNet approach)

**Problems:**
- Doesn't catch every single barnacle (especially in dense clusters)
- Still picks up some noise from edges and reflections
- Relies on barnacles being roughly oval-shaped
- Might not work as well with different lighting
- As you can see in the images, not perfect but definitely faster than deep learning 

## Technical details

The main steps are:
- Color range: (86,80,80) to (192,191,187) for brown barnacles (found manually )
- Morphology steps: opening -> erosion -. closing
- Filtering: area 20-2000 pixels, circularity > 0.03, aspect ratio 0.3-3.0 (mostly found through tweaking the values)

## What I learned

Traditional computer vision with OpenCV is fast and you can see exactly what it's doing, but it's not perfect for messy real-world images like this. The barnacles are all different shapes and sizes, and the lighting changes everything. Especially between the two original images, the barnacle colors and the lighting in general was completely different. In particular, there is a lot of noise and overdetection of barnacles as well as insufficient detection of barnacles. However, even though it is not as efficient as the machine learning model, it taught me a lot. Specifically, it  taught me a lot about data cleaning and the power of OpenCV in the process and now I can see how this tool can be used to enhance machine learning processes. 

## Next steps

Could try:
- Training a neural network on the mask data
- Adding confidence scores for each detection
- Making an interface where scientists can correct mistakes
- Testing on more images with different conditions 