import cv2
import numpy as np
import matplotlib.pyplot as plt

#Steps in the pipeline
# color segmentation
# morphology
# edge detection
# contour detection 
# blob detection

darkest = (86, 80, 80)   
lightest = (192, 191, 187)  
img = cv2.imread('imgs/img1.png')

if img is None:
    print("Error: Could not load image")
    exit()

print(f"Image shape: {img.shape}")
print(f"Image min/max values: {img.min()}/{img.max()}")

# Check if any pixels fall within our color range
mask = cv2.inRange(img, darkest, lightest)
print(f"Mask shape: {mask.shape}")
print(f"Pixels in range: {np.sum(mask > 0)} out of {mask.size}")

result = cv2.bitwise_and(img, img, mask=mask)

# Save the result instead of displaying
cv2.imwrite('opencv_result.png', result)
print("Saved result as opencv_result.png")

# Also save the mask for inspection
cv2.imwrite('opencv_mask.png', mask)
print("Saved mask as opencv_mask.png")

# Show some statistics
print(f"Result image min/max: {result.min()}/{result.max()}")
print(f"Non-zero pixels in result: {np.sum(result > 0)}")
