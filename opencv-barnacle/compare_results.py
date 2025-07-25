import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original mask and OpenCV result
original_mask = cv2.imread('imgs/mask1.png')
opencv_mask = cv2.imread('opencv_mask.png', cv2.IMREAD_GRAYSCALE)

print(f"Original mask shape: {original_mask.shape}")
print(f"OpenCV mask shape: {opencv_mask.shape}")

# Convert original mask to binary (blue to white)
hsv = cv2.cvtColor(original_mask, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])
original_binary = cv2.inRange(hsv, lower_blue, upper_blue)

print(f"Original binary mask shape: {original_binary.shape}")
print(f"Original binary pixels: {np.sum(original_binary > 0)}")
print(f"OpenCV mask pixels: {np.sum(opencv_mask > 0)}")

# Calculate overlap
overlap = np.sum((original_binary > 0) & (opencv_mask > 0))
total_original = np.sum(original_binary > 0)
total_opencv = np.sum(opencv_mask > 0)

if total_original > 0:
    recall = overlap / total_original
    print(f"Recall (OpenCV covers original): {recall:.3f}")

if total_opencv > 0:
    precision = overlap / total_opencv
    print(f"Precision (OpenCV is accurate): {precision:.3f}")

# Save comparison images
cv2.imwrite('original_binary.png', original_binary)
cv2.imwrite('comparison_overlay.png', cv2.addWeighted(original_binary, 0.5, opencv_mask, 0.5, 0))

print("Saved comparison images:")
print("- original_binary.png: Original mask converted to binary")
print("- comparison_overlay.png: Overlay of both masks")
print("- opencv_mask.png: Your OpenCV segmentation")
print("- opencv_result.png: Your OpenCV result with original colors") 