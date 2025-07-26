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

print(f"Image loaded: {img.shape}")
print(f"Color range: {darkest} to {lightest}")

# Color segmentation
mask = cv2.inRange(img, darkest, lightest)
pixels_detected = np.sum(mask > 0)
total_pixels = mask.size
print(f"Color segmentation: {pixels_detected:,} pixels detected out of {total_pixels:,} total ({pixels_detected/total_pixels*100:.2f}%)")

# Morphology to clean up
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
opening_pixels = np.sum(opening > 0)
print(f"After opening: {opening_pixels:,} pixels ({opening_pixels/total_pixels*100:.2f}%)")

cleaned = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
cleaned_pixels = np.sum(cleaned > 0)
print(f"After closing: {cleaned_pixels:,} pixels ({cleaned_pixels/total_pixels*100:.2f}%)")

# Find contours
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
barnacle_count = len(contours)

print(f"Found {barnacle_count} barnacles")

# Filter by area to remove tiny noise
min_area = 20
filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
filtered_count = len(filtered_contours)
print(f"After filtering (area > {min_area}): {filtered_count} barnacles")

# Draw results
result = img.copy()
cv2.drawContours(result, contours, -1, (0,255,0), 2)
cv2.drawContours(result, filtered_contours, -1, (0,0,255), 3)

# Save and show
cv2.imwrite('opencv_result.png', result)
cv2.imwrite('opencv_mask.png', cleaned)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cleaned, cmap='gray')
plt.title(f'Mask ({filtered_count} barnacles)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Detected (Green: all, Red: filtered)')
plt.axis('off')

plt.tight_layout()
plt.savefig('opencv_results.png')
plt.show()




