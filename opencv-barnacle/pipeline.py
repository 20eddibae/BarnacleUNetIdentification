import cv2
import numpy as np
import matplotlib.pyplot as plt

# Steps in the pipeline
# color segmentation
# morphology
# contour detection 

# Color boundaries manually found in BGR space through online tools 
darkest = (86, 80, 80)   
lightest = (192, 191, 187)  
img = cv2.imread('imgs/unseen_img2.png')

if img is None:
    print("Error: Could not load image")
    exit()

print(f"Image loaded: {img.shape}")

# Color segmentation
mask = cv2.inRange(img, darkest, lightest)
pixels_detected = np.sum(mask > 0)
total_pixels = mask.size
print(f"Color segmentation: {pixels_detected:,} pixels detected ({pixels_detected/total_pixels*100:.2f}%)")

# Morphology to clean up and separate barnacles
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Add erosion to separate connected barnacles
kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
eroded = cv2.erode(opening, kernel_erode, iterations=1)

# Close small gaps within individual barnacles
cleaned = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)
cleaned_pixels = np.sum(cleaned > 0)
print(f"After morphology: {cleaned_pixels:,} pixels ({cleaned_pixels/total_pixels*100:.2f}%)")

# Find contours
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Found {len(contours)} potential barnacles")

# Filter contours - better balance for middle vs edge barnacles
min_area = 20  # Slightly higher minimum to reduce edge noise
max_area = 2000  # Lower maximum to avoid large connected regions
filtered_contours = []

for contour in contours:
    area = cv2.contourArea(contour)
    if min_area < area < max_area:
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Better filtering: prefer circular/oval shapes, avoid very elongated
            if (circularity > 0.03 and 0.3 < aspect_ratio < 3.0) or (area > 100 and circularity > 0.02):
                filtered_contours.append(contour)

print(f"After filtering: {len(filtered_contours)} barnacles")

# Create visualization
result = img.copy()

# Draw filtered barnacles in red with numbers
for i, contour in enumerate(filtered_contours):
    cv2.drawContours(result, [contour], -1, (0,0,255), 3)  # Thicker red outlines
    
    # Add barnacle number
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(result, str(i+1), (cx-8, cy+8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)  # Black outline
        cv2.putText(result, str(i+1), (cx-8, cy+8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)  # White text

# Save results
cv2.imwrite('opencv_unseenimg2result.png', result)
cv2.imwrite('opencv_mask.png', cleaned)

# Display results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cleaned, cmap='gray')
plt.title(f'Mask ({len(filtered_contours)} barnacles)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title(f'Detected ({len(filtered_contours)} barnacles)')
plt.axis('off')

plt.tight_layout()
plt.savefig('opencv_results.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n=== Final Results ===")
print(f"Barnacles detected: {len(filtered_contours)}")
print(f"Results saved to: opencv_result.png, opencv_mask.png, opencv_results.png")




