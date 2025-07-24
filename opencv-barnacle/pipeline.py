import cv2
import numpy as np
import matplotlib.pyplot as plt

#Steps in the pipeline
# color segmentation
# morphology
# edge detection
# contour detection 
# blob detection

# Use BGR order for OpenCV
# Example: blueish barnacle color range (tune as needed)
darkest = (86, 80, 80)   
lightest = (192, 191, 187)  
img = cv2.imread('imgs/img1.png')
mask = cv2.inRange(img, darkest, lightest)
result = cv2.bitwise_and(img, img, mask=mask)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.show()
