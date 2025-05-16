# Implementation-of-Erosion-and-Dilation
## Aim
To implement Erosion and Dilation using Python and OpenCV.
## Software Required
1. Anaconda - Python 3.7
2. OpenCV
## Algorithm:
### Step1:
Import required libraries (OpenCV, NumPy) and load the image in grayscale.
### Step2:
Define a structuring element (kernel) for morphological operations.
### Step3:
Apply erosion using cv2.erode() on the image with the defined kernel.
### Step4:
Apply dilation using cv2.dilate() on the image with the same kernel.
### Step5:
Display and compare the original, eroded, and dilated images. 
## Program:
### Name : PRAJAN P
### Register Number : 212223240121

``` Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = np.zeros((300, 600, 3), dtype="uint8")

text = "RAKSHITHA"
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, text, (20, 200), font, 3, (255, 255, 255), 7)

kernel = np.ones((3, 3), np.uint8)

eroded_image = cv2.erode(image, kernel, iterations=1)

dilated_image = cv2.dilate(image, kernel, iterations=1)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
eroded_image_rgb = cv2.cvtColor(eroded_image, cv2.COLOR_BGR2RGB)
dilated_image_rgb = cv2.cvtColor(dilated_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 2)
plt.imshow(eroded_image_rgb)
plt.title("Eroded Image")
plt.axis("off")
plt.show()

plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 3)
plt.imshow(dilated_image_rgb)
plt.title("Dilated Image")
plt.axis("off")
plt.show()
```
## Output:

### Original Image:

![download](https://github.com/user-attachments/assets/47e29979-bacc-459a-a4be-e4e3692b250e)

### Eroded Image:

![download](https://github.com/user-attachments/assets/413be981-636a-4ef3-b636-dc57a42c9dc3)

### Dilated Image:

![download](https://github.com/user-attachments/assets/a1cfa3a0-4c59-458a-b8ae-a02969d2e327)

## Result
Thus the generated text image is eroded and dilated using python and OpenCV.
