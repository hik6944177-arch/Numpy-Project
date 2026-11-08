import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread

image = imread(r"D:\1. AI & ML\phase 1 and 2\numpy\Project\H.Heart.jpeg")
print("image shape", image.shape)

R = image[ : , : , 0]
G = image[ : , : , 1]
B = image[ : , : , 2]

grayscale = 0.2989 * R + 0.5870* G + 0.1140 * B
print("Grayscale shape", grayscale.shape)

threshold_value = 128
thresholded = np.where(grayscale > threshold_value, 255, 0)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(image.astype(np.uint8))

plt.subplot(1, 3, 2)
plt.title("Grayscale")
plt.imshow(grayscale, cmap ="gray")

plt.subplot(1, 3, 3)
plt.title("Threshold")
plt.imshow(thresholded, cmap = "gray")

plt.tight_layout()
plt.show()