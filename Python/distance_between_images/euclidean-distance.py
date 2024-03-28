import numpy as np

from loader import load_images

img1, img2 = load_images('image-3.png', 'image-3.png')

# Calculate the Euclidean distance between the two images
distance = np.linalg.norm(img1 - img2)
print(distance)

