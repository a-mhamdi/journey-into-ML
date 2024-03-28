import numpy as np

from loader import load_images

img1, img2 = load_images('image-3.png', 'image-3.png')

# Calculate the cosine similarity between the two images
similarity = np.dot(img1.flatten(), img2.flatten()) / (np.linalg.norm(img1) * np.linalg.norm(img2))
distance = 1 - similarity
print(distance)