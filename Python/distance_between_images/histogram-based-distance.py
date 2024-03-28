# %% Import the required modules
import numpy as np
from matplotlib import pyplot as plt

from loader import load_images

# %% Load the two images
lst_images = ['1', '3']
img1, img2 = load_images('image-'+lst_images[0]+'.png', 'image-'+lst_images[1]+'.png')
img = np.concatenate((img1[:, :, :, np.newaxis], img2[:, :, :, np.newaxis]), axis=3)
# %% Calculate the histogram-based distance between the two images
hist = np.zeros((256, 3, 2), dtype=np.uint64)
# Tuple to select colors of each channel line
colors = ('red', 'green', 'blue')

# Create the histogram plot, with three lines, one for each color
for i in range(0, 2):
    for channel_id, color in enumerate(colors):
        hist[:, channel_id, i], bin_edges = np.histogram(
            img[:, :, channel_id, i], bins=256, range=(0, 256)
        )


# %% Plot the histograms
fig = plt.figure(layout="constrained")
mosaic = """
        aa
        bb
        cc
        de
"""

ax = fig.subplot_mosaic(mosaic)
for i, loc in enumerate(['a', 'b', 'c']):
    for j in range(0, 2):    
        ax[loc].plot(hist[:, i, j], color=colors[i])

ax['a'].set_title('Color Histogram')
ax['c'].set_xlabel('Color value')
ax['a'].set_ylabel('Pixel count')
ax['b'].set_ylabel('Pixel count')
ax['c'].set_ylabel('Pixel count')

# %% Show the images
for i, loc in enumerate(['d', 'e']):
    ax[loc].axis('off')
    image = plt.imread('image-'+lst_images[i]+'.png')
    ax[loc].imshow(image)

plt.show()
