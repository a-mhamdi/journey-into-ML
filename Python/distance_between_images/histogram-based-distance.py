# %% Import the required modules
import numpy as np
from math import floor

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

# %% Histogram of multiple images
colors = ('red', 'green', 'blue')
img1, img2 = load_images('image-1.png', 'image-3.png')
width, height, _ = img1.shape
pas, nbins = 32, 256

# %%
data_hist = np.zeros((nbins*(floor(width/pas)-1), 3, 2), dtype=np.uint64)
for i in range(0, 2):
    img = img1 if i == 0 else img2
    for channel_id, color in enumerate(colors):
        hist = [[], [], []]
        debut, fin = 0, pas
        while True:
            img_trunc = img[debut:fin, :, channel_id]
            debut = fin
            fin += pas
            if fin > width:
                break
            tmp, bin_edges = np.histogram(
                img_trunc, bins=nbins, range=(0, nbins)
                )
            hist[channel_id].extend(list(tmp))
        data_hist[:, channel_id, i] = np.array(hist[channel_id])
# %%
_, ax = plt.subplots(3, 1)
cosine_sim = {}
for i in range(0, 3):
    ax[i].plot(data_hist[:, i, :], color=colors[i])
    cosine_sim[colors[i]] = np.dot(data_hist[:, i, 0], data_hist[:, i, 1]) / (np.linalg.norm(data_hist[:, i, 0]) * np.linalg.norm(data_hist[:, i, 1]))
    print(f'{colors[i]}: {cosine_sim[colors[i]]}')

# %% Divergence measures
# Calculate the KL-divergence
def kl_divergence(p, q): # KL(P || Q)
    return np.sum(p * np.log(p / q)) if q.all() != 0 else np.inf

# Calculate the JS-divergence
def js_divergence(p, q): # JS(P || Q)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

# %% Calculate (P || Q) and (Q || P)

p = data_hist[:, 0, 0] / (255 *width)
q = data_hist[:, 0, 1] / (255 *width)

print(kl_divergence(p, q), js_divergence(p, q))