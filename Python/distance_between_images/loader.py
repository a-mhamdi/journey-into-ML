import cv2
import numpy as np

def load_images(path1, path2):
    img1 = cv2.imread(path1).astype(np.float32)
    img2 = cv2.imread(path2).astype(np.float32)

    # img1 /= 255
    # img2 /= 255

    return img1, img2