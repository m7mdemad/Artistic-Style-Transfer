import numpy as np
import matplotlib.pyplot as plt
import image
import commonfunctions as cm
from PIL import ImageFilter
from cv2 import *
from scripy import ndimage as ndi

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
  # apply automatic Canny edge detection using the computed median
    lower =(max(0, (1.0 - sigma) * v)).astype(int)
    upper = (min(255, (1.0 + sigma) * v)).astype(int)
    edged = cv2.Canny(image, lower[0], upper[0])

    # return the edged image
    return edged
image = cv2.imread('circuit.tif')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
#print(blurred)
auto = auto_canny(blurred)
cv2.imshow("Original", image)
cv2.imshow("canny result", auto)
cm.show_images([image,auto])