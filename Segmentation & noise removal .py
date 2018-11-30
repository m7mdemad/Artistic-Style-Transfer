import numpy as np
import commonfunctions as cm
from cv2 import *
from cv2 import ximgproc as a
from skimage.util import random_noise


img = cv2.imread('abba.png',0)
edges3=np.copy(img)
###### Edge detection
edges = cv2.Canny(img,100,200)
######## face detection
cascPath = "haarcascade_frontalface_default.xml"

gray=cm.rgb2gray(img)
faceCascade = cv2.CascadeClassifier('image.xml')

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
# affinity segmentation

#edges2 = cv2.watershed(img,cv2.COLOR_BAYER_GB2GRAY)
#edges2= a.createDTFilter(img, sigmaSpatial=100, sigmaColor=0.03, mode= a.DTF_NC, numIters=3)
#noise1= random_noise(edges, mode = 's&p', amount = 0.05)

# domain transform
edges2= a.dtFilter(img, edges, sigmaSpatial=100, sigmaColor=0.03, mode=a.DTF_NC, numIters=3)
cm.show_images([img,edges,edges2],[" main"," segment1 ","transform"])



