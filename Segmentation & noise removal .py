import numpy as np
import commonfunctions as cm
from cv2 import *
from cv2 import ximgproc as a
from skimage.util import random_noise


img = cv2.imread('images/house 2-small.jpg',0)
edges3=np.copy(img)
###### Edge detection
edges = cv2.Canny(img,100,200)
######## face detection
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
edges2= a.dtFilter(edges3, edges, sigmaSpatial=100, sigmaColor=0.03, mode=a.DTF_NC, numIters=3)
cm.show_images([edges3,edges,img,edges2],[" main"," segment1 ","faces","transform"])



#face detection input gray ->img in gray level     output->faces (co ordinates of faces in image)
#domian transfer  input = edges3 (main image)   edges = noisy image
#edge detection input image and output image (nothing)