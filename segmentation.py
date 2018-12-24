#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 14:00:53 2018

@author: ibrahim
"""
from grabcut2 import *
from edge_segmentation import *
import scipy.ndimage as nd
from skimage.filters import gaussian
from skimage.segmentation import morphological_chan_vese
def get_segmentation(img_path,blur=7):
    gray = cv2.imread(img_path,0)
#    gray=rgb2gray(img)
    faceCascade = cv2.CascadeClassifier('image.xml')
    rect = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)
    face_mask = np.zeros_like(gray)
    #print(rect != 0)
    if (rect is not ()):
        #  todo face 
        face_mask = grabcut(img_path,rect)
    # egdesegmentation 
    edge_mask = edge_seg(img_path)
    new_mask = face_mask + edge_mask
    new_mask[new_mask > 1] = 1
    #new_mask = edge_mask
#    maximum = new_mask.max()
#    print(maximum)
#    new_mask = new_mask / maximum
    
    mask = gaussian((new_mask*255).astype('uint8'), blur)
    img = gray
    mask = mask[:,:]
#    mask = new_mask 
    return mask
    img = img*mask
    plt.imshow(img),plt.colorbar(),plt.show()

#img_path = 'images/house 2-small.jpg'
#segmentation(img_path)
   