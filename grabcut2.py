#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 18:46:00 2018

@author: ibrahim
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from commonfunctions import*
# from cv2 import ximgproc as a
#img = cv2.imread('images/me.jpg',0)
def grabcut(img_path ,rect):
    img = io.imread(img_path)
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    #rect = tuple(rect[0])
#    rect = rect[0] - 25, rect[1] - 25, rect[2] + 50 +rect[3] + 25
    #rect = (50,50,450,290)
    #img = io.imread('images/me.jpg')
    
    #print(img.shape)
    #mask2 = np.array(rect.shape[0]).astype(float)
    #mask = np.array(rect.shape[0]).astype(float)
    mask2 = [None]*rect.shape[0]
    mask3 = [None]*rect.shape[0]
    for i in range(rect.shape[0]):
        r = rect[i][0] - 25, rect[i][1] - 25, rect[i][2] + 50, rect[i][3] + 50
        
        mask3[i] = cv2.grabCut(img,mask,r,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)[0]
        mask2[i] = np.where((mask3[i]==2)|(mask3[i]==0),0,1).astype('uint8')
        
    #img = img*mask2[:,:,np.newaxis]
    #plt.imshow(img),plt.colorbar(),plt.show()
    mask333 = np.zeros_like(mask2[0])
    for m in mask2:
        mask333 += m
    
    return mask333

def test (img_path):
    gray = cv2.imread(img_path,0)
#    gray=rgb2gray(img)
    faceCascade = cv2.CascadeClassifier('image.xml')
    rect = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
    ) #print(rect!= 0)
    return grabcut(img_path,rect)
#img_path = 'images/me.jpg'
#out = test(img_path)
#print(out)