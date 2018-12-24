#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 15:53:25 2018

@author: ibrahim
"""

import numpy as np
from enum import Enum    
Trimap = Enum('Trimap', 'TrimapBackground TrimapUnknown TrimapForeground')
Matte = Enum('Matte', 'MatteBackground MatteForeground')

from commonfunctions import *
img = io.imread('images/house 2-small.jpg')

from sklearn.mixture import GaussianMixture
GMM = GaussianMixture(n_components=5)

img.shape

GMM.fit(img.reshape((img.shape[0]*img.shape[1], 3)))

#print(GMM.weights_.shape)
#print(GMM.weights_)
#print("tttt")
#print(GMM.means_.shape)
#print(GMM.means_)
#print("yyyy")
#print(GMM.covariances_.shape)
#print(GMM.covariances_)

x,y = 200,200
width,height = 100,100
# initialize all as BG
trimap = np.full((img.shape[0], img.shape[1]), Trimap.TrimapBackground)
trimap[y:y+height, x:x+width] = Trimap.TrimapUnknown
#print(trimap)


matte = np.full((img.shape[0],img.shape[1]), Matte.MatteBackground)
matte[trimap == Trimap.TrimapUnknown] = Matte.MatteForeground


foreground_pixels = img[matte == Matte.MatteForeground]
background_pixels = img[matte == Matte.MatteBackground]


#BG_GMM = GaussianMixture(n_components=5).fit(background_pixels)
#FG_GMM = GaussianMixture(n_components=5).fit(foreground_pixels)
#
#GMM_components = np.empty_like(matte).astype(int)
#
#for col in range(img.shape[0]):
#    for row in range(img.shape[1]):
#        pixel = img[col,row]
#        if matte[col,row] == Matte.MatteForeground:
#            proba = FG_GMM.predict_proba([pixel])
##            D = [-np.log10(FG_GMM.weights_[i] * (1/np.sqrt(np.linalg.det(FG_GMM.covariances_[i]))) * np.exp(-0.5*(pixel-FG_GMM.means_[i]).dot(np.linalg.inv(FG_GMM.covariances_[i])).dot(pixel - FG_GMM.means_[i]))) for i in range(5)] 
#        else:
#             proba = BG_GMM.predict_proba([pixel])
##             D = [-np.log10(BG_GMM.weights_[i] * (1/np.sqrt(np.linalg.det(BG_GMM.covariances_[i]))) * np.exp(-0.5*(pixel-BG_GMM.means_[i]).dot(np.linalg.inv(BG_GMM.covariances_[i])).dot(pixel - BG_GMM.means_[i]))) for i in range(5)] 
##        if(proba.argmax() != np.array(D).argmin()):
##            print("proba", proba)
##            print("D", D)
#        GMM_components[col,row] = proba.argmax()

all_GMM_components = np.empty_like(matte).astype(int)
for col in range(img.shape[0]):
    for row in range(img.shape[1]):
        pixel = img[col,row]
        proba = GMM.predict_proba([pixel])
        all_GMM_components[col,row] = proba.argmax()


gmm_mask = [all_GMM_components == i for i in range(5)]
fore_mask = matte == Matte.MatteForeground
#foreground_pixels = [ img[matte == Matte.MatteForeground and all_GMM_components == i] for i in range(5)] 
#background_pixels = [img[matte == Matte.MatteBackground and all_GMM_components == i] for i in range(5)] 
fore_pixels = [img[fore_mask * gmm_mask[i]] for i in range(5)]
back_pixels = [img[np.invert(fore_mask) * gmm_mask[i]] for i in range(5)]

#fore_GMM = [GaussianMixture(n_components=1).fit(fore_pixel) for fore_pixel in fore_pixels if fore_pixel.shape[0] != 0]
#back_GMM = [GaussianMixture(n_components=1).fit(back_pixel) for back_pixel in back_pixels if back_pixel.shape[0] != 0]

fore_GMM = []
back_GMM = []
for fore_pixel in fore_pixels:
    g = None
    if (fore_pixel.shape[0] != 0):
        g = GaussianMixture(n_components=1).fit(fore_pixel)
    fore_GMM.append(g)

for back_pixel in back_pixels:
    g = None
    if (back_pixel.shape[0] != 0):
        g = GaussianMixture(n_components=1).fit(back_pixel)
    back_GMM.append(g)

# iterate over and solve min-cut/max-flow for graph

