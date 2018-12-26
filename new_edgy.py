from commonfunctions import *
import cv2
import numpy as np
from skimage.filters import gaussian

def get_filter(size, sigma):
    m = (size-1)/2
    n = (size-1)/2
    std2 = sigma**3
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*std2) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    h1 = h*(x*x + y*y - 2*std2)/(std2**2)
    return h1 - h1.mean()
     
def edgy(img, thresh = 0, sigma_e = 1):
    [m, n] = img.shape
    e = np.zeros((m,n)) # edge map
    rr = np.arange(1, m-1)
    cc = np.arange(1, n-1)

    fsize = np.ceil(sigma_e*3) * 2 + 1
    op = get_filter(fsize, sigma_e)
    b = convolve2d(img, op)
#     show_images([b])
    if thresh == 0:
        thresh = np.absolute(b).mean() * 0.75
##
    for i in rr:
        for j in cc: 
            if b[i, j] < 0 and b[i, j+1] > 0 and abs( b[i, j]-b[i, j+1] ) > thresh:
                e[i,j] = 1
            if b[i, j-1] < 0 and b[i, j] > 0 and abs( b[i, j-1]-b[i, j] ) > thresh:
                e[i,j] = 1
            if b[i, j] < 0 and b[i+1, j] > 0 and abs( b[i+1, j]-b[i, j] ) > thresh:
                e[i,j] = 1
            if b[i-1, j] < 0 and b[i, j] > 0 and abs( b[i-1, j]-b[i, j] ) > thresh:
                e[i,j] = 1
                
#     show_images([e])
    
    bw = gaussian(0.5*e, 7)
#     show_images([bw])
    mask = np.zeros(bw.shape)
    val = bw.mean()

    mask = mask.flatten()
    bw = bw.flatten()

    for i in range (len(bw)):
        if bw[i] > val:
            mask[i] = 1
            
    mask = np.reshape(mask, img.shape)       
#     show_images([mask])
    img2 = cv2.convertScaleAbs(img)
#     show_images([img2])
    
    f = mask*e*10
#     show_images([f])
    
    #interpolate both edge detection output and segmentation output (can be modified later)
    img2 = img2.flatten()
    f = f.flatten()
    temp = np.zeros(f.shape)
    for i in range(len(f)):
        temp[i] = img2[i] + 10*f[i]
    temp = np.reshape(temp, (m,n))
    
    w = gaussian((temp).astype('uint8'), 7) #smoothing
    show_images([(temp).astype('uint8'), w], ['before blur', 'after blur'])

def test():
	content = io.imread('images/house 2-small.jpg')
	return edgy(content)

test()
