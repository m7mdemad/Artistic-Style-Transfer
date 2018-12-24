import numpy as np
from commonfunctions import *
import scipy.ndimage as nd
from skimage.filters import gaussian
from skimage.segmentation import morphological_chan_vese

def edge_seg(img_path, sigma=4): #sigma to be used in LoG, blur to be used in gaussian filter
    content = io.imread(img_path)
    content = rgb2gray(content)
    
    #to detect edges we compute zero-crossings after filtering content image with a Laplacian of Gaussian (LoG) filter.
    LoG = nd.gaussian_laplace(content, sigma)
    thres = np.absolute(LoG).mean() * 0.75  
    output = np.zeros(LoG.shape)
    w = output.shape[1]
    h = output.shape[0]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = LoG[y-1:y+2, x-1:x+2]
            p = LoG[y, x]
            maxP = patch.max()
            minP = patch.min()
            if (p > 0):
                if minP < 0:
                    zeroCross = True
                else:
                    zeroCross = False
            else:
                if maxP > 0:
                    zeroCross = True
                else:
                    zeroCross = False
            if ((maxP - minP) > thres) and zeroCross:
                output[y, x] = 1
                
    #show_images([output])
    
    #segmentation using morphological active contours with 50 iterations
    img = morphological_chan_vese(content, 50)
    #show_images([img])
    
    #interpolate both edge detection output and segmentation output (can be modified later)
    img = img.flatten()
    output = output.flatten()
    temp = np.zeros(output.shape)
    for i in range(len(output)):
        temp[i] = 10 * img[i] * output[i]
    mask = np.reshape(temp, content.shape)
#    mask = gaussian((temp*255).astype('uint8'), blur) #smoothing
   # show_images([(temp*255).astype('uint8'), w], ['before blur', 'after blur']) 
    maximum = mask.max()
    mask = mask/maximum
    return mask

#test
def test():
	content = io.imread('images/house 2-small.jpg')
	return edge_seg(content)

#w = test()