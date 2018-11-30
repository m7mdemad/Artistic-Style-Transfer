import numpy as np
from commonfunctions import *


def hist_match(content, style):
    oldshape = content.shape
    content = content.flatten()
    style = style.flatten()

    # get the set of unique pixel values and indices and counts
    c_values, indx, c_counts = np.unique(
        content, return_inverse=True, return_counts=True)  # for content image
    s_values, s_counts = np.unique(
        style, return_counts=True)  # for style image

    # get cum sum of counts and normalize it
    c_quantiles = np.cumsum(c_counts, dtype=float) / content.size
    #c_quantiles /= c_quantiles[-1]
    s_quantiles = np.cumsum(s_counts, dtype=float) / style.size
    #s_quantiles /= s_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interps_values = np.interp(c_quantiles, s_quantiles, s_values)

    return interps_values[indx].reshape(oldshape)


def color_transform(content, style):
    '''
    rcomp_C = rgb2gray(content[:,:,0])
    gcomp_C = rgb2gray(content[:,:,1])
    bcomp_C = rgb2gray(content[:,:,2])

    rcomp_S = rgb2gray(style[:,:,0])
    gcomp_S = rgb2gray(style[:,:,1])
    bcomp_S = rgb2gray(style[:,:,2])
    '''
    rcomp_C = content[:, :, 0]
    gcomp_C = content[:, :, 1]
    bcomp_C = content[:, :, 2]

    rcomp_S = style[:, :, 0]
    gcomp_S = style[:, :, 1]
    bcomp_S = style[:, :, 2]

    matchedr = hist_match(rcomp_C, rcomp_S)
    matchedg = hist_match(gcomp_C, gcomp_S)
    matchedb = hist_match(bcomp_C, bcomp_S)
    
    return np.stack([matchedr, matchedg, matchedb], axis=2).astype('uint8')


content = io.imread(r"images/house 2-small.jpg")
style = io.imread(r"images/starry-night - small.jpg")

show_images([content, style])

myimg = color_transform(content, style)

show_images([myimg])
# add gaussian noise
'''
row,col,ch= myimg.shape
mean = 0
sigma = 50
gauss = np.random.normal(mean,sigma,(row,col,ch))
gauss = gauss.reshape(row,col,ch)
noisy = myimg + gauss
show_images([noisy.astype('uint8')])
'''
