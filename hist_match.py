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
    s_quantiles = np.cumsum(s_counts, dtype=float) / style.size

    # interpolate linearly to find the pixel values in the style image
    # that correspond most closely to the quantiles in the content image
    interps_values = np.interp(c_quantiles, s_quantiles, s_values)

    return interps_values[indx].reshape(oldshape)


def color_transform(content, style):

    rcomp_C = content[:, :, 0]*255
    gcomp_C = content[:, :, 1]*255
    bcomp_C = content[:, :, 2]*255

    rcomp_S = style[:, :, 0]*255
    gcomp_S = style[:, :, 1]*255
    bcomp_S = style[:, :, 2]*255

    matchedr = hist_match(rcomp_C, rcomp_S)
    matchedg = hist_match(gcomp_C, gcomp_S)
    matchedb = hist_match(bcomp_C, bcomp_S)
    
    return (np.stack([matchedr, matchedg, matchedb], axis=2).astype('uint8'))/255

def test():
    content = io.imread(r"images/house 2-small.jpg")
    style = io.imread(r"images/starry-night - small.jpg")

    show_images([content, style])

    myimg = color_transform(content, style)

    show_images([myimg])