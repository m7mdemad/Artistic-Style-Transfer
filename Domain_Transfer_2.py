import numpy as np
from commonfunctions import *
import cv2

def image_transpose(I):

    h, w, num_channels = I.shape

    mT = np.zeros((w, h, num_channels), dtype=float)

    for c in range(0, num_channels):
        mT[:, :, c] = I[:, :, c].conj().T
    return mT

def RecursiveFilter(I, D, sigma):

    a = np.exp((-1 * np.sqrt(2) / sigma))
    F = I
    V = np.power(a, D)
    [h, w, num_channels] = np.shape(I)
    for i in range(1, w):
        for c in range(0, num_channels):
            F[:, i, c] += np.multiply(V[:, i], (F[:, i - 1, c] - F[:, i, c]))
    for i in range(w-2, -1, -1):
        for c in range(1, num_channels):
            F[:, i, c] += np.multiply(V[:, i+1], (F[:, i + 1, c] - F[:, i, c]))

    return F

def Iterative_C(img, sigma_s, sigma_r, N=3):
    I = np.copy(img.astype(float))
    h, w, c = I.shape

    dIcdx = np.diff(I, 1, 1)
    dIcdy = np.diff(I, 1, 0)

    dIdx = np.zeros((h, w))
    dIdy = np.zeros((h, w))
    for c in range(0, 3):
        dIdx[:, 1:w] = dIdx[:, 1:w] + abs(dIcdx[:, :, c])
        dIdy[1:h, :] = dIdy[1:h, :] + abs(dIcdy[:, :, c])

    dHdx = 1 + (sigma_s/sigma_r) * dIdx
    dVdy = 1 + (sigma_s/sigma_r) * dIdy
    dVdy = np.transpose(dVdy)
    F = I

    sigma_H = sigma_s
    for i in range(0, N):
        # Compute the sigma value for this iteration
        sigma_H_i = sigma_H * np.sqrt(3) * (2 ** (N - (i + 1))) / np.sqrt((4**N) - 1)
        F = RecursiveFilter(F, dHdx, sigma_H_i)
        F = image_transpose(F)
        F = RecursiveFilter(F, dVdy, sigma_H_i)
        F = image_transpose(F)

    return F

def test():
    src = io.imread(r"images/house 2-small.jpg")/255

    #src = cv2.resize(src, dsize=(400, 400), interpolation=cv2.INTER_CUBIC) #somtimes it gives weird error when the img resolution is too big XD
    f = Iterative_C(src, 5, 0.2)
    show_images([src, f], ["Original", "After Filter"])
