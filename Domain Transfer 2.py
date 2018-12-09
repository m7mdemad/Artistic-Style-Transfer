import numpy as np
import commonfunctions as cm
import cv2

def image_transpose(I):
    [h,w,num_channels] = np.shape(I)

    T = np.zeros([w,h,num_channels])

    for c in range (1,num_channels):
        T[:,:,c] = np.transpose(I[:,:,c])
    return T

def  RecursiveFilter(I, D, sigma):

    a = np.exp((-1 * np.sqrt(2) / sigma))
    F = I
    V = np.power(a,D)
    [h,w,num_channels] = np.shape(I)
    # bnsba kabera mo4kla hena
    for i in range (2,w-1):
        for c in range(1,num_channels):
            F[:, i, c] += np.multiply( V[:, i],(F[:, i - 1, c] - F[:, i, c]))
    for i in range (w-1,-1,1):
        for c in range(1, num_channels):
            F[:, i, c] += np.multiply( V[:, i+1],(F[:, i + 1, c] - F[:, i, c]))

    return F

def Iterative_C(img, sigma_s, sigma_r, N):
    I = img.astype(float)
    #[h,w,num_channels] = np.shape(I)
    dIcdx = abs(np.diff(I, 1, 1))
    dIcdy = abs(np.diff(I, 1, 0))

    # for c in range (1,num_channels):
    #      dIdx[:, 2: ] += abs(dIcdx[:,:, c])
    #      dIdy[2: ,:] += abs(dIcdy[:,:, c])
    dIdx = np.sum(dIcdx,axis=2)
    dIdy = np.sum(dIcdy, axis=2)
    dHdx = 1 + np.multiply((sigma_s/ sigma_r ),dIdx)
    dVdy = 1 + np.multiply((sigma_s / sigma_r) , dIdy)
    dVdy = np.transpose(dVdy)
    F = I
    s = np.multiply(sigma_s, np.sqrt(3))
    for i in range (0,N-1):
        # Compute the sigma value for this iteration 14
        sigma_H_i =np.multiply((2 ** (N - i-1)) / ((np.sqrt(((4 ** N) - 1)))),s)
        F = RecursiveFilter(F, dHdx, sigma_H_i)
        F = image_transpose(F)
        F = RecursiveFilter(F, dVdy, sigma_H_i)
        F = image_transpose(F)
    F = F.astype(dtype=int)
    return F

img = cv2.imread('statue.png')

sigma_s = 100
sigma_r = 0.4

f = Iterative_C(img, sigma_s, sigma_r,3)

cm.show_images([img,f],["img","y rab"])