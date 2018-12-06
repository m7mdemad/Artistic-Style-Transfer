import numpy as np
from skimage.util import view_as_windows
from commonfunctions import *

def IRLS(X, Z, patchSize, gap):
#     X estimate image
#     Z is the style patches
#     patchSize is the size of each patch
#     gap is the step in X  
    r = 0.8 # robust statistics value to use (from the research paper) 
    e = 1e-10 #some small value
    limit = 5 # no  . of iterations
    patchW = Z.shape[0]
    patchH = Z.shape[1]
    show_images([X])
    for k in range(0, limit):
        Xk = np.copy(X)
        W = np.ones_like(X)
        for x in range(0, patchW):
            for y in range(0, patchH):
                startRow = x * gap
                startcol = y * gap

                patch = Z[x,y]
                patch_weight = 1/(np.abs(patch - X[startRow:startRow+patchSize, startcol:startcol+patchSize]) + e)
#                print("patch", patch)
#                print("weight", patch_weight)
                Xk[startRow:startRow+patchSize, startcol:startcol+patchSize] += patch * patch_weight
                W[startRow:startRow+patchSize, startcol:startcol+patchSize] += patch_weight
        X = Xk/W
        show_images([X])
    return X
 
    
# a = np.arange(9).reshape(3,3)
# # print(a)
# Z = view_as_windows(a, (2,2), step = 1)
# print(R[0,0])
# X = [[8, 2, 5],
#      [1, 6, 4],
#      [9, 7, 3],
#     ]
# patchSize = 2
# gap = 1
# X = IRLS(np.asarray(X),np.asarray(Z), patchSize, gap) # can be IRLS(R,X,Z) directly if matrices are defined as np.array not list
# print(X)
