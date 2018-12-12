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
    patchH = Z.shape[0]
    patchW = Z.shape[1]
#    show_images([X])
    for k in range(0, limit):
        Xk = np.zeros_like(X)
        W = np.zeros_like(X)
        for y in range(0, patchW):
            for x in range(0, patchH):
                startRow = y * gap
                startcol = x * gap
                patch = Z[y,x]
#                show_images([patch, X[startRow:startRow+patchSize, startcol:startcol+patchSize]], ["nearest patch", "image patch"])
                patch_weight = (np.abs(patch - X[startRow:startRow+patchSize, startcol:startcol+patchSize]) + e)
#                print("patch", patch)
#                print("weight", patch_weight)
#                np.add(Xk[startRow:startRow+patchSize, startcol:startcol+patchSize], patch * patch_weight, out=Xk[startRow:startRow+patchSize, startcol:startcol+patchSize], casting="unsafe")
                Xk[startRow:startRow+patchSize, startcol:startcol+patchSize] += patch * patch_weight
#                np.add(W[startRow:startRow+patchSize, startcol:startcol+patchSize],  patch_weight, out=W[startRow:startRow+patchSize, startcol:startcol+patchSize], casting="unsafe")
                W[startRow:startRow+patchSize, startcol:startcol+patchSize] += patch_weight
        mask = W == 0
        W[mask] = 1
        X = Xk/W
        # show_images([X],'irls')
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
