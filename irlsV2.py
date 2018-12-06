import numpy as np
from skimage.util import view_as_windows

def IRLS(X, Z, patchSize, gap):
#     R is matrix of image patches
#     Z is the style patches
#     patchSize is the size of each patch
#     gap is the step in X  
    r = 0.8 # robust statistics value to use (from the research paper) 
    e = 1e-10 #some small value
    limit = 1 # no. of iterations 
    w,h = X.shape
    Xk = np.zeros([w, h])
    W = np.zeros([w, h])
    patchW = Z.shape[0]
    patchH = Z.shape[1]
    Xloop = X
    for k in range(0, limit):
        startRow = 0
        startcol = 0
        for x in range(0, patchW): 
            startRow = x*(patchSize - gap)
            for y in range(0, patchH): 
                startcol = y*(patchSize - gap)
                temp = Z[x,y]
                for i in range(startRow, startRow + patchSize): 
                    for j in range(startcol,  startcol + patchSize): 
                        W[i,j] += 1/(np.absolute(Xloop[i,j]-temp[i-startRow,j-startcol]) + e) #not sure  
                        Xk[i,j] += W[i,j]*Xloop[i,j]

        Xk = Xk/W
        Xloop = Xk                 
    return Xk
 
    
a = np.arange(9).reshape(3,3)
# print(a)
Z = view_as_windows(a, (2,2), step = 1)
print(R[0,0])
X = [[8, 2, 5],
     [1, 6, 4],
     [9, 7, 3],
    ]
patchSize = 2
gap = 1
X = IRLS(np.asarray(X),np.asarray(Z), patchSize, gap) # can be IRLS(R,X,Z) directly if matrices are defined as np.array not list
print(X)
