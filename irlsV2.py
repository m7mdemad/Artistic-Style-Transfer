def IRLS(X, Z, patchSize, gap):
#     R is matrix of image patches
#     Z is the style patches
#     patchSize is the size of each patch
#     gap is the step in X  
    r = 0.8 # robust statistics value to use (from the research paper) 
    e = 1e-10 #some small value
    limit = 5 # no. of iterations 
    w,h = X.shape
    Xk = np.zeros([w, h])
    W = np.zeros([w, h])
    patchW,patchH = Z.shape
    Xloop = X
    for k in range(0, limit):
        startRow = 0
        startcol = 0
        for x in range(0, patchW): 
            startRow = x*(patchSize - gap)
            for y in range(0, patchH): 
                startcol = y*(patchSize - gap)
                for i in range(startRow, startRow + patchSize): 
                    for j in range(startcol,  startcol + patchSize): 
                        W[i,j] += 1/(np.absolute(Xloop[i,j]-Z[i,j]) + e) #not sure  
                        Xk[i,j] += W[i,j]*Xloop[i,j]

        Xk = Xk/W
        Xloop = Xk                 
    return Xk
