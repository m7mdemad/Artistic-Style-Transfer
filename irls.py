import numpy as np

def IRLS(R,X,Z):
#     R is matrix of image patches
#     X is initial estimate
#     Z is matrix of style patches
    r = 0.8 # robust statistics value to use (from the research paper) 
    e = 1e-10 #some small value
    limit = 20 # no. of iterations 
    Xk = X
    for i in range(1, limit):
        for j in range(1, R.shape[0]): 
#           caculate the weight   
            w = np.power(np.power(R[:,j]*Xk-Z[:,j] + e,2),(r-2)/2)
            w = np.sum(np.asarray(w), axis=0) # just make calculations faster
        
        Xk = np.power(R.transpose()*w*R + e,-1)*R.transpose()*w*Z # update Xk         
#         print(Xk)
    return Xk
 
    
    
R = [[1,2,3],
    [4,5,6],
    [7,8,9]
    ]
X = [[11,21,37],
    [43,5,16],
    [75,82,9]
    ]
Z = [[1,2,3],
    [4,5,6],
    [7,8,9]
    ]
X = IRLS(np.asarray(R),np.asarray(X),np.asarray(Z)) # can be IRLS(R,X,Z) directly if matrices are defined as np.array not list
