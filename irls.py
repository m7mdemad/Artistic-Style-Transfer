import numpy as np

def IRLS(R,X,Z):
#     R is matrix of image patches
#     X is initial estimate
#     Z is matrix of style patches
    r = 0.8 # robust statistics value to use (from the research paper) 
    e = 1e-10 #some small value
    limit = 5 # no. of iterations 
    Xk = X
    w,h = R.shape
    unsampled_pixs = np.logical_not(R.sum(axis=1)>0) 
    unsampled_pixs = unsampled_pixs.astype(float) # prevent black bar artifacts from gap
    
    for i in range(0, limit):
        A=unsampled_pixs # prevent black bar artifacts from gap
        B=Xk*unsampled_pixs
        for j in range(0, h): 
            Rlogical = np.where(R[:,j]==0,0,1) 
            Xlogical = np.where(Rlogical==1,Xk,0)
            w = np.power(np.sum(np.power(Xlogical-Z[:,j],2) + e),(r-2)/2)
            A=A+w*R[:,j] # diag(R)
            B=B+w*Z[:,j]
            
        Xk=(1./(A+1e-10))*B;         
    return Xk
 
    
R = [[1,2,3],
    [4,5,6]
    ]
Z = [[11,2,13],
    [14,5,16]
    ]
X = [1, 21]
X = IRLS(np.asarray(R),np.asarray(X),np.asarray(Z)) # can be IRLS(R,X,Z) directly if matrices are defined as np.array not list
print(X)
