import numpy as np

def IRLS(R,X,Z):
    r = 0.8 # robust statistics value to use (from the research paper) 
    limit = 10 # no. of iterations 
    Xk = X
    for i in range(1, limit):
        sum = 0
        for j in range(1, R.shape[0]):  # not sure 
            norm = np.power(R[:,j]*Xk-Z[:,j],2)
            sum += norm
        Xk = np.argmin(np.power(sum,0.8-2))
        print(Xk)
    return Xk
 
