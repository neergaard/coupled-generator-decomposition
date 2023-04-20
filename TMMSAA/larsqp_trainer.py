# optimization loop
from tqdm import tqdm
import numpy as np
from TMMSAA import larsqp

def Optimizationloop(X,num_comp,lambda1,lambda2,max_iter=100, tol=1e-10):
    
    all_loss = []
    if type(X) is dict:
        B_shape = next(iter(X.values())).shape[-1]
        S_shape = (len(X),*next(iter(X.values())).shape)
        S_shape = (*S_shape[:-2],num_comp,B_shape)
        XtX_shape = (*S_shape[:-2],B_shape,B_shape)
        XtX = np.zeros(XtX_shape)
        for m,key in enumerate(X):
            XtX[m] = np.swapaxes(X[key],-2,-1)@X[key]
    else:
        B_shape = X.shape[-1]
        XtX = np.swapaxes(X,-2,-1)@X
    Bp = np.random.uniform(size=(B_shape, num_comp))
    Bn = np.random.uniform(size=(B_shape, num_comp))

    for epoch in tqdm(range(max_iter)):
        Bp,Bn,S = larsqp.larsqp(X,Bp=Bp,Bn=Bn,lambda1=lambda1,lambda2=lambda2,XtX=XtX)
        if type(X) is dict:
            loss = 0
            for m,key in enumerate(X):
                loss += np.linalg.norm(X[key]-X[key]@(Bp-Bn)@S[m])**2
        else:
            loss = np.linalg.norm(X-X@(Bp-Bn)@S)**2
        all_loss.append(loss.item())

        if epoch>100:
            if all_loss[-5] - loss < tol:
                break

    print("Tolerance reached at " + str(epoch) + " number of iterations")
    best_loss = min(all_loss)
    return all_loss, best_loss
