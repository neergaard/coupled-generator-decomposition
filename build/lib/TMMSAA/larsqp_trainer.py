# optimization loop
from tqdm import tqdm
import numpy as np
import cvxopt
cvxopt.solvers.options['show_progress'] = False

def Optimizationloop(X,num_comp,lambda1,lambda2,max_iter=100, tol=1e-10,Bp_init=None,Bn_init=None):
    
    all_loss = []
    if type(X) is dict: #if input data is multimodal
        T = next(iter(X.values())).shape[-1]
        S_shape = (len(X),*next(iter(X.values())).shape)
        S_shape = (*S_shape[:-2],num_comp,T)
        S = np.zeros(S_shape)
        
        XtX_shape = (*S_shape[:-2],T,T)
        XtX = np.zeros(XtX_shape)
        for m,key in enumerate(X):
            X[key] = np.array(X[key])
            XtX[m] = np.swapaxes(X[key],-2,-1)@X[key]
        # the sum is taken over modalities and subjects
        if XtX.ndim==4:
            XtX_sum = np.sum(XtX,axis=(0,1))
        elif XtX.ndim==3:
            XtX_sum = np.sum(XtX,axis=0)
        
    elif np.array(X).ndim==3:
        T = X.shape[-1]
        X = np.array(X)
        XtX = np.swapaxes(X,-2,-1)@X
        XtX_sum = np.sum(XtX,axis=0)
    elif np.array(X).ndim==2:
        T = X.shape[-1]
        X = np.array(X)
        XtX = X.T@X
        XtX_sum = XtX
    
    XtX_lambda2 = 2*XtX_sum + lambda2*np.eye(T)
    XtX_minus = -2*XtX_sum
    P = np.concatenate((XtX_lambda2,XtX_minus),axis=1)
    P = cvxopt.matrix(np.concatenate((P,np.concatenate((XtX_minus,XtX_lambda2),axis=-1)),axis=-2))
    G = cvxopt.matrix(-np.eye(2*T))
    h = cvxopt.matrix(np.zeros(2*T))

    if Bp_init is None and Bn_init is None:
        Bp = np.random.uniform(size=(T, num_comp))
        Bn = np.random.uniform(size=(T, num_comp))
    else:
        Bp = Bp_init 
        Bn = Bn_init
        
    for epoch in tqdm(range(max_iter)):
        initvals = cvxopt.matrix(np.concatenate((Bp,Bn),axis=0))

        if type(X) is dict:
            for m,key in enumerate(X):
                U,Sigma,Vt = np.linalg.svd(XtX[m] @ (Bp-Bn),full_matrices=False)
                S[m] = np.swapaxes(U@Vt,-2, -1)
            if XtX.ndim==4:
                SXtX = 2*np.sum(S@XtX,axis=(0,1))
            elif XtX.ndim==3:
                SXtX = 2*np.sum(S@XtX,axis=0)
        elif np.array(X).ndim==3:
            U,Sigma,Vt = np.linalg.svd(XtX@(Bp-Bn),full_matrices=False)
            S = np.swapaxes(U@Vt,-2, -1)
            SXtX = 2*np.sum(S@XtX,axis=0)
        elif np.array(X).ndim==2:
            U,Sigma,Vt = np.linalg.svd(XtX@(Bp-Bn),full_matrices=False)
            S = np.swapaxes(U@Vt,-2, -1)
            SXtX = 2*S@XtX

        Q = cvxopt.matrix(-np.concatenate((SXtX,-SXtX),axis=1)+lambda1) #vektor, 2T lang
        for k in range(num_comp):
            sol = np.array(cvxopt.solvers.qp(P,Q[k,:].T,G,h,initvals=initvals[:,k])['x']).reshape(2*T,)
            #sol = quadprog.solve_qp(P,-Q[k],-G,h)[0]
            Bp[:,k] = sol[:T] #første halvdel
            Bn[:,k] = sol[T:] #anden halvdel

        if type(X) is dict:
            loss = 0
            for m,key in enumerate(X):
                loss += np.linalg.norm(X[key]-X[key]@(Bp-Bn)@S[m])**2
        else:
            loss = np.linalg.norm(X-X@(Bp-Bn)@S)**2
        all_loss.append(loss.item())

        if epoch>5:
            if all_loss[-5] - loss < tol:
                break

    print("Tolerance reached at " + str(epoch) + " number of iterations")
    return all_loss,Bp,Bn,S
def larsqp_eval(X,Xtest,Bp,Bn,num_comp):

    if type(X) is dict: #if input data is multimodal
        T = next(iter(X.values())).shape[-1]
        S_shape = (len(X),*next(iter(X.values())).shape)
        S_shape = (*S_shape[:-2],num_comp,T)
        S = np.zeros(S_shape)
        
        XtX_shape = (*S_shape[:-2],T,T)
        XtX = np.zeros(XtX_shape)
        for m,key in enumerate(X):
            XtX[m] = np.swapaxes(X[key],-2,-1)@X[key]
            U,Sigma,Vt = np.linalg.svd(XtX[m] @ (Bp-Bn),full_matrices=False)
            S[m] = np.swapaxes(U@Vt,-2, -1)
        
    else:
        XtX = np.swapaxes(X,-2,-1)@X
        U,Sigma,Vt = np.linalg.svd(XtX@(Bp-Bn),full_matrices=False)
        S = np.swapaxes(U@Vt,-2, -1)

    if type(X) is dict:
        loss = 0
        for m,key in enumerate(X):
            loss += np.sum(np.linalg.norm(Xtest[key]-np.array(X[key])@(Bp-Bn)@S[m],axis=(-2,-1))**2)
    else:
        loss = np.sum(np.linalg.norm(Xtest-X@(Bp-Bn)@S,axis=(-2,-1))**2)
    return loss