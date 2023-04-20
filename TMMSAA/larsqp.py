import numpy as np
import cvxopt
cvxopt.solvers.options['show_progress'] = False
#import quadprog


def larsqp(X,Bp,Bn,lambda1,lambda2,XtX=None):
    T,K = Bp.shape #number of timepoints and components

    if type(X) is dict: #if input data is multimodal
        S_shape = (len(X),*next(iter(X.values())).shape)
        S_shape = (*S_shape[:-2],K,T)
        S = np.zeros(S_shape)
        
        if XtX is None:
            XtX_shape = (*S_shape[:-2],T,T)
            XtX = np.zeros(XtX_shape)
            for m,key in enumerate(X):
                XtX[m] = np.swapaxes(X[key],-2,-1)@X[key]

        for m,key in enumerate(X):
            U,Sigma,Vt = np.linalg.svd(XtX[m] @ (Bp-Bn),full_matrices=False)
            S[m] = np.swapaxes(U@Vt,-2, -1)
            

        # the sum is taken over modalities and subjects
        XtX_sum = np.sum(XtX,axis=(0,1))
        SXtX = 2*np.sum(S@XtX,axis=(0,1))
        
    else:
        if XtX is None:
            XtX = np.swapaxes(X,-2,-1)@X
        U,Sigma,Vt = np.linalg.svd(XtX@(Bp-Bn),full_matrices=False)
        S = np.swapaxes(U@Vt,-2, -1)

        XtX_sum = np.sum(XtX,axis=0)
        SXtX = 2*np.sum(S@XtX,axis=0)
    
    XtX_lambda2 = 2*XtX_sum + lambda2*np.eye(T)
    XtX_minus = -2*XtX_sum
    #SXtX = 2*S@XtX


    P = np.concatenate((XtX_lambda2,XtX_minus),axis=1)
    P = cvxopt.matrix(np.concatenate((P,np.concatenate((XtX_minus,XtX_lambda2),axis=-1)),axis=-2))
    #P = np.concatenate((P,np.concatenate((XtX_minus,XtX_lambda2),axis=-1)),axis=-2)

    Q = -np.concatenate((SXtX,-SXtX),axis=1)+lambda1 #vektor, 2T lang
    G = cvxopt.matrix(-np.eye(2*T))
    h = cvxopt.matrix(np.zeros(2*T))
    #G = -np.eye(2*T)
    #h = np.zeros(2*T)
    for k in range(K):
        sol = np.array(cvxopt.solvers.qp(P,cvxopt.matrix(Q[k,:].T),G,h)['x']).reshape(2*T,)
        #sol = quadprog.solve_qp(P,-Q[k],-G,h)[0]
        Bp[:,k] = sol[:T] #første halvdel
        Bn[:,k] = sol[T:] #anden halvdel
    return Bp, Bn, S